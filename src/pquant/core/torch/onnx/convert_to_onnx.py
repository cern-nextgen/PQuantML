"""Convert a PQuant torch model to ONNX or QONNX format."""

import operator

import onnx
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from pquant.core.onnx_common import (
    DEFAULT_TARGET,
    ExportOptions,
    GraphBuilder,
    add_activation,
    add_getitem,
    normalize_input_shapes,
    resolve_export_options,
    tensor_info,
)
from pquant.core.torch.activations import PQActivation
from pquant.core.torch.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQLayerNorm,
    PQMultiheadAttention,
)
from pquant.core.torch.onnx.layer_builders import (
    _add_avgpool,
    _add_batchnorm,
    _add_conv,
    _add_dense,
    _add_dense_integer,
    _add_layernorm,
    _add_maxpool,
    _add_mha,
    _add_upsample,
)
from pquant.core.torch.quantizer import Quantizer


class PQTracer(fx.Tracer):
    """Keeps the PQuant layers as leaves so they are exported as a whole instead of being traced into."""

    _LEAF_TYPES = (
        PQDense,
        PQConv2d,
        PQConv1d,
        PQBatchNorm1d,
        PQBatchNorm2d,
        PQLayerNorm,
        PQAvgPool1d,
        PQAvgPool2d,
        PQMultiheadAttention,
        PQActivation,
        Quantizer,
    )

    def is_leaf_module(self, m: nn.Module, qualname: str) -> bool:
        return isinstance(m, self._LEAF_TYPES) or super().is_leaf_module(m, qualname)


def _const_ints(args) -> list[int]:
    """Constant int arguments given as ``f(d0, d1, ...)`` or as ``f([d0, d1, ...])``."""
    values = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    if not all(isinstance(v, int) for v in values):
        raise TypeError("FX ONNX export: shape/dim arguments must be constant ints")
    return list(values)


class FxGraphEmitter:
    """Translate a shape-propagated fx.Graph into ONNX nodes and initializers."""

    def __init__(self, gm, g: GraphBuilder, input_names: dict):
        self.gm = gm
        self.g = g
        self.names: dict[fx.Node, str | tuple] = dict(input_names)
        self.outputs: list[tuple] = []

    def run(self):
        for node in self.gm.graph.nodes:
            if node.op == "get_attr":
                self._emit_get_attr(node)
            elif node.op == "call_module":
                self.names[node] = self._emit_module(node, self.gm.get_submodule(node.target))
            elif node.op in ("call_function", "call_method"):  # the tensor is args[0] either way
                self._emit_call(node)
            elif node.op == "output":
                self._collect_outputs(node)
        return self.outputs

    def _name(self, arg) -> str:
        if not isinstance(arg, fx.Node):
            raise TypeError(f"FX ONNX export: expected a tensor, got {type(arg).__name__}")
        return self.names[arg]

    def _shape(self, node: fx.Node) -> tuple:
        meta = node.meta.get("tensor_meta")
        if not hasattr(meta, "shape"):
            raise RuntimeError(f"FX ONNX export: ShapeProp did not produce tensor_meta for {node.name!r}")
        return tuple(meta.shape)

    def _rank(self, node: fx.Node) -> int:
        return len(self._shape(node))

    def _arg(self, node, pos, name, default=None):
        """A call argument given either positionally or by keyword."""
        return node.args[pos] if len(node.args) > pos else node.kwargs.get(name, default)

    def _add(self, node, op_type, inputs, **attrs):
        self.names[node] = self.g.node(op_type, inputs, f"{node.name}_{op_type.lower()}", **attrs)

    def _emit_get_attr(self, node):
        value = operator.attrgetter(node.target)(self.gm)
        if isinstance(value, torch.Tensor):
            self.g.initializer(node.name, value.detach().cpu().numpy())
        self.names[node] = node.name

    def _emit_module(self, node, module):
        """Emit a PQuant or standard torch.nn leaf module.  Returns its output name(s)."""
        g, prefix = self.g, node.name.replace(".", "_")
        if isinstance(module, PQMultiheadAttention):
            return self._emit_mha(node, module, prefix)
        x, rank = self._name(node.args[0]), self._rank(node.args[0])
        if isinstance(module, PQDense):
            if g.options.integer_ops:
                return _add_dense_integer(g, module, prefix, x)
            return _add_dense(g, module, prefix, x, rank)
        if isinstance(module, (PQConv2d, PQConv1d)):
            return _add_conv(g, module, prefix, x)
        if isinstance(module, (PQBatchNorm2d, PQBatchNorm1d)):
            return _add_batchnorm(g, module, prefix, x)
        if isinstance(module, PQLayerNorm):
            return _add_layernorm(g, module, prefix, x)
        if isinstance(module, (PQAvgPool2d, PQAvgPool1d)):
            return _add_avgpool(g, module, prefix, x, ndim=rank - 2)
        if isinstance(module, PQActivation):
            return add_activation(g, module, prefix, x)
        if isinstance(module, Quantizer):
            return g.quantize(module, prefix, x)
        if isinstance(module, nn.ReLU):
            return g.node("Relu", [x], f"{prefix}_relu")
        if isinstance(module, nn.LeakyReLU):
            return g.node("LeakyRelu", [x], f"{prefix}_leakyrelu", alpha=module.negative_slope)
        if isinstance(module, nn.Flatten):
            return self._flatten(node, prefix, module.start_dim, module.end_dim)
        if isinstance(module, nn.MaxPool2d):
            return _add_maxpool(g, module, prefix, x)
        if isinstance(module, nn.Upsample):
            return _add_upsample(g, module, prefix, x)
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            return x  # identity at inference
        raise TypeError(f"Unsupported module type for ONNX export: {type(module).__name__}")

    def _emit_mha(self, node, module, prefix) -> tuple:
        # forward(query, key, value, key_padding_mask=None, attn_mask=None, ...); the masks must be tensors if given
        params = ("query", "key", "value", "key_padding_mask", "attn_mask")
        args = [self._arg(node, pos, name) for pos, name in enumerate(params)]
        return _add_mha(self.g, module, prefix, *(arg if arg is None else self._name(arg) for arg in args))

    def _emit_call(self, node):
        target = node.target
        if target in self._BINARY_OPS:
            operands = [self._operand(node, idx, arg) for idx, arg in enumerate(node.args[:2])]
            self._add(node, self._BINARY_OPS[target], operands)
        elif target in self._UNARY_OPS:
            self._add(node, self._UNARY_OPS[target], [self._name(node.args[0])])
        elif target in self._HANDLERS:
            self._HANDLERS[target](self, node)
        else:
            raise TypeError(f"Unsupported {node.op} for FX ONNX export: {target!r}")

    def _operand(self, node, idx, arg) -> str:
        """Name of a binary-op operand; Python scalars become float constants."""
        if isinstance(arg, (int, float, bool)):
            return self.g.float_scalar(f"{node.name}_arg{idx}_const", float(arg))
        return self._name(arg)

    def _collect_outputs(self, node):
        returned = node.args[0]
        for ret in returned if isinstance(returned, (tuple, list)) else [returned]:
            if not isinstance(ret, fx.Node):
                raise TypeError("FX ONNX export: unsupported (non-tensor) model output")
            # A module with several outputs (PQMultiheadAttention) returned as is contributes all of them.
            names, metas = self.names[ret], ret.meta["tensor_meta"]
            if not isinstance(names, tuple):
                names, metas = (names,), (metas,)
            # meta is None for an output the module did not compute (attention weights with need_weights=False)
            self.outputs += [(name, meta) for name, meta in zip(names, metas) if meta is not None]

    def _emit_getitem(self, node):
        value, index = self.names[node.args[0]], node.args[1]
        if isinstance(value, tuple):  # unpacking a multi-output module
            self.names[node] = value[index]
        else:  # tensor slicing: x[:, 0], x[..., :4], ... → Slice (+ Squeeze)
            self.names[node] = add_getitem(self.g, node.name, value, index, self._rank(node.args[0]))

    def _emit_transpose(self, node):
        # transpose(d0, d1) swaps two dims; ONNX needs the full permutation.
        perm = list(range(self._rank(node.args[0])))
        d0, d1 = self._arg(node, 1, "dim0"), self._arg(node, 2, "dim1")
        perm[d0], perm[d1] = perm[d1], perm[d0]
        self._add(node, "Transpose", [self._name(node.args[0])], perm=perm)

    def _emit_permute(self, node):
        rank = self._rank(node.args[0])
        self._add(node, "Transpose", [self._name(node.args[0])], perm=[d % rank for d in _const_ints(node.args[1:])])

    def _emit_cat(self, node):
        self._add(node, "Concat", [self._name(t) for t in node.args[0]], axis=int(self._arg(node, 1, "dim", 0)))

    def _emit_flatten(self, node):
        start, end = self._arg(node, 1, "start_dim", 0), self._arg(node, 2, "end_dim", -1)
        self.names[node] = self._flatten(node, node.name, start, end)

    def _flatten(self, node, prefix, start_dim, end_dim):
        x, rank = self._name(node.args[0]), self._rank(node.args[0])
        if start_dim % rank == 1 and end_dim % rank == rank - 1:
            return self.g.node("Flatten", [x], f"{prefix}_flatten", axis=1)
        # ONNX Flatten always yields a rank-2 tensor, so any other range is a Reshape to the traced shape.
        return self._reshape(prefix, x, [-1, *self._shape(node)[1:]])

    def _emit_reshape(self, node):
        self.names[node] = self._reshape(node.name, self._name(node.args[0]), _const_ints(node.args[1:]))

    def _reshape(self, prefix, x, shape):
        return self.g.node("Reshape", [x, self.g.int64(f"{prefix}_shape", shape)], f"{prefix}_reshape")

    def _emit_squeeze(self, node):
        shape, dim = self._shape(node.args[0]), self._arg(node, 1, "dim")
        if dim is None:
            axes = [axis for axis, size in enumerate(shape) if size == 1 and axis != 0]  # never the batch axis
        else:
            axes = [dim % len(shape)] if shape[dim] == 1 else []
        self.names[node] = self.g.squeeze(node.name, self._name(node.args[0]), axes)

    def _emit_unsqueeze(self, node):
        axis = self._arg(node, 1, "dim") % (self._rank(node.args[0]) + 1)
        self.names[node] = self.g.unsqueeze(node.name, self._name(node.args[0]), [axis])

    _BINARY_OPS = {
        torch.add: "Add",
        operator.add: "Add",
        operator.iadd: "Add",
        torch.mul: "Mul",
        operator.mul: "Mul",
        torch.sub: "Sub",
        operator.sub: "Sub",
        operator.isub: "Sub",
        torch.div: "Div",
        operator.truediv: "Div",
        operator.itruediv: "Div",
        torch.matmul: "MatMul",
        operator.matmul: "MatMul",
        "matmul": "MatMul",
    }
    _UNARY_OPS = {
        F.relu: "Relu",
        torch.relu: "Relu",
        "relu": "Relu",
        F.sigmoid: "Sigmoid",
        torch.sigmoid: "Sigmoid",
    }
    # torch functions and Tensor method names (str) with a dedicated translation
    _HANDLERS = {
        operator.getitem: _emit_getitem,
        torch.transpose: _emit_transpose,
        "transpose": _emit_transpose,
        torch.permute: _emit_permute,
        "permute": _emit_permute,
        torch.cat: _emit_cat,
        torch.flatten: _emit_flatten,
        "flatten": _emit_flatten,
        torch.reshape: _emit_reshape,
        "reshape": _emit_reshape,
        "view": _emit_reshape,
        torch.squeeze: _emit_squeeze,
        "squeeze": _emit_squeeze,
        torch.unsqueeze: _emit_unsqueeze,
        "unsqueeze": _emit_unsqueeze,
    }


def _prune_trace_artifacts(gm):
    """Remove what specializing ``concrete_args`` leaves behind: assertions, dead comparisons, unused placeholders."""
    for node in reversed(list(gm.graph.nodes)):  # users before the values they consume
        if node.op == "call_function":
            is_assert = getattr(node.target, "__name__", "").startswith("_assert")
            removable = is_assert or (node.target is operator.eq and not node.users)
        else:
            removable = node.op == "placeholder" and node.args and not node.users  # has a baked default, now unused
        if removable:
            gm.graph.erase_node(node)
    gm.recompile()


def _graph_input_names(gm, n_expected: int) -> dict:
    """Map each tensor placeholder to its ONNX graph-input name."""
    placeholders = list(gm.graph.find_nodes(op="placeholder"))
    if len(placeholders) != n_expected:
        raise ValueError(
            f"FX export: model.forward has {len(placeholders)} tensor input(s) but "
            f"input_shape describes {n_expected}.  Specialize non-tensor "
            f"arguments via concrete_args={{...}}."
        )
    # A single input is named "input"; multiple inputs are named after their forward parameters.
    names = ["input"] if n_expected == 1 else [str(p.target) for p in placeholders]
    return dict(zip(placeholders, names))


def _input_dtypes(input_dtypes, n: int) -> list[torch.dtype]:
    """One torch dtype per graph input (float32 by default)."""
    if not isinstance(input_dtypes, (list, tuple)):
        input_dtypes = [input_dtypes or torch.float32] * n
    if len(input_dtypes) != n:
        raise ValueError(f"input_dtypes has {len(input_dtypes)} entries but there are {n} input(s)")
    dtypes = [getattr(torch, d, None) if isinstance(d, str) else d for d in input_dtypes]
    for given, dtype in zip(input_dtypes, dtypes):
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported input dtype {given!r}; expected a torch.dtype or its name, e.g. 'float32'")
    return dtypes


def _np_dtype(dtype: torch.dtype):
    return torch.empty(0, dtype=dtype, device="cpu").numpy().dtype


def convert_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: str = "model.onnx",
    target: str | ExportOptions = DEFAULT_TARGET,
    *,
    opset: int | None = None,
    include_clip: bool | None = None,
    batch_size: int | str | None = None,
    strict: bool | None = None,
    concrete_args: dict | None = None,
    input_dtypes=None,
) -> onnx.ModelProto:
    """
    Convert a PQuant nn.Module to ONNX or QONNX using torch.fx symbolic tracing.

    Args:
        model:          Trained nn.Module with apply_final_compression() called on its PQ modules.
        input_shape:    Shape of one sample without the batch dim, or one shape per input for a multi-input model.
        output_path:    Where to save the .onnx file.
        target:         Name from ``TARGETS`` (see ``list_targets()``) or an ``ExportOptions``; sets the graph format.
        opset:          ONNX opset version, at least 13; overrides the target's.
        include_clip:   Whether a Clip precedes each QuantizeLinear; overrides the target's.
        batch_size:     Fixed batch dimension, or ``"dynamic"``; overrides the target's.
        strict:         Turn export-check warnings into errors.
        concrete_args:  Non-tensor ``forward`` arguments to bake in as constants, by parameter name.
        input_dtypes:   torch.dtype (or its name) of each input; float32 by default.

    Returns:
        The constructed onnx.ModelProto.
    """
    options = resolve_export_options(target, include_clip=include_clip, opset=opset, batch_size=batch_size, strict=strict)
    model.eval()
    g = GraphBuilder(options)
    input_shapes = normalize_input_shapes(input_shape)
    dtypes = _input_dtypes(input_dtypes, len(input_shapes))

    gm = fx.GraphModule(model, PQTracer().trace(model, concrete_args=concrete_args))
    _prune_trace_artifacts(gm)
    input_names = _graph_input_names(gm, len(input_shapes))

    # Run one sample through the graph so that every node knows the shape of its value.
    device = next((p.device for p in model.parameters()), None)
    probes = [torch.zeros(1, *shape, device=device, dtype=dtype) for shape, dtype in zip(input_shapes, dtypes)]
    with torch.no_grad():
        ShapeProp(gm).propagate(*probes)

    outputs = FxGraphEmitter(gm, g, input_names).run()
    input_infos = [
        tensor_info(name, _np_dtype(dtype), shape, options.batch_size)
        for name, shape, dtype in zip(input_names.values(), input_shapes, dtypes)
    ]
    output_infos = [tensor_info(name, _np_dtype(meta.dtype), meta.shape[1:], options.batch_size) for name, meta in outputs]
    return g.save("pquant_onnx_fx", input_infos, output_infos, output_path)
