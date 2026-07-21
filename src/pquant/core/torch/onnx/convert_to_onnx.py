"""
Convert a PQuant model to ONNX or QONNX format.

Pass ``use_qonnx=True`` to emit QONNX ``Quant`` custom nodes (requires the
qonnx runtime).  Pass ``use_qonnx=False`` (default) to emit standard
``Clip + QuantizeLinear + DequantizeLinear`` nodes runnable with plain
onnxruntime.
"""

import functools
import logging
import operator
import os

import onnx
import onnx.helper as oh
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto
from torch.fx.passes.shape_prop import ShapeProp

os.environ["KERAS_BACKEND"] = "torch"  # must be set before any keras/pquant import

from pquant.core.onnx_common import (  # noqa: E402
    add_float_scalar,
    add_initializer,
    add_int64_array,
    apply_quantizer,
    emit_getitem,
    emit_squeeze,
    emit_unsqueeze,
    qdq_node,
    quant_node,
    save_model,
)
from pquant.core.torch.activations import PQActivation  # noqa: E402
from pquant.core.torch.layers import (  # noqa: E402
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
from pquant.core.torch.onnx.layer_builders import (  # noqa: E402
    _add_activation,
    _add_avgpool,
    _add_batchnorm,
    _add_conv,
    _add_dense,
    _add_dense_nd,
    _add_layernorm,
    _add_maxpool,
    _add_mha,
    _add_upsample,
)
from pquant.core.torch.quantizer import Quantizer  # noqa: E402


def _emit_module(
    module,
    prefix,
    current,
    nodes,
    initializers,
    quant_fn,
    use_qonnx,
    store_integer_weights,
    integer_ops=False,
    input_rank=None,
):
    """Emit ONNX nodes for a single PQuant or standard torch.nn module."""
    if isinstance(module, PQDense):
        # Gemm only accepts rank-2 inputs; higher ranks (e.g. [batch, seq, dim]) go through
        # MatMul + Add.  The integer_ops path is already MatMul-based and handles any rank.
        if input_rank is not None and input_rank > 2 and not (integer_ops and not use_qonnx):
            return _add_dense_nd(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
        return _add_dense(
            module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops
        )
    if isinstance(module, (PQConv2d, PQConv1d)):
        ndim = 2 if isinstance(module, PQConv2d) else 1
        return _add_conv(
            module,
            prefix,
            current,
            nodes,
            initializers,
            ndim=ndim,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )
    if isinstance(module, (PQBatchNorm2d, PQBatchNorm1d)):
        return _add_batchnorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
    if isinstance(module, PQLayerNorm):
        return _add_layernorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
    if isinstance(module, (PQAvgPool2d, PQAvgPool1d)):
        ndim = 2 if isinstance(module, PQAvgPool2d) else 1
        return _add_avgpool(module, prefix, current, nodes, initializers, ndim=ndim, quant_fn=quant_fn)
    if isinstance(module, PQActivation):
        return _add_activation(module, prefix, current, nodes, initializers, quant_fn)
    if isinstance(module, Quantizer):
        return apply_quantizer(module, prefix, current, nodes, initializers, quant_fn)
    if isinstance(module, nn.ReLU):
        out = f"{prefix}_relu"
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[out]))
        return out
    if isinstance(module, nn.LeakyReLU):
        out = f"{prefix}_leakyrelu"
        nodes.append(oh.make_node("LeakyRelu", inputs=[current], outputs=[out], alpha=module.negative_slope))
        return out
    if isinstance(module, nn.Flatten):
        out = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[current], outputs=[out], axis=module.start_dim))
        return out
    if isinstance(module, nn.MaxPool2d):
        return _add_maxpool(module, prefix, current, nodes)
    if isinstance(module, nn.Upsample):
        return _add_upsample(module, prefix, current, nodes, initializers)
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        return current  # identity at inference
    raise TypeError(f"Unsupported module type for ONNX export: {type(module).__name__}")


class PQTracer(fx.Tracer):
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


def _normalize_input_shapes(input_shape) -> list[tuple]:
    seq = list(input_shape)
    if len(seq) > 0 and all(isinstance(s, (list, tuple)) for s in seq):
        return [tuple(int(d) for d in s) for s in seq]
    return [tuple(int(d) for d in seq)]


def _normalize_input_dtypes(input_dtypes, n: int):
    torch_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "bool": torch.bool,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    tp_map = {
        torch.float32: TensorProto.FLOAT,
        torch.bool: TensorProto.BOOL,
        torch.int64: TensorProto.INT64,
        torch.int32: TensorProto.INT32,
    }

    if input_dtypes is None:
        items = [torch.float32] * n
    elif isinstance(input_dtypes, (list, tuple)):
        items = list(input_dtypes)
    else:
        items = [input_dtypes] * n

    if len(items) != n:
        raise ValueError(f"input_dtypes has {len(items)} entries but there are {n} input(s)")

    torch_dtypes, tp_dtypes = [], []
    for d in items:
        td = torch_map[d] if isinstance(d, str) else d
        if td not in tp_map:
            raise ValueError(f"Unsupported input dtype {d!r}; expected one of {list(torch_map)}")
        torch_dtypes.append(td)
        tp_dtypes.append(tp_map[td])
    return torch_dtypes, tp_dtypes


def _swap_perm(rank: int, d0: int, d1: int) -> list[int]:
    perm = list(range(rank))
    a, b = d0 % rank, d1 % rank
    perm[a], perm[b] = perm[b], perm[a]
    return perm


def _resolve_perm_dims(args, rank: int) -> list[int]:
    # Accept both permute(d0, d1, ...) and permute([d0, d1, ...]) shapes.
    dims = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return [int(d) % rank for d in dims]


class FxGraphEmitter:
    """Translate a shape-propagated fx.Graph into ONNX nodes and initializers.

    ``node_to_name`` maps each fx.Node to the name of the ONNX tensor holding
    its value — or, for modules with multiple outputs (PQMultiheadAttention),
    a tuple of names.
    """

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
    }
    _UNARY_OPS = {
        F.relu: "Relu",
        torch.relu: "Relu",
        F.sigmoid: "Sigmoid",
        torch.sigmoid: "Sigmoid",
    }

    def __init__(self, gm, ph_to_name, quant_fn, use_qonnx, store_integer_weights, integer_ops):
        self.gm = gm
        self.ph_to_name = ph_to_name
        self.quant_fn = quant_fn
        self.use_qonnx = use_qonnx
        self.store_integer_weights = store_integer_weights
        self.integer_ops = integer_ops
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.node_to_name: dict[fx.Node, str] = {}
        self.output_names: list[str] = []

    def _run(self) -> list[str]:
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self.node_to_name[node] = self.ph_to_name[node]
            elif node.op == "get_attr":
                self._emit_get_attr(node)
            elif node.op == "call_module":
                self._emit_call_module(node)
            elif node.op == "call_function":
                self._emit_call_function(node)
            elif node.op == "call_method":
                self._emit_call_method(node)
            elif node.op == "output":
                self._collect_outputs(node)
        return self.output_names

    def _name_of(self, arg) -> str:
        if isinstance(arg, fx.Node):
            return self.node_to_name[arg]
        raise TypeError(f"Expected fx.Node, got {type(arg)}")

    def _binop_inputs(self, node: fx.Node) -> list[str]:
        names: list[str] = []
        for idx, arg in enumerate(node.args[:2]):
            if isinstance(arg, fx.Node):
                names.append(self.node_to_name[arg])
            elif isinstance(arg, (int, float, bool)):
                names.append(add_float_scalar(self.initializers, f"{node.name}_arg{idx}_const", float(arg)))
            else:
                raise TypeError(f"FX export: unsupported binary-op arg type {type(arg).__name__}")
        return names

    def _node_shape(self, node: fx.Node) -> tuple:
        meta = node.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            raise RuntimeError(f"FX export: ShapeProp did not produce tensor_meta for {node.name!r}")
        return tuple(meta.shape)

    def _node_rank(self, node: fx.Node) -> int:
        return len(self._node_shape(node))

    def _emit_get_attr(self, node):
        obj = self.gm
        for part in node.target.split("."):
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor):
            add_initializer(self.initializers, node.name, obj.detach().cpu().numpy())
        self.node_to_name[node] = node.name

    def _emit_call_module(self, node):
        module = self.gm.get_submodule(node.target)
        prefix = node.name.replace(".", "_")
        if isinstance(module, PQMultiheadAttention):
            self.node_to_name[node] = self._emit_mha_module(module, node, prefix)
            return
        self.node_to_name[node] = _emit_module(
            module,
            prefix,
            self._name_of(node.args[0]),
            self.nodes,
            self.initializers,
            self.quant_fn,
            self.use_qonnx,
            self.store_integer_weights,
            self.integer_ops,
            input_rank=self._node_rank(node.args[0]),
        )

    def _emit_mha_module(self, module, node, prefix) -> tuple:
        # forward(query, key, value, key_padding_mask=None, attn_mask=None, ...)
        q_name = self._name_of(node.args[0])
        k_name = self._name_of(node.args[1]) if len(node.args) > 1 else q_name
        v_name = self._name_of(node.args[2]) if len(node.args) > 2 else q_name
        kpm_name = self._mask_name(node, 3, "key_padding_mask")
        attn_mask_name = self._mask_name(node, 4, "attn_mask")
        return _add_mha(
            module,
            prefix,
            q_name,
            k_name,
            v_name,
            self.nodes,
            self.initializers,
            self.quant_fn,
            self.use_qonnx,
            self.store_integer_weights,
            key_padding_mask=kpm_name,
            attn_mask=attn_mask_name,
        )

    def _mask_name(self, node, pos, kw):
        arg = node.args[pos] if len(node.args) > pos else node.kwargs.get(kw)
        if arg is None:
            return None
        if not isinstance(arg, fx.Node):
            raise TypeError(f"FX ONNX export: MHA {kw} must be a tensor (constant or input), got {type(arg)}")
        return self.node_to_name[arg]

    def _emit_call_function(self, node):
        fn = node.target
        if fn is torch._assert or getattr(fn, "__name__", "") == "_assert" or fn is operator.eq:
            return  # trace artifacts with no runtime effect
        if fn is operator.getitem:
            self._emit_getitem(node)
        elif fn in self._BINARY_OPS:
            self._add_simple_node(node, self._BINARY_OPS[fn], self._binop_inputs(node))
        elif fn in self._UNARY_OPS:
            self._add_simple_node(node, self._UNARY_OPS[fn], [self._name_of(node.args[0])])
        elif fn is torch.transpose:
            self._emit_transpose(node)
        elif fn is torch.permute:
            self._emit_permute(node)
        elif fn is torch.cat:
            tensors = [self._name_of(a) for a in node.args[0]]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            self._add_simple_node(node, "Concat", tensors, axis=int(dim))
        elif fn is torch.flatten:
            self._emit_flatten(node, default_start_dim=0)
        elif fn is torch.squeeze:
            self._emit_squeeze(node)
        elif fn is torch.unsqueeze:
            self._emit_unsqueeze(node)
        else:
            raise TypeError(f"Unsupported call_function for FX ONNX export: {fn}")

    def _emit_call_method(self, node):
        method = node.target
        if method == "relu":
            self._add_simple_node(node, "Relu", [self._name_of(node.args[0])])
        elif method == "flatten":
            self._emit_flatten(node, default_start_dim=1)
        elif method in ("view", "reshape"):
            self._emit_reshape(node)
        elif method == "transpose":
            self._emit_transpose(node)
        elif method == "permute":
            self._emit_permute(node)
        elif method == "matmul":
            self._add_simple_node(node, "MatMul", self._binop_inputs(node))
        elif method == "squeeze":
            self._emit_squeeze(node)
        elif method == "unsqueeze":
            self._emit_unsqueeze(node)
        else:
            raise TypeError(f"Unsupported call_method for FX ONNX export: {node.target!r}")

    def _collect_outputs(self, node):
        ret = node.args[0]
        rets = list(ret) if isinstance(ret, (tuple, list)) else [ret]
        for r in rets:
            if not isinstance(r, fx.Node):
                raise TypeError("FX ONNX export: unsupported (non-tensor) model output")
            val = self.node_to_name[r]
            # MHA nodes store a tuple (out, avg_attn); expose the attention output.
            self.output_names.append(val[0] if isinstance(val, tuple) else val)

    def _add_simple_node(self, node, op_type, inputs, **attrs):
        out = f"{node.name}_{op_type.lower()}"
        self.nodes.append(oh.make_node(op_type, inputs=inputs, outputs=[out], **attrs))
        self.node_to_name[node] = out

    def _emit_getitem(self, node):
        container = self.node_to_name[node.args[0]]
        if isinstance(container, tuple):
            # Unpack a tuple output (e.g. from PQMultiheadAttention).
            self.node_to_name[node] = container[node.args[1]]
        else:
            # Tensor slicing: x[:, 0], x[..., :4], ... → Slice (+ Squeeze)
            rank = self._node_rank(node.args[0])
            self.node_to_name[node] = emit_getitem(node.name, container, node.args[1], rank, self.nodes, self.initializers)

    def _emit_transpose(self, node):
        # torch.transpose(t, d0, d1) swaps two dims; ONNX needs a full perm.
        perm = _swap_perm(self._node_rank(node.args[0]), int(node.args[1]), int(node.args[2]))
        self._add_simple_node(node, "Transpose", [self._name_of(node.args[0])], perm=perm)

    def _emit_permute(self, node):
        perm = _resolve_perm_dims(node.args[1:], self._node_rank(node.args[0]))
        self._add_simple_node(node, "Transpose", [self._name_of(node.args[0])], perm=perm)

    def _emit_flatten(self, node, default_start_dim):
        start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", default_start_dim)
        self._add_simple_node(node, "Flatten", [self._name_of(node.args[0])], axis=int(start_dim))

    def _emit_reshape(self, node):
        shape_vals = []
        for a in node.args[1:]:
            if not isinstance(a, int):
                raise TypeError("Dynamic reshape (non-constant shape) is not supported in FX ONNX export")
            shape_vals.append(a)
        shape_name = add_int64_array(self.initializers, f"{node.name}_shape", shape_vals)
        self._add_simple_node(node, "Reshape", [self._name_of(node.args[0]), shape_name])

    def _emit_squeeze(self, node):
        axes = self._squeeze_axes(node)
        self.node_to_name[node] = emit_squeeze(node.name, self._name_of(node.args[0]), axes, self.nodes, self.initializers)

    def _squeeze_axes(self, node) -> list[int]:
        """Resolve the axes a torch squeeze()/.squeeze() call removes."""
        in_shape = self._node_shape(node.args[0])
        if len(node.args) > 1 or "dim" in node.kwargs:
            dim = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["dim"])
            dim %= len(in_shape)
            return [dim] if in_shape[dim] == 1 else []
        return [i for i, s in enumerate(in_shape) if s == 1 and i != 0]

    def _emit_unsqueeze(self, node):
        dim = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["dim"])
        axes = [dim % (self._node_rank(node.args[0]) + 1)]
        self.node_to_name[node] = emit_unsqueeze(node.name, self._name_of(node.args[0]), axes, self.nodes, self.initializers)


def _prune_untranslatable_nodes(gm):
    """Remove trace artifacts with no ONNX equivalent: assertions, dead comparisons,
    and placeholders specialized away by concrete_args."""
    for n in reversed(list(gm.graph.find_nodes(op="call_function", target=torch._assert))):
        gm.graph.erase_node(n)
    for n in reversed(list(gm.graph.find_nodes(op="call_function", target=operator.eq))):
        if len(n.users) == 0:
            gm.graph.erase_node(n)
    for n in reversed(list(gm.graph.find_nodes(op="placeholder"))):
        if len(n.users) == 0 and len(n.args) > 0:  # specialized: has a baked default, now unused
            gm.graph.erase_node(n)
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
    # A single input keeps the graph-input name "input"; with multiple inputs
    # each graph input is named after its forward parameter.
    names = ["input"] if n_expected == 1 else [str(p.target) for p in placeholders]
    return dict(zip(placeholders, names))


def _route_input_passthrough_outputs(output_names, input_names, nodes):
    """ONNX forbids a graph input from also being a graph output; insert Identity nodes."""
    graph_input_names = set(input_names)
    for idx, name in enumerate(output_names):
        if name in graph_input_names:
            identity_out = f"{name}_identity_out{idx}"
            nodes.append(oh.make_node("Identity", inputs=[name], outputs=[identity_out]))
            output_names[idx] = identity_out


def convert_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: str = "model.onnx",
    opset: int = 13,
    use_qonnx: bool = False,
    store_integer_weights: bool = False,
    integer_ops: bool = False,
    include_clip: bool = True,
    concrete_args: dict | None = None,
    input_dtypes=None,
    batch_size: int | None = None,
) -> onnx.ModelProto:
    """
    Convert a PQuant nn.Module to ONNX or QONNX using torch.fx symbolic tracing.

    Works with arbitrary model topologies including residual/skip connections,
    branches, and concatenations.  The model must be symbolically traceable
    (no data-dependent control flow).

    Multiple inputs are supported: pass a sequence of per-input shapes as
    ``input_shape`` (e.g. ``[(3, 32, 32), (16,)]``) and the model's ``forward``
    must take one tensor argument per shape, in the same order.  A single input
    keeps the graph-input name ``"input"``; with multiple inputs each graph input
    is named after its ``forward`` parameter.

    Non-tensor inputs (bool flags, int sizes, ``None`` masks, ...) are not ONNX
    graph inputs.  Specialize them to constants at trace time by passing
    ``concrete_args={"flag": False, ...}``; only the remaining tensor arguments
    become graph inputs (see ``concrete_args`` below).

    Args:
        model:                  Trained nn.Module. Call apply_final_compression()
                                on all PQ modules before passing here.
        input_shape:            Shape of a single sample (excluding batch), e.g. (3, 32, 32),
                                or a sequence of per-input shapes for multi-input models.
        output_path:            Where to save the .onnx file.
        opset:                  ONNX opset version (≥13 required for per-channel DequantizeLinear).
        use_qonnx:              If True, emit QONNX Quant custom nodes (requires qonnx runtime).
                                If False (default), emit Clip+QuantizeLinear+DequantizeLinear
                                nodes runnable with plain onnxruntime.
        store_integer_weights:  If True (and use_qonnx=False), store weight/bias initializers
                                as int8/uint8 followed by DequantizeLinear instead of float32.
                                Ignored when use_qonnx=True or integer_ops=True.
        integer_ops:            If True (and use_qonnx=False), use MatMulInteger for Dense layers
                                so the inner product runs in int32 arithmetic.  Weights are stored
                                as int8 (pre-transposed) and a single DequantizeLinear converts the
                                int32 accumulator back to float using the combined scale s_x * s_w.
                                Implies integer weight storage; store_integer_weights is ignored.
        include_clip:           Prepend a Clip node before each QuantizeLinear when True (default).
                                Set to False to emit bare QuantizeLinear+DequantizeLinear pairs —
                                safe when values are guaranteed in-range at inference time since
                                QuantizeLinear saturates naturally.  Ignored when use_qonnx=True.
        concrete_args:          Forwarded to ``torch.fx.Tracer.trace`` to bake non-tensor
                                ``forward`` arguments in as constants.  Keys are
                                ``forward`` parameter names.  Specialized arguments are
                                dropped from the ONNX graph inputs.
        input_dtypes:           Optional dtype per input (single value or a list parallel
                                to ``input_shape``).  Each is a torch.dtype or a string
                                (``"float32"``, ``"bool"``, ``"int64"``, ``"int32"``).
                                Defaults to float32.  Use ``"bool"`` for a runtime
                                attention ``key_padding_mask`` input, for example.
        batch_size:             If not None, fix the batch dimension of every graph input
                                and output to this value.  If None (default), the batch
                                dimension is left dynamic.

    Returns:
        The constructed onnx.ModelProto.
    """
    model.eval()
    quant_fn = quant_node if use_qonnx else functools.partial(qdq_node, include_clip=include_clip)

    input_shapes = _normalize_input_shapes(input_shape)
    input_torch_dtypes, input_tp_dtypes = _normalize_input_dtypes(input_dtypes, len(input_shapes))

    gm = fx.GraphModule(model, PQTracer().trace(model, concrete_args=concrete_args))
    _prune_untranslatable_nodes(gm)
    ph_to_name = _graph_input_names(gm, len(input_shapes))
    input_names = list(ph_to_name.values())

    device = next((p.device for p in model.parameters()), None)
    probes = [torch.zeros(1, *shp, device=device, dtype=dt) for shp, dt in zip(input_shapes, input_torch_dtypes)]
    with torch.no_grad():
        ShapeProp(gm).propagate(*probes)

    emitter = FxGraphEmitter(gm, ph_to_name, quant_fn, use_qonnx, store_integer_weights, integer_ops)
    output_names = emitter._run()
    _route_input_passthrough_outputs(output_names, input_names, emitter.nodes)

    with torch.no_grad():
        dummy_out = model(*probes, **(concrete_args or {}))
    dummy_outs = list(dummy_out) if isinstance(dummy_out, (tuple, list)) else [dummy_out]

    batch_dim = batch_size  # None → dynamic, int → fixed
    input_vis = [
        oh.make_tensor_value_info(name, tp, [batch_dim, *shp])
        for name, shp, tp in zip(input_names, input_shapes, input_tp_dtypes)
    ]
    output_vis = [
        oh.make_tensor_value_info(name, TensorProto.FLOAT, [batch_dim] + list(t.shape[1:]))
        for name, t in zip(output_names, dummy_outs)
    ]

    graph = oh.make_graph(
        nodes=emitter.nodes,
        name="pquant_onnx_fx",
        inputs=input_vis,
        outputs=output_vis,
        initializer=emitter.initializers,
    )
    model_proto = save_model(graph, output_path, opset, use_qonnx=use_qonnx)
    logging.info("Saved %s model (FX) → %s", "QONNX" if use_qonnx else "ONNX (QDQ)", output_path)
    return model_proto
