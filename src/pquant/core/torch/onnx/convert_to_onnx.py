"""
Convert a PQuant model to ONNX or QONNX format.

Pass ``use_qonnx=True`` to emit QONNX ``Quant`` custom nodes (requires the
qonnx runtime).  Pass ``use_qonnx=False`` (default) to emit standard
``Clip + QuantizeLinear + DequantizeLinear`` nodes runnable with plain
onnxruntime.
"""

import functools
import logging
import operator as _operator
import os

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as _F
from onnx import TensorProto

os.environ["KERAS_BACKEND"] = "torch"  # must be set before any keras/pquant import

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
from pquant.core.torch.onnx.helpers import (  # noqa: E402
    emit_getitem,
    emit_squeeze,
    emit_unsqueeze,
    maybe_quant_input,
    maybe_quant_output,
    qdq_node,
    quant_node,
)
from pquant.core.torch.onnx.layers import (  # noqa: E402
    add_avgpool,
    add_batchnorm,
    add_conv,
    add_dense,
    add_layernorm,
    add_mha,
)
from pquant.core.torch.quantizer import Quantizer  # noqa: E402


def emit_module(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops=False):
    """Emit ONNX nodes for a single PQuant or standard torch.nn module."""
    if isinstance(module, PQDense):
        return add_dense(
            module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops
        )
    if isinstance(module, PQConv2d):
        return add_conv(
            module,
            prefix,
            current,
            nodes,
            initializers,
            ndim=2,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )
    if isinstance(module, PQConv1d):
        return add_conv(
            module,
            prefix,
            current,
            nodes,
            initializers,
            ndim=1,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )
    if isinstance(module, (PQBatchNorm2d, PQBatchNorm1d)):
        return add_batchnorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
    if isinstance(module, PQLayerNorm):
        return add_layernorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
    if isinstance(module, PQAvgPool2d):
        return add_avgpool(module, prefix, current, nodes, initializers, ndim=2, quant_fn=quant_fn)
    if isinstance(module, PQAvgPool1d):
        return add_avgpool(module, prefix, current, nodes, initializers, ndim=1, quant_fn=quant_fn)
    if isinstance(module, nn.ReLU):
        out = f"{prefix}_relu"
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[out]))
        return out
    if isinstance(module, nn.Flatten):
        out = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[current], outputs=[out], axis=module.start_dim))
        return out
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        return current  # identity at inference
    if isinstance(module, nn.LeakyReLU):
        out = f"{prefix}_leakyrelu"
        nodes.append(oh.make_node("LeakyRelu", inputs=[current], outputs=[out], alpha=module.negative_slope))
        return out
    if isinstance(module, nn.MaxPool2d):
        out = f"{prefix}_maxpool"
        kernel = module.kernel_size if isinstance(module.kernel_size, (list, tuple)) else [module.kernel_size] * 2
        stride = module.stride if isinstance(module.stride, (list, tuple)) else [module.stride] * 2
        pad = module.padding if isinstance(module.padding, (list, tuple)) else [module.padding] * 2
        nodes.append(
            oh.make_node(
                "MaxPool",
                inputs=[current],
                outputs=[out],
                kernel_shape=list(kernel),
                strides=list(stride),
                pads=[pad[0], pad[1], pad[0], pad[1]],
            )
        )
        return out
    if isinstance(module, nn.Upsample):
        # Emit a Resize node with nearest/bilinear mode and scale factors.
        roi_name = f"{prefix}_upsample_roi"
        scales_name = f"{prefix}_upsample_scales"
        initializers.append(onh.from_array(np.array([], dtype=np.float32), name=roi_name))
        scale_factor = module.scale_factor
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor, scale_factor)
        scales = np.array([1.0, 1.0, float(scale_factor[0]), float(scale_factor[1])], dtype=np.float32)
        initializers.append(onh.from_array(scales, name=scales_name))
        mode = "nearest" if module.mode == "nearest" else "linear"
        out = f"{prefix}_upsample"
        nodes.append(
            oh.make_node(
                "Resize",
                inputs=[current, roi_name, scales_name],
                outputs=[out],
                mode=mode,
                coordinate_transformation_mode="asymmetric",
            )
        )
        return out
    if isinstance(module, PQActivation):
        current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)
        act = module.activation_name
        act_out = f"{prefix}_act"
        if act == "relu":
            nodes.append(oh.make_node("Relu", inputs=[current], outputs=[act_out]))
        elif act == "tanh":
            nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[act_out]))
        elif act == "hard_tanh":
            cmin_name = f"{prefix}_htanh_min"
            cmax_name = f"{prefix}_htanh_max"
            initializers += [
                onh.from_array(np.array(-1.0, dtype=np.float32), name=cmin_name),
                onh.from_array(np.array(1.0, dtype=np.float32), name=cmax_name),
            ]
            nodes.append(oh.make_node("Clip", inputs=[current, cmin_name, cmax_name], outputs=[act_out]))
        elif act == "leaky_relu":
            nodes.append(
                oh.make_node(
                    "LeakyRelu", inputs=[current], outputs=[act_out], alpha=module.activation_function.negative_slope
                )
            )
        elif act == "gelu":
            # Decompose so the default opset (13) works; ONNX added a Gelu op only in opset 20.
            approximate = getattr(module.activation_function, "approximate", "none")
            half_name = f"{prefix}_gelu_half"
            one_name = f"{prefix}_gelu_one"
            initializers += [
                onh.from_array(np.array(0.5, dtype=np.float32), name=half_name),
                onh.from_array(np.array(1.0, dtype=np.float32), name=one_name),
            ]
            if approximate == "tanh":
                # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                c0_name = f"{prefix}_gelu_sqrt2_over_pi"
                c1_name = f"{prefix}_gelu_c1"
                three_name = f"{prefix}_gelu_three"
                initializers += [
                    onh.from_array(np.array(np.sqrt(2.0 / np.pi), dtype=np.float32), name=c0_name),
                    onh.from_array(np.array(0.044715, dtype=np.float32), name=c1_name),
                    onh.from_array(np.array(3.0, dtype=np.float32), name=three_name),
                ]
                x3 = f"{prefix}_gelu_x3"
                cx3 = f"{prefix}_gelu_cx3"
                inner = f"{prefix}_gelu_inner"
                scaled = f"{prefix}_gelu_scaled"
                tanh_out = f"{prefix}_gelu_tanh"
                plus_one = f"{prefix}_gelu_plus1"
                x_times = f"{prefix}_gelu_xprod"
                nodes += [
                    oh.make_node("Pow", inputs=[current, three_name], outputs=[x3]),
                    oh.make_node("Mul", inputs=[x3, c1_name], outputs=[cx3]),
                    oh.make_node("Add", inputs=[current, cx3], outputs=[inner]),
                    oh.make_node("Mul", inputs=[inner, c0_name], outputs=[scaled]),
                    oh.make_node("Tanh", inputs=[scaled], outputs=[tanh_out]),
                    oh.make_node("Add", inputs=[tanh_out, one_name], outputs=[plus_one]),
                    oh.make_node("Mul", inputs=[current, plus_one], outputs=[x_times]),
                    oh.make_node("Mul", inputs=[x_times, half_name], outputs=[act_out]),
                ]
            else:
                # Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
                inv_sqrt2_name = f"{prefix}_gelu_inv_sqrt2"
                initializers.append(onh.from_array(np.array(1.0 / np.sqrt(2.0), dtype=np.float32), name=inv_sqrt2_name))
                scaled = f"{prefix}_gelu_scaled"
                erf_out = f"{prefix}_gelu_erf"
                plus_one = f"{prefix}_gelu_plus1"
                x_times = f"{prefix}_gelu_xprod"
                nodes += [
                    oh.make_node("Mul", inputs=[current, inv_sqrt2_name], outputs=[scaled]),
                    oh.make_node("Erf", inputs=[scaled], outputs=[erf_out]),
                    oh.make_node("Add", inputs=[erf_out, one_name], outputs=[plus_one]),
                    oh.make_node("Mul", inputs=[current, plus_one], outputs=[x_times]),
                    oh.make_node("Mul", inputs=[x_times, half_name], outputs=[act_out]),
                ]
        else:
            raise TypeError(f"PQActivation: unsupported activation {act!r} for ONNX export")
        current = act_out
        current = maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
        return current
    if isinstance(module, PQMultiheadAttention):
        out, _ = add_mha(
            module, prefix, current, current, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
        )
        return out
    if isinstance(module, Quantizer):
        k, i, f = module.get_quantization_bits()
        new_nodes, out = quant_fn(prefix, current, module.round_mode, k, i, f, initializers, overflow_mode=module.overflow)
        nodes.extend(new_nodes)
        return out
    raise TypeError(f"Unsupported module type for ONNX export: {type(module).__name__}")


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def export_qdq_layernorm(
    output_path: str,
    input_shape,
    gamma: np.ndarray,
    beta: np.ndarray,
    input_scale_log2: int,
    output_scale_log2: int,
    eps_q0: int = 1,
    opset: int = 17,
) -> onnx.ModelProto:
    # ----- validate shape -----
    input_shape = tuple(int(d) for d in input_shape)
    if len(input_shape) not in (2, 3):
        raise ValueError(f"input_shape rank must be 2 or 3, got {len(input_shape)} ({input_shape})")
    for d in input_shape:
        if d <= 0:
            raise ValueError(f"input_shape must be fully static and positive, got {input_shape}")
    D = input_shape[-1]
    if not is_pow2(D):
        raise ValueError(f"last dim must be a power of two, got {D}")
    if D % 32 != 0:
        raise ValueError(f"last dim must be a multiple of 32, got {D}")

    # ----- validate gamma / beta -----
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    if gamma.shape != (D,):
        raise ValueError(f"gamma must have shape ({D},), got {gamma.shape}")
    if beta.shape != (D,):
        raise ValueError(f"beta must have shape ({D},), got {beta.shape}")

    GAMMA_F = 7  # Q7  in int16 -> scale = 2**-7
    BETA_F = 15  # Q15 in int16 -> scale = 2**-15
    INT16_MIN, INT16_MAX = -(2**15), 2**15 - 1

    def check_q_int16(arr: np.ndarray, frac_bits: int, name: str) -> None:
        scaled = arr.astype(np.float64) * (2**frac_bits)
        rounded = np.round(scaled)
        # Exactly representable: rounding is a no-op (within fp slack).
        if not np.allclose(scaled, rounded, atol=1e-4):
            raise ValueError(
                f"{name} not exactly representable as int16 Q{frac_bits} "
                f"(max abs round error = {np.max(np.abs(scaled - rounded)):.6g})"
            )
        if rounded.min() < INT16_MIN or rounded.max() > INT16_MAX:
            raise ValueError(f"{name} overflows int16 at Q{frac_bits} " f"(range [{rounded.min()}, {rounded.max()}])")

    check_q_int16(gamma, GAMMA_F, "gamma")
    check_q_int16(beta, BETA_F, "beta")

    # ----- validate quant params -----
    input_scale_log2 = int(input_scale_log2)
    output_scale_log2 = int(output_scale_log2)
    eps_q0 = int(eps_q0)
    if eps_q0 < 1:
        raise ValueError(f"eps_q0 must be >= 1, got {eps_q0}")

    if opset < 17:
        raise ValueError(f"opset must be >= 17 for LayerNormalization, got {opset}")

    input_scale = float(2.0**input_scale_log2)
    output_scale = float(2.0**output_scale_log2)
    epsilon = float(eps_q0) * input_scale * input_scale

    # ----- build initializers -----
    initializers = [
        onh.from_array(np.array(input_scale, dtype=np.float32), name="input_scale"),
        onh.from_array(np.array(0, dtype=np.int8), name="input_zero_point"),
        onh.from_array(np.array(output_scale, dtype=np.float32), name="output_scale"),
        onh.from_array(np.array(0, dtype=np.int8), name="output_zero_point"),
        onh.from_array(gamma.astype(np.float32), name="gamma"),
        onh.from_array(beta.astype(np.float32), name="beta"),
    ]

    # ----- build nodes -----
    nodes = [
        oh.make_node(
            "DequantizeLinear",
            inputs=["input_q", "input_scale", "input_zero_point"],
            outputs=["x_dq"],
            name="input_dq",
        ),
        oh.make_node(
            "LayerNormalization",
            inputs=["x_dq", "gamma", "beta"],
            outputs=["ln_out"],
            name="layernorm",
            axis=-1,
            epsilon=epsilon,
        ),
        oh.make_node(
            "QuantizeLinear",
            inputs=["ln_out", "output_scale", "output_zero_point"],
            outputs=["y_q"],
            name="output_q",
        ),
        oh.make_node(
            "DequantizeLinear",
            inputs=["y_q", "output_scale", "output_zero_point"],
            outputs=["output"],
            name="output_dq",
        ),
    ]

    # ----- build graph + model -----
    input_vi = oh.make_tensor_value_info("input_q", TensorProto.INT8, list(input_shape))
    output_vi = oh.make_tensor_value_info("output", TensorProto.FLOAT, list(input_shape))

    graph = oh.make_graph(
        nodes=nodes,
        name="qdq_layernorm",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=initializers,
    )

    model_proto = oh.make_model(graph, opset_imports=[oh.make_opsetid("", opset)])
    model_proto.ir_version = 8

    # Strip any initializer names that the onnx library may have added to graph.input.
    _init_names = {t.name for t in model_proto.graph.initializer}
    _data_inputs = [vi for vi in model_proto.graph.input if vi.name not in _init_names]
    del model_proto.graph.input[:]
    model_proto.graph.input.extend(_data_inputs)

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    return model_proto


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


def normalize_input_shapes(input_shape) -> list[tuple]:
    seq = list(input_shape)
    if len(seq) > 0 and all(isinstance(s, (list, tuple)) for s in seq):
        return [tuple(int(d) for d in s) for s in seq]
    return [tuple(int(d) for d in seq)]


def normalize_input_dtypes(input_dtypes, n: int):
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

    input_shapes = normalize_input_shapes(input_shape)
    input_torch_dtypes, input_tp_dtypes = normalize_input_dtypes(input_dtypes, len(input_shapes))

    graph = PQTracer().trace(model, concrete_args=concrete_args)
    gm = fx.GraphModule(model, graph)

    for n in reversed(list(gm.graph.find_nodes(op="call_function", target=torch._assert))):
        gm.graph.erase_node(n)
    for n in reversed(list(gm.graph.find_nodes(op="call_function", target=_operator.eq))):
        if len(n.users) == 0:
            gm.graph.erase_node(n)
    for n in reversed(list(gm.graph.find_nodes(op="placeholder"))):
        if len(n.users) == 0 and len(n.args) > 0:  # specialized: has a baked default, now unused
            gm.graph.erase_node(n)
    gm.recompile()

    tensor_phs = list(gm.graph.find_nodes(op="placeholder"))
    if len(tensor_phs) != len(input_shapes):
        raise ValueError(
            f"FX export: model.forward has {len(tensor_phs)} tensor input(s) but "
            f"input_shape describes {len(input_shapes)}.  Specialize non-tensor "
            f"arguments via concrete_args={{...}}."
        )
    input_names = ["input"] if len(tensor_phs) == 1 else [str(p.target) for p in tensor_phs]
    ph_to_name = {p: n for p, n in zip(tensor_phs, input_names)}

    from torch.fx.passes.shape_prop import ShapeProp

    device = next((p.device for p in model.parameters()), None)
    probes = [torch.zeros(1, *shp, device=device, dtype=dt) for shp, dt in zip(input_shapes, input_torch_dtypes)]
    with torch.no_grad():
        ShapeProp(gm).propagate(*probes)

    onnx_nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    node_to_name: dict[fx.Node, str] = {}
    output_names: list[str] = []

    def res(arg) -> str:
        if isinstance(arg, fx.Node):
            return node_to_name[arg]
        raise TypeError(f"Expected fx.Node, got {type(arg)}")

    def binop_inputs(node: fx.Node) -> list[str]:
        names: list[str] = []
        for i, a in enumerate(node.args[:2]):
            if isinstance(a, fx.Node):
                names.append(node_to_name[a])
            elif isinstance(a, (int, float, bool)):
                cname = f"{node.name}_arg{i}_const"
                initializers.append(onh.from_array(np.array(float(a), dtype=np.float32), name=cname))
                names.append(cname)
            else:
                raise TypeError(f"FX export: unsupported binary-op arg type {type(a).__name__}")
        return names

    def node_shape(n: fx.Node) -> tuple:
        meta = n.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            raise RuntimeError(f"FX export: ShapeProp did not produce tensor_meta for {n.name!r}")
        return tuple(meta.shape)

    def node_rank(n: fx.Node) -> int:
        return len(node_shape(n))

    def squeeze_axes_for(node: fx.Node) -> list[int]:
        """Resolve the axes a torch squeeze()/​.squeeze() call removes."""
        in_shape = node_shape(node.args[0])
        if len(node.args) > 1 or "dim" in node.kwargs:
            dim = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["dim"])
            dim %= len(in_shape)
            return [dim] if in_shape[dim] == 1 else []
        return [i for i, s in enumerate(in_shape) if s == 1 and i != 0]

    def swap_perm(rank: int, d0: int, d1: int) -> list[int]:
        perm = list(range(rank))
        a, b = d0 % rank, d1 % rank
        perm[a], perm[b] = perm[b], perm[a]
        return perm

    def resolve_perm_dims(args, rank: int) -> list[int]:
        # Accept both permute(d0, d1, ...) and permute([d0, d1, ...]) shapes.
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            dims = args[0]
        else:
            dims = args
        return [int(d) % rank for d in dims]

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node_to_name[node] = ph_to_name[node]

        elif node.op == "get_attr":
            obj = gm
            for part in node.target.split("."):
                obj = getattr(obj, part)
            attr_name = node.name
            if isinstance(obj, torch.Tensor):
                initializers.append(onh.from_array(obj.detach().cpu().numpy(), name=attr_name))
            node_to_name[node] = attr_name

        elif node.op == "call_module":
            mod = gm.get_submodule(node.target)
            mod_prefix = node.name.replace(".", "_")
            if isinstance(mod, PQMultiheadAttention):
                # forward(query, key, value, key_padding_mask=None, attn_mask=None, ...)
                q_name = node_to_name[node.args[0]]
                k_name = node_to_name[node.args[1]] if len(node.args) > 1 else q_name
                v_name = node_to_name[node.args[2]] if len(node.args) > 2 else q_name

                def mask_name(pos, kw, node=node):
                    arg = node.args[pos] if len(node.args) > pos else node.kwargs.get(kw)
                    if arg is None:
                        return None
                    if not isinstance(arg, fx.Node):
                        raise TypeError(f"FX ONNX export: MHA {kw} must be a tensor (constant or input), got {type(arg)}")
                    return node_to_name[arg]

                kpm_name = mask_name(3, "key_padding_mask")
                attn_mask_name = mask_name(4, "attn_mask")
                out_name, avg_attn_name = add_mha(
                    mod,
                    mod_prefix,
                    q_name,
                    k_name,
                    v_name,
                    onnx_nodes,
                    initializers,
                    quant_fn,
                    use_qonnx,
                    store_integer_weights,
                    key_padding_mask=kpm_name,
                    attn_mask=attn_mask_name,
                )
                node_to_name[node] = (out_name, avg_attn_name)
            else:
                current = emit_module(
                    mod,
                    mod_prefix,
                    node_to_name[node.args[0]],
                    onnx_nodes,
                    initializers,
                    quant_fn,
                    use_qonnx,
                    store_integer_weights,
                    integer_ops,
                )
                node_to_name[node] = current

        elif node.op == "call_function":
            fn = node.target

            if fn is torch._assert or getattr(fn, "__name__", "") == "_assert" or fn is _operator.eq:
                continue

            if fn is _operator.getitem:
                container = node_to_name[node.args[0]]
                if isinstance(container, tuple):
                    # Unpack a tuple output (e.g. from PQMultiheadAttention).
                    node_to_name[node] = container[node.args[1]]
                else:
                    # Tensor slicing: x[:, 0], x[..., :4], ... → Slice (+ Squeeze)
                    rank = node_rank(node.args[0])
                    node_to_name[node] = emit_getitem(node.name, container, node.args[1], rank, onnx_nodes, initializers)
                continue

            if fn in (torch.add, _operator.add, _operator.iadd):
                out = f"{node.name}_add"
                onnx_nodes.append(oh.make_node("Add", inputs=binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.mul, _operator.mul):
                out = f"{node.name}_mul"
                onnx_nodes.append(oh.make_node("Mul", inputs=binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.sub, _operator.sub, _operator.isub):
                out = f"{node.name}_sub"
                onnx_nodes.append(oh.make_node("Sub", inputs=binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.div, _operator.truediv, _operator.itruediv):
                out = f"{node.name}_div"
                onnx_nodes.append(oh.make_node("Div", inputs=binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.matmul, _operator.matmul):
                out = f"{node.name}_matmul"
                onnx_nodes.append(oh.make_node("MatMul", inputs=binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn is torch.transpose:
                # torch.transpose(t, d0, d1) swaps two dims; ONNX needs a full perm.
                rank = node_rank(node.args[0])
                perm = swap_perm(rank, int(node.args[1]), int(node.args[2]))
                out = f"{node.name}_transpose"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[res(node.args[0])], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif fn is torch.permute:
                rank = node_rank(node.args[0])
                perm = resolve_perm_dims(node.args[1:], rank)
                out = f"{node.name}_permute"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[res(node.args[0])], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif fn is torch.cat:
                tensors = [res(a) for a in node.args[0]]
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
                out = f"{node.name}_concat"
                onnx_nodes.append(oh.make_node("Concat", inputs=tensors, outputs=[out], axis=int(dim)))
                node_to_name[node] = out

            elif fn in (_F.relu, torch.relu):
                out = f"{node.name}_relu"
                onnx_nodes.append(oh.make_node("Relu", inputs=[res(node.args[0])], outputs=[out]))
                node_to_name[node] = out

            elif fn in (_F.sigmoid, torch.sigmoid):
                out = f"{node.name}_sigmoid"
                onnx_nodes.append(oh.make_node("Sigmoid", inputs=[res(node.args[0])], outputs=[out]))
                node_to_name[node] = out

            elif fn is torch.flatten:
                start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 0)
                out = f"{node.name}_flatten"
                onnx_nodes.append(oh.make_node("Flatten", inputs=[res(node.args[0])], outputs=[out], axis=int(start_dim)))
                node_to_name[node] = out

            elif fn is torch.squeeze:
                node_to_name[node] = emit_squeeze(
                    node.name, res(node.args[0]), squeeze_axes_for(node), onnx_nodes, initializers
                )

            elif fn is torch.unsqueeze:
                dim = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["dim"])
                axes = [dim % (node_rank(node.args[0]) + 1)]
                node_to_name[node] = emit_unsqueeze(node.name, res(node.args[0]), axes, onnx_nodes, initializers)

            else:
                raise TypeError(f"Unsupported call_function for FX ONNX export: {fn}")

        elif node.op == "call_method":
            x = res(node.args[0])

            if node.target == "relu":
                out = f"{node.name}_relu"
                onnx_nodes.append(oh.make_node("Relu", inputs=[x], outputs=[out]))
                node_to_name[node] = out

            elif node.target == "flatten":
                start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 1)
                out = f"{node.name}_flatten"
                onnx_nodes.append(oh.make_node("Flatten", inputs=[x], outputs=[out], axis=int(start_dim)))
                node_to_name[node] = out

            elif node.target in ("view", "reshape"):
                shape_vals = []
                for a in node.args[1:]:
                    if not isinstance(a, int):
                        raise TypeError("Dynamic reshape (non-constant shape) is not supported in FX ONNX export")
                    shape_vals.append(a)
                shape_name = f"{node.name}_shape"
                out = f"{node.name}_reshape"
                initializers.append(onh.from_array(np.array(shape_vals, dtype=np.int64), name=shape_name))
                onnx_nodes.append(oh.make_node("Reshape", inputs=[x, shape_name], outputs=[out]))
                node_to_name[node] = out

            elif node.target == "transpose":
                rank = node_rank(node.args[0])
                perm = swap_perm(rank, int(node.args[1]), int(node.args[2]))
                out = f"{node.name}_transpose"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[x], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif node.target == "permute":
                rank = node_rank(node.args[0])
                perm = resolve_perm_dims(node.args[1:], rank)
                out = f"{node.name}_permute"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[x], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif node.target == "matmul":
                out = f"{node.name}_matmul"
                onnx_nodes.append(oh.make_node("MatMul", inputs=[x, res(node.args[1])], outputs=[out]))
                node_to_name[node] = out

            elif node.target == "squeeze":
                node_to_name[node] = emit_squeeze(node.name, x, squeeze_axes_for(node), onnx_nodes, initializers)

            elif node.target == "unsqueeze":
                dim = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["dim"])
                axes = [dim % (node_rank(node.args[0]) + 1)]
                node_to_name[node] = emit_unsqueeze(node.name, x, axes, onnx_nodes, initializers)

            else:
                raise TypeError(f"Unsupported call_method for FX ONNX export: {node.target!r}")

        elif node.op == "output":
            ret = node.args[0]
            rets = list(ret) if isinstance(ret, (tuple, list)) else [ret]
            for r in rets:
                if not isinstance(r, fx.Node):
                    raise TypeError("FX ONNX export: unsupported (non-tensor) model output")
                val = node_to_name[r]
                # MHA nodes store a tuple (out, avg_attn); expose the attention output.
                output_names.append(val[0] if isinstance(val, tuple) else val)

    graph_input_names = set(input_names)
    for idx, nm in enumerate(output_names):
        if nm in graph_input_names:
            ident = f"{nm}_identity_out{idx}"
            onnx_nodes.append(oh.make_node("Identity", inputs=[nm], outputs=[ident]))
            output_names[idx] = ident

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

    onnx_graph = oh.make_graph(
        nodes=onnx_nodes,
        name="pquant_onnx_fx",
        inputs=input_vis,
        outputs=output_vis,
        initializer=initializers,
    )

    opset_imports = [oh.make_opsetid("", opset)]
    if use_qonnx:
        opset_imports.append(oh.make_opsetid("qonnx.custom_op.general", 1))
    model_proto = oh.make_model(onnx_graph, opset_imports=opset_imports)
    model_proto.ir_version = 6

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    fmt = "QONNX" if use_qonnx else "ONNX (QDQ)"
    logging.info("Saved %s model (FX) → %s", fmt, output_path)
    return model_proto
