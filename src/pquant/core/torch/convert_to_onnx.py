"""
Convert a PQuant model to ONNX or QONNX format.

Pass ``use_qonnx=True`` to emit QONNX ``Quant`` custom nodes (requires the
qonnx runtime).  Pass ``use_qonnx=False`` (default) to emit standard
``Clip + QuantizeLinear + DequantizeLinear`` nodes runnable with plain
onnxruntime.

Fixed-point (k, i, f) mapping
------------------------------
QONNX:
  scale      = 2^(-f)
  zero_point = 0
  bit_width  = k + i + f
  signed     = int(k)

Standard ONNX (QDQ):
  scale      = 2^(-f)
  zero_point = 0  (int8 signed, uint8 unsigned)
  clip range = [-2^i,  2^i - 2^(-f)]  signed
             = [0,     2^i - 2^(-f)]  unsigned
  Rounding is always nearest-even (QuantizeLinear behaviour).
  Weights are stored as plain float32 initializers — after
  apply_final_compression() they are already on the fixed-point grid.
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
import torch.fx as _fx
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
from pquant.core.torch.quantizer import Quantizer  # noqa: E402

# ---------------------------------------------------------------------------
# QONNX Quant node
# ---------------------------------------------------------------------------

ROUND_MODE_MAP = {
    "TRN": "FLOOR",
    "RND": "ROUND",
    "RND_CONV": "ROUND",
    "TRN_ZERO": "TRUNCATE",
    "RND_ZERO": "ROUND",
    "RND_MIN_INF": "FLOOR",
    "RND_INF": "ROUND",
}


def _quant_node(name_prefix, input_name, rounding_mode, k, i, f, initializers, overflow_mode="SAT"):
    k_val = int(k.item())
    if f.numel() > 1:
        i = i.reshape(-1).max()
        f = f.reshape(-1).min()
    i_val = float(i.item())
    f_val = float(f.item())
    scale = float(2.0 ** (-f_val))
    bit_width = float(k_val + i_val + f_val)
    qonnx_rnd = ROUND_MODE_MAP.get(rounding_mode, "ROUND")
    narrow = 1 if (k_val == 1 and overflow_mode == "SAT_SYM") else 0

    scale_name = f"{name_prefix}_scale"
    zp_name = f"{name_prefix}_zero_point"
    bw_name = f"{name_prefix}_bit_width"
    out_name = f"{name_prefix}_quantized"

    initializers.append(onh.from_array(np.array(scale, dtype=np.float32), name=scale_name))
    initializers.append(onh.from_array(np.array(0.0, dtype=np.float32), name=zp_name))
    initializers.append(onh.from_array(np.array(bit_width, dtype=np.float32), name=bw_name))

    node = oh.make_node(
        op_type="Quant",
        inputs=[input_name, scale_name, zp_name, bw_name],
        outputs=[out_name],
        domain="qonnx.custom_op.general",
        signed=k_val,
        narrow=narrow,
        rounding_mode=qonnx_rnd,
    )
    return [node], out_name


# ---------------------------------------------------------------------------
# Standard ONNX QDQ triple
# ---------------------------------------------------------------------------


def _qdq_node(
    name_prefix, input_name, rounding_mode, k, i, f, initializers, overflow_mode="SAT", include_clip=True
):  # noqa: ARG001
    k_val = int(k.item())
    i_val = float(i.item())
    f_val = float(f.item())
    scale = float(2.0 ** (-f_val))
    signed = k_val == 1

    clip_max = float(2.0**i_val - 2.0 ** (-f_val))
    if not signed:
        clip_min = 0.0
    elif overflow_mode == "SAT_SYM":
        clip_min = -clip_max
    else:
        clip_min = float(-(2.0**i_val))
    zp_val = np.int8(0) if signed else np.uint8(0)

    scale_name = f"{name_prefix}_scale"
    zp_name = f"{name_prefix}_zero_point"
    quantized_name = f"{name_prefix}_quantized"
    out_name = f"{name_prefix}_dequantized"

    initializers += [
        onh.from_array(np.array(scale, dtype=np.float32), name=scale_name),
        onh.from_array(np.array(zp_val), name=zp_name),
    ]

    if include_clip:
        clip_min_name = f"{name_prefix}_clip_min"
        clip_max_name = f"{name_prefix}_clip_max"
        clipped_name = f"{name_prefix}_clipped"
        initializers += [
            onh.from_array(np.array(clip_min, dtype=np.float32), name=clip_min_name),
            onh.from_array(np.array(clip_max, dtype=np.float32), name=clip_max_name),
        ]
        nodes = [
            oh.make_node("Clip", inputs=[input_name, clip_min_name, clip_max_name], outputs=[clipped_name]),
            oh.make_node("QuantizeLinear", inputs=[clipped_name, scale_name, zp_name], outputs=[quantized_name]),
        ]
    else:
        nodes = [
            oh.make_node("QuantizeLinear", inputs=[input_name, scale_name, zp_name], outputs=[quantized_name]),
        ]
    nodes.append(oh.make_node("DequantizeLinear", inputs=[quantized_name, scale_name, zp_name], outputs=[out_name]))
    return nodes, out_name


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _int_weight_node(name_prefix, weight_np, k, i, f, initializers):  # noqa: ARG001 (i unused)
    """
    Store a weight tensor as int8/uint8 + DequantizeLinear.

    weight_np must already be on the fixed-point grid (guaranteed after
    apply_final_compression).  Converts by dividing by the scale and casting —
    no re-rounding needed.

    Granularity handling:
    - per-tensor  (f is scalar): single scale, standard DequantizeLinear.
    - per-channel (f has shape [out, 1, ...]): 1D scale with axis=0.
      All weights in a channel share the same f so the conversion is exact.
    - per-weight  (f is fully per-element): ONNX has no per-weight quantization;
      falls back to float32 storage (no DequantizeLinear node).

    Returns ([node], output_name).
    """
    k_val = int(k.item())
    dtype = np.int8 if k_val == 1 else np.uint8
    out_channels = weight_np.shape[0]
    out_name = f"{name_prefix}_dequantized"

    f_t = f.detach().cpu()

    if f_t.numel() == 1:
        # per-tensor
        scale_np = np.array(float(2.0 ** (-f_t.item())), dtype=np.float32)
        int_weights = np.round(weight_np / float(scale_np)).astype(dtype)
        per_channel = False
    else:
        f_np = f_t.float().numpy().reshape(out_channels, -1)
        if np.allclose(f_np, f_np[:, :1]):
            # per-channel: all elements within an output channel share one f
            f_1d = f_np[:, 0]
            scale_np = (2.0 ** (-f_1d)).astype(np.float32)
            bcast = scale_np.reshape((out_channels,) + (1,) * (weight_np.ndim - 1))
            int_weights = np.round(weight_np / bcast).astype(dtype)
            per_channel = True
        else:
            # per-weight: ONNX cannot represent this; store as float32
            float_name = f"{name_prefix}_float"
            initializers.append(onh.from_array(weight_np, name=float_name))
            return [], float_name

    int_name = f"{name_prefix}_int"
    scale_name = f"{name_prefix}_dq_scale"
    zp_name = f"{name_prefix}_dq_zp"

    zp_np = np.zeros(out_channels if per_channel else 1, dtype=dtype)
    initializers += [
        onh.from_array(int_weights, name=int_name),
        onh.from_array(scale_np, name=scale_name),
        onh.from_array(zp_np if per_channel else np.array(dtype(0)), name=zp_name),
    ]
    node_kwargs = {"axis": 0} if per_channel else {}
    node = oh.make_node("DequantizeLinear", inputs=[int_name, scale_name, zp_name], outputs=[out_name], **node_kwargs)
    return [node], out_name


def _torch_padding_to_onnx(padding, ndim):
    if isinstance(padding, int):
        padding = (padding,) * ndim
    return list(padding) + list(padding)


def _to_list(v, n):
    """Normalize a scalar-or-sequence layer attribute (kernel/stride/...) to an n-length list."""
    return list(v) if hasattr(v, "__iter__") else [v] * n


def _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn):
    # input_quantizer is created conditionally, so guard it; the bool flags are always present.
    if getattr(module, "input_quantizer", None) is not None and module.quantize_input and module.enable_quantization:
        q = module.input_quantizer
        k, i, f = q.get_quantization_bits()
        new_nodes, current = quant_fn(f"{prefix}_in", current, q.round_mode, k, i, f, initializers, overflow_mode=q.overflow)
        nodes.extend(new_nodes)
    return current


def _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn):
    if getattr(module, "output_quantizer", None) is not None and module.quantize_output and module.enable_quantization:
        q = module.output_quantizer
        k, i, f = q.get_quantization_bits()
        new_nodes, current = quant_fn(
            f"{prefix}_out", current, q.round_mode, k, i, f, initializers, overflow_mode=q.overflow
        )
        nodes.extend(new_nodes)
    return current


def _emit_param(prefix, name, arr, quantizer, nodes, initializers, use_qonnx, store_integer_weights):
    if use_qonnx:
        fp_name = f"{prefix}_{name}_fp"
        initializers.append(onh.from_array(arr, name=fp_name))
        k, i, f = quantizer.get_quantization_bits()
        q_nodes, out = _quant_node(
            f"{prefix}_{name}", fp_name, quantizer.round_mode, k, i, f, initializers, overflow_mode=quantizer.overflow
        )
        nodes.extend(q_nodes)
        return out
    if store_integer_weights:
        k, i, f = quantizer.get_quantization_bits()
        q_nodes, out = _int_weight_node(f"{prefix}_{name}", arr, k, i, f, initializers)
        nodes.extend(q_nodes)
        return out
    out = f"{prefix}_{name}"
    initializers.append(onh.from_array(arr, name=out))
    return out


# ---------------------------------------------------------------------------
# per-layer graph builders
# ---------------------------------------------------------------------------


def _add_dense_integer(module, prefix, current, nodes, initializers):
    if getattr(module, "input_quantizer", None) is None or not module.quantize_input:
        raise ValueError(f"{prefix}: integer_ops requires quantize_input=True on the layer")

    # --- Input: Clip + QuantizeLinear → int8 (stop before DequantizeLinear) ---
    k_x, i_x, f_x = module.input_quantizer.get_quantization_bits()
    k_x_val = int(k_x.item())
    i_x_val = float(i_x.item())
    f_x_val = float(f_x.item())
    s_x = float(2.0 ** (-f_x_val))
    signed_x = k_x_val == 1

    clip_min_x = float(-(2.0**i_x_val)) if signed_x else 0.0
    clip_max_x = float(2.0**i_x_val - 2.0 ** (-f_x_val))
    zp_x_np = np.int8(0) if signed_x else np.uint8(0)

    clip_min_name = f"{prefix}_in_clip_min"
    clip_max_name = f"{prefix}_in_clip_max"
    scale_x_name = f"{prefix}_in_scale"
    zp_x_name = f"{prefix}_in_zp"
    x_int_name = f"{prefix}_in_int"

    initializers += [
        onh.from_array(np.array(clip_min_x, dtype=np.float32), name=clip_min_name),
        onh.from_array(np.array(clip_max_x, dtype=np.float32), name=clip_max_name),
        onh.from_array(np.array(s_x, dtype=np.float32), name=scale_x_name),
        onh.from_array(np.array(zp_x_np), name=zp_x_name),
    ]
    nodes += [
        oh.make_node("Clip", inputs=[current, clip_min_name, clip_max_name], outputs=[f"{prefix}_in_clipped"]),
        oh.make_node("QuantizeLinear", inputs=[f"{prefix}_in_clipped", scale_x_name, zp_x_name], outputs=[x_int_name]),
    ]

    # --- Weights: stored pre-transposed as int8 so MatMulInteger needs no Transpose ---
    # PyTorch weight shape: [out, in].  MatMulInteger(A, B) = A @ B, so we need [in, out].
    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)
    k_w, _, f_w = module.weight_quantizer.get_quantization_bits()
    k_w_val = int(k_w.item())  # get_quantization_bits() always returns tensors
    dtype_w = np.int8 if k_w_val == 1 else np.uint8
    out_ch = weight_np.shape[0]

    f_w_t = f_w.detach().cpu()
    if f_w_t.numel() == 1:
        f_w_1d = np.array([float(f_w_t.item())])
        per_channel_w = False
    else:
        f_w_2d = f_w_t.float().numpy().reshape(out_ch, -1)
        f_w_1d = f_w_2d.min(axis=1)  # min f → max scale → covers all values
        per_channel_w = True

    s_w_1d = (2.0 ** (-f_w_1d)).astype(np.float32)  # shape [1] or [out]
    bcast_s_w = s_w_1d.reshape((out_ch,) + (1,) * (weight_np.ndim - 1)) if per_channel_w else float(s_w_1d[0])
    # Transpose before storing so MatMulInteger can use it without a runtime Transpose node
    int_weights_T = np.round(weight_np / bcast_s_w).astype(dtype_w).T  # [in, out]

    zp_w_np = np.array(dtype_w(0))  # scalar zero-point; zero for symmetric quantization
    w_int_name = f"{prefix}_weight_int"
    w_zp_name = f"{prefix}_weight_zp"
    initializers += [
        onh.from_array(int_weights_T, name=w_int_name),
        onh.from_array(zp_w_np, name=w_zp_name),
    ]

    # --- MatMulInteger([batch, in], [in, out]) → int32 [batch, out] ---
    y_int_name = f"{prefix}_matmul_int"
    nodes.append(
        oh.make_node(
            "MatMulInteger",
            inputs=[x_int_name, w_int_name, zp_x_name, w_zp_name],
            outputs=[y_int_name],
        )
    )

    # --- Bias added in int32 domain: bias_int[c] = round(bias[c] / (s_x * s_w[c])) ---
    current_int32 = y_int_name
    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        combined_s = s_x * s_w_1d  # shape [1] or [out]
        bias_int32 = np.round(bias_np / (combined_s if per_channel_w else float(combined_s[0]))).astype(np.int32)
        bias_int_name = f"{prefix}_bias_int"
        y_biased_name = f"{prefix}_matmul_biased"
        initializers.append(onh.from_array(bias_int32, name=bias_int_name))
        nodes.append(oh.make_node("Add", inputs=[current_int32, bias_int_name], outputs=[y_biased_name]))
        current_int32 = y_biased_name

    # --- DequantizeLinear: int32 → float32 using combined scale s_x * s_w ---
    # Per-channel: axis=1 because the output tensor is [batch, out] and out is axis 1.
    combined_scale_name = f"{prefix}_combined_scale"
    combined_zp_name = f"{prefix}_combined_zp"

    if per_channel_w:
        combined_scale_np = (s_x * s_w_1d).astype(np.float32)  # [out]
        combined_zp_np = np.zeros(out_ch, dtype=np.int32)
        dql_kwargs = {"axis": 1}
    else:
        combined_scale_np = np.array(float(s_x * s_w_1d[0]), dtype=np.float32)
        combined_zp_np = np.array(np.int32(0))
        dql_kwargs = {}

    initializers += [
        onh.from_array(combined_scale_np, name=combined_scale_name),
        onh.from_array(combined_zp_np, name=combined_zp_name),
    ]
    y_float_name = f"{prefix}_dequantized"
    nodes.append(
        oh.make_node(
            "DequantizeLinear",
            inputs=[current_int32, combined_scale_name, combined_zp_name],
            outputs=[y_float_name],
            **dql_kwargs,
        )
    )
    current = y_float_name

    # Optional output quantization (e.g. last layer with quantize_output=True)
    current = _maybe_quant_output(module, prefix, current, nodes, initializers, _qdq_node)
    return current


def _add_dense_nd(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)  # [out, in]
    if use_qonnx or store_integer_weights:
        # Quantized/int-stored weight is emitted in native [out, in] layout, then transposed.
        q_weight_native = _emit_param(
            prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        q_weight = f"{prefix}_weight_T"
        nodes.append(oh.make_node("Transpose", inputs=[q_weight_native], outputs=[q_weight], perm=[1, 0]))
    else:
        q_weight = f"{prefix}_weight_T"
        initializers.append(onh.from_array(weight_np.T, name=q_weight))  # pre-transposed [in, out]

    matmul_out = f"{prefix}_matmul"
    nodes.append(oh.make_node("MatMul", inputs=[current, q_weight], outputs=[matmul_out]))
    current = matmul_out

    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        q_bias = _emit_param(
            prefix, "bias", bias_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        biased_out = f"{prefix}_biased"
        nodes.append(oh.make_node("Add", inputs=[matmul_out, q_bias], outputs=[biased_out]))
        current = biased_out

    current = _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def _add_dense(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops=False):
    if integer_ops and not use_qonnx:
        return _add_dense_integer(module, prefix, current, nodes, initializers)
    current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)
    q_weight = _emit_param(
        prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )

    gemm_inputs = [current, q_weight]

    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        q_bias = _emit_param(
            prefix, "bias", bias_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        gemm_inputs.append(q_bias)

    gemm_out = f"{prefix}_gemm"
    nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_out], transB=1))
    current = gemm_out

    current = _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def _add_conv(module, prefix, current, nodes, initializers, ndim, quant_fn, use_qonnx, store_integer_weights):
    current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)
    q_weight = _emit_param(
        prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )

    conv_inputs = [current, q_weight]

    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        q_bias = _emit_param(
            prefix, "bias", bias_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        conv_inputs.append(q_bias)

    padding = module.padding
    if isinstance(padding, str):
        auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
        pads = None
    else:
        auto_pad = "NOTSET"
        pads = _torch_padding_to_onnx(padding, ndim)

    conv_attrs = dict(
        kernel_shape=_to_list(module.kernel_size, ndim),
        strides=_to_list(module.stride, ndim),
        dilations=_to_list(module.dilation, ndim),
        group=module.groups,
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    current = _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def _add_batchnorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    gamma_np = module._weight.detach().cpu().numpy().astype(np.float32)
    beta_np = module._bias.detach().cpu().numpy().astype(np.float32)

    q_gamma = _emit_param(
        prefix, "gamma", gamma_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )
    q_beta = _emit_param(
        prefix, "beta", beta_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )

    mean_name = f"{prefix}_running_mean"
    var_name = f"{prefix}_running_var"
    initializers.append(onh.from_array(module.running_mean.detach().cpu().numpy().astype(np.float32), name=mean_name))
    initializers.append(onh.from_array(module.running_var.detach().cpu().numpy().astype(np.float32), name=var_name))

    bn_out = f"{prefix}_bn"
    nodes.append(
        oh.make_node(
            "BatchNormalization",
            inputs=[current, q_gamma, q_beta, mean_name, var_name],
            outputs=[bn_out],
            epsilon=float(module.eps),
        )
    )
    return bn_out


def _add_layernorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    ns = (
        tuple(int(d) for d in module.normalized_shape)
        if hasattr(module.normalized_shape, "__iter__")
        else (int(module.normalized_shape),)
    )
    axis = -len(ns)

    has_weight = module._weight is not None
    has_bias = module._bias is not None

    gamma_np = module._weight.detach().cpu().numpy().astype(np.float32) if has_weight else np.ones(ns, dtype=np.float32)
    beta_np = module._bias.detach().cpu().numpy().astype(np.float32) if has_bias else None

    qonnx_p = use_qonnx and has_weight
    intstore_p = store_integer_weights and has_weight
    q_gamma = _emit_param(
        prefix,
        "gamma",
        gamma_np,
        module.weight_quantizer if has_weight else None,
        nodes,
        initializers,
        qonnx_p,
        intstore_p,
    )
    if has_bias:
        q_beta = _emit_param(
            prefix,
            "beta",
            beta_np,
            module.bias_quantizer if has_weight else None,
            nodes,
            initializers,
            qonnx_p,
            intstore_p,
        )

    ln_inputs = [current, q_gamma]
    if has_bias:
        ln_inputs.append(q_beta)
    ln_out = f"{prefix}_ln"
    nodes.append(
        oh.make_node(
            "LayerNormalization",
            inputs=ln_inputs,
            outputs=[ln_out],
            axis=axis,
            epsilon=float(module.eps),
        )
    )
    current = ln_out
    current = _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def _add_avgpool(module, prefix, current, nodes, initializers, ndim, quant_fn):
    current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    pool_out = f"{prefix}_pool"
    nodes.append(
        oh.make_node(
            "AveragePool",
            inputs=[current],
            outputs=[pool_out],
            kernel_shape=_to_list(module.kernel_size, ndim),
            strides=_to_list(module.stride, ndim),
            pads=_torch_padding_to_onnx(module.padding, ndim),
            ceil_mode=int(module.ceil_mode),
            count_include_pad=int(module.count_include_pad),
        )
    )
    current = pool_out

    current = _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


# ---------------------------------------------------------------------------
# multi-head attention graph builder
# ---------------------------------------------------------------------------


def _add_quantized_softmax(sm, prefix, current, nodes, initializers, quant_fn, kpm_mask=None):
    enable = sm.enable_quantization
    scaler = float(sm.input_scaler)
    stable = bool(sm.stable)
    eps = float(sm.epsilon)

    def qdq(q, pfx, x):
        k, i, f = q.get_quantization_bits()
        q_nodes, out = quant_fn(pfx, x, q.round_mode, k, i, f, initializers, overflow_mode=q.overflow)
        nodes.extend(q_nodes)
        return out

    if sm.quantize_input and enable:
        current = qdq(sm.input_quantizer, f"{prefix}_sm_in_q", current)

    if stable:
        m_name = f"{prefix}_sm_max"
        nodes.append(oh.make_node("ReduceMax", inputs=[current], outputs=[m_name], axes=[-1], keepdims=1))
        exp_in = f"{prefix}_sm_sub"
        nodes.append(oh.make_node("Sub", inputs=[m_name, current], outputs=[exp_in]))
    else:
        exp_in = current

    exp_t = sm.exp_table
    if exp_t.quantize_input and enable:
        exp_in = qdq(exp_t.input_quantizer, f"{prefix}_sm_exp_in_q", exp_in)
    coeff = -scaler if stable else scaler
    exp_arg = exp_in
    if coeff != 1.0:
        coeff_name = f"{prefix}_sm_exp_coeff"
        initializers.append(onh.from_array(np.array(coeff, dtype=np.float32), name=coeff_name))
        exp_arg = f"{prefix}_sm_exp_arg"
        nodes.append(oh.make_node("Mul", inputs=[exp_in, coeff_name], outputs=[exp_arg]))
    exp_inp = f"{prefix}_sm_exp"
    nodes.append(oh.make_node("Exp", inputs=[exp_arg], outputs=[exp_inp]))
    if exp_t.quantize_output and enable:
        exp_inp = qdq(exp_t.output_quantizer, f"{prefix}_sm_exp_out_q", exp_inp)

    if kpm_mask is not None:
        kpm_f = f"{prefix}_sm_mask_f"
        nodes.append(oh.make_node("Cast", inputs=[kpm_mask], outputs=[kpm_f], to=TensorProto.FLOAT))
        masked = f"{prefix}_sm_masked"
        nodes.append(oh.make_node("Mul", inputs=[kpm_f, exp_inp], outputs=[masked]))
        exp_inp = masked

    sum_axes = f"{prefix}_sm_sum_axes"
    initializers.append(onh.from_array(np.array([-1], dtype=np.int64), name=sum_axes))
    sums = f"{prefix}_sm_sum"
    nodes.append(oh.make_node("ReduceSum", inputs=[exp_inp, sum_axes], outputs=[sums], keepdims=1))

    inv_t = sm.inv_table
    inv_in = sums
    if inv_t.quantize_input and enable:
        inv_in = qdq(inv_t.input_quantizer, f"{prefix}_sm_inv_in_q", inv_in)
    eps_name = f"{prefix}_sm_eps"
    initializers.append(onh.from_array(np.array(eps, dtype=np.float32), name=eps_name))
    inv_add = f"{prefix}_sm_inv_add"
    nodes.append(oh.make_node("Add", inputs=[inv_in, eps_name], outputs=[inv_add]))
    divisor = f"{prefix}_sm_inv"
    nodes.append(oh.make_node("Reciprocal", inputs=[inv_add], outputs=[divisor]))
    if inv_t.quantize_output and enable:
        divisor = qdq(inv_t.output_quantizer, f"{prefix}_sm_inv_out_q", divisor)

    out = f"{prefix}_sm_out"
    nodes.append(oh.make_node("Mul", inputs=[exp_inp, divisor], outputs=[out]))
    current = out

    if sm.quantize_output and enable:
        current = qdq(sm.output_quantizer, f"{prefix}_sm_out_q", current)
    return current


def _add_mha(
    module,
    prefix,
    q_input,
    k_input,
    v_input,
    nodes,
    initializers,
    quant_fn,
    use_qonnx,
    store_integer_weights,
    key_padding_mask=None,
    attn_mask=None,
):
    H = module.num_heads
    head_dim = module.head_dim
    E = module.embed_dim
    scale_val = float(module.scale)

    # --- Optional transpose for seq-first inputs (T, B, E) → (B, T, E) ---
    if not module.batch_first:
        q_t = f"{prefix}_q_in_t"
        k_t = f"{prefix}_k_in_t"
        v_t = f"{prefix}_v_in_t"
        nodes.append(oh.make_node("Transpose", inputs=[q_input], outputs=[q_t], perm=[1, 0, 2]))
        nodes.append(oh.make_node("Transpose", inputs=[k_input], outputs=[k_t], perm=[1, 0, 2]))
        nodes.append(oh.make_node("Transpose", inputs=[v_input], outputs=[v_t], perm=[1, 0, 2]))
        q_input, k_input, v_input = q_t, k_t, v_t

    # --- Q / K / V projections: (B, L, E) → (B, L, E) via MatMul (input is rank-3) ---
    q_proj_out = _add_dense_nd(
        module.q_proj, f"{prefix}_q_proj", q_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    k_proj_out = _add_dense_nd(
        module.k_proj, f"{prefix}_k_proj", k_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    v_proj_out = _add_dense_nd(
        module.v_proj, f"{prefix}_v_proj", v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    # --- Helper: (B, L, E) → (B, H, L, head_dim) using dynamic shapes ---
    def _split_heads(x_name, pfx):
        shape_out = f"{pfx}_shape"
        b_scalar = f"{pfx}_b_sc"
        l_scalar = f"{pfx}_l_sc"
        b_1d = f"{pfx}_b_1d"
        l_1d = f"{pfx}_l_1d"
        h_1d_const = f"{pfx}_H_1d"
        hd_1d_const = f"{pfx}_hd_1d"
        shape_4d = f"{pfx}_shape4d"
        reshaped = f"{pfx}_reshaped"
        transposed = f"{pfx}_transposed"
        idx0 = f"{pfx}_gi0"
        idx1 = f"{pfx}_gi1"
        ax0 = f"{pfx}_ax0"

        nodes.append(oh.make_node("Shape", inputs=[x_name], outputs=[shape_out]))
        initializers.extend(
            [
                onh.from_array(np.array(0, dtype=np.int64), name=idx0),
                onh.from_array(np.array(1, dtype=np.int64), name=idx1),
                onh.from_array(np.array([0], dtype=np.int64), name=ax0),
                onh.from_array(np.array([H], dtype=np.int64), name=h_1d_const),
                onh.from_array(np.array([head_dim], dtype=np.int64), name=hd_1d_const),
            ]
        )
        nodes.append(oh.make_node("Gather", inputs=[shape_out, idx0], outputs=[b_scalar]))
        nodes.append(oh.make_node("Gather", inputs=[shape_out, idx1], outputs=[l_scalar]))
        nodes.append(oh.make_node("Unsqueeze", inputs=[b_scalar, ax0], outputs=[b_1d]))
        nodes.append(oh.make_node("Unsqueeze", inputs=[l_scalar, ax0], outputs=[l_1d]))
        nodes.append(oh.make_node("Concat", inputs=[b_1d, l_1d, h_1d_const, hd_1d_const], outputs=[shape_4d], axis=0))
        nodes.append(oh.make_node("Reshape", inputs=[x_name, shape_4d], outputs=[reshaped]))
        # (B, L, H, head_dim) → (B, H, L, head_dim)
        nodes.append(oh.make_node("Transpose", inputs=[reshaped], outputs=[transposed], perm=[0, 2, 1, 3]))
        return transposed

    q_h = _split_heads(q_proj_out, f"{prefix}_q")
    k_h = _split_heads(k_proj_out, f"{prefix}_k")
    v_h = _split_heads(v_proj_out, f"{prefix}_v")

    # --- k^T: (B, H, S, head_dim) → (B, H, head_dim, S) ---
    k_t_name = f"{prefix}_k_T"
    nodes.append(oh.make_node("Transpose", inputs=[k_h], outputs=[k_t_name], perm=[0, 1, 3, 2]))

    # --- Scaled dot-product scores: (B, H, T, head_dim) @ (B, H, head_dim, S) → (B, H, T, S) ---
    raw_scores = f"{prefix}_scores_raw"
    scaled_scores = f"{prefix}_scores_scaled"
    scale_cst = f"{prefix}_attn_scale"
    nodes.append(oh.make_node("MatMul", inputs=[q_h, k_t_name], outputs=[raw_scores]))
    initializers.append(onh.from_array(np.array(scale_val, dtype=np.float32), name=scale_cst))
    nodes.append(oh.make_node("Mul", inputs=[raw_scores, scale_cst], outputs=[scaled_scores]))
    current = scaled_scores

    if attn_mask is not None:
        masked_scores = f"{prefix}_scores_masked"
        nodes.append(oh.make_node("Add", inputs=[current, attn_mask], outputs=[masked_scores]))
        current = masked_scores

    kpm_mult = None
    if key_padding_mask is not None:
        kpm_not = f"{prefix}_kpm_not"
        nodes.append(oh.make_node("Not", inputs=[key_padding_mask], outputs=[kpm_not]))
        kpm_axes = f"{prefix}_kpm_axes"
        initializers.append(onh.from_array(np.array([1, 2], dtype=np.int64), name=kpm_axes))
        kpm_mult = f"{prefix}_kpm_mask"  # (B, 1, 1, S) bool, cast to float inside the softmax
        nodes.append(oh.make_node("Unsqueeze", inputs=[kpm_not, kpm_axes], outputs=[kpm_mult]))

    current = _add_quantized_softmax(
        module.softmax, f"{prefix}_attn", current, nodes, initializers, quant_fn, kpm_mask=kpm_mult
    )
    attn_w_name = current  # softmax output = attention weights (also averaged over heads below)

    ctx_raw = f"{prefix}_ctx_raw"
    nodes.append(oh.make_node("MatMul", inputs=[current, v_h], outputs=[ctx_raw]))
    current_ctx = ctx_raw

    ctx_t = f"{prefix}_ctx_t"  # after Transpose → (B, T, H, head_dim)
    ctx_shape = f"{prefix}_ctx_shape"
    ctx_b_sc = f"{prefix}_ctx_b_sc"
    ctx_t_sc = f"{prefix}_ctx_t_sc"
    ctx_b_1d = f"{prefix}_ctx_b_1d"
    ctx_t_1d = f"{prefix}_ctx_t_1d"
    ctx_E_1d = f"{prefix}_ctx_E_1d"
    ctx_ax0 = f"{prefix}_ctx_ax0"
    ctx_gi0 = f"{prefix}_ctx_gi0"
    ctx_gi1 = f"{prefix}_ctx_gi1"
    ctx_3d = f"{prefix}_ctx_shape3d"
    ctx_merged = f"{prefix}_ctx_merged"

    nodes.append(oh.make_node("Transpose", inputs=[current_ctx], outputs=[ctx_t], perm=[0, 2, 1, 3]))
    nodes.append(oh.make_node("Shape", inputs=[ctx_t], outputs=[ctx_shape]))
    initializers += [
        onh.from_array(np.array(0, dtype=np.int64), name=ctx_gi0),
        onh.from_array(np.array(1, dtype=np.int64), name=ctx_gi1),
        onh.from_array(np.array([0], dtype=np.int64), name=ctx_ax0),
        onh.from_array(np.array([E], dtype=np.int64), name=ctx_E_1d),
    ]
    nodes.append(oh.make_node("Gather", inputs=[ctx_shape, ctx_gi0], outputs=[ctx_b_sc]))
    nodes.append(oh.make_node("Gather", inputs=[ctx_shape, ctx_gi1], outputs=[ctx_t_sc]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[ctx_b_sc, ctx_ax0], outputs=[ctx_b_1d]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[ctx_t_sc, ctx_ax0], outputs=[ctx_t_1d]))
    nodes.append(oh.make_node("Concat", inputs=[ctx_b_1d, ctx_t_1d, ctx_E_1d], outputs=[ctx_3d], axis=0))
    nodes.append(oh.make_node("Reshape", inputs=[ctx_t, ctx_3d], outputs=[ctx_merged]))

    # --- Output projection (rank-3 input: (B, T, E)) ---
    out = _add_dense_nd(
        module.out_proj, f"{prefix}_out_proj", ctx_merged, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    # --- Average attention weights over heads: (B, H, T, S) → (B, T, S) ---
    # Emitted so that getitem(mha, 1) has a valid ONNX value name.
    avg_attn = f"{prefix}_avg_attn_weights"
    nodes.append(oh.make_node("ReduceMean", inputs=[attn_w_name], outputs=[avg_attn], axes=[1], keepdims=0))

    # --- Optional transpose back for seq-first output ---
    if not module.batch_first:
        out_final = f"{prefix}_out_seq_first"
        nodes.append(oh.make_node("Transpose", inputs=[out], outputs=[out_final], perm=[1, 0, 2]))
        return out_final, avg_attn

    return out, avg_attn


# ---------------------------------------------------------------------------
# shared module dispatch (used by both sequential and FX converters)
# ---------------------------------------------------------------------------


def _emit_module(
    module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops=False
):
    """Emit ONNX nodes for a single PQuant or standard torch.nn module."""
    if isinstance(module, PQDense):
        return _add_dense(
            module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops
        )
    if isinstance(module, PQConv2d):
        return _add_conv(
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
        return _add_conv(
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
        return _add_batchnorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
    if isinstance(module, PQLayerNorm):
        return _add_layernorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)
    if isinstance(module, PQAvgPool2d):
        return _add_avgpool(module, prefix, current, nodes, initializers, ndim=2, quant_fn=quant_fn)
    if isinstance(module, PQAvgPool1d):
        return _add_avgpool(module, prefix, current, nodes, initializers, ndim=1, quant_fn=quant_fn)
    if isinstance(module, nn.ReLU):
        out = f"{prefix}_relu"
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[out]))
        return out
    if isinstance(module, nn.Flatten):
        out = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[current], outputs=[out], axis=module.start_dim))
        return out
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        gamma_name = f"{prefix}_bn_gamma"
        beta_name = f"{prefix}_bn_beta"
        mean_name = f"{prefix}_bn_mean"
        var_name = f"{prefix}_bn_var"
        initializers += [
            onh.from_array(module.weight.detach().cpu().numpy().astype(np.float32), name=gamma_name),
            onh.from_array(module.bias.detach().cpu().numpy().astype(np.float32), name=beta_name),
            onh.from_array(module.running_mean.detach().cpu().numpy().astype(np.float32), name=mean_name),
            onh.from_array(module.running_var.detach().cpu().numpy().astype(np.float32), name=var_name),
        ]
        out = f"{prefix}_bn"
        nodes.append(
            oh.make_node(
                "BatchNormalization",
                inputs=[current, gamma_name, beta_name, mean_name, var_name],
                outputs=[out],
                epsilon=float(module.eps),
            )
        )
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
        current = _maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)
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
        current = _maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
        return current
    if isinstance(module, PQMultiheadAttention):
        # Sequential converter: treat as self-attention (Q = K = V = current).
        # Returns (out_name, avg_attn_name); expose only the attention output.
        out, _ = _add_mha(
            module, prefix, current, current, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
        )
        return out
    if isinstance(module, Quantizer):
        # Standalone quantizer (e.g. an auto-inserted missing quantizer or a
        # constant-matrix quantizer): emit a single QDQ node from its k/i/f.
        k, i, f = module.get_quantization_bits()
        new_nodes, out = quant_fn(prefix, current, module.round_mode, k, i, f, initializers, overflow_mode=module.overflow)
        nodes.extend(new_nodes)
        return out
    raise TypeError(f"Unsupported module type for ONNX export: {type(module).__name__}")


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------


def convert_to_onnx(
    model: nn.Sequential,
    input_shape: tuple,
    output_path: str = "model.onnx",
    opset: int = 13,
    use_qonnx: bool = False,
    store_integer_weights: bool = False,
    integer_ops: bool = False,
    include_clip: bool = True,
    batch_size: int | None = None,
) -> onnx.ModelProto:
    """
    Convert a Sequential model of PQuant layers to ONNX or QONNX.

    Args:
        model:                  Trained nn.Sequential. Call apply_final_compression()
                                on all PQ modules before passing here.
        input_shape:            Shape of a single sample (excluding batch), e.g. (3, 32, 32).
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
        batch_size:             If not None, fix the batch dimension of all graph inputs and
                                outputs to this value.  If None (default), the batch dimension
                                is left dynamic.

    Returns:
        The constructed onnx.ModelProto.

    Note:
        This is a thin wrapper over convert_to_onnx_fx().  An ``nn.Sequential`` is a
        plain linear chain, so torch.fx always traces it successfully; routing through
        the FX converter keeps a single code path for both linear and branched models.
    """
    return convert_to_onnx_fx(
        model,
        input_shape,
        output_path=output_path,
        opset=opset,
        use_qonnx=use_qonnx,
        store_integer_weights=store_integer_weights,
        integer_ops=integer_ops,
        include_clip=include_clip,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Hardware-targeted static-QDQ LayerNormalization graph
# ---------------------------------------------------------------------------


def _is_pow2(n: int) -> bool:
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
    if not _is_pow2(D):
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

    def _check_q_int16(arr: np.ndarray, frac_bits: int, name: str) -> None:
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

    _check_q_int16(gamma, GAMMA_F, "gamma")
    _check_q_int16(beta, BETA_F, "beta")

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


# ---------------------------------------------------------------------------
# FX-based conversion (supports arbitrary nn.Module topology / skip connections)
# ---------------------------------------------------------------------------


class _PQTracer(_fx.Tracer):
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


def convert_to_onnx_fx(
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
    Convert any PQuant nn.Module to ONNX using torch.fx symbolic tracing.

    Unlike convert_to_onnx(), this function works with arbitrary model topologies
    including residual/skip connections, branches, and concatenations.  It requires
    the model to be symbolically traceable (no data-dependent control flow).

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
        concrete_args:  Forwarded to ``torch.fx.Tracer.trace`` to bake non-tensor
                        ``forward`` arguments in as constants.  Keys are
                        ``forward`` parameter names.  Specialized arguments are
                        dropped from the ONNX graph inputs.
        input_dtypes:   Optional dtype per input (single value or a list parallel
                        to ``input_shape``).  Each is a torch.dtype or a string
                        (``"float32"``, ``"bool"``, ``"int64"``, ``"int32"``).
                        Defaults to float32.  Use ``"bool"`` for a runtime
                        attention ``key_padding_mask`` input, for example.
        batch_size:     If not None, fix the batch dimension of every graph input
                        and output to this value.  If None (default), the batch
                        dimension is left dynamic.

    Remaining args match the per-layer quantization behaviour described in the module docstring.
    """
    model.eval()
    quant_fn = _quant_node if use_qonnx else functools.partial(_qdq_node, include_clip=include_clip)

    input_shapes = _normalize_input_shapes(input_shape)
    input_torch_dtypes, input_tp_dtypes = _normalize_input_dtypes(input_dtypes, len(input_shapes))

    graph = _PQTracer().trace(model, concrete_args=concrete_args)
    gm = _fx.GraphModule(model, graph)

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
    node_to_name: dict[_fx.Node, str] = {}
    output_names: list[str] = []

    def _res(arg) -> str:
        if isinstance(arg, _fx.Node):
            return node_to_name[arg]
        raise TypeError(f"Expected fx.Node, got {type(arg)}")

    def _binop_inputs(node: _fx.Node) -> list[str]:
        names: list[str] = []
        for i, a in enumerate(node.args[:2]):
            if isinstance(a, _fx.Node):
                names.append(node_to_name[a])
            elif isinstance(a, (int, float, bool)):
                cname = f"{node.name}_arg{i}_const"
                initializers.append(onh.from_array(np.array(float(a), dtype=np.float32), name=cname))
                names.append(cname)
            else:
                raise TypeError(f"FX export: unsupported binary-op arg type {type(a).__name__}")
        return names

    def _rank(n: _fx.Node) -> int:
        meta = n.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            raise RuntimeError(f"FX export: ShapeProp did not produce tensor_meta for {n.name!r}")
        return len(meta.shape)

    def _swap_perm(rank: int, d0: int, d1: int) -> list[int]:
        perm = list(range(rank))
        a, b = d0 % rank, d1 % rank
        perm[a], perm[b] = perm[b], perm[a]
        return perm

    def _resolve_perm_dims(args, rank: int) -> list[int]:
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

                def _mask_name(pos, kw, node=node):
                    arg = node.args[pos] if len(node.args) > pos else node.kwargs.get(kw)
                    if arg is None:
                        return None
                    if not isinstance(arg, _fx.Node):
                        raise TypeError(f"FX ONNX export: MHA {kw} must be a tensor (constant or input), got {type(arg)}")
                    return node_to_name[arg]

                kpm_name = _mask_name(3, "key_padding_mask")
                attn_mask_name = _mask_name(4, "attn_mask")
                out_name, avg_attn_name = _add_mha(
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
                # Store tuple so operator.getitem(node, 0/1) resolves correctly.
                node_to_name[node] = (out_name, avg_attn_name)
            else:
                current = _emit_module(
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
                # Unpack a tuple output (e.g. from PQMultiheadAttention).
                container = node_to_name[node.args[0]]
                if not isinstance(container, tuple):
                    raise TypeError(
                        f"operator.getitem on non-tuple node {node.args[0].name!r} " f"is not supported in FX ONNX export"
                    )
                node_to_name[node] = container[node.args[1]]
                continue

            if fn in (torch.add, _operator.add, _operator.iadd):
                out = f"{node.name}_add"
                onnx_nodes.append(oh.make_node("Add", inputs=_binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.mul, _operator.mul):
                out = f"{node.name}_mul"
                onnx_nodes.append(oh.make_node("Mul", inputs=_binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.sub, _operator.sub, _operator.isub):
                out = f"{node.name}_sub"
                onnx_nodes.append(oh.make_node("Sub", inputs=_binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.div, _operator.truediv, _operator.itruediv):
                out = f"{node.name}_div"
                onnx_nodes.append(oh.make_node("Div", inputs=_binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn in (torch.matmul, _operator.matmul):
                out = f"{node.name}_matmul"
                onnx_nodes.append(oh.make_node("MatMul", inputs=_binop_inputs(node), outputs=[out]))
                node_to_name[node] = out

            elif fn is torch.transpose:
                # torch.transpose(t, d0, d1) swaps two dims; ONNX needs a full perm.
                rank = _rank(node.args[0])
                perm = _swap_perm(rank, int(node.args[1]), int(node.args[2]))
                out = f"{node.name}_transpose"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[_res(node.args[0])], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif fn is torch.permute:
                rank = _rank(node.args[0])
                perm = _resolve_perm_dims(node.args[1:], rank)
                out = f"{node.name}_permute"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[_res(node.args[0])], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif fn is torch.cat:
                tensors = [_res(a) for a in node.args[0]]
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
                out = f"{node.name}_concat"
                onnx_nodes.append(oh.make_node("Concat", inputs=tensors, outputs=[out], axis=int(dim)))
                node_to_name[node] = out

            elif fn in (_F.relu, torch.relu):
                out = f"{node.name}_relu"
                onnx_nodes.append(oh.make_node("Relu", inputs=[_res(node.args[0])], outputs=[out]))
                node_to_name[node] = out

            elif fn in (_F.sigmoid, torch.sigmoid):
                out = f"{node.name}_sigmoid"
                onnx_nodes.append(oh.make_node("Sigmoid", inputs=[_res(node.args[0])], outputs=[out]))
                node_to_name[node] = out

            elif fn is torch.flatten:
                start_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 0)
                out = f"{node.name}_flatten"
                onnx_nodes.append(oh.make_node("Flatten", inputs=[_res(node.args[0])], outputs=[out], axis=int(start_dim)))
                node_to_name[node] = out

            else:
                raise TypeError(f"Unsupported call_function for FX ONNX export: {fn}")

        elif node.op == "call_method":
            x = _res(node.args[0])

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
                rank = _rank(node.args[0])
                perm = _swap_perm(rank, int(node.args[1]), int(node.args[2]))
                out = f"{node.name}_transpose"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[x], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif node.target == "permute":
                rank = _rank(node.args[0])
                perm = _resolve_perm_dims(node.args[1:], rank)
                out = f"{node.name}_permute"
                onnx_nodes.append(oh.make_node("Transpose", inputs=[x], outputs=[out], perm=perm))
                node_to_name[node] = out

            elif node.target == "matmul":
                out = f"{node.name}_matmul"
                onnx_nodes.append(oh.make_node("MatMul", inputs=[x, _res(node.args[1])], outputs=[out]))
                node_to_name[node] = out

            else:
                raise TypeError(f"Unsupported call_method for FX ONNX export: {node.target!r}")

        elif node.op == "output":
            ret = node.args[0]
            rets = list(ret) if isinstance(ret, (tuple, list)) else [ret]
            for r in rets:
                if not isinstance(r, _fx.Node):
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
