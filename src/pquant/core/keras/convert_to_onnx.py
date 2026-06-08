"""
Convert a PQuant Keras functional model to ONNX or QONNX format.

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

Keras weight layout (kernel always stored as HWIO regardless of data_format):
  Conv2D kernel:          [kH, kW, in/g, out] → [out, in/g, kH, kW] for ONNX
  Conv1D kernel:          [kL, in/g, out]     → [out, in/g, kL]     for ONNX
  DepthwiseConv2D kernel: [kH, kW, in, dm]    → [in*dm, 1, kH, kW]  for ONNX
  Dense kernel:           [in, out]           → stored as [out, in] for Gemm (transB=1)

Data format:
  channels_first  Data flows as NCHW; Conv/Pool ONNX ops work naturally.
  channels_last   Transpose(NHWC→NCHW) is inserted before each Conv/Pool/BN
                  node and Transpose(NCHW→NHWC) is inserted after.  The
                  logical data format in the ONNX graph therefore stays NHWC
                  at every inter-layer edge; only the PQ ops run internally in
                  NCHW.  ONNX-aware optimisers (e.g. onnxsim) can fold the
                  redundant back-to-back transposes away.
"""

import functools
import logging

import keras
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from keras import ops
from onnx import TensorProto

from pquant.core.keras.activations import PQActivation
from pquant.core.keras.layers import (
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
)

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
    """Build a QONNX Quant node.  k/i/f are numpy arrays.  Returns ([node], output_name)."""
    k_val = int(float(np.array(k).ravel()[0]))
    f_arr = np.array(f, dtype=np.float32)
    i_arr = np.array(i, dtype=np.float32)
    if f_arr.size > 1:
        i_arr = i_arr.ravel().max()
        f_arr = f_arr.ravel().min()
    i_val = float(i_arr)
    f_val = float(f_arr)
    scale = float(2.0 ** (-f_val))
    bit_width = float(k_val + i_val + f_val)
    qonnx_rnd = ROUND_MODE_MAP.get(rounding_mode, "ROUND")
    # SAT_SYM excludes the most-negative value → QONNX narrow=1
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
    """Build QuantizeLinear+DequantizeLinear nodes, optionally preceded by a Clip.

    Returns ([nodes], output_name).  Set include_clip=False to skip the Clip node
    (safe when values are guaranteed to be in-range at inference time, since
    QuantizeLinear saturates naturally).
    """
    k_val = int(float(np.array(k).ravel()[0]))
    i_val = float(np.array(i, dtype=np.float32).ravel()[0])
    f_val = float(np.array(f, dtype=np.float32).ravel()[0])
    scale = float(2.0 ** (-f_val))
    signed = k_val == 1

    clip_max = float(2.0**i_val - 2.0 ** (-f_val))
    if not signed:
        clip_min = 0.0
    elif overflow_mode == "SAT_SYM":
        clip_min = -clip_max  # symmetric: -(2^i - 2^(-f))
    else:
        clip_min = float(-(2.0**i_val))  # SAT: -2^i
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
# integer weight storage helper
# ---------------------------------------------------------------------------


def _int_weight_node(name_prefix, weight_np, k, i, f, initializers):  # noqa: ARG001 (i unused)
    """
    Store a weight tensor as int8/uint8 + DequantizeLinear.

    weight_np must already be in ONNX layout (transposed from Keras) and on the
    fixed-point grid after apply_final_compression().

    k/i/f are numpy arrays (may be per-tensor scalar or per-channel 1-D after
    caller has already squeezed/reshaped appropriately).

    Granularity:
    - per-tensor  (f is scalar): single scale.
    - per-channel (f is 1-D of length out_channels): axis=0 on weight tensor.
    - per-weight  (fully per-element): falls back to float32 storage.

    Returns ([node], output_name).
    """
    k_np = np.array(k, dtype=np.float32)
    f_np = np.array(f, dtype=np.float32)
    k_val = int(float(k_np.ravel()[0]))
    dtype = np.int8 if k_val == 1 else np.uint8
    out_channels = weight_np.shape[0]
    out_name = f"{name_prefix}_dequantized"

    if f_np.size == 1:
        # per-tensor
        scale_np = np.array(float(2.0 ** (-float(f_np.ravel()[0]))), dtype=np.float32)
        int_w = np.round(weight_np / float(scale_np)).astype(dtype)
        per_ch = False
    else:
        f_1d = f_np.ravel()
        if f_1d.size == out_channels:
            # per-channel: one f value per output channel
            scale_1d = (2.0 ** (-f_1d)).astype(np.float32)
            bcast = scale_1d.reshape((out_channels,) + (1,) * (weight_np.ndim - 1))
            int_w = np.round(weight_np / bcast).astype(dtype)
            scale_np = scale_1d
            per_ch = True
        else:
            # per-weight: ONNX cannot represent; fall back to float32
            float_name = f"{name_prefix}_float"
            initializers.append(onh.from_array(weight_np, name=float_name))
            return [], float_name

    int_name = f"{name_prefix}_int"
    scale_name = f"{name_prefix}_dq_scale"
    zp_name = f"{name_prefix}_dq_zp"

    zp_np = np.zeros(out_channels if per_ch else 1, dtype=dtype)
    initializers += [
        onh.from_array(int_w, name=int_name),
        onh.from_array(scale_np, name=scale_name),
        onh.from_array(zp_np if per_ch else np.array(dtype(0)), name=zp_name),
    ]
    node_kwargs = {"axis": 0} if per_ch else {}
    node = oh.make_node("DequantizeLinear", inputs=[int_name, scale_name, zp_name], outputs=[out_name], **node_kwargs)
    return [node], out_name


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(tensor):
    """Convert a Keras tensor / variable / scalar to a float32 numpy array."""
    return np.array(tensor, dtype=np.float32)


def _bn_transpose_info(layer):
    """
    Return (need_transpose, perm_fwd, perm_bwd) for a BatchNormalization layer.

    ONNX BN (opset < 14) always normalises on axis 1 (NCHW).  If the Keras
    layer uses axis=-1 (channels_last), we must insert Transpose nodes around
    the BN op.  We infer ndim from the layer's stored input_shape.
    """
    axis = getattr(layer, "axis", 1)
    stored = getattr(layer, "input_shape", None)
    ndim = len(stored) if stored is not None else 4
    eff_axis = axis if axis >= 0 else (ndim + axis)

    if eff_axis == 1 or ndim <= 2:
        # channels already at position 1, or 2-D input — no transpose needed
        return False, None, None

    if ndim == 4 and eff_axis == 3:
        return True, [0, 3, 1, 2], [0, 2, 3, 1]

    if ndim == 3 and eff_axis == 2:
        return True, [0, 2, 1], [0, 2, 1]

    # Fallback: general permutation that moves eff_axis to position 1
    perm_fwd = [0, eff_axis] + [i for i in range(1, ndim) if i != eff_axis]
    # Inverse permutation
    perm_bwd = [0] * ndim
    for i, p in enumerate(perm_fwd):
        perm_bwd[p] = i
    return True, perm_fwd, perm_bwd


def _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn):
    if (
        getattr(layer, "input_quantizer", None) is not None
        and getattr(layer, "quantize_input", True)
        and getattr(layer, "enable_quantization", True)
    ):
        q = layer.input_quantizer
        k, i, f = q.get_quantization_bits()
        new_nodes, current = quant_fn(
            f"{prefix}_in",
            current,
            q.round_mode,
            _np(k),
            _np(i),
            _np(f),
            initializers,
            overflow_mode=getattr(q, "overflow", "SAT"),
        )
        nodes.extend(new_nodes)
    return current


def _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn):
    if (
        getattr(layer, "output_quantizer", None) is not None
        and getattr(layer, "quantize_output", False)
        and getattr(layer, "enable_quantization", True)
    ):
        q = layer.output_quantizer
        k, i, f = q.get_quantization_bits()
        new_nodes, current = quant_fn(
            f"{prefix}_out",
            current,
            q.round_mode,
            _np(k),
            _np(i),
            _np(f),
            initializers,
            overflow_mode=getattr(q, "overflow", "SAT"),
        )
        nodes.extend(new_nodes)
    return current


def _add_transpose(name, input_name, perm, nodes):
    """Emit a Transpose node and return the output name."""
    out = f"{name}_transpose_{''.join(str(p) for p in perm)}"
    nodes.append(oh.make_node("Transpose", inputs=[input_name], outputs=[out], perm=list(perm)))
    return out


def _channels_last(layer):
    return getattr(layer, "data_format", "channels_first") == "channels_last"


def _weight_f_for_onnx(f_np, out_channels):
    """Squeeze/ravel a Keras per-channel f array to shape (out_channels,) for ONNX."""
    f_flat = f_np.ravel()
    if f_flat.size == 1:
        return f_flat  # scalar, return as-is
    if f_flat.size == out_channels:
        return f_flat
    # Per-element or mismatched: take the minimum to avoid overflow
    return np.array([f_flat.min()], dtype=np.float32)


# ---------------------------------------------------------------------------
# per-layer graph builders
# ---------------------------------------------------------------------------


def _add_dense(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """Dense / PQDense.  Keras kernel [in, out] stored as [out, in]; Gemm uses transB=1.

    Storing the weight pre-transposed means axis=0 is always the output dimension,
    which is required for per-channel DequantizeLinear and avoids a runtime Transpose.
    """
    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    # Transpose kernel to [out, in]; Gemm will use transB=1 so Y = X @ W^T = X @ kernel.
    kernel_np = _np(layer._kernel).T  # [out, in]
    out_units = kernel_np.shape[0]

    if use_qonnx:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        wfp_name = f"{prefix}_weight_fp"
        initializers.append(onh.from_array(kernel_np, name=wfp_name))
        w_nodes, q_weight = _quant_node(
            f"{prefix}_weight",
            wfp_name,
            layer.weight_quantizer.round_mode,
            _np(k_w),
            _np(i_w),
            _np(f_w),
            initializers,
            overflow_mode=getattr(layer.weight_quantizer, "overflow", "SAT"),
        )
        nodes.extend(w_nodes)
    elif store_integer_weights:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        f_np_w = _np(f_w)
        f_for_onnx = _weight_f_for_onnx(f_np_w, out_units)
        k_for_onnx = _weight_f_for_onnx(_np(k_w), out_units)
        i_for_onnx = _weight_f_for_onnx(_np(i_w), out_units)
        w_nodes, q_weight = _int_weight_node(f"{prefix}_weight", kernel_np, k_for_onnx, i_for_onnx, f_for_onnx, initializers)
        nodes.extend(w_nodes)
    else:
        q_weight = f"{prefix}_weight"
        initializers.append(onh.from_array(kernel_np, name=q_weight))

    gemm_inputs = [current, q_weight]

    if layer._bias is not None:
        bias_np = _np(layer._bias)
        if use_qonnx:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            bfp_name = f"{prefix}_bias_fp"
            initializers.append(onh.from_array(bias_np, name=bfp_name))
            b_nodes, q_bias = _quant_node(
                f"{prefix}_bias",
                bfp_name,
                layer.bias_quantizer.round_mode,
                _np(k_b),
                _np(i_b),
                _np(f_b),
                initializers,
                overflow_mode=getattr(layer.bias_quantizer, "overflow", "SAT"),
            )
            nodes.extend(b_nodes)
        elif store_integer_weights:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            b_nodes, q_bias = _int_weight_node(f"{prefix}_bias", bias_np, _np(k_b), _np(i_b), _np(f_b), initializers)
            nodes.extend(b_nodes)
        else:
            q_bias = f"{prefix}_bias"
            initializers.append(onh.from_array(bias_np, name=q_bias))
        gemm_inputs.append(q_bias)

    gemm_out = f"{prefix}_gemm"
    nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_out], transB=1))
    current = gemm_out

    current = _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)
    return current


def _add_conv(layer, prefix, current, nodes, initializers, ndim, quant_fn, use_qonnx, store_integer_weights):
    """PQConv2d / PQConv1d.  Keras kernel: [*kernel, in/g, out] → ONNX [out, in/g, *kernel]."""
    cl = _channels_last(layer)

    if cl:
        perm_to_nchw = [0, 3, 1, 2] if ndim == 2 else [0, 2, 1]
        perm_to_nhwx = [0, 2, 3, 1] if ndim == 2 else [0, 2, 1]
        current = _add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = _np(layer._kernel)
    # Transpose kernel from Keras HWIO to ONNX OIHW
    if ndim == 2:
        kernel_onnx = np.transpose(kernel_np, (3, 2, 0, 1))  # [kH,kW,in,out] → [out,in,kH,kW]
    else:
        kernel_onnx = np.transpose(kernel_np, (2, 1, 0))  # [kL,in,out]    → [out,in,kL]

    out_channels = kernel_onnx.shape[0]

    if use_qonnx:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        wfp_name = f"{prefix}_weight_fp"
        initializers.append(onh.from_array(kernel_onnx, name=wfp_name))
        w_nodes, q_weight = _quant_node(
            f"{prefix}_weight",
            wfp_name,
            layer.weight_quantizer.round_mode,
            _np(k_w),
            _np(i_w),
            _np(f_w),
            initializers,
            overflow_mode=getattr(layer.weight_quantizer, "overflow", "SAT"),
        )
        nodes.extend(w_nodes)
    elif store_integer_weights:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        f_for_onnx = _weight_f_for_onnx(_np(f_w), out_channels)
        k_for_onnx = _weight_f_for_onnx(_np(k_w), out_channels)
        i_for_onnx = _weight_f_for_onnx(_np(i_w), out_channels)
        w_nodes, q_weight = _int_weight_node(
            f"{prefix}_weight", kernel_onnx, k_for_onnx, i_for_onnx, f_for_onnx, initializers
        )
        nodes.extend(w_nodes)
    else:
        q_weight = f"{prefix}_weight"
        initializers.append(onh.from_array(kernel_onnx, name=q_weight))

    conv_inputs = [current, q_weight]

    if layer._bias is not None:
        bias_np = _np(layer._bias)
        if use_qonnx:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            bfp_name = f"{prefix}_bias_fp"
            initializers.append(onh.from_array(bias_np, name=bfp_name))
            b_nodes, q_bias = _quant_node(
                f"{prefix}_bias",
                bfp_name,
                layer.bias_quantizer.round_mode,
                _np(k_b),
                _np(i_b),
                _np(f_b),
                initializers,
                overflow_mode=getattr(layer.bias_quantizer, "overflow", "SAT"),
            )
            nodes.extend(b_nodes)
        elif store_integer_weights:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            b_nodes, q_bias = _int_weight_node(f"{prefix}_bias", bias_np, _np(k_b), _np(i_b), _np(f_b), initializers)
            nodes.extend(b_nodes)
        else:
            q_bias = f"{prefix}_bias"
            initializers.append(onh.from_array(bias_np, name=q_bias))
        conv_inputs.append(q_bias)

    # Padding
    padding = layer.padding
    if isinstance(padding, str):
        auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
        pads = None
    else:
        p = list(padding) if hasattr(padding, "__iter__") else [padding] * ndim
        pads = p + p  # ONNX format: [begin_0, begin_1, ..., end_0, end_1, ...]
        auto_pad = "NOTSET"

    to_list = lambda v, n: list(v) if hasattr(v, "__iter__") else [v] * n  # noqa: E731
    conv_attrs = dict(
        kernel_shape=to_list(layer.kernel_size, ndim),
        strides=to_list(layer.strides, ndim),
        dilations=to_list(layer.dilation_rate, ndim),
        group=getattr(layer, "groups", 1),
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    current = _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if cl:
        current = _add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def _add_depthwise_conv(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """PQDepthwiseConv2d.

    Keras kernel: [kH, kW, in, depth_mult]
    ONNX Conv with groups=in: weight [in*depth_mult, 1, kH, kW]
    """
    cl = _channels_last(layer)

    if cl:
        current = _add_transpose(f"{prefix}_pre", current, [0, 3, 1, 2], nodes)

    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = _np(layer._kernel)  # [kH, kW, in, depth_mult]
    in_ch, depth_mult = kernel_np.shape[2], kernel_np.shape[3]
    # Rearrange to [in*depth_mult, 1, kH, kW]
    kernel_onnx = np.transpose(kernel_np, (2, 3, 0, 1)).reshape(in_ch * depth_mult, 1, *kernel_np.shape[:2])

    out_channels = kernel_onnx.shape[0]

    if use_qonnx:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        wfp_name = f"{prefix}_weight_fp"
        initializers.append(onh.from_array(kernel_onnx, name=wfp_name))
        w_nodes, q_weight = _quant_node(
            f"{prefix}_weight",
            wfp_name,
            layer.weight_quantizer.round_mode,
            _np(k_w),
            _np(i_w),
            _np(f_w),
            initializers,
            overflow_mode=getattr(layer.weight_quantizer, "overflow", "SAT"),
        )
        nodes.extend(w_nodes)
    elif store_integer_weights:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        f_for_onnx = _weight_f_for_onnx(_np(f_w), out_channels)
        k_for_onnx = _weight_f_for_onnx(_np(k_w), out_channels)
        i_for_onnx = _weight_f_for_onnx(_np(i_w), out_channels)
        w_nodes, q_weight = _int_weight_node(
            f"{prefix}_weight", kernel_onnx, k_for_onnx, i_for_onnx, f_for_onnx, initializers
        )
        nodes.extend(w_nodes)
    else:
        q_weight = f"{prefix}_weight"
        initializers.append(onh.from_array(kernel_onnx, name=q_weight))

    conv_inputs = [current, q_weight]

    if layer._bias is not None:
        bias_np = _np(layer._bias)
        if use_qonnx:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            bfp_name = f"{prefix}_bias_fp"
            initializers.append(onh.from_array(bias_np, name=bfp_name))
            b_nodes, q_bias = _quant_node(
                f"{prefix}_bias",
                bfp_name,
                layer.bias_quantizer.round_mode,
                _np(k_b),
                _np(i_b),
                _np(f_b),
                initializers,
                overflow_mode=getattr(layer.bias_quantizer, "overflow", "SAT"),
            )
            nodes.extend(b_nodes)
        elif store_integer_weights:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            b_nodes, q_bias = _int_weight_node(f"{prefix}_bias", bias_np, _np(k_b), _np(i_b), _np(f_b), initializers)
            nodes.extend(b_nodes)
        else:
            q_bias = f"{prefix}_bias"
            initializers.append(onh.from_array(bias_np, name=q_bias))
        conv_inputs.append(q_bias)

    padding = layer.padding
    if isinstance(padding, str):
        auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
        pads = None
    else:
        p = list(padding) if hasattr(padding, "__iter__") else [padding, padding]
        pads = p + p
        auto_pad = "NOTSET"

    to_list = lambda v, n: list(v) if hasattr(v, "__iter__") else [v] * n  # noqa: E731
    conv_attrs = dict(
        kernel_shape=to_list(layer.kernel_size, 2),
        strides=to_list(layer.strides, 2),
        dilations=to_list(layer.dilation_rate, 2),
        group=in_ch,
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    current = _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if cl:
        current = _add_transpose(f"{prefix}_post", current, [0, 2, 3, 1], nodes)
    return current


def _add_batchnorm(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """PQBatchNormalization / standard BatchNormalization."""
    need_tr, perm_to_nchw, perm_to_nhwx = _bn_transpose_info(layer)

    if need_tr:
        current = _add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    is_pq = isinstance(layer, PQBatchNormalization)

    gamma_np = _np(layer.gamma) if layer.gamma is not None else None
    beta_np = _np(layer.beta) if layer.beta is not None else None

    if gamma_np is None:
        # scale=False: use ones
        n_ch = _np(layer.moving_mean).shape[0]
        gamma_np = np.ones(n_ch, dtype=np.float32)
    if beta_np is None:
        # center=False: use zeros
        n_ch = _np(layer.moving_mean).shape[0]
        beta_np = np.zeros(n_ch, dtype=np.float32)

    if is_pq and use_qonnx:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        gfp = f"{prefix}_gamma_fp"
        initializers.append(onh.from_array(gamma_np, name=gfp))
        g_nodes, q_gamma = _quant_node(
            f"{prefix}_gamma",
            gfp,
            layer.weight_quantizer.round_mode,
            _np(k_w),
            _np(i_w),
            _np(f_w),
            initializers,
            overflow_mode=getattr(layer.weight_quantizer, "overflow", "SAT"),
        )
        nodes.extend(g_nodes)

        k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
        bfp = f"{prefix}_beta_fp"
        initializers.append(onh.from_array(beta_np, name=bfp))
        b_nodes, q_beta = _quant_node(
            f"{prefix}_beta",
            bfp,
            layer.bias_quantizer.round_mode,
            _np(k_b),
            _np(i_b),
            _np(f_b),
            initializers,
            overflow_mode=getattr(layer.bias_quantizer, "overflow", "SAT"),
        )
        nodes.extend(b_nodes)
    elif is_pq and store_integer_weights:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        g_nodes, q_gamma = _int_weight_node(f"{prefix}_gamma", gamma_np, _np(k_w), _np(i_w), _np(f_w), initializers)
        nodes.extend(g_nodes)
        k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
        b_nodes, q_beta = _int_weight_node(f"{prefix}_beta", beta_np, _np(k_b), _np(i_b), _np(f_b), initializers)
        nodes.extend(b_nodes)
    else:
        q_gamma = f"{prefix}_gamma"
        q_beta = f"{prefix}_beta"
        initializers.append(onh.from_array(gamma_np, name=q_gamma))
        initializers.append(onh.from_array(beta_np, name=q_beta))

    mean_name = f"{prefix}_running_mean"
    var_name = f"{prefix}_running_var"
    initializers.append(onh.from_array(_np(layer.moving_mean), name=mean_name))
    initializers.append(onh.from_array(_np(layer.moving_variance), name=var_name))

    bn_out = f"{prefix}_bn"
    nodes.append(
        oh.make_node(
            "BatchNormalization",
            inputs=[current, q_gamma, q_beta, mean_name, var_name],
            outputs=[bn_out],
            epsilon=float(layer.epsilon),
        )
    )
    current = bn_out

    if need_tr:
        current = _add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def _add_dense_nd(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """PQDense for rank-3 inputs (B, T, E).

    Uses MatMul + Add instead of Gemm so the op works for any rank ≥ 2.
    The kernel is stored as [out, in] (same layout as _add_dense / _int_weight_node),
    then transposed to [in, out] at runtime via a Transpose node so that
    MatMul(input, kernel_t) broadcasts correctly over the sequence dimension.
    """
    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = _np(layer._kernel).T  # [out, in]
    out_units = kernel_np.shape[0]

    if use_qonnx:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        wfp_name = f"{prefix}_weight_fp"
        initializers.append(onh.from_array(kernel_np, name=wfp_name))
        w_nodes, q_weight = _quant_node(
            f"{prefix}_weight",
            wfp_name,
            layer.weight_quantizer.round_mode,
            _np(k_w),
            _np(i_w),
            _np(f_w),
            initializers,
            overflow_mode=getattr(layer.weight_quantizer, "overflow", "SAT"),
        )
        nodes.extend(w_nodes)
    elif store_integer_weights:
        k_w, i_w, f_w = layer.weight_quantizer.get_quantization_bits()
        f_for_onnx = _weight_f_for_onnx(_np(f_w), out_units)
        k_for_onnx = _weight_f_for_onnx(_np(k_w), out_units)
        i_for_onnx = _weight_f_for_onnx(_np(i_w), out_units)
        w_nodes, q_weight = _int_weight_node(f"{prefix}_weight", kernel_np, k_for_onnx, i_for_onnx, f_for_onnx, initializers)
        nodes.extend(w_nodes)
    else:
        q_weight = f"{prefix}_weight"
        initializers.append(onh.from_array(kernel_np, name=q_weight))

    # Transpose [out, in] → [in, out] so MatMul(input[..., in], kernel_t[in, out]) works
    kernel_t_name = f"{prefix}_weight_t"
    nodes.append(oh.make_node("Transpose", inputs=[q_weight], outputs=[kernel_t_name], perm=[1, 0]))

    mm_out = f"{prefix}_mm"
    nodes.append(oh.make_node("MatMul", inputs=[current, kernel_t_name], outputs=[mm_out]))
    current = mm_out

    if layer._bias is not None:
        bias_np = _np(layer._bias)
        if use_qonnx:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            bfp_name = f"{prefix}_bias_fp"
            initializers.append(onh.from_array(bias_np, name=bfp_name))
            b_nodes, q_bias = _quant_node(
                f"{prefix}_bias",
                bfp_name,
                layer.bias_quantizer.round_mode,
                _np(k_b),
                _np(i_b),
                _np(f_b),
                initializers,
                overflow_mode=getattr(layer.bias_quantizer, "overflow", "SAT"),
            )
            nodes.extend(b_nodes)
        elif store_integer_weights:
            k_b, i_b, f_b = layer.bias_quantizer.get_quantization_bits()
            b_nodes, q_bias = _int_weight_node(f"{prefix}_bias", bias_np, _np(k_b), _np(i_b), _np(f_b), initializers)
            nodes.extend(b_nodes)
        else:
            q_bias = f"{prefix}_bias"
            initializers.append(onh.from_array(bias_np, name=q_bias))
        add_out = f"{prefix}_bias_add"
        nodes.append(oh.make_node("Add", inputs=[current, q_bias], outputs=[add_out]))
        current = add_out

    current = _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)
    return current


def _add_mha(layer, prefix, q_input, k_input, v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """Build ONNX nodes for PQMultiheadAttention (Keras version, always batch-first).

    Decomposes multi-head attention into primitive ONNX ops:

      Q/K/V MatMul projections  (rank-3 MatMul via _add_dense_nd)
      Reshape (B, L, E) → (B, H, L, head_dim) + Transpose
      MatMul(Q, K^T) * scale  →  optional Quant
      Softmax  →  optional Quant
      MatMul(attn_weights, V)  →  optional context Quant
      Transpose + Reshape → (B, T, E)
      out_proj MatMul

    Returns (out_name, avg_attn_weights_name).
    """
    H = layer.num_heads
    head_dim = layer.head_dim
    E = layer.embed_dim
    scale_val = float(layer.scale)

    # --- Q / K / V projections: (B, L, E) → (B, L, E) ---
    q_proj_out = _add_dense_nd(
        layer.q_proj, f"{prefix}_q_proj", q_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    k_proj_out = _add_dense_nd(
        layer.k_proj, f"{prefix}_k_proj", k_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    v_proj_out = _add_dense_nd(
        layer.v_proj, f"{prefix}_v_proj", v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
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

    # --- Optional attn-score quantization ---
    if (
        getattr(layer, "quantize_attn_scores", False)
        and hasattr(layer, "attn_score_quantizer")
        and getattr(layer, "enable_quantization", True)
    ):
        q = layer.attn_score_quantizer
        k_q, i_q, f_q = q.get_quantization_bits()
        q_nodes, current = quant_fn(
            f"{prefix}_attn_score_q",
            current,
            q.round_mode,
            _np(k_q),
            _np(i_q),
            _np(f_q),
            initializers,
            overflow_mode=getattr(q, "overflow", "SAT"),
        )
        nodes.extend(q_nodes)

    # --- Softmax (axis=-1); approximate_softmax falls back to standard Softmax in ONNX ---
    attn_w_name = f"{prefix}_attn_weights"
    nodes.append(oh.make_node("Softmax", inputs=[current], outputs=[attn_w_name], axis=-1))
    current = attn_w_name

    # --- Optional attn-weight quantization ---
    if (
        getattr(layer, "quantize_attn_weights", False)
        and hasattr(layer, "attn_weight_quantizer")
        and getattr(layer, "enable_quantization", True)
    ):
        q = layer.attn_weight_quantizer
        k_q, i_q, f_q = q.get_quantization_bits()
        q_nodes, current = quant_fn(
            f"{prefix}_attn_weight_q",
            current,
            q.round_mode,
            _np(k_q),
            _np(i_q),
            _np(f_q),
            initializers,
            overflow_mode=getattr(q, "overflow", "SAT"),
        )
        nodes.extend(q_nodes)

    # --- Context: (B, H, T, S) @ (B, H, S, head_dim) → (B, H, T, head_dim) ---
    ctx_raw = f"{prefix}_ctx_raw"
    nodes.append(oh.make_node("MatMul", inputs=[current, v_h], outputs=[ctx_raw]))
    current_ctx = ctx_raw

    # --- Optional context quantization ---
    if (
        getattr(layer, "quantize_context", False)
        and hasattr(layer, "context_quantizer")
        and getattr(layer, "enable_quantization", True)
    ):
        q = layer.context_quantizer
        k_q, i_q, f_q = q.get_quantization_bits()
        q_nodes, current_ctx = quant_fn(
            f"{prefix}_context_q",
            current_ctx,
            q.round_mode,
            _np(k_q),
            _np(i_q),
            _np(f_q),
            initializers,
            overflow_mode=getattr(q, "overflow", "SAT"),
        )
        nodes.extend(q_nodes)

    # --- Merge heads: (B, H, T, head_dim) → (B, T, E) using dynamic shapes ---
    ctx_t = f"{prefix}_ctx_t"
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

    # --- Output projection: (B, T, E) → (B, T, E) ---
    out = _add_dense_nd(
        layer.out_proj, f"{prefix}_out_proj", ctx_merged, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    # --- Average attention weights over heads: (B, H, T, S) → (B, T, S) ---
    avg_attn = f"{prefix}_avg_attn_weights"
    nodes.append(oh.make_node("ReduceMean", inputs=[attn_w_name], outputs=[avg_attn], axes=[1], keepdims=0))

    return out, avg_attn


def _add_avgpool(layer, prefix, current, nodes, initializers, ndim, quant_fn):
    cl = _channels_last(layer)

    if cl:
        perm_to_nchw = [0, 3, 1, 2] if ndim == 2 else [0, 2, 1]
        perm_to_nhwx = [0, 2, 3, 1] if ndim == 2 else [0, 2, 1]
        current = _add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    to_list = lambda v, n: list(v) if hasattr(v, "__iter__") else [v] * n  # noqa: E731

    pool_out = f"{prefix}_pool"
    nodes.append(
        oh.make_node(
            "AveragePool",
            inputs=[current],
            outputs=[pool_out],
            kernel_shape=to_list(layer.pool_size, ndim),
            strides=to_list(layer.strides, ndim),
            pads=[0] * (ndim * 2),
            count_include_pad=0,
        )
    )
    current = pool_out

    current = _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if cl:
        current = _add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def _add_global_avgpool(layer, prefix, current, nodes, ndim):
    cl = _channels_last(layer)

    if cl:
        perm_to_nchw = [0, 3, 1, 2] if ndim == 2 else [0, 2, 1]
        current = _add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    pool_out = f"{prefix}_global_pool"
    nodes.append(oh.make_node("GlobalAveragePool", inputs=[current], outputs=[pool_out]))
    current = pool_out

    if cl:
        # GlobalAveragePool returns [N, C, 1, 1]; emit Flatten to [N, C].
        # Actually after GlobalAveragePool output is [N, C, 1, 1]; transpose back would give
        # [N, 1, 1, C] which then needs squeezing — that's the same as just squeezing [N, C].
        # Emit Flatten to [N, C] instead of bothering with transpose.
        flatten_name = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[pool_out], outputs=[flatten_name], axis=1))
        current = flatten_name

    return current


def _add_pq_activation(layer, prefix, current, nodes, initializers, quant_fn):
    """PQActivation: [input QDQ] → [multiplier scale] → activation → [output QDQ].

    Supported activations: relu, tanh, hard_tanh (= Clip(-1, 1)).

    The optional relu multiplier is baked to a constant: 2^round(m).
    """
    # --- optional input quantization ---
    current = _maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    # --- optional learnable multiplier (relu only) ---
    if (
        getattr(layer, "use_multiplier", False)
        and getattr(layer, "activation_name", "") == "relu"
        and hasattr(layer, "multiplier")
    ):
        m_val = float(np.array(layer.multiplier).ravel()[0])
        scale = float(2.0 ** round(m_val))
        scale_name = f"{prefix}_mul_scale"
        scaled_out = f"{prefix}_scaled"
        initializers.append(onh.from_array(np.array(scale, dtype=np.float32), name=scale_name))
        nodes.append(oh.make_node("Mul", inputs=[current, scale_name], outputs=[scaled_out]))
        current = scaled_out

    # --- activation ---
    act = getattr(layer, "activation_name", "relu")
    act_out = f"{prefix}_act"
    if act == "relu":
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[act_out]))
    elif act == "tanh":
        nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[act_out]))
    elif act == "hard_tanh":
        # hard_tanh(x) = clip(x, -1, 1)
        cmin_name = f"{prefix}_htanh_min"
        cmax_name = f"{prefix}_htanh_max"
        initializers += [
            onh.from_array(np.array(-1.0, dtype=np.float32), name=cmin_name),
            onh.from_array(np.array(1.0, dtype=np.float32), name=cmax_name),
        ]
        nodes.append(oh.make_node("Clip", inputs=[current, cmin_name, cmax_name], outputs=[act_out]))
    else:
        raise TypeError(f"PQActivation: unsupported activation {act!r} for ONNX export")
    current = act_out

    # --- optional output quantization ---
    current = _maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)
    return current


# ---------------------------------------------------------------------------
# shared layer dispatcher
# ---------------------------------------------------------------------------


def _emit_layer(
    layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, input_onnx_names=None
):
    """Emit ONNX nodes for a single Keras layer.  Returns the ONNX output name."""

    # --- PQuant layers ---
    if isinstance(layer, PQMultiheadAttention):
        # input_onnx_names = [query, key, value] or [single_input] for self-attention
        if len(input_onnx_names) >= 3:
            q_in, k_in, v_in = input_onnx_names[0], input_onnx_names[1], input_onnx_names[2]
        elif len(input_onnx_names) == 2:
            q_in, k_in, v_in = input_onnx_names[0], input_onnx_names[1], input_onnx_names[1]
        else:
            q_in = k_in = v_in = input_onnx_names[0]
        return _add_mha(layer, prefix, q_in, k_in, v_in, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, PQActivation):
        return _add_pq_activation(layer, prefix, current, nodes, initializers, quant_fn)

    if isinstance(layer, PQDense):
        return _add_dense(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, PQDepthwiseConv2d):
        return _add_depthwise_conv(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, PQConv2d):
        return _add_conv(
            layer,
            prefix,
            current,
            nodes,
            initializers,
            ndim=2,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )

    if isinstance(layer, PQConv1d):
        return _add_conv(
            layer,
            prefix,
            current,
            nodes,
            initializers,
            ndim=1,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )

    if isinstance(layer, PQBatchNormalization):
        return _add_batchnorm(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    # --- Standard Keras layers ---
    if isinstance(layer, keras.layers.BatchNormalization):
        return _add_batchnorm(
            layer, prefix, current, nodes, initializers, quant_fn=quant_fn, use_qonnx=False, store_integer_weights=False
        )

    if isinstance(layer, keras.layers.Conv2D):
        # Plain Conv2D (non-PQ): wrap in a minimal shim
        _layer = layer
        _layer._bias = layer.bias
        return _add_conv_plain(layer, prefix, current, nodes, initializers)

    if isinstance(layer, keras.layers.Dense):
        out = f"{prefix}_gemm"
        w_name = f"{prefix}_weight"
        initializers.append(onh.from_array(_np(layer.kernel).T, name=w_name))  # store [out, in]
        gemm_inputs = [current, w_name]
        if layer.bias is not None:
            b_name = f"{prefix}_bias"
            initializers.append(onh.from_array(_np(layer.bias), name=b_name))
            gemm_inputs.append(b_name)
        nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[out], transB=1))
        return out

    if isinstance(layer, (keras.layers.ReLU, keras.layers.Activation)):
        activation = (
            layer.activation.__name__
            if isinstance(layer, keras.layers.Activation) and callable(layer.activation)
            else getattr(layer, "activation", "relu")
        )
        act_name = activation if isinstance(activation, str) else "relu"
        out = f"{prefix}_act"
        if "relu" in act_name.lower():
            nodes.append(oh.make_node("Relu", inputs=[current], outputs=[out]))
        elif "sigmoid" in act_name.lower():
            nodes.append(oh.make_node("Sigmoid", inputs=[current], outputs=[out]))
        elif "tanh" in act_name.lower():
            nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[out]))
        else:
            raise TypeError(f"Unsupported Activation for ONNX export: {act_name!r}")
        return out

    if isinstance(layer, keras.layers.Flatten):
        out = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[current], outputs=[out], axis=1))
        return out

    if isinstance(layer, keras.layers.Reshape):
        target_shape = list(layer.target_shape)
        # Prepend batch dim (-1 means dynamic)
        full_shape = [-1] + target_shape
        shape_name = f"{prefix}_shape"
        out = f"{prefix}_reshape"
        initializers.append(onh.from_array(np.array(full_shape, dtype=np.int64), name=shape_name))
        nodes.append(oh.make_node("Reshape", inputs=[current, shape_name], outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Add):
        assert input_onnx_names is not None and len(input_onnx_names) == 2
        out = f"{prefix}_add"
        nodes.append(oh.make_node("Add", inputs=input_onnx_names, outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Concatenate):
        assert input_onnx_names is not None
        axis = layer.axis
        # Negative axis: leave as-is; onnx Concat supports negative axes
        out = f"{prefix}_concat"
        nodes.append(oh.make_node("Concat", inputs=input_onnx_names, outputs=[out], axis=axis))
        return out

    if isinstance(layer, keras.layers.Multiply):
        assert input_onnx_names is not None and len(input_onnx_names) == 2
        out = f"{prefix}_mul"
        nodes.append(oh.make_node("Mul", inputs=input_onnx_names, outputs=[out]))
        return out

    if isinstance(layer, keras.layers.AveragePooling2D):
        return _add_avgpool(layer, prefix, current, nodes, initializers, ndim=2, quant_fn=quant_fn)

    if isinstance(layer, keras.layers.AveragePooling1D):
        return _add_avgpool(layer, prefix, current, nodes, initializers, ndim=1, quant_fn=quant_fn)

    if isinstance(layer, keras.layers.GlobalAveragePooling2D):
        return _add_global_avgpool(layer, prefix, current, nodes, ndim=2)

    if isinstance(layer, keras.layers.GlobalAveragePooling1D):
        return _add_global_avgpool(layer, prefix, current, nodes, ndim=1)

    if isinstance(layer, (keras.layers.Dropout,)):
        return current  # identity at inference

    raise TypeError(f"Unsupported Keras layer type for ONNX export: {type(layer).__name__!r}")


def _add_conv_plain(layer, prefix, current, nodes, initializers):
    """Emit a plain (non-PQ) Conv2D layer."""
    cl = _channels_last(layer)
    if cl:
        current = _add_transpose(f"{prefix}_pre", current, [0, 3, 1, 2], nodes)

    kernel_np = _np(layer.kernel)
    kernel_onnx = np.transpose(kernel_np, (3, 2, 0, 1))
    w_name = f"{prefix}_weight"
    initializers.append(onh.from_array(kernel_onnx, name=w_name))
    conv_inputs = [current, w_name]

    if layer.bias is not None:
        b_name = f"{prefix}_bias"
        initializers.append(onh.from_array(_np(layer.bias), name=b_name))
        conv_inputs.append(b_name)

    padding = layer.padding
    auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
    to_list = lambda v, n: list(v) if hasattr(v, "__iter__") else [v] * n  # noqa: E731
    conv_attrs = dict(
        kernel_shape=to_list(layer.kernel_size, 2),
        strides=to_list(layer.strides, 2),
        dilations=to_list(layer.dilation_rate, 2),
        group=layer.groups,
        auto_pad=auto_pad,
    )
    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    if cl:
        current = _add_transpose(f"{prefix}_post", current, [0, 2, 3, 1], nodes)
    return current


# ---------------------------------------------------------------------------
# Keras functional model graph traversal
# ---------------------------------------------------------------------------


def _build_tensor_onnx_map(model):
    """
    Return a dict mapping id(KerasTensor) → ONNX tensor name for model.inputs.
    Multi-input models are supported; inputs are named "input_0", "input_1", etc.
    (or just "input" for single-input models).
    """
    tensor_to_onnx = {}
    for i, inp in enumerate(model.inputs):
        name = "input" if len(model.inputs) == 1 else f"input_{i}"
        tensor_to_onnx[id(inp)] = name
    return tensor_to_onnx


def _inbound_input_names(layer, tensor_to_onnx):
    """Return the list of ONNX input names for this layer based on its inbound node."""
    if not layer._inbound_nodes:
        return []
    node = layer._inbound_nodes[0]
    input_tensors = node.input_tensors
    if not isinstance(input_tensors, (list, tuple)):
        input_tensors = [input_tensors]
    result = []
    for t in input_tensors:
        key = id(t)
        if key not in tensor_to_onnx:
            raise RuntimeError(
                f"Layer {layer.name!r}: input tensor not found in tensor_to_onnx map. "
                "Ensure model.layers is in topological order."
            )
        result.append(tensor_to_onnx[key])
    return result


def _register_layer_output(layer, onnx_name, tensor_to_onnx):
    """Register the ONNX output name for a layer's output tensor(s).

    onnx_name may be a plain string (single-output layer) or a tuple of strings
    (multi-output layer, e.g. PQMultiheadAttention returns (out, avg_attn_weights)).
    """
    if not layer._inbound_nodes:
        return
    node = layer._inbound_nodes[0]
    out_tensors = node.output_tensors
    if not isinstance(out_tensors, (list, tuple)):
        out_tensors = [out_tensors]
    if isinstance(onnx_name, (list, tuple)):
        for tensor, name in zip(out_tensors, onnx_name):
            tensor_to_onnx[id(tensor)] = name
    else:
        tensor_to_onnx[id(out_tensors[0])] = onnx_name


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------


def convert_to_onnx(
    model: keras.Model,
    input_shape: tuple,
    output_path: str = "model.onnx",
    opset: int = 13,
    use_qonnx: bool = False,
    store_integer_weights: bool = False,
    include_clip: bool = True,
    batch_size: int | None = None,
) -> onnx.ModelProto:
    """
    Convert a Keras functional model of PQuant layers to ONNX or QONNX.

    The model must have apply_final_compression() called on all PQ layers
    before calling this function.  Only inference-mode semantics are exported.

    Args:
        model:                  Trained keras.Model.  Must be a functional model
                                (built with the Keras functional API or subclassed
                                models whose layers are accessible via model.layers).
        input_shape:            Shape of a single sample excluding batch, e.g. (3, 32, 32).
                                For channels_last Conv models use e.g. (32, 32, 3).
        output_path:            Where to save the .onnx file.
        opset:                  ONNX opset version (≥13 required for per-channel
                                DequantizeLinear).
        use_qonnx:              Emit QONNX Quant custom nodes if True.
        store_integer_weights:  Store weight initializers as int8/uint8 +
                                DequantizeLinear instead of float32 (ignored when
                                use_qonnx=True).
        include_clip:           Prepend a Clip node before each QuantizeLinear when
                                True (default).  Set to False to emit bare
                                QuantizeLinear+DequantizeLinear pairs — safe when
                                values are guaranteed in-range at inference time since
                                QuantizeLinear saturates naturally.  Ignored when
                                use_qonnx=True.
        batch_size:             If not None, fix the batch dimension of all graph
                                inputs and outputs to this value.  If None (default),
                                the batch dimension is left dynamic.

    Returns:
        The constructed onnx.ModelProto.
    """
    quant_fn = _quant_node if use_qonnx else functools.partial(_qdq_node, include_clip=include_clip)

    onnx_nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    tensor_to_onnx = _build_tensor_onnx_map(model)
    last_output_name: str = ""

    for layer in model.layers:
        # Skip InputLayer — already seeded in tensor_to_onnx
        if isinstance(layer, keras.layers.InputLayer):
            continue

        input_onnx_names = _inbound_input_names(layer, tensor_to_onnx)
        if not input_onnx_names:
            continue

        current = input_onnx_names[0]  # primary input (used by single-input layers)
        prefix = layer.name.replace("/", "_").replace(":", "_")

        output_name = _emit_layer(
            layer,
            prefix,
            current,
            onnx_nodes,
            initializers,
            quant_fn,
            use_qonnx,
            store_integer_weights,
            input_onnx_names=input_onnx_names,
        )

        _register_layer_output(layer, output_name, tensor_to_onnx)
        # For multi-output layers (e.g. MHA returns (out, avg_attn)), track only the
        # primary output as the graph's last output name.
        last_output_name = output_name[0] if isinstance(output_name, tuple) else output_name

    # Determine output shape via a forward pass
    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    dummy_out = model(dummy, training=False)
    dummy_out_np = np.array(ops.convert_to_numpy(dummy_out))
    batch_dim = batch_size  # None → dynamic, int → fixed
    output_shape = [batch_dim] + list(dummy_out_np.shape[1:])

    # Build ONNX graph
    if len(model.inputs) == 1:
        input_vis = [oh.make_tensor_value_info("input", TensorProto.FLOAT, [batch_dim, *input_shape])]
    else:
        input_vis = [
            oh.make_tensor_value_info(f"input_{i}", TensorProto.FLOAT, [batch_dim, *input_shape])
            for i in range(len(model.inputs))
        ]
    output_vi = oh.make_tensor_value_info(last_output_name, TensorProto.FLOAT, output_shape)

    graph = oh.make_graph(
        nodes=onnx_nodes,
        name="pquant_keras_onnx",
        inputs=input_vis,
        outputs=[output_vi],
        initializer=initializers,
    )

    opset_imports = [oh.make_opsetid("", opset)]
    if use_qonnx:
        opset_imports.append(oh.make_opsetid("qonnx.custom_op.general", 1))
    model_proto = oh.make_model(graph, opset_imports=opset_imports)
    model_proto.ir_version = 6

    # ONNX opset >= 9: initializers are implicit constants and must NOT appear in
    # graph.input — otherwise tools treat weight tensors as runtime inputs.
    # Some onnx library versions add them automatically for backward compatibility;
    # strip them here so only the actual data inputs remain.
    _init_names = {t.name for t in model_proto.graph.initializer}
    _data_inputs = [vi for vi in model_proto.graph.input if vi.name not in _init_names]
    del model_proto.graph.input[:]
    model_proto.graph.input.extend(_data_inputs)

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    fmt = "QONNX" if use_qonnx else "ONNX (QDQ)"
    logging.info("Saved %s Keras model → %s", fmt, output_path)
    return model_proto


# ---------------------------------------------------------------------------
# usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pquant
    from pquant import apply_final_compression

    cfg = pquant.pdp_config()

    inp = keras.Input(shape=(3, 32, 32))
    x = PQConv2d(cfg, filters=16, kernel_size=3, padding="same")(inp)
    x = PQBatchNormalization(cfg)(x)
    x = keras.layers.ReLU()(x)
    x = PQConv2d(cfg, filters=32, kernel_size=3, padding="same")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = PQDense(cfg, units=10)(x)
    model = keras.Model(inp, x)

    apply_final_compression(model)

    convert_to_onnx(model, input_shape=(3, 32, 32), output_path="model_keras.onnx")

    import onnxruntime as ort

    sess = ort.InferenceSession("model_keras.onnx")
    out = sess.run(None, {"input": np.random.randn(2, 3, 32, 32).astype(np.float32)})
    print("Output shape:", out[0].shape)  # noqa: T201
