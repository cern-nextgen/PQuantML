"""
Low-level ONNX node emitters and small utilities shared by the PQuant
torch → ONNX converter.

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
"""

import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh

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


def quant_node(name_prefix, input_name, rounding_mode, k, i, f, initializers, overflow_mode="SAT"):
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


def qdq_node(
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


def int_weight_node(name_prefix, weight_np, k, i, f, initializers):  # noqa: ARG001 (i unused)
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


def torch_padding_to_onnx(padding, ndim):
    if isinstance(padding, int):
        padding = (padding,) * ndim
    return list(padding) + list(padding)


def to_list(v, n):
    """Normalize a scalar-or-sequence layer attribute (kernel/stride/...) to an n-length list."""
    return list(v) if hasattr(v, "__iter__") else [v] * n


def maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn):
    # input_quantizer is created conditionally, so guard it; the bool flags are always present.
    if getattr(module, "input_quantizer", None) is not None and module.quantize_input and module.enable_quantization:
        q = module.input_quantizer
        k, i, f = q.get_quantization_bits()
        new_nodes, current = quant_fn(f"{prefix}_in", current, q.round_mode, k, i, f, initializers, overflow_mode=q.overflow)
        nodes.extend(new_nodes)
    return current


def maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn):
    if getattr(module, "output_quantizer", None) is not None and module.quantize_output and module.enable_quantization:
        q = module.output_quantizer
        k, i, f = q.get_quantization_bits()
        new_nodes, current = quant_fn(
            f"{prefix}_out", current, q.round_mode, k, i, f, initializers, overflow_mode=q.overflow
        )
        nodes.extend(new_nodes)
    return current


def emit_param(prefix, name, arr, quantizer, nodes, initializers, use_qonnx, store_integer_weights):
    if use_qonnx:
        fp_name = f"{prefix}_{name}_fp"
        initializers.append(onh.from_array(arr, name=fp_name))
        k, i, f = quantizer.get_quantization_bits()
        q_nodes, out = quant_node(
            f"{prefix}_{name}", fp_name, quantizer.round_mode, k, i, f, initializers, overflow_mode=quantizer.overflow
        )
        nodes.extend(q_nodes)
        return out
    if store_integer_weights:
        k, i, f = quantizer.get_quantization_bits()
        q_nodes, out = int_weight_node(f"{prefix}_{name}", arr, k, i, f, initializers)
        nodes.extend(q_nodes)
        return out
    out = f"{prefix}_{name}"
    initializers.append(onh.from_array(arr, name=out))
    return out


def emit_getitem(prefix, input_name, spec, rank, nodes, initializers):
    """Translate a constant Python indexing spec into ONNX Slice (+ Squeeze)."""
    if not isinstance(spec, tuple):
        spec = (spec,)
    n_ellipsis = sum(1 for s in spec if s is Ellipsis)
    if n_ellipsis > 1:
        raise TypeError("indexing with more than one Ellipsis is not supported in ONNX export")
    if n_ellipsis:
        pos = spec.index(Ellipsis)
        fill = rank - (len(spec) - 1)
        spec = spec[:pos] + (slice(None),) * fill + spec[pos + 1 :]
    if len(spec) > rank:
        raise TypeError(f"indexing spec has {len(spec)} dims but tensor rank is {rank}")

    int64_max = np.iinfo(np.int64).max
    starts, ends, axes, steps, squeeze_axes = [], [], [], [], []
    for axis, s in enumerate(spec):
        if isinstance(s, slice):
            if s.start is None and s.stop is None and s.step in (None, 1):
                continue  # full slice: no-op on this axis
            step = 1 if s.step is None else int(s.step)
            if step < 1:
                raise TypeError("slice steps < 1 are not supported in ONNX export")
            starts.append(0 if s.start is None else int(s.start))
            ends.append(int64_max if s.stop is None else int(s.stop))
            axes.append(axis)
            steps.append(step)
        elif isinstance(s, int):
            starts.append(s)
            ends.append(int64_max if s == -1 else s + 1)
            axes.append(axis)
            steps.append(1)
            squeeze_axes.append(axis)
        else:
            raise TypeError(f"unsupported index element {s!r} for ONNX export (constant int/slice/Ellipsis only)")

    current = input_name
    if axes:
        slice_inputs = [current]
        for part, vals in (("starts", starts), ("ends", ends), ("axes", axes), ("steps", steps)):
            name = f"{prefix}_slice_{part}"
            initializers.append(onh.from_array(np.array(vals, dtype=np.int64), name=name))
            slice_inputs.append(name)
        current = f"{prefix}_slice"
        nodes.append(oh.make_node("Slice", inputs=slice_inputs, outputs=[current]))
    if squeeze_axes:
        # Squeeze takes axes as an input tensor from opset 13 on (the converter minimum).
        ax_name = f"{prefix}_squeeze_axes"
        initializers.append(onh.from_array(np.array(squeeze_axes, dtype=np.int64), name=ax_name))
        out = f"{prefix}_squeeze"
        nodes.append(oh.make_node("Squeeze", inputs=[current, ax_name], outputs=[out]))
        current = out
    return current


def emit_squeeze(prefix, input_name, axes, nodes, initializers):
    """Emit an ONNX Squeeze removing the given size-1 axes (no-op if axes is empty)."""
    if not axes:
        return input_name
    ax_name = f"{prefix}_squeeze_axes"
    initializers.append(onh.from_array(np.array(sorted(axes), dtype=np.int64), name=ax_name))
    out = f"{prefix}_squeeze"
    nodes.append(oh.make_node("Squeeze", inputs=[input_name, ax_name], outputs=[out]))
    return out


def emit_unsqueeze(prefix, input_name, axes, nodes, initializers):
    """Emit an ONNX Unsqueeze inserting size-1 dims at the given axes."""
    ax_name = f"{prefix}_unsqueeze_axes"
    initializers.append(onh.from_array(np.array(axes, dtype=np.int64), name=ax_name))
    out = f"{prefix}_unsqueeze"
    nodes.append(oh.make_node("Unsqueeze", inputs=[input_name, ax_name], outputs=[out]))
    return out
