"""
Backend-agnostic ONNX node emitters shared by the PQuant Keras and torch
ONNX converters.

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

All quantization parameters (k, i, f) are accepted as anything ``to_np`` can
convert: torch tensors, Keras/TF tensors, numpy arrays, or Python scalars.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

ROUND_MODE_MAP = {
    "TRN": "FLOOR",
    "RND": "ROUND",
    "RND_CONV": "ROUND",
    "TRN_ZERO": "TRUNCATE",
    "RND_ZERO": "ROUND",
    "RND_MIN_INF": "FLOOR",
    "RND_INF": "ROUND",
}


def to_np(tensor):
    """Convert a torch/Keras/TF tensor (or scalar) to a float32 numpy array."""
    if hasattr(tensor, "detach"):  # torch tensor, possibly on GPU
        tensor = tensor.detach().cpu()
    return np.asarray(tensor, dtype=np.float32)


def add_initializer(initializers, name, array):
    """Register a constant tensor and return its name."""
    initializers.append(onh.from_array(array, name=name))
    return name


def add_float_scalar(initializers, name, value):
    return add_initializer(initializers, name, np.array(value, dtype=np.float32))


def add_int64_array(initializers, name, values):
    return add_initializer(initializers, name, np.array(values, dtype=np.int64))


def add_transpose(name, input_name, perm, nodes):
    """Emit a Transpose node and return the output name."""
    out = f"{name}_transpose_{''.join(str(p) for p in perm)}"
    nodes.append(oh.make_node("Transpose", inputs=[input_name], outputs=[out], perm=list(perm)))
    return out


def to_list(v, n):
    """Normalize a scalar-or-sequence layer attribute (kernel/stride/...) to an n-length list."""
    return list(v) if hasattr(v, "__iter__") else [v] * n


def symmetric_pads(padding, ndim):
    """Expand a symmetric padding spec to the ONNX [begin_0, ..., end_0, ...] form."""
    per_axis = to_list(padding, ndim)
    return per_axis + per_axis


def conv_padding_attrs(padding, ndim):
    """Map a Keras/torch conv padding spec to ONNX (auto_pad, pads) attributes."""
    if isinstance(padding, str):
        return ("SAME_UPPER" if padding == "same" else "VALID"), None
    return "NOTSET", symmetric_pads(padding, ndim)


def fixed_point_clip_range(signed, i_val, f_val, overflow_mode):
    """Representable [min, max] of a fixed-point grid with i integer and f fractional bits."""
    clip_max = float(2.0**i_val - 2.0 ** (-f_val))
    if not signed:
        return 0.0, clip_max
    if overflow_mode == "SAT_SYM":
        return -clip_max, clip_max  # symmetric: excludes -2^i
    return float(-(2.0**i_val)), clip_max  # SAT: -2^i


def quant_node(name_prefix, input_name, rounding_mode, k, i, f, initializers, overflow_mode="SAT"):
    """Build a QONNX Quant node.  Returns ([node], output_name)."""
    k_np, i_np, f_np = to_np(k), to_np(i), to_np(f)
    k_val = int(k_np.ravel()[0])
    if f_np.size > 1:
        # The Quant node holds a single scale/bit_width; widen to cover every element.
        i_np = i_np.ravel().max()
        f_np = f_np.ravel().min()
    i_val = float(i_np)
    f_val = float(f_np)
    scale = float(2.0 ** (-f_val))
    bit_width = float(k_val + i_val + f_val)
    qonnx_rnd = ROUND_MODE_MAP.get(rounding_mode, "ROUND")
    # SAT_SYM excludes the most-negative value → QONNX narrow=1
    narrow = 1 if (k_val == 1 and overflow_mode == "SAT_SYM") else 0

    scale_name = add_float_scalar(initializers, f"{name_prefix}_scale", scale)
    zp_name = add_float_scalar(initializers, f"{name_prefix}_zero_point", 0.0)
    bw_name = add_float_scalar(initializers, f"{name_prefix}_bit_width", bit_width)
    out_name = f"{name_prefix}_quantized"

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
):  # noqa: ARG001 (rounding_mode kept for the shared quant_fn signature)
    """Build QuantizeLinear+DequantizeLinear nodes, optionally preceded by a Clip.

    Returns ([nodes], output_name).  Set include_clip=False to skip the Clip node
    (safe when values are guaranteed to be in-range at inference time).
    """
    k_val = int(to_np(k).ravel()[0])
    i_val = float(to_np(i).ravel()[0])
    f_val = float(to_np(f).ravel()[0])
    signed = k_val == 1
    clip_min, clip_max = fixed_point_clip_range(signed, i_val, f_val, overflow_mode)
    zp_val = np.int8(0) if signed else np.uint8(0)

    scale_name = add_float_scalar(initializers, f"{name_prefix}_scale", 2.0 ** (-f_val))
    zp_name = add_initializer(initializers, f"{name_prefix}_zero_point", np.array(zp_val))
    quantized_name = f"{name_prefix}_quantized"
    out_name = f"{name_prefix}_dequantized"

    nodes = []
    quantize_input = input_name
    if include_clip:
        clip_min_name = add_float_scalar(initializers, f"{name_prefix}_clip_min", clip_min)
        clip_max_name = add_float_scalar(initializers, f"{name_prefix}_clip_max", clip_max)
        quantize_input = f"{name_prefix}_clipped"
        nodes.append(oh.make_node("Clip", inputs=[input_name, clip_min_name, clip_max_name], outputs=[quantize_input]))
    nodes.append(oh.make_node("QuantizeLinear", inputs=[quantize_input, scale_name, zp_name], outputs=[quantized_name]))
    nodes.append(oh.make_node("DequantizeLinear", inputs=[quantized_name, scale_name, zp_name], outputs=[out_name]))
    return nodes, out_name


def per_channel_scale(f_np, out_channels):
    """Return the (out_channels,) scale vector, or None when f varies within a channel."""
    if f_np.size % out_channels != 0:
        return None
    f_per_channel = f_np.reshape(out_channels, -1)
    if not np.allclose(f_per_channel, f_per_channel[:, :1]):
        return None
    return (2.0 ** (-f_per_channel[:, 0])).astype(np.float32)


def int_weight_node(name_prefix, weight_np, k, f, initializers):
    """
    Store a weight tensor as int8/uint8 + DequantizeLinear.

    weight_np must already be in ONNX layout and on the fixed-point grid
    (guaranteed after apply_final_compression).  Converts by dividing by the
    scale and casting — no re-rounding needed.

    Granularity handling:
    - per-tensor  (f has one element): single scale, standard DequantizeLinear.
    - per-channel (f constant within each output channel): 1D scale with axis=0.
    - per-weight  (f fully per-element): ONNX has no per-weight quantization;
      falls back to float32 storage (no DequantizeLinear node).

    Returns ([nodes], output_name).
    """
    k_val = int(to_np(k).ravel()[0])
    dtype = np.int8 if k_val == 1 else np.uint8
    out_channels = weight_np.shape[0]
    f_np = to_np(f)

    if f_np.size == 1:
        scale_np = np.array(2.0 ** (-float(f_np.ravel()[0])), dtype=np.float32)
        int_weights = np.round(weight_np / float(scale_np)).astype(dtype)
        per_channel = False
    else:
        scale_np = per_channel_scale(f_np, out_channels)
        if scale_np is None:
            float_name = add_initializer(initializers, f"{name_prefix}_float", weight_np)
            return [], float_name
        broadcast_scale = scale_np.reshape((out_channels,) + (1,) * (weight_np.ndim - 1))
        int_weights = np.round(weight_np / broadcast_scale).astype(dtype)
        per_channel = True

    int_name = add_initializer(initializers, f"{name_prefix}_int", int_weights)
    scale_name = add_initializer(initializers, f"{name_prefix}_dq_scale", scale_np)
    zp_np = np.zeros(out_channels, dtype=dtype) if per_channel else np.array(dtype(0))
    zp_name = add_initializer(initializers, f"{name_prefix}_dq_zp", zp_np)

    out_name = f"{name_prefix}_dequantized"
    node_kwargs = {"axis": 0} if per_channel else {}
    node = oh.make_node("DequantizeLinear", inputs=[int_name, scale_name, zp_name], outputs=[out_name], **node_kwargs)
    return [node], out_name


def emit_param(prefix, name, arr, quantizer, nodes, initializers, use_qonnx, store_integer_weights):
    """Emit the ONNX value for a learnable parameter (kernel/bias/gamma/beta) and return its name.

    arr must already be in ONNX layout (e.g. OIHW for conv kernels).
    """
    if use_qonnx:
        fp_name = add_initializer(initializers, f"{prefix}_{name}_fp", arr)
        k, i, f = quantizer.get_quantization_bits()
        q_nodes, out = quant_node(
            f"{prefix}_{name}", fp_name, quantizer.round_mode, k, i, f, initializers, overflow_mode=quantizer.overflow
        )
        nodes.extend(q_nodes)
        return out
    if store_integer_weights:
        k, _, f = quantizer.get_quantization_bits()
        q_nodes, out = int_weight_node(f"{prefix}_{name}", arr, k, f, initializers)
        nodes.extend(q_nodes)
        return out
    return add_initializer(initializers, f"{prefix}_{name}", arr)


def apply_quantizer(quantizer, prefix, current, nodes, initializers, quant_fn):
    """Emit quant_fn nodes for one Quantizer and return the new tensor name."""
    k, i, f = quantizer.get_quantization_bits()
    new_nodes, out = quant_fn(prefix, current, quantizer.round_mode, k, i, f, initializers, overflow_mode=quantizer.overflow)
    nodes.extend(new_nodes)
    return out


def maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn):
    # input_quantizer is created conditionally, so guard it; the bool flags are always present.
    if getattr(layer, "input_quantizer", None) is not None and layer.quantize_input and layer.enable_quantization:
        current = apply_quantizer(layer.input_quantizer, f"{prefix}_in", current, nodes, initializers, quant_fn)
    return current


def maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn):
    if getattr(layer, "output_quantizer", None) is not None and layer.quantize_output and layer.enable_quantization:
        current = apply_quantizer(layer.output_quantizer, f"{prefix}_out", current, nodes, initializers, quant_fn)
    return current


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
            slice_inputs.append(add_int64_array(initializers, f"{prefix}_slice_{part}", vals))
        current = f"{prefix}_slice"
        nodes.append(oh.make_node("Slice", inputs=slice_inputs, outputs=[current]))
    if squeeze_axes:
        current = emit_squeeze(prefix, current, squeeze_axes, nodes, initializers)
    return current


def emit_squeeze(prefix, input_name, axes, nodes, initializers):
    """Emit an ONNX Squeeze removing the given size-1 axes (no-op if axes is empty).

    Squeeze takes axes as an input tensor from opset 13 on (the converter minimum).
    """
    if not axes:
        return input_name
    ax_name = add_int64_array(initializers, f"{prefix}_squeeze_axes", sorted(axes))
    out = f"{prefix}_squeeze"
    nodes.append(oh.make_node("Squeeze", inputs=[input_name, ax_name], outputs=[out]))
    return out


def emit_unsqueeze(prefix, input_name, axes, nodes, initializers):
    """Emit an ONNX Unsqueeze inserting size-1 dims at the given axes."""
    ax_name = add_int64_array(initializers, f"{prefix}_unsqueeze_axes", axes)
    out = f"{prefix}_unsqueeze"
    nodes.append(oh.make_node("Unsqueeze", inputs=[input_name, ax_name], outputs=[out]))
    return out


def add_quantized_softmax(sm, prefix, current, nodes, initializers, quant_fn, kpm_mask=None):
    """Emit the PQuant quantized-softmax decomposition over the last axis."""
    enable = sm.enable_quantization
    scaler = float(sm.input_scaler)
    stable = bool(sm.stable)

    # 1) Softmax input quantizer.
    if sm.quantize_input and enable:
        current = apply_quantizer(sm.input_quantizer, f"{prefix}_sm_in_q", current, nodes, initializers, quant_fn)

    # 2) Stable max-subtract over the last axis (ReduceMax keeps axes as an attribute).
    if stable:
        m_name = f"{prefix}_sm_max"
        nodes.append(oh.make_node("ReduceMax", inputs=[current], outputs=[m_name], axes=[-1], keepdims=1))
        exp_in = f"{prefix}_sm_sub"
        nodes.append(oh.make_node("Sub", inputs=[m_name, current], outputs=[exp_in]))
    else:
        exp_in = current

    # 3) Quantized exp table: optional input QDQ, Exp of (-scaler * x) for the
    #    stable branch (+scaler otherwise), optional output QDQ.
    exp_table = sm.exp_table
    if exp_table.quantize_input and enable:
        exp_in = apply_quantizer(exp_table.input_quantizer, f"{prefix}_sm_exp_in_q", exp_in, nodes, initializers, quant_fn)
    coeff = -scaler if stable else scaler
    exp_arg = exp_in
    if coeff != 1.0:
        coeff_name = add_float_scalar(initializers, f"{prefix}_sm_exp_coeff", coeff)
        exp_arg = f"{prefix}_sm_exp_arg"
        nodes.append(oh.make_node("Mul", inputs=[exp_in, coeff_name], outputs=[exp_arg]))
    numerator = f"{prefix}_sm_exp"
    nodes.append(oh.make_node("Exp", inputs=[exp_arg], outputs=[numerator]))
    if exp_table.quantize_output and enable:
        numerator = apply_quantizer(
            exp_table.output_quantizer, f"{prefix}_sm_exp_out_q", numerator, nodes, initializers, quant_fn
        )

    # 3b) Optional key-padding mask: zero the exp-numerator at masked positions.
    if kpm_mask is not None:
        kpm_f = f"{prefix}_sm_mask_f"
        nodes.append(oh.make_node("Cast", inputs=[kpm_mask], outputs=[kpm_f], to=onnx.TensorProto.FLOAT))
        masked = f"{prefix}_sm_masked"
        nodes.append(oh.make_node("Mul", inputs=[kpm_f, numerator], outputs=[masked]))
        numerator = masked

    # 4) Sum over the last axis (ReduceSum takes axes as an input from opset 13).
    sum_axes = add_int64_array(initializers, f"{prefix}_sm_sum_axes", [-1])
    sums = f"{prefix}_sm_sum"
    nodes.append(oh.make_node("ReduceSum", inputs=[numerator, sum_axes], outputs=[sums], keepdims=1))

    # 5) Quantized reciprocal table: input QDQ, 1/(x+eps), output QDQ.
    inv_table = sm.inv_table
    inv_in = sums
    if inv_table.quantize_input and enable:
        inv_in = apply_quantizer(inv_table.input_quantizer, f"{prefix}_sm_inv_in_q", inv_in, nodes, initializers, quant_fn)
    eps_name = add_float_scalar(initializers, f"{prefix}_sm_eps", float(sm.epsilon))
    inv_add = f"{prefix}_sm_inv_add"
    nodes.append(oh.make_node("Add", inputs=[inv_in, eps_name], outputs=[inv_add]))
    divisor = f"{prefix}_sm_inv"
    nodes.append(oh.make_node("Reciprocal", inputs=[inv_add], outputs=[divisor]))
    if inv_table.quantize_output and enable:
        divisor = apply_quantizer(
            inv_table.output_quantizer, f"{prefix}_sm_inv_out_q", divisor, nodes, initializers, quant_fn
        )

    # 6) Multiply numerator by reciprocal.
    out = f"{prefix}_sm_out"
    nodes.append(oh.make_node("Mul", inputs=[numerator, divisor], outputs=[out]))
    current = out

    # 7) Softmax output quantizer.
    if sm.quantize_output and enable:
        current = apply_quantizer(sm.output_quantizer, f"{prefix}_sm_out_q", current, nodes, initializers, quant_fn)
    return current


def split_heads(x_name, pfx, num_heads, head_dim, nodes, initializers):
    """(B, L, E) → (B, num_heads, L, head_dim) using runtime Shape ops so B and L stay dynamic."""
    shape_out = f"{pfx}_shape"
    nodes.append(oh.make_node("Shape", inputs=[x_name], outputs=[shape_out]))

    idx0 = add_int64_array(initializers, f"{pfx}_gi0", 0)
    idx1 = add_int64_array(initializers, f"{pfx}_gi1", 1)
    ax0 = add_int64_array(initializers, f"{pfx}_ax0", [0])
    heads_1d = add_int64_array(initializers, f"{pfx}_H_1d", [num_heads])
    head_dim_1d = add_int64_array(initializers, f"{pfx}_hd_1d", [head_dim])

    batch_scalar = f"{pfx}_b_sc"
    length_scalar = f"{pfx}_l_sc"
    batch_1d = f"{pfx}_b_1d"
    length_1d = f"{pfx}_l_1d"
    shape_4d = f"{pfx}_shape4d"
    reshaped = f"{pfx}_reshaped"
    transposed = f"{pfx}_transposed"

    nodes.append(oh.make_node("Gather", inputs=[shape_out, idx0], outputs=[batch_scalar]))
    nodes.append(oh.make_node("Gather", inputs=[shape_out, idx1], outputs=[length_scalar]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[batch_scalar, ax0], outputs=[batch_1d]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[length_scalar, ax0], outputs=[length_1d]))
    nodes.append(oh.make_node("Concat", inputs=[batch_1d, length_1d, heads_1d, head_dim_1d], outputs=[shape_4d], axis=0))
    nodes.append(oh.make_node("Reshape", inputs=[x_name, shape_4d], outputs=[reshaped]))
    # (B, L, H, head_dim) → (B, H, L, head_dim)
    nodes.append(oh.make_node("Transpose", inputs=[reshaped], outputs=[transposed], perm=[0, 2, 1, 3]))
    return transposed


def merge_heads(x_name, pfx, embed_dim, nodes, initializers):
    """(B, H, T, head_dim) → (B, T, embed_dim): the inverse of split_heads."""
    transposed = f"{pfx}_t"  # (B, T, H, head_dim)
    shape_out = f"{pfx}_shape"
    nodes.append(oh.make_node("Transpose", inputs=[x_name], outputs=[transposed], perm=[0, 2, 1, 3]))
    nodes.append(oh.make_node("Shape", inputs=[transposed], outputs=[shape_out]))

    idx0 = add_int64_array(initializers, f"{pfx}_gi0", 0)
    idx1 = add_int64_array(initializers, f"{pfx}_gi1", 1)
    ax0 = add_int64_array(initializers, f"{pfx}_ax0", [0])
    embed_1d = add_int64_array(initializers, f"{pfx}_E_1d", [embed_dim])

    batch_scalar = f"{pfx}_b_sc"
    length_scalar = f"{pfx}_t_sc"
    batch_1d = f"{pfx}_b_1d"
    length_1d = f"{pfx}_t_1d"
    shape_3d = f"{pfx}_shape3d"
    merged = f"{pfx}_merged"

    nodes.append(oh.make_node("Gather", inputs=[shape_out, idx0], outputs=[batch_scalar]))
    nodes.append(oh.make_node("Gather", inputs=[shape_out, idx1], outputs=[length_scalar]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[batch_scalar, ax0], outputs=[batch_1d]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[length_scalar, ax0], outputs=[length_1d]))
    nodes.append(oh.make_node("Concat", inputs=[batch_1d, length_1d, embed_1d], outputs=[shape_3d], axis=0))
    nodes.append(oh.make_node("Reshape", inputs=[transposed, shape_3d], outputs=[merged]))
    return merged


def emit_mha_core(
    mha, prefix, q_proj_out, k_proj_out, v_proj_out, nodes, initializers, quant_fn, key_padding_mask, attn_mask
):
    """Scaled-dot-product attention between projected Q/K/V, quantized softmax included.

    Returns (context_name, avg_attn_name): the merged (B, T, E) context ready for
    the output projection, and the attention weights averaged over heads.
    """
    q_heads = split_heads(q_proj_out, f"{prefix}_q", mha.num_heads, mha.head_dim, nodes, initializers)
    k_heads = split_heads(k_proj_out, f"{prefix}_k", mha.num_heads, mha.head_dim, nodes, initializers)
    v_heads = split_heads(v_proj_out, f"{prefix}_v", mha.num_heads, mha.head_dim, nodes, initializers)

    k_transposed = f"{prefix}_k_T"
    nodes.append(oh.make_node("Transpose", inputs=[k_heads], outputs=[k_transposed], perm=[0, 1, 3, 2]))

    raw_scores = f"{prefix}_scores_raw"
    scaled_scores = f"{prefix}_scores_scaled"
    scale_name = add_float_scalar(initializers, f"{prefix}_attn_scale", float(mha.scale))
    nodes.append(oh.make_node("MatMul", inputs=[q_heads, k_transposed], outputs=[raw_scores]))
    nodes.append(oh.make_node("Mul", inputs=[raw_scores, scale_name], outputs=[scaled_scores]))
    current = scaled_scores

    if attn_mask is not None:
        masked_scores = f"{prefix}_scores_masked"
        nodes.append(oh.make_node("Add", inputs=[current, attn_mask], outputs=[masked_scores]))
        current = masked_scores

    kpm_mult = None
    if key_padding_mask is not None:
        kpm_not = f"{prefix}_kpm_not"
        nodes.append(oh.make_node("Not", inputs=[key_padding_mask], outputs=[kpm_not]))
        kpm_axes = add_int64_array(initializers, f"{prefix}_kpm_axes", [1, 2])
        kpm_mult = f"{prefix}_kpm_mask"  # (B, 1, 1, S) bool, cast to float inside the softmax
        nodes.append(oh.make_node("Unsqueeze", inputs=[kpm_not, kpm_axes], outputs=[kpm_mult]))

    attn_weights = add_quantized_softmax(
        mha.softmax, f"{prefix}_attn", current, nodes, initializers, quant_fn, kpm_mask=kpm_mult
    )

    context = f"{prefix}_ctx_raw"
    nodes.append(oh.make_node("MatMul", inputs=[attn_weights, v_heads], outputs=[context]))
    merged = merge_heads(context, f"{prefix}_ctx", mha.embed_dim, nodes, initializers)

    # Average attention weights over heads: (B, H, T, S) → (B, T, S)
    avg_attn = f"{prefix}_avg_attn_weights"
    nodes.append(oh.make_node("ReduceMean", inputs=[attn_weights], outputs=[avg_attn], axes=[1], keepdims=0))
    return merged, avg_attn


def save_model(graph, output_path, opset, use_qonnx=False, ir_version=6):
    """Assemble the ModelProto for a finished graph, validate it, and save it."""
    opset_imports = [oh.make_opsetid("", opset)]
    if use_qonnx:
        opset_imports.append(oh.make_opsetid("qonnx.custom_op.general", 1))
    model_proto = oh.make_model(graph, opset_imports=opset_imports)
    model_proto.ir_version = ir_version
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    return model_proto
