"""Per-layer ONNX graph builders (Dense/Conv/BN/LN/Pool/Activation/MHA) for the PQuant torch converter."""

import numpy as np
import onnx.helper as oh

from pquant.core.onnx_common import (
    add_float_scalar,
    add_initializer,
    add_transpose,
    conv_padding_attrs,
    emit_mha_core,
    emit_param,
    fixed_point_clip_range,
    maybe_quant_input,
    maybe_quant_output,
    qdq_node,
    symmetric_pads,
    to_list,
    to_np,
)


def quantize_input_to_int(quantizer, prefix, current, nodes, initializers):
    """Clip + QuantizeLinear the input to int8/uint8, stopping before DequantizeLinear.

    Returns (int_tensor_name, zero_point_name, input_scale).
    """
    k, i, f = quantizer.get_quantization_bits()
    signed = int(to_np(k).ravel()[0]) == 1
    i_val = float(to_np(i).ravel()[0])
    f_val = float(to_np(f).ravel()[0])
    scale = float(2.0 ** (-f_val))
    clip_min, clip_max = fixed_point_clip_range(signed, i_val, f_val, "SAT")

    clip_min_name = add_float_scalar(initializers, f"{prefix}_in_clip_min", clip_min)
    clip_max_name = add_float_scalar(initializers, f"{prefix}_in_clip_max", clip_max)
    scale_name = add_float_scalar(initializers, f"{prefix}_in_scale", scale)
    zp_name = add_initializer(initializers, f"{prefix}_in_zp", np.array(np.int8(0) if signed else np.uint8(0)))

    clipped_name = f"{prefix}_in_clipped"
    int_name = f"{prefix}_in_int"
    nodes.append(oh.make_node("Clip", inputs=[current, clip_min_name, clip_max_name], outputs=[clipped_name]))
    nodes.append(oh.make_node("QuantizeLinear", inputs=[clipped_name, scale_name, zp_name], outputs=[int_name]))
    return int_name, zp_name, scale


def integer_weights_transposed(module, prefix, initializers):
    """Quantize the dense weight to int8/uint8, pre-transposed to [in, out] so
    MatMulInteger needs no runtime Transpose node.

    Returns (weight_name, weight_zp_name, scale_1d, per_channel) where scale_1d
    has shape [1] (per-tensor) or [out] (per-channel).
    """
    weight_np = to_np(module._weight)  # PyTorch layout: [out, in]
    k, _, f = module.weight_quantizer.get_quantization_bits()
    dtype = np.int8 if int(to_np(k).ravel()[0]) == 1 else np.uint8
    out_channels = weight_np.shape[0]

    f_np = to_np(f)
    if f_np.size == 1:
        f_1d = np.array([float(f_np.ravel()[0])])
        per_channel = False
    else:
        f_1d = f_np.reshape(out_channels, -1).min(axis=1)  # min f → max scale → covers all values
        per_channel = True

    scale_1d = (2.0 ** (-f_1d)).astype(np.float32)
    broadcast = scale_1d.reshape((out_channels,) + (1,) * (weight_np.ndim - 1)) if per_channel else float(scale_1d[0])
    int_weights = np.round(weight_np / broadcast).astype(dtype).T  # [in, out]

    weight_name = add_initializer(initializers, f"{prefix}_weight_int", int_weights)
    zp_name = add_initializer(initializers, f"{prefix}_weight_zp", np.array(dtype(0)))  # zero for symmetric quantization
    return weight_name, zp_name, scale_1d, per_channel


def dequantize_accumulator(prefix, current, combined_scale_1d, per_channel, nodes, initializers):
    """DequantizeLinear the int32 accumulator back to float32 with the combined scale s_x * s_w.

    Per-channel: axis=1 because the output tensor is [batch, out] and out is axis 1.
    """
    if per_channel:
        scale_np = combined_scale_1d.astype(np.float32)
        zp_np = np.zeros(len(combined_scale_1d), dtype=np.int32)
        dql_kwargs = {"axis": 1}
    else:
        scale_np = np.array(float(combined_scale_1d[0]), dtype=np.float32)
        zp_np = np.array(np.int32(0))
        dql_kwargs = {}

    scale_name = add_initializer(initializers, f"{prefix}_combined_scale", scale_np)
    zp_name = add_initializer(initializers, f"{prefix}_combined_zp", zp_np)
    out = f"{prefix}_dequantized"
    nodes.append(oh.make_node("DequantizeLinear", inputs=[current, scale_name, zp_name], outputs=[out], **dql_kwargs))
    return out


def add_dense_integer(module, prefix, current, nodes, initializers):
    """Dense layer whose inner product runs in int32 via MatMulInteger."""
    if getattr(module, "input_quantizer", None) is None or not module.quantize_input:
        raise ValueError(f"{prefix}: integer_ops requires quantize_input=True on the layer")

    x_int, x_zp, input_scale = quantize_input_to_int(module.input_quantizer, prefix, current, nodes, initializers)
    w_int, w_zp, weight_scale_1d, per_channel = integer_weights_transposed(module, prefix, initializers)
    combined_scale_1d = input_scale * weight_scale_1d

    current = f"{prefix}_matmul_int"  # MatMulInteger([batch, in], [in, out]) → int32 [batch, out]
    nodes.append(oh.make_node("MatMulInteger", inputs=[x_int, w_int, x_zp, w_zp], outputs=[current]))

    if module._bias is not None:
        bias_scale = combined_scale_1d if per_channel else float(combined_scale_1d[0])
        bias_int32 = np.round(to_np(module._bias) / bias_scale).astype(np.int32)
        bias_name = add_initializer(initializers, f"{prefix}_bias_int", bias_int32)
        biased_name = f"{prefix}_matmul_biased"
        nodes.append(oh.make_node("Add", inputs=[current, bias_name], outputs=[biased_name]))
        current = biased_name

    current = dequantize_accumulator(prefix, current, combined_scale_1d, per_channel, nodes, initializers)

    # Optional output quantization (e.g. last layer with quantize_output=True)
    return maybe_quant_output(module, prefix, current, nodes, initializers, qdq_node)


def add_dense_nd(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """Dense layer as MatMul + Add, for inputs of rank > 2 (Gemm only takes rank-2)."""
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = to_np(module._weight)  # [out, in]
    if use_qonnx or store_integer_weights:
        # Quantized/int-stored weight is emitted in native [out, in] layout, then transposed.
        q_weight_native = emit_param(
            prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        q_weight = f"{prefix}_weight_T"
        nodes.append(oh.make_node("Transpose", inputs=[q_weight_native], outputs=[q_weight], perm=[1, 0]))
    else:
        q_weight = f"{prefix}_weight_T"
        add_initializer(initializers, q_weight, weight_np.T)  # pre-transposed [in, out]

    matmul_out = f"{prefix}_matmul"
    nodes.append(oh.make_node("MatMul", inputs=[current, q_weight], outputs=[matmul_out]))
    current = matmul_out

    if module._bias is not None:
        q_bias = emit_param(
            prefix, "bias", to_np(module._bias), module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        biased_out = f"{prefix}_biased"
        nodes.append(oh.make_node("Add", inputs=[matmul_out, q_bias], outputs=[biased_out]))
        current = biased_out

    return maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)


def add_dense(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops=False):
    if integer_ops and not use_qonnx:
        return add_dense_integer(module, prefix, current, nodes, initializers)
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    q_weight = emit_param(
        prefix,
        "weight",
        to_np(module._weight),
        module.weight_quantizer,
        nodes,
        initializers,
        use_qonnx,
        store_integer_weights,
    )
    gemm_inputs = [current, q_weight]

    if module._bias is not None:
        gemm_inputs.append(
            emit_param(
                prefix,
                "bias",
                to_np(module._bias),
                module.bias_quantizer,
                nodes,
                initializers,
                use_qonnx,
                store_integer_weights,
            )
        )

    gemm_out = f"{prefix}_gemm"
    nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_out], transB=1))

    return maybe_quant_output(module, prefix, gemm_out, nodes, initializers, quant_fn)


def add_conv(module, prefix, current, nodes, initializers, ndim, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    q_weight = emit_param(
        prefix,
        "weight",
        to_np(module._weight),
        module.weight_quantizer,
        nodes,
        initializers,
        use_qonnx,
        store_integer_weights,
    )
    conv_inputs = [current, q_weight]

    if module._bias is not None:
        conv_inputs.append(
            emit_param(
                prefix,
                "bias",
                to_np(module._bias),
                module.bias_quantizer,
                nodes,
                initializers,
                use_qonnx,
                store_integer_weights,
            )
        )

    auto_pad, pads = conv_padding_attrs(module.padding, ndim)
    conv_attrs = dict(
        kernel_shape=to_list(module.kernel_size, ndim),
        strides=to_list(module.stride, ndim),
        dilations=to_list(module.dilation, ndim),
        group=module.groups,
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))

    return maybe_quant_output(module, prefix, conv_out, nodes, initializers, quant_fn)


def add_batchnorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    q_gamma = emit_param(
        prefix,
        "gamma",
        to_np(module._weight),
        module.weight_quantizer,
        nodes,
        initializers,
        use_qonnx,
        store_integer_weights,
    )
    q_beta = emit_param(
        prefix, "beta", to_np(module._bias), module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )
    mean_name = add_initializer(initializers, f"{prefix}_running_mean", to_np(module.running_mean))
    var_name = add_initializer(initializers, f"{prefix}_running_var", to_np(module.running_var))

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


def add_layernorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    normalized_shape = tuple(to_list(module.normalized_shape, 1))
    axis = -len(normalized_shape)

    has_weight = module._weight is not None
    has_bias = module._bias is not None
    # elementwise_affine=False has no quantizers; emit plain float parameters.
    use_qonnx = use_qonnx and has_weight
    store_integer_weights = store_integer_weights and has_weight

    gamma_np = to_np(module._weight) if has_weight else np.ones(normalized_shape, dtype=np.float32)
    quantizer = module.weight_quantizer if has_weight else None
    q_gamma = emit_param(prefix, "gamma", gamma_np, quantizer, nodes, initializers, use_qonnx, store_integer_weights)

    ln_inputs = [current, q_gamma]
    if has_bias:
        bias_quantizer = module.bias_quantizer if has_weight else None
        q_beta = emit_param(
            prefix, "beta", to_np(module._bias), bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        ln_inputs.append(q_beta)

    ln_out = f"{prefix}_ln"
    nodes.append(
        oh.make_node("LayerNormalization", inputs=ln_inputs, outputs=[ln_out], axis=axis, epsilon=float(module.eps))
    )
    return maybe_quant_output(module, prefix, ln_out, nodes, initializers, quant_fn)


def add_avgpool(module, prefix, current, nodes, initializers, ndim, quant_fn):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    pool_out = f"{prefix}_pool"
    nodes.append(
        oh.make_node(
            "AveragePool",
            inputs=[current],
            outputs=[pool_out],
            kernel_shape=to_list(module.kernel_size, ndim),
            strides=to_list(module.stride, ndim),
            pads=symmetric_pads(module.padding, ndim),
            ceil_mode=int(module.ceil_mode),
            count_include_pad=int(module.count_include_pad),
        )
    )
    return maybe_quant_output(module, prefix, pool_out, nodes, initializers, quant_fn)


def add_maxpool(module, prefix, current, nodes):
    out = f"{prefix}_maxpool"
    nodes.append(
        oh.make_node(
            "MaxPool",
            inputs=[current],
            outputs=[out],
            kernel_shape=to_list(module.kernel_size, 2),
            strides=to_list(module.stride, 2),
            pads=symmetric_pads(module.padding, 2),
        )
    )
    return out


def add_upsample(module, prefix, current, nodes, initializers):
    """Emit a Resize node with nearest/linear mode and constant scale factors."""
    roi_name = add_initializer(initializers, f"{prefix}_upsample_roi", np.array([], dtype=np.float32))
    scale_factor = module.scale_factor
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)
    scales = np.array([1.0, 1.0, float(scale_factor[0]), float(scale_factor[1])], dtype=np.float32)
    scales_name = add_initializer(initializers, f"{prefix}_upsample_scales", scales)
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


def add_activation(module, prefix, current, nodes, initializers, quant_fn):
    """PQActivation: optional input quantization, the activation itself, optional output quantization."""
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    activation = module.activation_name
    act_out = f"{prefix}_act"
    if activation == "relu":
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[act_out]))
    elif activation == "tanh":
        nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[act_out]))
    elif activation == "hard_tanh":
        cmin_name = add_float_scalar(initializers, f"{prefix}_htanh_min", -1.0)
        cmax_name = add_float_scalar(initializers, f"{prefix}_htanh_max", 1.0)
        nodes.append(oh.make_node("Clip", inputs=[current, cmin_name, cmax_name], outputs=[act_out]))
    elif activation == "leaky_relu":
        alpha = module.activation_function.negative_slope
        nodes.append(oh.make_node("LeakyRelu", inputs=[current], outputs=[act_out], alpha=alpha))
    elif activation == "gelu":
        add_gelu(module, prefix, current, act_out, nodes, initializers)
    else:
        raise TypeError(f"PQActivation: unsupported activation {activation!r} for ONNX export")

    return maybe_quant_output(module, prefix, act_out, nodes, initializers, quant_fn)


def add_gelu(module, prefix, current, act_out, nodes, initializers):
    """Decompose gelu so the default opset (13) works; ONNX added a Gelu op only in opset 20."""
    approximate = getattr(module.activation_function, "approximate", "none")
    half_name = add_float_scalar(initializers, f"{prefix}_gelu_half", 0.5)
    one_name = add_float_scalar(initializers, f"{prefix}_gelu_one", 1.0)
    plus_one = f"{prefix}_gelu_plus1"

    if approximate == "tanh":
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi_name = add_float_scalar(initializers, f"{prefix}_gelu_sqrt2_over_pi", np.sqrt(2.0 / np.pi))
        cubic_coeff_name = add_float_scalar(initializers, f"{prefix}_gelu_c1", 0.044715)
        three_name = add_float_scalar(initializers, f"{prefix}_gelu_three", 3.0)
        x3 = f"{prefix}_gelu_x3"
        cx3 = f"{prefix}_gelu_cx3"
        inner = f"{prefix}_gelu_inner"
        scaled = f"{prefix}_gelu_scaled"
        tanh_out = f"{prefix}_gelu_tanh"
        nodes += [
            oh.make_node("Pow", inputs=[current, three_name], outputs=[x3]),
            oh.make_node("Mul", inputs=[x3, cubic_coeff_name], outputs=[cx3]),
            oh.make_node("Add", inputs=[current, cx3], outputs=[inner]),
            oh.make_node("Mul", inputs=[inner, sqrt_2_over_pi_name], outputs=[scaled]),
            oh.make_node("Tanh", inputs=[scaled], outputs=[tanh_out]),
            oh.make_node("Add", inputs=[tanh_out, one_name], outputs=[plus_one]),
        ]
    else:
        # Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt2_name = add_float_scalar(initializers, f"{prefix}_gelu_inv_sqrt2", 1.0 / np.sqrt(2.0))
        scaled = f"{prefix}_gelu_scaled"
        erf_out = f"{prefix}_gelu_erf"
        nodes += [
            oh.make_node("Mul", inputs=[current, inv_sqrt2_name], outputs=[scaled]),
            oh.make_node("Erf", inputs=[scaled], outputs=[erf_out]),
            oh.make_node("Add", inputs=[erf_out, one_name], outputs=[plus_one]),
        ]

    x_times = f"{prefix}_gelu_xprod"
    nodes += [
        oh.make_node("Mul", inputs=[current, plus_one], outputs=[x_times]),
        oh.make_node("Mul", inputs=[x_times, half_name], outputs=[act_out]),
    ]


def add_mha(
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
    if not module.batch_first:
        q_input = add_transpose(f"{prefix}_q_in", q_input, [1, 0, 2], nodes)
        k_input = add_transpose(f"{prefix}_k_in", k_input, [1, 0, 2], nodes)
        v_input = add_transpose(f"{prefix}_v_in", v_input, [1, 0, 2], nodes)

    # Q / K / V projections: (B, L, E) → (B, L, E) via MatMul (input is rank-3)
    q_proj_out = add_dense_nd(
        module.q_proj, f"{prefix}_q_proj", q_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    k_proj_out = add_dense_nd(
        module.k_proj, f"{prefix}_k_proj", k_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    v_proj_out = add_dense_nd(
        module.v_proj, f"{prefix}_v_proj", v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    context, avg_attn = emit_mha_core(
        module, prefix, q_proj_out, k_proj_out, v_proj_out, nodes, initializers, quant_fn, key_padding_mask, attn_mask
    )
    out = add_dense_nd(
        module.out_proj, f"{prefix}_out_proj", context, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    if not module.batch_first:
        out = add_transpose(f"{prefix}_out_seq_first", out, [1, 0, 2], nodes)
    return out, avg_attn
