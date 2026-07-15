"""Per-layer ONNX graph builders (Dense/Conv/BN/Pool/Activation/MHA) for the PQuant Keras converter."""

import numpy as np
import onnx.helper as oh

from pquant.core.keras.layers import PQBatchNormalization
from pquant.core.keras.onnx.helpers import bn_transpose_info, channels_last, nchw_perms
from pquant.core.onnx_common import (
    add_float_scalar,
    add_initializer,
    add_transpose,
    conv_padding_attrs,
    emit_mha_core,
    emit_param,
    maybe_quant_input,
    maybe_quant_output,
    to_list,
    to_np,
)


def add_dense(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = to_np(layer._kernel).T  # [in, out] → [out, in] for Gemm (transB=1)
    q_weight = emit_param(
        prefix, "weight", kernel_np, layer.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )
    gemm_inputs = [current, q_weight]

    if layer._bias is not None:
        gemm_inputs.append(
            emit_param(
                prefix,
                "bias",
                to_np(layer._bias),
                layer.bias_quantizer,
                nodes,
                initializers,
                use_qonnx,
                store_integer_weights,
            )
        )

    gemm_out = f"{prefix}_gemm"
    nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_out], transB=1))

    return maybe_quant_output(layer, prefix, gemm_out, nodes, initializers, quant_fn)


def add_dense_nd(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """Dense layer as MatMul + Add, for inputs of rank > 2 (Gemm only takes rank-2)."""
    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = to_np(layer._kernel).T  # [out, in]
    if use_qonnx or store_integer_weights:
        # Quantized/int-stored weight is emitted in native [out, in] layout, then transposed.
        q_weight_native = emit_param(
            prefix, "weight", kernel_np, layer.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        q_weight = f"{prefix}_weight_t"
        nodes.append(oh.make_node("Transpose", inputs=[q_weight_native], outputs=[q_weight], perm=[1, 0]))
    else:
        q_weight = f"{prefix}_weight_t"
        add_initializer(initializers, q_weight, kernel_np.T)  # pre-transposed [in, out]

    mm_out = f"{prefix}_mm"
    nodes.append(oh.make_node("MatMul", inputs=[current, q_weight], outputs=[mm_out]))
    current = mm_out

    if layer._bias is not None:
        q_bias = emit_param(
            prefix, "bias", to_np(layer._bias), layer.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        add_out = f"{prefix}_bias_add"
        nodes.append(oh.make_node("Add", inputs=[current, q_bias], outputs=[add_out]))
        current = add_out

    return maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)


def add_conv_node(layer, prefix, conv_inputs, groups, ndim, nodes):
    """Emit the Conv node shared by the regular and depthwise builders."""
    auto_pad, pads = conv_padding_attrs(layer.padding, ndim)
    conv_attrs = dict(
        kernel_shape=to_list(layer.kernel_size, ndim),
        strides=to_list(layer.strides, ndim),
        dilations=to_list(layer.dilation_rate, ndim),
        group=groups,
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    return conv_out


def add_conv_common(layer, prefix, current, kernel_onnx, groups, ndim, nodes, initializers, quant_fn, use_qonnx, store_int):
    """Shared body of the conv builders: layout transposes, param emission, Conv, quantization."""
    is_channels_last = channels_last(layer)
    if is_channels_last:
        perm_to_nchw, perm_to_nhwx = nchw_perms(ndim)
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    q_weight = emit_param(prefix, "weight", kernel_onnx, layer.weight_quantizer, nodes, initializers, use_qonnx, store_int)
    conv_inputs = [current, q_weight]
    if layer._bias is not None:
        conv_inputs.append(
            emit_param(prefix, "bias", to_np(layer._bias), layer.bias_quantizer, nodes, initializers, use_qonnx, store_int)
        )

    current = add_conv_node(layer, prefix, conv_inputs, groups, ndim, nodes)
    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if is_channels_last:
        current = add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def add_conv(layer, prefix, current, nodes, initializers, ndim, quant_fn, use_qonnx, store_integer_weights):
    kernel_np = to_np(layer._kernel)
    # Transpose kernel from Keras HWIO to ONNX OIHW
    if ndim == 2:
        kernel_onnx = np.transpose(kernel_np, (3, 2, 0, 1))  # [kH,kW,in,out] → [out,in,kH,kW]
    else:
        kernel_onnx = np.transpose(kernel_np, (2, 1, 0))  # [kL,in,out]    → [out,in,kL]
    groups = getattr(layer, "groups", 1)
    return add_conv_common(
        layer, prefix, current, kernel_onnx, groups, ndim, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )


def add_depthwise_conv(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    kernel_np = to_np(layer._kernel)  # [kH, kW, in, depth_mult]
    in_ch, depth_mult = kernel_np.shape[2], kernel_np.shape[3]
    # ONNX depthwise = Conv with groups=in and weight [in*depth_mult, 1, kH, kW]
    kernel_onnx = np.transpose(kernel_np, (2, 3, 0, 1)).reshape(in_ch * depth_mult, 1, *kernel_np.shape[:2])
    return add_conv_common(
        layer, prefix, current, kernel_onnx, in_ch, 2, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )


def add_batchnorm(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """PQBatchNormalization (also handles plain keras BatchNormalization,
    but emit_layer currently only dispatches the PQ variant here)."""
    need_transpose, perm_to_nchw, perm_to_nhwx = bn_transpose_info(layer)
    if need_transpose:
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    n_ch = to_np(layer.moving_mean).shape[0]
    gamma_np = to_np(layer.gamma) if layer.gamma is not None else np.ones(n_ch, dtype=np.float32)  # scale=False
    beta_np = to_np(layer.beta) if layer.beta is not None else np.zeros(n_ch, dtype=np.float32)  # center=False

    # Plain (non-PQ) BatchNormalization has no quantizers; emit plain float parameters.
    is_pq = isinstance(layer, PQBatchNormalization)
    use_qonnx = use_qonnx and is_pq
    store_integer_weights = store_integer_weights and is_pq
    weight_quantizer = layer.weight_quantizer if is_pq else None
    bias_quantizer = layer.bias_quantizer if is_pq else None

    q_gamma = emit_param(prefix, "gamma", gamma_np, weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights)
    q_beta = emit_param(prefix, "beta", beta_np, bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights)
    mean_name = add_initializer(initializers, f"{prefix}_running_mean", to_np(layer.moving_mean))
    var_name = add_initializer(initializers, f"{prefix}_running_var", to_np(layer.moving_variance))

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

    if need_transpose:
        current = add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def add_mha(
    layer,
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
    # Q / K / V projections: (B, L, E) → (B, L, E)
    q_proj_out = add_dense_nd(
        layer.q_proj, f"{prefix}_q_proj", q_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    k_proj_out = add_dense_nd(
        layer.k_proj, f"{prefix}_k_proj", k_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    v_proj_out = add_dense_nd(
        layer.v_proj, f"{prefix}_v_proj", v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    context, avg_attn = emit_mha_core(
        layer, prefix, q_proj_out, k_proj_out, v_proj_out, nodes, initializers, quant_fn, key_padding_mask, attn_mask
    )

    # Output projection: (B, T, E) → (B, T, E)
    out = add_dense_nd(
        layer.out_proj, f"{prefix}_out_proj", context, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    return out, avg_attn


def add_avgpool(layer, prefix, current, nodes, initializers, ndim, quant_fn):
    is_channels_last = channels_last(layer)
    if is_channels_last:
        perm_to_nchw, perm_to_nhwx = nchw_perms(ndim)
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

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
    current = maybe_quant_output(layer, prefix, pool_out, nodes, initializers, quant_fn)

    if is_channels_last:
        current = add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def add_global_avgpool(layer, prefix, current, nodes, ndim):
    is_channels_last = channels_last(layer)
    if is_channels_last:
        perm_to_nchw, _ = nchw_perms(ndim)
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    pool_out = f"{prefix}_global_pool"
    nodes.append(oh.make_node("GlobalAveragePool", inputs=[current], outputs=[pool_out]))
    current = pool_out

    if is_channels_last:
        flatten_name = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[pool_out], outputs=[flatten_name], axis=1))
        current = flatten_name
    return current


def add_pq_activation(layer, prefix, current, nodes, initializers, quant_fn):
    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    if layer.use_multiplier and layer.activation_name == "relu" and hasattr(layer, "multiplier"):
        multiplier = float(np.array(layer.multiplier).ravel()[0])
        scale_name = add_float_scalar(initializers, f"{prefix}_mul_scale", 2.0 ** round(multiplier))
        scaled_out = f"{prefix}_scaled"
        nodes.append(oh.make_node("Mul", inputs=[current, scale_name], outputs=[scaled_out]))
        current = scaled_out

    activation = layer.activation_name
    act_out = f"{prefix}_act"
    if activation == "relu":
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[act_out]))
    elif activation == "tanh":
        nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[act_out]))
    elif activation == "hard_tanh":
        cmin_name = add_float_scalar(initializers, f"{prefix}_htanh_min", -1.0)
        cmax_name = add_float_scalar(initializers, f"{prefix}_htanh_max", 1.0)
        nodes.append(oh.make_node("Clip", inputs=[current, cmin_name, cmax_name], outputs=[act_out]))
    else:
        raise TypeError(f"PQActivation: unsupported activation {activation!r} for ONNX export")

    return maybe_quant_output(layer, prefix, act_out, nodes, initializers, quant_fn)
