"""ONNX builders for the torch PQuant layers; the backend-agnostic ones live in ``pquant.core.onnx_common``."""

import numpy as np

from pquant.core.onnx_common import (
    add_conv,
    add_dense,
    add_mha,
    cast_to_int_container,
    quantizer_bits,
    scalar_bits,
    to_list,
    to_np,
)


def _bias(module):
    return None if module._bias is None else to_np(module._bias)


def _add_dense(g, module, prefix, x, rank):
    return add_dense(g, module, prefix, x, to_np(module._weight), _bias(module), rank)


def _add_dense_integer(g, module, prefix, x):
    """Dense layer whose inner product runs in int32: QuantizeLinear → MatMulInteger (+ bias) → DequantizeLinear."""
    if getattr(module, "input_quantizer", None) is None or not (module.quantize_input and module.enable_quantization):
        raise ValueError(f"{prefix}: integer_ops requires a quantized layer with quantize_input=True")
    x, _, x_zero_point = g.quantize_linear(module.input_quantizer, f"{prefix}_in", x)
    input_scale = 2.0 ** -scalar_bits(module.input_quantizer)[2]

    weight = to_np(module._weight)  # [out, in]
    k, _, f = quantizer_bits(module.weight_quantizer)
    dtype = np.int8 if int(k.ravel()[0]) == 1 else np.uint8
    per_channel = f.size > 1
    # One scale per output channel; the smallest f of a channel gives the scale covering all of its weights.
    weight_scale = 2.0 ** -(f.reshape(len(weight), -1).min(axis=1) if per_channel else f.reshape(1))
    scale = (input_scale * weight_scale).astype(np.float32)  # of the int32 accumulator: s_x * s_w

    # Stored pre-transposed as [in, out] so that MatMulInteger needs no runtime Transpose.
    codes = cast_to_int_container(f"{prefix}_weight", weight / weight_scale[:, None], dtype)
    codes = g.initializer(f"{prefix}_weight_int", codes.T)
    weight_zero_point = g.initializer(f"{prefix}_weight_zp", np.array(0, dtype=dtype))
    out = g.node("MatMulInteger", [x, codes, x_zero_point, weight_zero_point], f"{prefix}_matmul_int")
    if module._bias is not None:
        bias = g.initializer(f"{prefix}_bias_int", np.round(to_np(module._bias) / scale).astype(np.int32))
        out = g.node("Add", [out, bias], f"{prefix}_matmul_biased")

    # The out features are the last axis of the accumulator, whatever the input rank.
    scale, attrs = (scale, {"axis": -1}) if per_channel else (scale.reshape(()), {})
    inputs = [
        out,
        g.initializer(f"{prefix}_combined_scale", scale),
        g.initializer(f"{prefix}_combined_zp", np.zeros(scale.shape, dtype=np.int32)),
    ]
    out = g.node("DequantizeLinear", inputs, f"{prefix}_dequantized", **attrs)
    return g.quantize_output(module, prefix, out)


def _add_conv(g, module, prefix, x):
    return add_conv(
        g,
        module,
        prefix,
        x,
        to_np(module._weight),
        _bias(module),
        strides=module.stride,
        dilations=module.dilation,
        padding=module.padding,
        groups=module.groups,
    )


def _add_batchnorm(g, module, prefix, x):
    inputs = [
        g.quantize_input(module, prefix, x),
        g.param(module, prefix, "gamma", to_np(module._weight), module.weight_quantizer),
        g.param(module, prefix, "beta", to_np(module._bias), module.bias_quantizer),
        g.initializer(f"{prefix}_running_mean", to_np(module.running_mean)),
        g.initializer(f"{prefix}_running_var", to_np(module.running_var)),
    ]
    return g.node("BatchNormalization", inputs, f"{prefix}_bn", epsilon=float(module.eps))


def _add_layernorm(g, module, prefix, x):
    shape = tuple(module.normalized_shape)
    inputs = [g.quantize_input(module, prefix, x)]
    if module._weight is None:  # elementwise_affine=False: ONNX still requires the scale input
        inputs.append(g.initializer(f"{prefix}_gamma", np.ones(shape, dtype=np.float32)))
    else:
        inputs.append(g.param(module, prefix, "gamma", to_np(module._weight), module.weight_quantizer))
    if module._bias is not None:
        inputs.append(g.param(module, prefix, "beta", to_np(module._bias), module.bias_quantizer))
    out = g.node("LayerNormalization", inputs, f"{prefix}_ln", axis=-len(shape), epsilon=float(module.eps))
    return g.quantize_output(module, prefix, out)


def _pool_attrs(module, ndim):
    return dict(
        kernel_shape=to_list(module.kernel_size, ndim),
        strides=to_list(module.stride, ndim),
        pads=to_list(module.padding, ndim) * 2,  # [begin_0, ..., end_0, ...]
    )


def _add_avgpool(g, module, prefix, x, ndim):
    out = g.node(
        "AveragePool",
        [g.quantize_input(module, prefix, x)],
        f"{prefix}_pool",
        ceil_mode=int(module.ceil_mode),
        count_include_pad=int(module.count_include_pad),
        **_pool_attrs(module, ndim),
    )
    return g.quantize_output(module, prefix, out)


def _add_maxpool(g, module, prefix, x):
    return g.node("MaxPool", [x], f"{prefix}_maxpool", **_pool_attrs(module, 2))


def _add_upsample(g, module, prefix, x):
    """Resize with nearest/linear interpolation and constant scale factors."""
    scales = np.array([1.0, 1.0, *to_list(module.scale_factor, 2)], dtype=np.float32)
    inputs = [
        x,
        g.initializer(f"{prefix}_upsample_roi", np.array([], dtype=np.float32)),
        g.initializer(f"{prefix}_upsample_scales", scales),
    ]
    mode = "nearest" if module.mode == "nearest" else "linear"
    return g.node("Resize", inputs, f"{prefix}_upsample", mode=mode, coordinate_transformation_mode="asymmetric")


def _add_mha(g, module, prefix, query, key, value, key_padding_mask=None, attn_mask=None):
    if not module.batch_first:  # (L, B, E) → (B, L, E)
        query, key, value = (g.transpose(f"{prefix}_{n}_in", x, [1, 0, 2]) for n, x in zip("qkv", (query, key, value)))
    out, attn = add_mha(g, module, prefix, query, key, value, _add_dense, key_padding_mask, attn_mask)
    if not module.batch_first:
        out = g.transpose(f"{prefix}_out_seq_first", out, [1, 0, 2])
    return out, attn
