"""ONNX builders for the Keras PQuant layers; the backend-agnostic ones live in ``pquant.core.onnx_common``."""

import numpy as np

from pquant.core.keras.layers import PQDepthwiseConv2d
from pquant.core.onnx_common import add_conv, add_dense, to_list, to_np


def _bias(layer):
    return None if layer._bias is None else to_np(layer._bias)


def _channel_axis(layer):
    return -1 if layer.data_format == "channels_last" else 1


def _channels_first(g, prefix, x, rank, channel_axis):
    """Move ``channel_axis`` to position 1.  Returns (x, the permutation that restores the layout, or None)."""
    channel_axis %= rank
    if rank <= 2 or channel_axis == 1:
        return x, None
    perm = [0, channel_axis] + [axis for axis in range(1, rank) if axis != channel_axis]
    return g.transpose(f"{prefix}_pre", x, perm), [perm.index(axis) for axis in range(rank)]


def _restore_layout(g, prefix, x, restore):
    return x if restore is None else g.transpose(f"{prefix}_post", x, restore)


def _add_dense(g, layer, prefix, x, rank):
    return add_dense(g, layer, prefix, x, to_np(layer._kernel).T, _bias(layer), rank)


def _add_conv(g, layer, prefix, x, rank):
    """PQConv1d, PQConv2d and PQDepthwiseConv2d."""
    kernel = to_np(layer._kernel)
    if isinstance(layer, PQDepthwiseConv2d):
        # ONNX depthwise convolution is a Conv with one group per input channel.
        groups = kernel.shape[2]
        kernel = np.transpose(kernel, (2, 3, 0, 1)).reshape(-1, 1, *kernel.shape[:2])  # → [in * mult, 1, kH, kW]
    else:
        groups = layer.groups
        kernel = np.moveaxis(kernel, (-1, -2), (0, 1))  # → [out, in/groups, *spatial]
    x, restore = _channels_first(g, prefix, x, rank, _channel_axis(layer))
    x = add_conv(g, layer, prefix, x, kernel, _bias(layer), layer.strides, layer.dilation_rate, layer.padding, groups)
    return _restore_layout(g, prefix, x, restore)


def _add_batchnorm(g, layer, prefix, x, rank):
    channels = layer.moving_mean.shape[0]

    def affine(name, variable, default, quantizer):
        if variable is None:  # scale=False / center=False: ONNX still requires the input
            return g.initializer(f"{prefix}_{name}", np.full(channels, default, dtype=np.float32))
        return g.param(layer, prefix, name, to_np(variable), quantizer if layer.quantize_parameters else None)

    x, restore = _channels_first(g, prefix, x, rank, layer.axis)
    inputs = [
        g.quantize_input(layer, prefix, x),
        affine("gamma", layer.gamma, 1.0, layer.weight_quantizer),
        affine("beta", layer.beta, 0.0, layer.bias_quantizer),
        g.initializer(f"{prefix}_running_mean", to_np(layer.moving_mean)),
        g.initializer(f"{prefix}_running_var", to_np(layer.moving_variance)),
    ]
    x = g.node("BatchNormalization", inputs, f"{prefix}_bn", epsilon=float(layer.epsilon))
    return _restore_layout(g, prefix, x, restore)


def _add_avgpool(g, layer, prefix, x, rank):
    """AveragePooling1D/2D, with the quantizers of the PQAvgPool variants."""
    ndim = rank - 2
    padding = {"auto_pad": "SAME_UPPER"} if layer.padding == "same" else {"pads": [0] * (2 * ndim)}
    x, restore = _channels_first(g, prefix, x, rank, _channel_axis(layer))
    x = g.node(
        "AveragePool",
        [g.quantize_input(layer, prefix, x)],
        f"{prefix}_pool",
        kernel_shape=to_list(layer.pool_size, ndim),
        strides=to_list(layer.strides, ndim),
        count_include_pad=0,
        **padding,
    )
    return _restore_layout(g, prefix, g.quantize_output(layer, prefix, x), restore)


def _add_global_avgpool(g, layer, prefix, x, rank):
    x, restore = _channels_first(g, prefix, x, rank, _channel_axis(layer))
    x = g.node("GlobalAveragePool", [x], f"{prefix}_global_pool")  # keeps the spatial dims with size 1
    if layer.keepdims:
        return _restore_layout(g, prefix, x, restore)
    return g.node("Flatten", [x], f"{prefix}_flatten", axis=1)
