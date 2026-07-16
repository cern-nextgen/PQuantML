from __future__ import annotations

from math import prod

import keras
import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase
from alkaid.converter.builtin.keras.layers.activation import keras_numpy_unary_map
from alkaid.converter.builtin.keras.layers.batchnorm import ReplayBatchNormalization
from alkaid.converter.builtin.keras.layers.conv import _conv
from alkaid.converter.builtin.keras.layers.pool import ReplayPool
from alkaid.trace import FVArray
from alkaid.trace.ops import einsum, extract_patches
from keras.layers import DepthwiseConv1D, DepthwiseConv2D

from pquant._alkaid_plugin._alkaid_common import (
    PQuantAlkaidError,
    _assert_final_compression,
    _final_bias,
    _mark_plugin_loaded,
    _replay_quantizer,
    _replay_quantizer_if_enabled,
    _replay_softmax,
    _scale_by_relu_multiplier,
    _to_numpy,
)
from pquant.core.keras.activations import PQActivation
from pquant.core.keras.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
    PQSeparableConv2d,
    PQSoftmax,
)
from pquant.core.keras.quantizer import Quantizer


def _final_kernel(layer) -> np.ndarray:
    """The layer's final (compressed) kernel as numpy. Asserts compression was applied."""
    _assert_final_compression(layer)
    return _to_numpy(layer._kernel)


def _table_fn(table):
    """Numpy-callable for a PQActivation lookup table, evaluated in float32 like the keras runtime."""
    fn = table.activation_function

    def apply_fn(v: np.ndarray) -> np.ndarray:
        t = keras.ops.cast(keras.ops.convert_to_tensor(v), 'float32')
        return np.asarray(keras.ops.convert_to_numpy(fn(t)), dtype=np.float64)

    return apply_fn


def _unpack_query_key_value(inputs):
    if not isinstance(inputs, (list, tuple)):
        return inputs, inputs, inputs
    if len(inputs) == 3:
        return inputs[0], inputs[1], inputs[2]
    if len(inputs) == 2:
        return inputs[0], inputs[1], inputs[1]
    return inputs[0], inputs[0], inputs[0]


class ReplayPQuantQuantizer(ReplayOperationBase):
    __activation_handled__ = True
    handles = (Quantizer,)

    def call(self, x: FVArray) -> FVArray:
        return _replay_quantizer(self.op, x)


class ReplayPQuantDense(ReplayOperationBase):
    handles = (PQDense,)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        inputs = _replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
        out = np.einsum('...c,cC->...C', inputs, _final_kernel(layer)) + _final_bias(layer)
        return _replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantConv(ReplayOperationBase):
    handles = (PQConv1d, PQConv2d, PQDepthwiseConv2d)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        inputs = _replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
        kernel = _final_kernel(layer)
        bias = _final_bias(layer)

        if isinstance(layer, (DepthwiseConv1D, DepthwiseConv2D)):
            ch_in, dm = kernel.shape[-2:]
            kernel = kernel.reshape(*kernel.shape[:-2], 1, ch_in * dm)
            groups = ch_in
        else:
            groups = layer.groups

        x = extract_patches(
            inputs,
            size=layer.kernel_size,
            strides=layer.strides,
            dilation_rate=layer.dilation_rate,
            padding=layer.padding,
            data_format=layer.data_format,
        )
        ch_out = kernel.shape[-1]
        ch_in_per_g = kernel.shape[-2]
        k_vol = int(prod(layer.kernel_size))
        out = _conv(
            x,
            kernel,
            k_vol=k_vol,
            groups=groups,
            ch_in_per_g=ch_in_per_g,
            out_per_g=ch_out // groups,
        )
        if bias.shape != ():
            out = out + bias
        if layer.data_format == 'channels_first':
            out = np.moveaxis(out, -1, 1)  # type: ignore
        return _replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantSeparableConv(ReplayOperationBase):
    handles = (PQSeparableConv2d,)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        x = ReplayPQuantConv(layer.depthwise_conv).call(inputs)
        return ReplayPQuantConv(layer.pointwise_conv).call(x)


class ReplayPQuantBatchNormalization(ReplayBatchNormalization):
    handles = (PQBatchNormalization,)

    def fused_scale_offset(self) -> tuple[np.ndarray, np.ndarray]:
        layer = self.op
        _assert_final_compression(layer)
        mean = _to_numpy(keras.ops.cast(layer.moving_mean, layer.dtype))
        variance = _to_numpy(keras.ops.cast(layer.moving_variance, layer.dtype))
        if layer.scale:
            gamma = _to_numpy(keras.ops.cast(layer.gamma, layer.dtype))
        else:
            gamma = np.ones_like(mean)
        if layer.center:
            beta = _to_numpy(keras.ops.cast(layer.beta, layer.dtype))
        else:
            beta = np.zeros_like(mean)
        scale = gamma / np.sqrt(variance + layer.epsilon)
        offset = beta - mean * scale
        return scale, offset

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        layer = self.op
        inputs = _replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
        scale, offset = self.fused_scale_offset()
        shape = [1] * inputs.ndim
        axis = layer.axis if isinstance(layer.axis, (list, tuple)) else [layer.axis]
        for a in axis:
            aa = a if a >= 0 else inputs.ndim + a
            shape[aa] = inputs.shape[aa]
        out = inputs
        if not np.all(scale == 1):
            out = out * scale.reshape(shape)  # type: ignore
        if not np.all(offset == 0):
            out = out + offset.reshape(shape)  # type: ignore
        return out


class ReplayPQuantAvgPool(ReplayPool):
    __activation_handled__ = True
    handles = (PQAvgPool1d, PQAvgPool2d)

    def call(self, inputs: FVArray, mask: None = None) -> FVArray:
        layer = self.op
        inputs = _replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
        out = super().call(inputs, mask=mask)
        return _replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantActivation(ReplayOperationBase):
    __activation_handled__ = True
    handles = (PQActivation,)

    def call(self, inputs: FVArray) -> FVArray:
        layer = self.op
        inputs = _scale_by_relu_multiplier(layer, inputs)
        inputs = _replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
        if layer.activation_name not in keras_numpy_unary_map:
            raise PQuantAlkaidError(f'Unsupported PQuant activation for Alkaid conversion: {layer.activation_name!r}')
        out = keras_numpy_unary_map[layer.activation_name](inputs)
        return _replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantSoftmax(ReplayOperationBase):
    __activation_handled__ = True
    handles = (PQSoftmax,)

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        if mask is not None:
            raise PQuantAlkaidError('PQSoftmax masks are not supported in Alkaid conversion.')
        return _replay_softmax(self.op, inputs, _table_fn)


class ReplayPQuantMultiheadAttention(ReplayOperationBase):
    __activation_handled__ = True
    handles = (PQMultiheadAttention,)

    def call(self, inputs, key_padding_mask=None, attn_mask=None, need_weights=True):
        layer = self.op
        if key_padding_mask is not None or attn_mask is not None:
            raise PQuantAlkaidError('Attention masks are not supported in Alkaid conversion.')

        query, key, value = _unpack_query_key_value(inputs)
        batch_size, query_len = query.shape[0], query.shape[1]
        key_len = key.shape[1]
        num_heads, head_dim = layer.num_heads, layer.head_dim

        q = ReplayPQuantDense(layer.q_proj).call(query)  # (B, T, E)
        k = ReplayPQuantDense(layer.k_proj).call(key)  # (B, S, E)
        v = ReplayPQuantDense(layer.v_proj).call(value)  # (B, S, E)

        # Reshape to (B, H, T/S, head_dim)
        q = q.reshape(batch_size, query_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, key_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, key_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        scale = float(np.float32(layer.scale))
        attn_scores = einsum('bhtd,bhsd->bhts', q, k) * scale

        # The softmax's own input/output quantizers handle the scores and the attention weights
        attn_weights = ReplayPQuantSoftmax(layer.softmax).call(attn_scores)

        # Weighted sum of values (dropout is an inference no-op): (B, H, T, head_dim)
        out = einsum('bhts,bhsd->bhtd', attn_weights, v)

        # Merge heads: (B, T, E)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, query_len, layer.embed_dim)
        out = ReplayPQuantDense(layer.out_proj).call(out)

        if need_weights:
            # Average attention weights over heads: (B, T, S)
            return out, np.mean(attn_weights, axis=1)
        return (out,)


def register() -> None:
    """Entry point for Alkaid's ``alkaid_keras`` second-level plugin group."""
    _mark_plugin_loaded('keras')
