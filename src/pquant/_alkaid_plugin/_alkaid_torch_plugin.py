from __future__ import annotations

import builtins
import operator
from functools import wraps
from typing import Any

import numpy as np
import torch
from alkaid.converter.builtin.torch.layers.direct import torch_numpy_unary_map
from alkaid.converter.builtin.torch.layers.functional import (
    _functional_map,
    conv_nd_replay,
    replay_avg_pool,
)
from alkaid.converter.builtin.torch.layers.methods import _method_map
from alkaid.converter.builtin.torch.layers.modules import (
    ReplayBatchNorm,
    ReplayModuleBase,
)
from alkaid.trace import FVArray

from pquant._alkaid_plugin._alkaid_common import (
    PQuantAlkaidError,
    assert_final_compression,
    final_bias,
    mark_plugin_loaded,
    replay_quantizer,
    replay_quantizer_if_enabled,
    replay_softmax,
    scale_by_relu_multiplier,
    to_numpy,
)
from pquant.core.torch.activations import PQActivation, PQSoftmax
from pquant.core.torch.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
)
from pquant.core.torch.quantizer import Quantizer


def final_weight(layer: torch.nn.Module) -> np.ndarray:
    """The layer's final (compressed) weight as numpy. Asserts compression was applied."""
    assert_final_compression(layer)
    return to_numpy(layer._weight)


def activation_numpy_fn(layer: PQActivation):
    """The numpy elementwise function for a named PQActivation (relu/tanh/gelu/...)."""
    name = layer.activation_name
    fn = torch_numpy_unary_map.get(name) or torch_numpy_unary_map.get(name.replace('_', ''))
    if fn is not None:
        return fn
    if name == 'leaky_relu':
        slope = float(getattr(layer.activation_function, 'negative_slope', 0.1015625))
        return lambda x: np.where(x < 0, x * slope, x)  # type: ignore
    raise PQuantAlkaidError(f'Unsupported PQuant activation for Alkaid conversion: {name!r}')


def table_fn(table):
    """Numpy-callable for a PQActivation lookup table, evaluated in float32 like the torch runtime."""
    fn = table.activation_function

    def apply_fn(v: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(v, dtype=torch.float32, device='cpu')
            return fn(t).detach().cpu().numpy().astype(np.float64)

    return apply_fn


class ReplayPQuantQuantizer(ReplayModuleBase):
    handles = (Quantizer,)

    def call(self, input: FVArray) -> FVArray:
        return replay_quantizer(self.module, input)


class ReplayPQuantDense(ReplayModuleBase):
    handles = (PQDense,)

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        out = input @ final_weight(layer).T
        bias = final_bias(layer)
        if bias.shape != ():
            out = out + bias
        return replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantConv(ReplayModuleBase):
    handles = (PQConv1d, PQConv2d)

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        out = conv_nd_replay(
            input,
            final_weight(layer),
            final_bias(layer),
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        return replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantBatchNorm(ReplayBatchNorm):
    handles = (PQBatchNorm1d, PQBatchNorm2d)

    def fused_scale_offset(self) -> tuple[np.ndarray, np.ndarray]:
        layer = self.module
        assert_final_compression(layer)
        mean = to_numpy(layer.running_mean)
        variance = to_numpy(layer.running_var)
        gamma = to_numpy(layer._weight) if layer._weight is not None else np.ones_like(mean)
        beta = to_numpy(layer._bias) if layer._bias is not None else np.zeros_like(mean)
        scale = gamma / np.sqrt(variance + layer.eps)
        offset = beta - mean * scale
        return scale, offset

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        return super().call(input)


class ReplayPQuantAvgPool(ReplayModuleBase):
    handles = (PQAvgPool1d, PQAvgPool2d)

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        out = replay_avg_pool(
            input,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.ceil_mode,
            layer.count_include_pad,
        )
        return replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantActivation(ReplayModuleBase):
    handles = (PQActivation,)

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        input = scale_by_relu_multiplier(layer, input)
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        out = activation_numpy_fn(layer)(input)
        return replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


class ReplayPQuantSoftmax(ReplayModuleBase):
    """Replay PQSoftmax as a single fx-leaf module"""

    handles = (PQSoftmax,)

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        module = self.module
        if mask is not None:
            raise PQuantAlkaidError('PQSoftmax masks are not supported in Alkaid conversion.')
        if not module.built:
            raise PQuantAlkaidError('PQSoftmax must be built (one real forward) before Alkaid conversion.')
        return replay_softmax(module, inputs, table_fn)


def patch_root_quantizer_trace() -> None:
    """Let a bare Quantizer be traced as the model root, which Alkaid's fx tracer cannot handle."""
    import alkaid.converter.builtin.torch.main as torch_main

    tracer_cls = torch_main.TorchALIRTracer
    marker = '__alkaid_pquant_patched_root_quantizer__'
    if getattr(tracer_cls, marker, False):
        return
    original = tracer_cls.apply_model

    @wraps(original)
    def wrapped(self, verbose: bool, inputs: tuple[FVArray, ...]):
        if isinstance(self.model, Quantizer):
            if isinstance(inputs, FVArray):
                inputs = (inputs,)
            replay = ReplayPQuantQuantizer(self.model)
            dump = replay(*inputs)
            return {'inputs': tuple(inputs), 'quantizer/final': dump['final'], 'final': dump['final']}, ['final']
        return original(self, verbose, inputs)

    tracer_cls.apply_model = wrapped
    setattr(tracer_cls, marker, True)


def replay_getattr(obj: Any, name: str, *default: Any) -> Any:
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def replay_tensor(data: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(data, FVArray):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def normalize_shape(args: tuple[Any, ...]) -> tuple[int, ...]:
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(int(v) for v in args[0])
    return tuple(int(v) for v in args)


def replay_zeros(*size: Any, **kwargs: Any) -> np.ndarray:
    return np.zeros(normalize_shape(size), dtype=np.float32)


def replay_ones(*size: Any, **kwargs: Any) -> np.ndarray:
    return np.ones(normalize_shape(size), dtype=np.float32)


def replay_full(size: Any, fill_value: Any, **kwargs: Any) -> np.ndarray:
    return np.full(normalize_shape((size,)), fill_value, dtype=np.float32)


def replay_zeros_like(x: Any, **kwargs: Any) -> np.ndarray:
    return np.zeros(tuple(x.shape), dtype=np.float32)


def replay_ones_like(x: Any, **kwargs: Any) -> np.ndarray:
    return np.ones(tuple(x.shape), dtype=np.float32)


def register_functional_helpers() -> None:
    _functional_map.setdefault(operator.pow, lambda a, b: a**b)
    _functional_map.setdefault(torch.pow, lambda a, b: a**b)
    _functional_map.setdefault(builtins.getattr, replay_getattr)
    _functional_map.setdefault(torch.tensor, replay_tensor)
    _functional_map.setdefault(torch.as_tensor, replay_tensor)
    _functional_map.setdefault(torch.zeros, replay_zeros)
    _functional_map.setdefault(torch.ones, replay_ones)
    _functional_map.setdefault(torch.full, replay_full)
    _functional_map.setdefault(torch.zeros_like, replay_zeros_like)
    _functional_map.setdefault(torch.ones_like, replay_ones_like)
    _method_map.setdefault('pow', lambda receiver, exponent, **_kwargs: receiver**exponent)


def register() -> None:
    """Entry point for Alkaid's ``alkaid_torch`` second-level plugin group."""
    patch_root_quantizer_trace()
    register_functional_helpers()
    mark_plugin_loaded('torch')
