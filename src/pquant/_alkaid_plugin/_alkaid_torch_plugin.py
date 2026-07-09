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
    replay_quantizer,
    replay_quantizer_if_enabled,
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


def _module_bool(module: torch.nn.Module, name: str, default: bool = False) -> bool:
    value = module._parameters.get(name, getattr(module, name, default))
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    return bool(value)


def _assert_final_compression(module: torch.nn.Module) -> None:
    if not _module_bool(module, 'final_compression_done'):
        raise PQuantAlkaidError(
            f'{type(module).__name__} must have apply_final_compression() applied before Alkaid conversion.'
        )


def _weight(layer: torch.nn.Module) -> np.ndarray:
    """The layer's final (compressed) weight as numpy. Asserts compression was applied."""
    _assert_final_compression(layer)
    return to_numpy(layer._weight)


def _bias(layer: torch.nn.Module) -> np.ndarray:
    """The layer's final (compressed) bias as numpy, or ``np.array(0.0)`` when absent."""
    _assert_final_compression(layer)
    bias = getattr(layer, '_bias', None)
    if bias is None:
        return np.array(0.0)
    return to_numpy(bias)


class ReplayPQuantQuantizer(ReplayModuleBase):
    handles = (Quantizer,)

    def call(self, input: FVArray) -> FVArray:
        return replay_quantizer(self.module, input)


class ReplayPQuantDense(ReplayModuleBase):
    handles = (PQDense,)

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        out = input @ _weight(layer).T
        bias = _bias(layer)
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
            _weight(layer),
            _bias(layer),
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
        _assert_final_compression(layer)
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


def _activation_numpy_fn(layer: PQActivation):
    """The numpy elementwise function for a named PQActivation (relu/tanh/gelu/...)."""
    name = layer.activation_name
    fn = torch_numpy_unary_map.get(name) or torch_numpy_unary_map.get(name.replace('_', ''))
    if fn is not None:
        return fn
    if name == 'leaky_relu':
        slope = float(getattr(layer.activation_function, 'negative_slope', 0.1015625))
        return lambda x: np.where(x < 0, x * slope, x)  # type: ignore
    raise PQuantAlkaidError(f'Unsupported PQuant activation for Alkaid conversion: {name!r}')


class ReplayPQuantActivation(ReplayModuleBase):
    handles = (PQActivation,)

    def call(self, input: FVArray) -> FVArray:
        layer = self.module
        if (
            not bool(getattr(layer, 'use_hgq', False))
            and bool(getattr(layer, 'use_multiplier', False))
            and layer.activation_name == 'relu'
            and hasattr(layer, 'multiplier')
        ):
            input = input * (2.0 ** np.rint(to_numpy(layer.multiplier)))  # type: ignore
        input = replay_quantizer_if_enabled(layer, 'input_quantizer', input, 'quantize_input')
        out = _activation_numpy_fn(layer)(input)
        return replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


def _table_fn(table):
    """Numpy-callable for a PQActivation lookup table, evaluated in float32 like the torch runtime."""
    fn = table.activation_function

    def apply_fn(v: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(v, dtype=torch.float32, device='cpu')
            return fn(t).detach().cpu().numpy().astype(np.float64)

    return apply_fn


class ReplayPQuantSoftmax(ReplayModuleBase):
    """Replay PQSoftmax as a single fx-leaf module"""

    handles = (PQSoftmax,)

    @staticmethod
    def _replay_table(table, x: FVArray) -> FVArray:
        if not (table.quantize_output and table.enable_quantization):
            raise PQuantAlkaidError(
                f'PQSoftmax table {type(table).__name__} must have an enabled output quantizer for Alkaid conversion.'
            )
        x = replay_quantizer_if_enabled(table, 'input_quantizer', x, 'quantize_input')
        out = x.apply(_table_fn(table))
        return replay_quantizer(table.output_quantizer, out)

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        module = self.module
        if mask is not None:
            raise PQuantAlkaidError('PQSoftmax masks are not supported in Alkaid conversion.')
        if not module.built:
            raise PQuantAlkaidError('PQSoftmax must be built (one real forward) before Alkaid conversion.')
        inputs = replay_quantizer_if_enabled(module, 'input_quantizer', inputs, 'quantize_input')
        if module.stable:
            inputs = np.max(inputs, axis=module.axes, keepdims=True) - inputs  # type: ignore
        exp_inp = self._replay_table(module.exp_table, inputs)
        sums = np.sum(exp_inp, axis=module.axes, keepdims=True)
        divisor = self._replay_table(module.inv_table, sums)
        out = exp_inp * divisor
        return replay_quantizer_if_enabled(module, 'output_quantizer', out, 'quantize_output')


def _patch_root_quantizer_trace() -> None:
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


def _replay_getattr(obj: Any, name: str, *default: Any) -> Any:
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def _tensor(data: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(data, FVArray):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def _normalize_shape(args: tuple[Any, ...]) -> tuple[int, ...]:
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(int(v) for v in args[0])
    return tuple(int(v) for v in args)


def _zeros(*size: Any, **kwargs: Any) -> np.ndarray:
    return np.zeros(_normalize_shape(size), dtype=np.float32)


def _ones(*size: Any, **kwargs: Any) -> np.ndarray:
    return np.ones(_normalize_shape(size), dtype=np.float32)


def _full(size: Any, fill_value: Any, **kwargs: Any) -> np.ndarray:
    return np.full(_normalize_shape((size,)), fill_value, dtype=np.float32)


def _zeros_like(x: Any, **kwargs: Any) -> np.ndarray:
    return np.zeros(tuple(x.shape), dtype=np.float32)


def _ones_like(x: Any, **kwargs: Any) -> np.ndarray:
    return np.ones(tuple(x.shape), dtype=np.float32)


def _register_functional_helpers() -> None:
    _functional_map.setdefault(operator.pow, lambda a, b: a**b)
    _functional_map.setdefault(torch.pow, lambda a, b: a**b)
    _functional_map.setdefault(builtins.getattr, _replay_getattr)
    _functional_map.setdefault(torch.tensor, _tensor)
    _functional_map.setdefault(torch.as_tensor, _tensor)
    _functional_map.setdefault(torch.zeros, _zeros)
    _functional_map.setdefault(torch.ones, _ones)
    _functional_map.setdefault(torch.full, _full)
    _functional_map.setdefault(torch.zeros_like, _zeros_like)
    _functional_map.setdefault(torch.ones_like, _ones_like)
    _method_map.setdefault('pow', lambda receiver, exponent, **_kwargs: receiver**exponent)


def register() -> None:
    """Entry point for Alkaid's ``alkaid_torch`` second-level plugin group."""
    _patch_root_quantizer_trace()
    _register_functional_helpers()
    try:
        from alkaid.converter import _plugin_loader

        _plugin_loader._LOADED.add(('pquant', 'torch'))
    except Exception:
        pass
