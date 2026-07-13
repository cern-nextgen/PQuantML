from __future__ import annotations

import builtins
import operator
from functools import wraps
from typing import Any

import numpy as np
import torch

try:
    from torch.fx._symbolic_trace import is_fx_symbolic_tracing
except ImportError:  # torch < 2.8 exposes it as is_fx_tracing
    from torch.fx._symbolic_trace import is_fx_tracing as is_fx_symbolic_tracing
from alkaid.converter.builtin.torch.layers.functional import _functional_map
from alkaid.converter.builtin.torch.layers.methods import _method_map
from alkaid.converter.builtin.torch.layers.modules import ReplayModuleBase
from alkaid.trace import FVArray

from pquant._alkaid_plugin._alkaid_common import (
    PQuantAlkaidError,
    replay_quantizer,
    replay_quantizer_if_enabled,
)
from pquant.core.torch.activations import PQActivation
from pquant.core.torch.layers import (
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQSoftmax,
    PQWeightBiasBase,
)
from pquant.core.torch.quantizer import Quantizer


def _contains_fx_proxy(value: Any) -> bool:
    from torch.fx.proxy import Proxy

    if isinstance(value, Proxy):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_fx_proxy(v) for v in value)
    if isinstance(value, dict):
        return any(_contains_fx_proxy(v) for v in value.values())
    return False


def _patch_once(cls: type, name: str, wrapper_factory) -> None:
    marker = f"__alkaid_pquant_patched_{name}__"
    if getattr(cls, marker, False):
        return
    original = getattr(cls, name)
    setattr(cls, f"__alkaid_pquant_original_{name}__", original)
    setattr(cls, name, wrapper_factory(original))
    setattr(cls, marker, True)


def _module_parameter(module: torch.nn.Module, name: str) -> Any:
    if name in module._parameters:
        return module._parameters[name]
    return getattr(module, name)


def _module_bool(module: torch.nn.Module, name: str, default: bool = False) -> bool:
    value = module._parameters.get(name, getattr(module, name, default))
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    return bool(value)


def _assert_final_compression(module: torch.nn.Module) -> None:
    if not _module_bool(module, "final_compression_done"):
        raise PQuantAlkaidError(
            f"{type(module).__name__} must have apply_final_compression() applied before Alkaid conversion."
        )


def _patch_weight_bias_properties() -> None:
    for cls in (PQDense, PQConv1d, PQConv2d, PQBatchNorm1d, PQBatchNorm2d):
        marker = "__alkaid_pquant_patched_weight_bias__"
        if getattr(cls, marker, False):
            continue
        original_weight = cls.weight.fget
        original_bias = cls.bias.fget

        def weight(self, _original_weight=original_weight):
            if not is_fx_symbolic_tracing():
                return _original_weight(self)
            _assert_final_compression(self)
            return _module_parameter(self, "_weight")

        def bias(self, _original_bias=original_bias):
            if not is_fx_symbolic_tracing():
                return _original_bias(self)
            _assert_final_compression(self)
            return _module_parameter(self, "_bias")

        cls.weight = property(weight)
        cls.bias = property(bias)
        setattr(cls, marker, True)


def _patch_lazy_build_assertions() -> None:
    def wrap_pre_forward(original):
        @wraps(original)
        def wrapped(self, x):
            if not _contains_fx_proxy(x):
                return original(self, x)
            if self.quantize_input:
                x = self.quantize(x, self.input_quantizer)
            return x

        return wrapped

    _patch_once(PQWeightBiasBase, "pre_forward", wrap_pre_forward)

    def wrap_pre_activation(original):
        @wraps(original)
        def wrapped(self, x):
            if not _contains_fx_proxy(x):
                return original(self, x)
            if not self.use_hgq and self.use_multiplier and self.activation_name == "relu" and hasattr(self, "multiplier"):
                multiplier = _module_parameter(self, "multiplier")
                x = x * (2.0 ** torch.round(multiplier.detach()).item())
            if self.quantize_input and self.enable_quantization:
                x = self.input_quantizer(x)
            return x

        return wrapped

    _patch_once(PQActivation, "pre_activation", wrap_pre_activation)

    def wrap_bn_forward(original):
        @wraps(original)
        def wrapped(self, input):
            if not _contains_fx_proxy(input):
                return original(self, input)
            if self.quantize_input and self.enable_quantization:
                input = self.input_quantizer(input)
            return torch.nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )

        return wrapped

    _patch_once(PQBatchNorm1d, "forward", wrap_bn_forward)
    _patch_once(PQBatchNorm2d, "forward", wrap_bn_forward)


class ReplayPQuantQuantizer(ReplayModuleBase):
    handles = (Quantizer,)

    def call(self, input: FVArray) -> FVArray:
        return replay_quantizer(self.module, input)


def _table_fn(table):
    """Numpy-callable for a PQActivation lookup table, evaluated in float32 like the torch runtime."""
    fn = table.activation_function

    def apply_fn(v: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(v, dtype=torch.float32, device="cpu")
            return fn(t).detach().cpu().numpy().astype(np.float64)

    return apply_fn


class ReplayPQuantSoftmax(ReplayModuleBase):
    """Replay PQSoftmax as a single fx-leaf module"""

    handles = (PQSoftmax,)

    @staticmethod
    def _replay_table(table, x: FVArray) -> FVArray:
        if not (table.quantize_output and table.enable_quantization):
            raise PQuantAlkaidError(
                f"PQSoftmax table {type(table).__name__} must have an enabled output quantizer for Alkaid conversion."
            )
        x = replay_quantizer_if_enabled(table, "input_quantizer", x, "quantize_input")
        out = x.apply(_table_fn(table))
        return replay_quantizer(table.output_quantizer, out)

    def call(self, inputs: FVArray, mask=None) -> FVArray:
        module = self.module
        if mask is not None:
            raise PQuantAlkaidError("PQSoftmax masks are not supported in Alkaid conversion.")
        if not module.built:
            raise PQuantAlkaidError("PQSoftmax must be built (one real forward) before Alkaid conversion.")
        inputs = replay_quantizer_if_enabled(module, "input_quantizer", inputs, "quantize_input")
        if module.stable:
            inputs = np.max(inputs, axis=module.axes, keepdims=True) - inputs  # type: ignore
        exp_inp = self._replay_table(module.exp_table, inputs)
        sums = np.sum(exp_inp, axis=module.axes, keepdims=True)
        divisor = self._replay_table(module.inv_table, sums)
        out = exp_inp * divisor
        return replay_quantizer_if_enabled(module, "output_quantizer", out, "quantize_output")


def _patch_root_quantizer_trace() -> None:
    import alkaid.converter.builtin.torch.main as torch_main

    tracer_cls = torch_main.TorchALIRTracer
    marker = "__alkaid_pquant_patched_root_quantizer__"
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
            return {"inputs": tuple(inputs), "quantizer/final": dump["final"], "final": dump["final"]}, ["final"]
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
    _method_map.setdefault("pow", lambda receiver, exponent, **_kwargs: receiver**exponent)


def register() -> None:
    """Entry point for Alkaid's ``alkaid_torch`` second-level plugin group."""
    _patch_lazy_build_assertions()
    _patch_weight_bias_properties()
    _patch_root_quantizer_trace()
    _register_functional_helpers()
    try:
        from alkaid.converter import _plugin_loader

        _plugin_loader._LOADED.add(("pquant", "torch"))
    except Exception:
        pass
