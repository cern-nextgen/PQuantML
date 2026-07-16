from __future__ import annotations

from typing import Any

import numpy as np
from alkaid.trace.ops import quantize as alkaid_quantize


class PQuantAlkaidError(ValueError):
    """Raised for PQuant states that cannot be replayed by Alkaid."""


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array(0.0)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, 'detach'):
        value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        return value.numpy()
    try:
        import keras

        return np.asarray(keras.ops.convert_to_numpy(value))
    except Exception:
        return np.asarray(value)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        arr = _to_numpy(value)
    except Exception:
        return bool(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(np.all(arr))


def _to_int_bits(value: Any) -> np.ndarray:
    return np.rint(_to_numpy(value)).astype(np.int64)


def _raw_module_attr(obj: Any, name: str, default: Any = None) -> Any:
    for storage_name in ('_parameters', '_buffers', '_modules'):
        storage = getattr(obj, storage_name, None)
        if isinstance(storage, dict) and name in storage:
            return storage[name]
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return getattr(obj, name, default)


def _quantizer_kif(quantizer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(quantizer, '_parameters'):
        if not bool(_raw_module_attr(quantizer, 'use_hgq', False)):
            return (
                _to_int_bits(_raw_module_attr(quantizer, 'k')),
                _to_int_bits(_raw_module_attr(quantizer, 'i')),
                _to_int_bits(_raw_module_attr(quantizer, 'f')),
            )
        inner = _raw_module_attr(quantizer, 'quantizer')
        if hasattr(inner, '_parameters') or hasattr(inner, '_buffers'):
            k = _raw_module_attr(inner, '_k')
            i = _raw_module_attr(inner, '_i_raw', None)
            if i is None:
                i = _raw_module_attr(inner, '_i')
            f = _raw_module_attr(inner, '_f')
            return _to_int_bits(k), _to_int_bits(i), _to_int_bits(f)
    k, i, f = quantizer.get_quantization_bits()
    return _to_int_bits(k), _to_int_bits(i), _to_int_bits(f)


def _replay_quantizer(quantizer: Any, x: Any) -> Any:
    k, i, f = _quantizer_kif(quantizer)
    inner = _raw_module_attr(quantizer, 'quantizer', None)
    overflow = _raw_module_attr(quantizer, 'overflow', _raw_module_attr(inner, 'overflow_mode', 'WRAP'))
    round_mode = _raw_module_attr(quantizer, 'round_mode', _raw_module_attr(inner, 'round_mode', 'TRN'))
    return alkaid_quantize(x, k=k, i=i, f=f, overflow_mode=str(overflow).upper(), round_mode=str(round_mode).upper())


def _replay_quantizer_if_enabled(layer: Any, quantizer_name: str, x: Any, flag_name: str) -> Any:
    if not bool(getattr(layer, 'enable_quantization', True)):
        return x
    if not bool(getattr(layer, flag_name, True)):
        return x
    quantizer = getattr(layer, quantizer_name, None)
    return _replay_quantizer(quantizer, x)


def _assert_final_compression(layer: Any) -> None:
    if not _to_bool(_raw_module_attr(layer, 'final_compression_done', False)):
        raise PQuantAlkaidError(
            f'{type(layer).__name__} must have apply_final_compression() applied before Alkaid conversion.'
        )


def _final_bias(layer: Any) -> np.ndarray:
    """The layer's final (compressed) bias as numpy, or a scalar zero when absent."""
    _assert_final_compression(layer)
    bias = getattr(layer, '_bias', None)
    if bias is None:
        return np.array(0.0)
    return _to_numpy(bias)


def _scale_by_relu_multiplier(layer: Any, x: Any) -> Any:
    """Apply PQActivation's power-of-two ReLU multiplier, which is used only without HGQ."""
    applies = (
        not _to_bool(getattr(layer, 'use_hgq', False))
        and _to_bool(getattr(layer, 'use_multiplier', False))
        and layer.activation_name == 'relu'
        and hasattr(layer, 'multiplier')
    )
    if not applies:
        return x
    return x * (2.0 ** np.rint(_to_numpy(layer.multiplier)))


def _replay_table(table: Any, x: Any, table_fn: Any) -> Any:
    """Replay a lookup-table activation: quantize input, apply the table, quantize output."""
    if not (table.quantize_output and table.enable_quantization):
        name = getattr(table, 'name', None) or type(table).__name__
        raise PQuantAlkaidError(f'PQSoftmax table {name!r} must have an enabled output quantizer for Alkaid conversion.')
    x = _replay_quantizer_if_enabled(table, 'input_quantizer', x, 'quantize_input')
    out = x.apply(table_fn(table))
    return _replay_quantizer(table.output_quantizer, out)


def _replay_softmax(layer: Any, inputs: Any, table_fn: Any) -> Any:
    """Replay PQSoftmax through its exp and inverse-sum lookup tables."""
    inputs = _replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
    if layer.stable:
        inputs = np.max(inputs, axis=layer.axes, keepdims=True) - inputs  # type: ignore
    exponents = _replay_table(layer.exp_table, inputs, table_fn)
    sums = np.sum(exponents, axis=layer.axes, keepdims=True)
    inverse_sums = _replay_table(layer.inv_table, sums, table_fn)
    out = exponents * inverse_sums
    return _replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


def _mark_plugin_loaded(framework: str) -> None:
    """Record the pquant plugin as loaded in Alkaid's plugin loader, if that loader exists."""
    try:
        from alkaid.converter import _plugin_loader

        _plugin_loader._LOADED.add(('pquant', framework))
    except Exception:
        pass
