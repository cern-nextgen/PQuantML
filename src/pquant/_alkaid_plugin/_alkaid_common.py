from __future__ import annotations

from typing import Any

import numpy as np
from alkaid.trace.ops import quantize as alkaid_quantize


class PQuantAlkaidError(ValueError):
    """Raised for PQuant states that cannot be replayed by Alkaid."""


def to_numpy(value: Any) -> np.ndarray:
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


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        arr = to_numpy(value)
    except Exception:
        return bool(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(np.all(arr))


def to_int_bits(value: Any) -> np.ndarray:
    return np.rint(to_numpy(value)).astype(np.int64)


def raw_module_attr(obj: Any, name: str, default: Any = None) -> Any:
    for storage_name in ('_parameters', '_buffers', '_modules'):
        storage = getattr(obj, storage_name, None)
        if isinstance(storage, dict) and name in storage:
            return storage[name]
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return getattr(obj, name, default)


def quantizer_kif(quantizer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(quantizer, '_parameters'):
        if not bool(raw_module_attr(quantizer, 'use_hgq', False)):
            return (
                to_int_bits(raw_module_attr(quantizer, 'k')),
                to_int_bits(raw_module_attr(quantizer, 'i')),
                to_int_bits(raw_module_attr(quantizer, 'f')),
            )
        inner = raw_module_attr(quantizer, 'quantizer')
        if hasattr(inner, '_parameters') or hasattr(inner, '_buffers'):
            k = raw_module_attr(inner, '_k')
            i = raw_module_attr(inner, '_i_raw', None)
            if i is None:
                i = raw_module_attr(inner, '_i')
            f = raw_module_attr(inner, '_f')
            return to_int_bits(k), to_int_bits(i), to_int_bits(f)
    k, i, f = quantizer.get_quantization_bits()
    return to_int_bits(k), to_int_bits(i), to_int_bits(f)


def replay_quantizer(quantizer: Any, x: Any) -> Any:
    k, i, f = quantizer_kif(quantizer)
    inner = raw_module_attr(quantizer, 'quantizer', None)
    overflow = raw_module_attr(quantizer, 'overflow', raw_module_attr(inner, 'overflow_mode', 'WRAP'))
    round_mode = raw_module_attr(quantizer, 'round_mode', raw_module_attr(inner, 'round_mode', 'TRN'))
    return alkaid_quantize(x, k=k, i=i, f=f, overflow_mode=str(overflow).upper(), round_mode=str(round_mode).upper())


def replay_quantizer_if_enabled(layer: Any, quantizer_name: str, x: Any, flag_name: str) -> Any:
    if not bool(getattr(layer, 'enable_quantization', True)):
        return x
    if not bool(getattr(layer, flag_name, True)):
        return x
    quantizer = getattr(layer, quantizer_name, None)
    return replay_quantizer(quantizer, x)


def assert_final_compression(layer: Any) -> None:
    if not to_bool(raw_module_attr(layer, 'final_compression_done', False)):
        raise PQuantAlkaidError(
            f'{type(layer).__name__} must have apply_final_compression() applied before Alkaid conversion.'
        )


def final_bias(layer: Any) -> np.ndarray:
    """The layer's final (compressed) bias as numpy, or a scalar zero when absent."""
    assert_final_compression(layer)
    bias = getattr(layer, '_bias', None)
    if bias is None:
        return np.array(0.0)
    return to_numpy(bias)


def scale_by_relu_multiplier(layer: Any, x: Any) -> Any:
    """Apply PQActivation's power-of-two ReLU multiplier, which is used only without HGQ."""
    applies = (
        not to_bool(getattr(layer, 'use_hgq', False))
        and to_bool(getattr(layer, 'use_multiplier', False))
        and layer.activation_name == 'relu'
        and hasattr(layer, 'multiplier')
    )
    if not applies:
        return x
    return x * (2.0 ** np.rint(to_numpy(layer.multiplier)))


def replay_table(table: Any, x: Any, table_fn: Any) -> Any:
    """Replay a lookup-table activation: quantize input, apply the table, quantize output."""
    if not (table.quantize_output and table.enable_quantization):
        name = getattr(table, 'name', None) or type(table).__name__
        raise PQuantAlkaidError(f'PQSoftmax table {name!r} must have an enabled output quantizer for Alkaid conversion.')
    x = replay_quantizer_if_enabled(table, 'input_quantizer', x, 'quantize_input')
    out = x.apply(table_fn(table))
    return replay_quantizer(table.output_quantizer, out)


def replay_softmax(layer: Any, inputs: Any, table_fn: Any) -> Any:
    """Replay PQSoftmax through its exp and inverse-sum lookup tables."""
    inputs = replay_quantizer_if_enabled(layer, 'input_quantizer', inputs, 'quantize_input')
    if layer.stable:
        inputs = np.max(inputs, axis=layer.axes, keepdims=True) - inputs  # type: ignore
    exponents = replay_table(layer.exp_table, inputs, table_fn)
    sums = np.sum(exponents, axis=layer.axes, keepdims=True)
    inverse_sums = replay_table(layer.inv_table, sums, table_fn)
    out = exponents * inverse_sums
    return replay_quantizer_if_enabled(layer, 'output_quantizer', out, 'quantize_output')


def mark_plugin_loaded(framework: str) -> None:
    """Record the pquant plugin as loaded in Alkaid's plugin loader, if that loader exists."""
    try:
        from alkaid.converter import _plugin_loader

        _plugin_loader._LOADED.add(('pquant', framework))
    except Exception:
        pass
