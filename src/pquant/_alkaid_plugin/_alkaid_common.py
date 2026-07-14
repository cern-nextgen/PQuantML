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
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
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
    for storage_name in ("_parameters", "_buffers", "_modules"):
        storage = getattr(obj, storage_name, None)
        if isinstance(storage, dict) and name in storage:
            return storage[name]
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return getattr(obj, name, default)


def quantizer_kif(quantizer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(quantizer, "_parameters"):
        if not bool(raw_module_attr(quantizer, "use_hgq", False)):
            return (
                to_int_bits(raw_module_attr(quantizer, "k")),
                to_int_bits(raw_module_attr(quantizer, "i")),
                to_int_bits(raw_module_attr(quantizer, "f")),
            )
        inner = raw_module_attr(quantizer, "quantizer")
        if hasattr(inner, "_parameters") or hasattr(inner, "_buffers"):
            k = raw_module_attr(inner, "_k")
            i = raw_module_attr(inner, "_i_raw", None)
            if i is None:
                i = raw_module_attr(inner, "_i")
            f = raw_module_attr(inner, "_f")
            return to_int_bits(k), to_int_bits(i), to_int_bits(f)
    k, i, f = quantizer.get_quantization_bits()
    return to_int_bits(k), to_int_bits(i), to_int_bits(f)


def replay_quantizer(quantizer: Any, x: Any) -> Any:
    k, i, f = quantizer_kif(quantizer)
    inner = raw_module_attr(quantizer, "quantizer", None)
    overflow = raw_module_attr(quantizer, "overflow", raw_module_attr(inner, "overflow_mode", "WRAP"))
    round_mode = raw_module_attr(quantizer, "round_mode", raw_module_attr(inner, "round_mode", "TRN"))
    return alkaid_quantize(x, k=k, i=i, f=f, overflow_mode=str(overflow).upper(), round_mode=str(round_mode).upper())


def replay_quantizer_if_enabled(layer: Any, quantizer_name: str, x: Any, flag_name: str) -> Any:
    if not bool(getattr(layer, "enable_quantization", True)):
        return x
    if not bool(getattr(layer, flag_name, True)):
        return x
    quantizer = getattr(layer, quantizer_name, None)
    return replay_quantizer(quantizer, x)
