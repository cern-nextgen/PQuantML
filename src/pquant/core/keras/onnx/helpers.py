"""Keras-specific utilities for the PQuant Keras → ONNX converter.

The backend-agnostic node emitters live in ``pquant.core.onnx_common``.
"""

import keras
from onnx import TensorProto


def _keras_dtype_to_tp(dtype):
    """Map a Keras/numpy dtype string to an ONNX TensorProto dtype (default float32)."""
    return {
        "float32": TensorProto.FLOAT,
        "float64": TensorProto.DOUBLE,
        "float16": TensorProto.FLOAT16,
        "bool": TensorProto.BOOL,
        "int64": TensorProto.INT64,
        "int32": TensorProto.INT32,
    }.get(str(dtype), TensorProto.FLOAT)


def _channels_last(layer):
    return getattr(layer, "data_format", keras.config.image_data_format()) == "channels_last"


def _nchw_perms(ndim):
    """Permutations between the Keras channels_last and ONNX channels_first layouts."""
    if ndim == 2:
        return [0, 3, 1, 2], [0, 2, 3, 1]
    return [0, 2, 1], [0, 2, 1]


def _bn_transpose_info(layer):
    """
    Return (need_transpose, perm_fwd, perm_bwd) for a BatchNormalization layer.

    ONNX BN always normalises on axis 1 (NCHW; true in every opset), so
    channels_last inputs need Transpose nodes around the BN op.
    """
    axis = getattr(layer, "axis", 1)
    stored = getattr(layer, "input_shape", None)
    ndim = len(stored) if stored is not None else len(layer.input.shape)
    eff_axis = axis if axis >= 0 else (ndim + axis)

    if eff_axis == 1 or ndim <= 2:
        # channels already at position 1, or 2-D input — no transpose needed
        return False, None, None

    if ndim == 4 and eff_axis == 3:
        return True, [0, 3, 1, 2], [0, 2, 3, 1]

    if ndim == 3 and eff_axis == 2:
        return True, [0, 2, 1], [0, 2, 1]

    # Fallback: general permutation that moves eff_axis to position 1
    perm_fwd = [0, eff_axis] + [i for i in range(1, ndim) if i != eff_axis]
    perm_bwd = [0] * ndim
    for i, p in enumerate(perm_fwd):
        perm_bwd[p] = i
    return True, perm_fwd, perm_bwd
