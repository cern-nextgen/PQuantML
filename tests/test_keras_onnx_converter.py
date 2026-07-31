"""Tests for the Keras → ONNX converter (convert_to_onnx).

Each test builds a small functional Keras model, runs a forward pass to
initialise all sublayer state, calls apply_final_compression, exports to ONNX
via convert_to_onnx(), and verifies that onnxruntime produces the same output
as the Keras model.

bias=True/False is tested via parametrize where applicable.
"""

import keras
import numpy as np
import pytest

import pquant
from pquant.core.keras.convert_to_onnx import convert_to_onnx
from pquant.core.keras.layers import (
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    apply_final_compression,
)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

ATOL = 1e-4


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = False
    return c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _channels_first():
    return keras.backend.image_data_format() == "channels_first"


def _keras_out(model, x: np.ndarray) -> np.ndarray:
    from keras import ops

    return ops.convert_to_numpy(model(x, training=False))


def _onnx_run(model, x: np.ndarray, input_shape: tuple, tmp_path) -> np.ndarray:
    path = str(tmp_path / "model.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x})[0]


# ---------------------------------------------------------------------------
# PQDense
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_dense_onnx(cfg, bias, tmp_path):
    IN, OUT = 16, 8
    inputs = keras.Input(shape=(IN,))
    x = PQDense(cfg, units=OUT, use_bias=bias)(inputs)
    model = keras.Model(inputs, x)

    dummy = np.zeros((1, IN), dtype=np.float32)
    model(dummy)
    apply_final_compression(model)

    x_np = np.random.randn(4, IN).astype(np.float32)
    keras_out = _keras_out(model, x_np)
    onnx_out = _onnx_run(model, x_np, input_shape=(IN,), tmp_path=tmp_path)
    np.testing.assert_allclose(keras_out, onnx_out, atol=ATOL, err_msg=f"PQDense bias={bias}: keras vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQConv2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_onnx(cfg, bias, tmp_path):
    IN_C, OUT_C, H, W = 3, 8, 8, 8
    if _channels_first():
        input_shape = (IN_C, H, W)
        x_np = np.random.randn(2, IN_C, H, W).astype(np.float32)
    else:
        input_shape = (H, W, IN_C)
        x_np = np.random.randn(2, H, W, IN_C).astype(np.float32)

    inputs = keras.Input(shape=input_shape)
    x = PQConv2d(cfg, OUT_C, kernel_size=3, padding="same", use_bias=bias)(inputs)
    model = keras.Model(inputs, x)

    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    model(dummy)
    apply_final_compression(model)

    keras_out = _keras_out(model, x_np)
    onnx_out = _onnx_run(model, x_np, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(keras_out, onnx_out, atol=ATOL, err_msg=f"PQConv2d bias={bias}: keras vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQConv1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_conv1d_onnx(cfg, bias, tmp_path):
    IN_C, OUT_C, L = 4, 8, 16
    if _channels_first():
        input_shape = (IN_C, L)
        x_np = np.random.randn(2, IN_C, L).astype(np.float32)
    else:
        input_shape = (L, IN_C)
        x_np = np.random.randn(2, L, IN_C).astype(np.float32)

    inputs = keras.Input(shape=input_shape)
    x = PQConv1d(cfg, OUT_C, kernel_size=3, padding="same", use_bias=bias)(inputs)
    model = keras.Model(inputs, x)

    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    model(dummy)
    apply_final_compression(model)

    keras_out = _keras_out(model, x_np)
    onnx_out = _onnx_run(model, x_np, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(keras_out, onnx_out, atol=ATOL, err_msg=f"PQConv1d bias={bias}: keras vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQBatchNormalization
# ---------------------------------------------------------------------------


def test_batchnorm_onnx(cfg, tmp_path):
    IN_C, H, W = 8, 4, 4
    if _channels_first():
        input_shape = (IN_C, H, W)
        x_np = np.random.randn(4, IN_C, H, W).astype(np.float32)
        bn_axis = 1
    else:
        input_shape = (H, W, IN_C)
        x_np = np.random.randn(4, H, W, IN_C).astype(np.float32)
        bn_axis = -1

    inputs = keras.Input(shape=input_shape)
    x = PQBatchNormalization(cfg, axis=bn_axis)(inputs)
    model = keras.Model(inputs, x)

    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    model(dummy, training=True)  # warm up running stats
    apply_final_compression(model)

    keras_out = _keras_out(model, x_np)
    onnx_out = _onnx_run(model, x_np, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(keras_out, onnx_out, atol=ATOL, err_msg="PQBatchNormalization: keras vs ONNX mismatch")


# ---------------------------------------------------------------------------
# PQDepthwiseConv2d
# ---------------------------------------------------------------------------


def test_depthwise_conv2d_onnx(cfg, tmp_path):
    IN_C, H, W = 4, 8, 8
    if _channels_first():
        input_shape = (IN_C, H, W)
        x_np = np.random.randn(2, IN_C, H, W).astype(np.float32)
    else:
        input_shape = (H, W, IN_C)
        x_np = np.random.randn(2, H, W, IN_C).astype(np.float32)

    inputs = keras.Input(shape=input_shape)
    x = PQDepthwiseConv2d(cfg, kernel_size=3, padding="same")(inputs)
    model = keras.Model(inputs, x)

    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    model(dummy)
    apply_final_compression(model)

    keras_out = _keras_out(model, x_np)
    onnx_out = _onnx_run(model, x_np, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(keras_out, onnx_out, atol=ATOL, err_msg="PQDepthwiseConv2d: keras vs ONNX mismatch")
