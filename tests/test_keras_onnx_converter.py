"""Tests for the Keras → ONNX converter (convert_to_onnx).

Each test builds a small functional Keras model, runs a forward pass to
initialise all sublayer state, calls apply_final_compression, exports to ONNX
via convert_to_onnx(), and verifies that onnxruntime produces the same output
as the Keras model.

bias=True/False is tested via parametrize where applicable.
"""

import keras
import numpy as np
import onnxruntime as ort
import pytest

import pquant
from pquant.core.keras.layers import (
    PQActivation,
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
    apply_final_compression,
)
from pquant.core.keras.onnx import convert_to_onnx

ATOL = 1e-4
QUANT_ATOL = 5e-3


def atol(cfg):
    return QUANT_ATOL if cfg.quantization_parameters.enable_quantization else ATOL


@pytest.fixture(params=[False, True], ids=["float", "quant"])
def cfg(request):
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = request.param
    return c


def channels_first():
    return keras.backend.image_data_format() == "channels_first"


def keras_out(model, x: np.ndarray) -> np.ndarray:
    from keras import ops

    return ops.convert_to_numpy(model(x, training=False))


def onnx_run(model, x: np.ndarray, input_shape: tuple, tmp_path) -> np.ndarray:
    path = str(tmp_path / "model.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x})[0]


# (layer factory, channels, spatial dims, batch size, warm-up call kwargs)
SINGLE_LAYER_CASES = [
    pytest.param(lambda cfg: PQDense(cfg, units=8, use_bias=True), 16, (), 4, {}, id="dense-bias"),
    pytest.param(lambda cfg: PQDense(cfg, units=8, use_bias=False), 16, (), 4, {}, id="dense-nobias"),
    pytest.param(
        lambda cfg: PQConv2d(cfg, 8, kernel_size=3, padding="same", use_bias=True), 3, (8, 8), 2, {}, id="conv2d-bias"
    ),
    pytest.param(
        lambda cfg: PQConv2d(cfg, 8, kernel_size=3, padding="same", use_bias=False), 3, (8, 8), 2, {}, id="conv2d-nobias"
    ),
    pytest.param(
        lambda cfg: PQConv1d(cfg, 8, kernel_size=3, padding="same", use_bias=True), 4, (16,), 2, {}, id="conv1d-bias"
    ),
    pytest.param(
        lambda cfg: PQConv1d(cfg, 8, kernel_size=3, padding="same", use_bias=False), 4, (16,), 2, {}, id="conv1d-nobias"
    ),
    pytest.param(
        lambda cfg: PQBatchNormalization(cfg, axis=1 if channels_first() else -1),
        8,
        (4, 4),
        4,
        {"training": True},  # warm up running stats
        id="batchnorm",
    ),
    pytest.param(lambda cfg: PQDepthwiseConv2d(cfg, kernel_size=3, padding="same"), 4, (8, 8), 2, {}, id="depthwise_conv2d"),
]


@pytest.mark.parametrize("make_layer,channels,spatial,batch,warmup_kwargs", SINGLE_LAYER_CASES)
def test_single_layer_onnx(cfg, make_layer, channels, spatial, batch, warmup_kwargs, tmp_path):
    input_shape = (channels, *spatial) if channels_first() else (*spatial, channels)
    x_np = np.random.randn(batch, *input_shape).astype(np.float32)

    inputs = keras.Input(shape=input_shape)
    x = make_layer(cfg)(inputs)
    model = keras.Model(inputs, x)

    model(np.zeros((1, *input_shape), dtype=np.float32), **warmup_kwargs)
    apply_final_compression(model)

    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(keras_output, onnx_out, atol=atol(cfg), err_msg="keras vs ONNX mismatch")


@pytest.mark.parametrize("activation", ["relu", "tanh", "hard_tanh"])
def test_pqactivation_onnx(cfg, activation, tmp_path):
    DIM = 16
    inputs = keras.Input(shape=(DIM,))
    x = PQActivation(cfg, activation)(inputs)
    model = keras.Model(inputs, x)

    model(np.zeros((1, DIM), dtype=np.float32))
    apply_final_compression(model)

    x_np = np.random.randn(4, DIM).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=(DIM,), tmp_path=tmp_path)
    np.testing.assert_allclose(
        keras_output, onnx_out, atol=atol(cfg), err_msg=f"PQActivation {activation}: keras vs ONNX mismatch"
    )


def test_residual_concat_onnx(cfg, tmp_path):
    DIM = 16
    inputs = keras.Input(shape=(DIM,))
    h = PQDense(cfg, units=DIM)(inputs)
    h2 = PQDense(cfg, units=DIM)(h)
    add = keras.layers.Add()([h, h2])  # residual / skip add
    cat = keras.layers.Concatenate(axis=-1)([add, inputs])  # branch merge
    out = PQDense(cfg, units=8)(cat)
    model = keras.Model(inputs, out)

    model(np.zeros((1, DIM), dtype=np.float32))
    apply_final_compression(model)

    x_np = np.random.randn(4, DIM).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=(DIM,), tmp_path=tmp_path)
    np.testing.assert_allclose(keras_output, onnx_out, atol=atol(cfg), err_msg="residual+concat: keras vs ONNX mismatch")


def test_two_input_onnx(cfg, tmp_path):
    IN_A, IN_B, OUT = 16, 4, 8
    a = keras.Input(shape=(IN_A,))
    b = keras.Input(shape=(IN_B,))
    ha = PQDense(cfg, units=OUT)(a)
    hb = PQDense(cfg, units=OUT)(b)
    out = keras.layers.Add()([ha, hb])
    model = keras.Model([a, b], out)

    model([np.zeros((1, IN_A), np.float32), np.zeros((1, IN_B), np.float32)])
    apply_final_compression(model)

    xa = np.random.randn(3, IN_A).astype(np.float32)
    xb = np.random.randn(3, IN_B).astype(np.float32)
    keras_output = keras_out(model, [xa, xb])

    path = str(tmp_path / "two_input.onnx")
    proto = convert_to_onnx(model, input_shape=[(IN_A,), (IN_B,)], output_path=path)
    in_shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in proto.graph.input}
    assert in_shapes["input_0"] == [0, IN_A]  # dim_value 0 == dynamic batch
    assert in_shapes["input_1"] == [0, IN_B]

    sess = ort.InferenceSession(path)
    names = [i.name for i in sess.get_inputs()]
    onnx_out = sess.run(None, {names[0]: xa, names[1]: xb})[0]
    np.testing.assert_allclose(keras_output, onnx_out, atol=atol(cfg), err_msg="two-input: keras vs ONNX mismatch")


@pytest.mark.parametrize("bias", [True, False])
def test_mha_onnx(cfg, bias, tmp_path):
    E, H, T = 16, 4, 8
    inputs = keras.Input(shape=(T, E))
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H, bias=bias)
    out, _ = mha([inputs, inputs, inputs])
    model = keras.Model(inputs, out)

    model(np.zeros((1, T, E), dtype=np.float32))
    apply_final_compression(model)

    x_np = np.random.randn(2, T, E).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        keras_output, onnx_out, atol=atol(cfg), err_msg=f"PQMultiheadAttention bias={bias}: keras vs ONNX mismatch"
    )


def test_mha_causal_attn_mask_onnx(cfg, tmp_path):
    E, H, T = 16, 4, 8
    inputs = keras.Input(shape=(T, E))
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H)
    # (T, S) additive causal mask: 0 on/below the diagonal, large-negative above it.
    attn_mask = np.triu(np.full((T, T), -1e4, dtype=np.float32), k=1)
    out, _ = mha([inputs, inputs, inputs], attn_mask=attn_mask)
    model = keras.Model(inputs, out)

    model(np.zeros((1, T, E), dtype=np.float32))
    apply_final_compression(model)

    x_np = np.random.randn(2, T, E).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        keras_output, onnx_out, atol=atol(cfg), err_msg="MHA causal attn_mask: keras vs ONNX mismatch"
    )


def test_mha_key_padding_mask_onnx(cfg, tmp_path):
    import onnx

    E, H, T = 16, 4, 8
    inputs = keras.Input(shape=(T, E))
    kpm = keras.Input(shape=(T,), dtype="bool")  # runtime bool padding mask, True == padding
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H)
    out, _ = mha([inputs, inputs, inputs], key_padding_mask=kpm)
    model = keras.Model([inputs, kpm], out)

    model([np.zeros((1, T, E), np.float32), np.zeros((1, T), bool)])
    apply_final_compression(model)

    x_np = np.random.randn(2, T, E).astype(np.float32)
    mask_np = np.zeros((2, T), dtype=bool)
    mask_np[:, -2:] = True  # last two key positions are padding
    keras_output = keras_out(model, [x_np, mask_np])

    path = str(tmp_path / "mha_kpm.onnx")
    proto = convert_to_onnx(model, input_shape=[(T, E), (T,)], output_path=path)
    # The padding mask must be a genuine bool graph input, not baked away.
    kpm_vi = next(i for i in proto.graph.input if i.name == "input_1")
    assert kpm_vi.type.tensor_type.elem_type == onnx.TensorProto.BOOL

    sess = ort.InferenceSession(path)
    names = [i.name for i in sess.get_inputs()]
    onnx_out = sess.run(None, {names[0]: x_np, names[1]: mask_np})[0]
    np.testing.assert_allclose(
        keras_output, onnx_out, atol=atol(cfg), err_msg="MHA key_padding_mask: keras vs ONNX mismatch"
    )


@pytest.mark.parametrize(
    "slicer",
    [
        lambda y: y[:, 2:6],
        lambda y: y[:, 0],
        lambda y: y[..., 1:8:2],
        lambda y: y[:, -1],
    ],
    ids=["range", "int_squeeze", "ellipsis_step", "neg_int"],
)
def test_tensor_slicing_onnx(cfg, slicer, tmp_path):
    """KerasTensor slicing (GetItem ops) must export as ONNX Slice (+ Squeeze)."""
    IN, OUT = 16, 8
    inputs = keras.Input(shape=(IN,))
    y = PQDense(cfg, units=OUT)(inputs)
    model = keras.Model(inputs, slicer(y))

    dummy = np.zeros((1, IN), dtype=np.float32)
    model(dummy)
    apply_final_compression(model)

    x_np = np.random.randn(4, IN).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=(IN,), tmp_path=tmp_path)
    np.testing.assert_allclose(keras_output, onnx_out, atol=atol(cfg), err_msg="tensor slicing: keras vs ONNX mismatch")


@pytest.mark.parametrize(
    "reshaper",
    [
        lambda y: keras.ops.expand_dims(y, 1),
        lambda y: keras.ops.expand_dims(y, -1),
        lambda y: keras.ops.squeeze(keras.ops.expand_dims(y, 2), 2),
        lambda y: keras.ops.squeeze(keras.ops.expand_dims(y, 1)),
    ],
    ids=["expand_dims", "expand_dims_neg", "roundtrip", "squeeze_all"],
)
def test_squeeze_unsqueeze_onnx(cfg, reshaper, tmp_path):
    """keras.ops.squeeze / expand_dims must export as ONNX Squeeze/Unsqueeze."""
    IN, OUT = 16, 8
    inputs = keras.Input(shape=(IN,))
    y = PQDense(cfg, units=OUT)(inputs)
    model = keras.Model(inputs, reshaper(y))

    dummy = np.zeros((1, IN), dtype=np.float32)
    model(dummy)
    apply_final_compression(model)

    x_np = np.random.randn(4, IN).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=(IN,), tmp_path=tmp_path)
    assert keras_output.shape == onnx_out.shape
    np.testing.assert_allclose(keras_output, onnx_out, atol=atol(cfg), err_msg="squeeze/expand_dims: keras vs ONNX mismatch")


def rank3_dense_model(cfg):
    """PQDense applied to a [batch, seq, dim] input (Dense maps over the last axis)."""
    SEQ, DIM, OUT = 5, 16, 8
    inputs = keras.Input(shape=(SEQ, DIM))
    out = PQDense(cfg, units=OUT)(inputs)
    model = keras.Model(inputs, out)
    model(np.zeros((1, SEQ, DIM), dtype=np.float32))
    apply_final_compression(model)
    return model, (SEQ, DIM)


def test_dense_rank3_input_onnx(cfg, tmp_path):
    """A standalone PQDense on a rank-3 input must export as MatMul + Add (Gemm is rank-2 only)."""
    model, input_shape = rank3_dense_model(cfg)

    x_np = np.random.randn(4, *input_shape).astype(np.float32)
    keras_output = keras_out(model, x_np)
    onnx_out = onnx_run(model, x_np, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(keras_output, onnx_out, atol=atol(cfg), err_msg="rank-3 dense: keras vs ONNX mismatch")


def test_dense_rank3_integer_weights_onnx(tmp_path):
    """store_integer_weights on a rank-3 dense exercises the quantized-weight Transpose branch."""
    cfg = pquant.cs_config()
    cfg.quantization_parameters.enable_quantization = True
    model, input_shape = rank3_dense_model(cfg)

    x_np = np.random.randn(4, *input_shape).astype(np.float32)
    keras_output = keras_out(model, x_np)

    path = str(tmp_path / "rank3_int.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path, store_integer_weights=True)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x_np})[0]
    np.testing.assert_allclose(keras_output, onnx_out, atol=QUANT_ATOL, err_msg="rank-3 dense int weights: mismatch")
