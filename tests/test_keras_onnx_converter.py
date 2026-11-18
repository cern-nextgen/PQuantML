"""Tests for the Keras → ONNX converter: onnxruntime must reproduce the Keras outputs of every exported model."""

import warnings

import keras
import numpy as np
import onnx
import onnxruntime as ort
import pytest

import pquant
from pquant.core.keras.layers import (
    PQActivation,
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
    apply_final_compression,
)
from pquant.core.keras.onnx import OnnxExportWarning, convert_to_onnx, resolve_export_options

ATOL = 1e-4
MHA_ATOL = 5e-3  # exp and reciprocal differ in the last bit between runtimes, which can flip a quantizer after them


def make_cfg(quantize=True, **quantization_parameters):
    """cs_config that rounds half to even like QuantizeLinear, so that quantized exports are exact."""
    cfg = pquant.cs_config()
    cfg.quantization_parameters.enable_quantization = quantize
    cfg.quantization_parameters.round_mode = "RND_CONV"
    for name, value in quantization_parameters.items():
        setattr(cfg.quantization_parameters, name, value)
    return cfg


@pytest.fixture(params=[False, True], ids=["float", "quant"])
def cfg(request):
    return make_cfg(request.param)


@pytest.fixture
def cfg_quant():
    return make_cfg()


@pytest.fixture(autouse=True)
def seed():
    keras.utils.set_random_seed(0)  # the session-wide seed makes tests depend on their order


def rand(*shape):
    return np.random.randn(*shape).astype(np.float32)


def channels_first():
    return keras.backend.image_data_format() == "channels_first"


def compressed(model, *inputs):
    """Warm up ``model`` in training mode and apply the final compression."""
    model(list(inputs) if len(inputs) > 1 else inputs[0], training=True)
    apply_final_compression(model)
    return model


def run_onnx(path, *inputs):
    """Outputs of the graph as exported; onnxruntime's optimizers would re-quantize float weights."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path, options)
    return session.run(None, {i.name: x for i, x in zip(session.get_inputs(), inputs)})


def check_export(model, *inputs, tmp_path, atol=ATOL, compare=True, **export_options):
    """Export ``model`` and compare all of its outputs with onnxruntime.  Returns the ModelProto."""
    compressed(model, *inputs)
    path = str(tmp_path / "model.onnx")
    proto = convert_to_onnx(model, output_path=path, **export_options)
    if compare:
        expected = model(list(inputs) if len(inputs) > 1 else inputs[0], training=False)
        expected = [keras.ops.convert_to_numpy(t) for t in (expected if isinstance(expected, (list, tuple)) else [expected])]
        outputs = run_onnx(path, *inputs)
        assert [t.shape for t in expected] == [o.shape for o in outputs]
        for want, got in zip(expected, outputs):
            np.testing.assert_allclose(want, got, atol=atol)
    return proto


def single_layer(layer, channels, spatial):
    """(model, input) for one layer on an input laid out according to the image data format."""
    shape = (channels, *spatial) if channels_first() else (*spatial, channels)
    x = keras.Input(shape)
    return keras.Model(x, layer(x)), rand(2, *shape)


def dense_relu(cfg):
    x = keras.Input((16,))
    return keras.Model(x, keras.layers.ReLU()(PQDense(cfg, units=8)(x)))


# name: (layer, channels, spatial dims)
LAYERS = {
    "dense": (lambda c: PQDense(c, units=8), 16, ()),
    "dense-nobias": (lambda c: PQDense(c, units=8, use_bias=False), 16, ()),
    "dense-rank3": (lambda c: PQDense(c, units=8), 16, (5,)),
    "conv2d": (lambda c: PQConv2d(c, 8, kernel_size=3, padding="same"), 3, (8, 8)),
    "conv2d-nobias": (lambda c: PQConv2d(c, 8, kernel_size=3, padding="same", use_bias=False), 3, (8, 8)),
    "conv1d": (lambda c: PQConv1d(c, 8, kernel_size=3, padding="same"), 4, (16,)),
    "conv1d-nobias": (lambda c: PQConv1d(c, 8, kernel_size=3, padding="same", use_bias=False), 4, (16,)),
    "depthwise-conv2d": (lambda c: PQDepthwiseConv2d(c, kernel_size=3, padding="same"), 4, (8, 8)),
    "batchnorm": (lambda c: PQBatchNormalization(c, axis=1 if channels_first() else -1), 8, (4, 4)),
    "avgpool2d": (lambda c: PQAvgPool2d(c, 2), 4, (8, 8)),
    "avgpool2d-same": (lambda c: PQAvgPool2d(c, 3, strides=1, padding="same"), 4, (8, 8)),
    "avgpool1d": (lambda c: PQAvgPool1d(c, 2), 4, (16,)),
}


@pytest.mark.parametrize("name", LAYERS)
def test_single_layer_onnx(cfg, name, tmp_path):
    make_layer, channels, spatial = LAYERS[name]
    check_export(*single_layer(make_layer(cfg), channels, spatial), tmp_path=tmp_path)


PLAIN_LAYERS = {
    "global-avgpool2d": lambda: keras.layers.GlobalAveragePooling2D(),
    "global-avgpool2d-keepdims": lambda: keras.layers.GlobalAveragePooling2D(keepdims=True),
    "flatten": lambda: keras.layers.Flatten(),
}


@pytest.mark.parametrize("name", PLAIN_LAYERS)
def test_plain_layer_onnx(name, tmp_path):
    """Keras layers whose output layout depends on the image data format."""
    check_export(*single_layer(PLAIN_LAYERS[name](), 4, (8, 8)), tmp_path=tmp_path)


@pytest.mark.parametrize("activation, multiplier", [("relu", False), ("tanh", False), ("hard_tanh", False), ("relu", True)])
def test_pqactivation_onnx(cfg, activation, multiplier, tmp_path):
    cfg.quantization_parameters.use_relu_multiplier = multiplier
    x = keras.Input((16,))
    check_export(keras.Model(x, PQActivation(cfg, activation)(x)), rand(4, 16), tmp_path=tmp_path)


def test_plain_activation_onnx(tmp_path):
    """keras.layers.Activation is matched by its exact name; anything else raises instead of exporting another function."""
    for name in ("relu", "tanh", "sigmoid"):
        x = keras.Input((16,))
        check_export(keras.Model(x, keras.layers.Activation(name)(x)), rand(4, 16), tmp_path=tmp_path)

    layers = keras.layers
    for layer in (layers.Activation("hard_tanh"), layers.Activation("leaky_relu"), layers.ReLU(max_value=6)):
        x = keras.Input((16,))
        with pytest.raises(TypeError, match="Unsupported activation"):
            convert_to_onnx(keras.Model(x, layer(x)), output_path=str(tmp_path / "unsupported.onnx"))


def mha_atol(cfg):
    return MHA_ATOL if cfg.quantization_parameters.enable_quantization else ATOL


@pytest.mark.parametrize("variant", ["plain", "no-bias", "causal-mask", "with-weights"])
def test_mha_onnx(cfg, variant, tmp_path):
    x = keras.Input((8, 16))
    mha = PQMultiheadAttention(cfg, embed_dim=16, num_heads=4, bias=variant != "no-bias")
    causal_mask = np.triu(np.full((8, 8), -1e4, dtype=np.float32), k=1) if variant == "causal-mask" else None
    out, weights = mha([x, x, x], attn_mask=causal_mask)
    model = keras.Model(x, [out, weights] if variant == "with-weights" else out)
    # ReduceMax/ReduceMean take their axes as an input from opset 18 on
    opset = 18 if variant == "with-weights" else None
    check_export(model, rand(2, 8, 16), tmp_path=tmp_path, atol=mha_atol(cfg), opset=opset)


def test_mha_key_padding_mask_onnx(cfg, tmp_path):
    """A runtime key_padding_mask (True = padding) stays a bool graph input."""
    x, mask = keras.Input((8, 16)), keras.Input((8,), dtype="bool")
    out, _ = PQMultiheadAttention(cfg, embed_dim=16, num_heads=4)([x, x, x], key_padding_mask=mask)
    padding = np.zeros((2, 8), dtype=bool)
    padding[:, -2:] = True
    proto = check_export(keras.Model([x, mask], out), rand(2, 8, 16), padding, tmp_path=tmp_path, atol=mha_atol(cfg))
    assert proto.graph.input[1].type.tensor_type.elem_type == onnx.TensorProto.BOOL


def test_two_input_onnx(cfg_quant, tmp_path):
    """Inputs are named input_0, input_1, ...; an input_shape that contradicts the model is an error."""
    a, b = keras.Input((16,)), keras.Input((4,))
    model = keras.Model([a, b], keras.layers.Add()([PQDense(cfg_quant, units=8)(a), PQDense(cfg_quant, units=8)(b)]))
    proto = check_export(model, rand(3, 16), rand(3, 4), tmp_path=tmp_path, input_shape=[(16,), (4,)])
    assert [i.name for i in proto.graph.input] == ["input_0", "input_1"]
    shapes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in proto.graph.input]
    assert shapes == [[0, 16], [0, 4]]  # 0 means dynamic
    for bad_shape in [(16,), [(16,), (5,)], [(16,), (4, 1)]]:
        with pytest.raises(ValueError, match="does not match the model inputs"):
            convert_to_onnx(model, input_shape=bad_shape, output_path=str(tmp_path / "bad_shape.onnx"))


def test_residual_concat_onnx(cfg_quant, tmp_path):
    x = keras.Input((16,))
    h = PQDense(cfg_quant, units=16)(x)
    h = keras.layers.Add()([h, PQDense(cfg_quant, units=16)(h)])
    out = PQDense(cfg_quant, units=8)(keras.layers.Concatenate()([h, x]))
    check_export(keras.Model(x, out), rand(4, 16), tmp_path=tmp_path)


def test_multi_output_onnx(cfg_quant, tmp_path):
    """The graph outputs are the model outputs, in the model's order."""
    x = keras.Input((16,))
    wide, narrow = PQDense(cfg_quant, units=8)(x), PQDense(cfg_quant, units=4)(x)
    check_export(keras.Model(x, [narrow, wide]), rand(3, 16), tmp_path=tmp_path)


def test_shared_layer_onnx(cfg_quant, tmp_path):
    """A layer called more than once is exported once per call."""
    a, b, dense = keras.Input((16,)), keras.Input((16,)), PQDense(cfg_quant, units=8)
    model = keras.Model([a, b], keras.layers.Add()([dense(a), dense(b)]))
    proto = check_export(model, rand(4, 16), rand(4, 16), tmp_path=tmp_path)
    assert [node.op_type for node in proto.graph.node].count("Gemm") == 2


# Operations on the (N, 8) output of a dense layer
TENSOR_OPS = {
    "slice-range": lambda y: y[:, 2:6],
    "slice-int": lambda y: y[:, 0],
    "slice-ellipsis-step": lambda y: y[..., 1:8:2],
    "slice-negative-int": lambda y: y[:, -1],
    "expand-dims": lambda y: keras.ops.expand_dims(y, 1),
    "expand-dims-negative": lambda y: keras.ops.expand_dims(y, -1),
    "expand-dims-squeeze": lambda y: keras.ops.squeeze(keras.ops.expand_dims(y, 2), 2),
    "squeeze-all": lambda y: keras.ops.squeeze(keras.ops.expand_dims(y, 1)),
}


@pytest.mark.parametrize("name", TENSOR_OPS)
def test_tensor_op_onnx(cfg_quant, name, tmp_path):
    x = keras.Input((16,))
    check_export(keras.Model(x, TENSOR_OPS[name](PQDense(cfg_quant, units=8)(x))), rand(4, 16), tmp_path=tmp_path)


@pytest.mark.parametrize("input_shape", [(16,), (5, 16)], ids=["rank2", "rank3"])
def test_integer_weights_onnx(cfg_quant, input_shape, tmp_path):
    """Integer weight storage stays exact; rank 3 goes through the quantized-weight Transpose."""
    x = keras.Input(input_shape)
    model = keras.Model(x, PQDense(cfg_quant, units=8)(x))
    check_export(model, rand(4, *input_shape), tmp_path=tmp_path, target="qdq-int8", include_clip=True)


def test_unquantized_parameters_stay_float(tmp_path):
    """Parameters the model does not quantize must not be stored as integers."""
    int_weights = {"target": "qdq-int8", "include_clip": True}
    x = keras.Input((16,))
    model = keras.Model(x, PQDense(make_cfg(quantize=False), units=8)(x))
    check_export(model, rand(4, 16), tmp_path=tmp_path, **int_weights)

    x = keras.Input((8,))  # a batch normalization that leaves gamma and beta out
    model = keras.Model(x, PQBatchNormalization(make_cfg(), axis=-1, scale=False, center=False)(x))
    check_export(model, rand(8, 8), tmp_path=tmp_path, **int_weights)


@pytest.mark.parametrize("target", ["reference", "qonnx", "aie4ml", "qdq-int8"])
def test_target_export_onnx(cfg_quant, target, tmp_path):
    """Each target yields a valid graph with the structure its options promise, which onnxruntime reproduces."""
    options = resolve_export_options(target)
    # inside the unsigned 7-bit input range: exact without Clip too
    x = (np.random.rand(options.batch_size or 4, 16) * 0.9).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OnnxExportWarning)  # 7-bit activations without Clip, see test_export_checks
        # onnxruntime cannot execute the Quant custom op
        proto = check_export(dense_relu(cfg_quant), x, tmp_path=tmp_path, target=target, compare=not options.use_qonnx)
    onnx.checker.check_model(proto)

    op_types = {node.op_type for node in proto.graph.node}
    assert ("Quant" in op_types) == options.use_qonnx
    assert ("Clip" in op_types) == (options.include_clip and not options.use_qonnx)
    assert any(t.name.endswith("_weight_int") for t in proto.graph.initializer) == options.store_integer_weights
    assert proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value == (options.batch_size or 0)  # 0 means dynamic


def test_integer_ops_target_is_torch_only(cfg_quant, tmp_path):
    model = compressed(dense_relu(cfg_quant), rand(4, 16))
    with pytest.raises(ValueError, match="only supported by the torch"):
        convert_to_onnx(model, output_path=str(tmp_path / "model.onnx"), target="integer-ops")


def test_export_checks(tmp_path):
    """The export checks of the shared builder apply to Keras models too: errors raise, warnings follow ``strict``."""
    path = str(tmp_path / "model.onnx")
    # the default config: unsigned 7-bit activations, narrower than uint8, and a tie rule that is only logged
    model = compressed(dense_relu(make_cfg(round_mode="RND")), rand(4, 16))
    with pytest.warns(OnnxExportWarning, match="saturates") as caught:
        convert_to_onnx(model, output_path=path, target="qdq-int8")
    assert caught[0].filename == __file__  # reported at the convert_to_onnx call, not inside the converter
    with pytest.raises(ValueError, match="saturates"):
        convert_to_onnx(model, output_path=path, target="qdq-int8", strict=True)

    wide = compressed(dense_relu(make_cfg(default_data_fractional_bits=9.0)), rand(4, 16))  # 9-bit activations
    with pytest.raises(ValueError, match="container"):
        convert_to_onnx(wide, output_path=path)
    convert_to_onnx(wide, output_path=path, target="qonnx")  # Quant carries the bit width
