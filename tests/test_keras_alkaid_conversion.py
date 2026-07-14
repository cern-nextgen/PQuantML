"""Convert a pruned + quantized PQuant Keras model with Alkaid"""

import keras
import numpy as np
import pytest
from alkaid.codegen import RTLModel
from alkaid.converter import trace_model
from alkaid.trace import trace
from pquant.activations import PQActivation
from pquant.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
    PQSeparableConv2d,
    PQSoftmax,
    apply_final_compression,
)

from pquant import pdp_config
from pquant._alkaid_plugin import _alkaid_keras_plugin
from pquant.core.keras.quantizer import Quantizer

_alkaid_keras_plugin.register()

IN_FEATURES = 3
OUT_FEATURES = 4
KERNEL_SIZE = 3
H = W = 6
SEQ_LEN = H * W

PRUNE_FRACTION = 0.9
INPUT_KIF = (1, 4, 4)

IMG_SHAPE = (H, W, IN_FEATURES)
SEQ_SHAPE = (SEQ_LEN, IN_FEATURES)


@pytest.fixture(autouse=True)
def _channels_last():
    # Override conftest's default (channels_first); see module docstring.
    keras.backend.set_image_data_format("channels_last")


def _build_model(config):
    img_in = keras.Input(shape=IMG_SHAPE, name="img")
    a = PQConv2d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(img_in)
    a = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(a)
    a = keras.layers.Flatten()(a)

    seq_in = keras.Input(shape=SEQ_SHAPE, name="seq")
    b = PQConv1d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(seq_in)
    b = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(b)
    b = keras.layers.Flatten()(b)

    x = keras.layers.Add()([a, b])
    x = PQDense(config, units=OUT_FEATURES)(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    return keras.Model([img_in, seq_in], x)


def _rtl_predict(comb, path, data):
    """Write the RTL project, compile the simulation emulator, and run bit-accurate inference."""
    rtl_model = RTLModel(comb, str(path), "model", flavor="verilog", latency_cutoff=5, clock_period=5.0, print_latency=False)
    rtl_model.write()
    rtl_model.compile()
    data = [a.astype(np.float64) for a in data] if isinstance(data, list) else data.astype(np.float64)
    return rtl_model.predict(data)


def _random_prune(layer, fraction, rng):
    """Zero exactly ``fraction`` of the layer's weights via its pruning mask."""
    mask = layer.pruning_layer.mask
    numel = int(np.prod(mask.shape))
    n_zero = round(fraction * numel)
    flat = np.ones(numel, dtype="float32")
    flat[rng.permutation(numel)[:n_zero]] = 0.0
    mask.assign(flat.reshape(mask.shape))
    return n_zero / numel


def _build_pruned_compressed_model(config, rng):
    """Build the model, build it (one forward), prune 90%, and apply final compression."""
    model = _build_model(config)

    img = np.zeros((1, *IMG_SHAPE), dtype="float32")
    seq = np.zeros((1, *SEQ_SHAPE), dtype="float32")

    # Call once to build the quantizers and pruning masks.
    model([img, seq])

    pq_layers = [layer for layer in model.layers if isinstance(layer, (PQConv2d, PQConv1d, PQDense))]

    for layer in pq_layers:
        layer._kernel.assign(rng.standard_normal(layer._kernel.shape).astype("float32"))
    expected_sparsity = {layer.name: _random_prune(layer, PRUNE_FRACTION, rng) for layer in pq_layers}

    apply_final_compression(model)
    return model, pq_layers, expected_sparsity


def test_alkaid_conversion_pruned_quantized_model():
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model, pq_layers, _expected_sparsity = _build_pruned_compressed_model(config, rng)
    assert {type(layer).__name__ for layer in pq_layers} == {"PQConv2d", "PQConv1d", "PQDense"}

    inp, out = trace_model(model, inputs_kif=INPUT_KIF)

    assert out.shape == (OUT_FEATURES,)
    assert inp.shape == (int(np.prod(IMG_SHAPE)) + int(np.prod(SEQ_SHAPE)),)


def test_alkaid_rtl_matches_model(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model, _, _ = _build_pruned_compressed_model(config, rng)

    inp_fv, out_fv = trace_model(model, inputs_kif=INPUT_KIF)
    comb = trace(inp_fv, out_fv, optimize=True)

    n_samples = 16
    img = rng.integers(0, 16, size=(n_samples, *IMG_SHAPE)).astype("float32") / 16.0
    seq = rng.integers(0, 16, size=(n_samples, *SEQ_SHAPE)).astype("float32") / 16.0

    reference = np.asarray(model([img, seq]), dtype=np.float64)  # (n_samples, OUT_FEATURES)
    emulated = _rtl_predict(comb, tmp_path, [img, seq])
    assert (tmp_path / "src" / "model.v").exists()

    assert np.any(reference != 0)  # the comparison is non-trivial
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)


# --- Coverage of every PQ layer the keras Alkaid plugin handles ---------------

ALL_C = 4
ALL_H = ALL_W = 8
ALL_LIN = (ALL_H // 2) * (ALL_W // 2) * 2

ALL_IMG_SHAPE = (ALL_H, ALL_W, IN_FEATURES)
ALL_SEQ_SHAPE = (ALL_LIN, IN_FEATURES)

ALL_KERAS_LAYER_TYPES = {
    "PQConv2d",
    "PQBatchNormalization",
    "PQDepthwiseConv2d",
    "PQSeparableConv2d",
    "PQAvgPool2d",
    "PQConv1d",
    "PQAvgPool1d",
    "PQDense",
    "PQActivation",
}


def _build_all_layers_model(config):
    """Model exercising every PQ layer type the keras Alkaid plugin handles."""
    img_in = keras.Input(shape=ALL_IMG_SHAPE, name="img")
    a = PQConv2d(config, ALL_C, KERNEL_SIZE, padding="same")(img_in)
    a = PQBatchNormalization(config, axis=-1)(a)
    a = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(a)
    a = PQDepthwiseConv2d(config, KERNEL_SIZE, padding="same")(a)
    a = PQSeparableConv2d(config, ALL_C, KERNEL_SIZE, padding="same")(a)
    a = PQAvgPool2d(config, pool_size=2, strides=2, padding="valid")(a)
    a = keras.layers.Flatten()(a)

    seq_in = keras.Input(shape=ALL_SEQ_SHAPE, name="seq")
    b = PQConv1d(config, ALL_C, KERNEL_SIZE, padding="same")(seq_in)
    b = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(b)
    b = PQAvgPool1d(config, pool_size=2, strides=2, padding="valid")(b)
    b = keras.layers.Flatten()(b)

    x = keras.layers.Add()([a, b])
    x = PQDense(config, units=OUT_FEATURES)(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    return keras.Model([img_in, seq_in], x)


def _all_prunable_layers(model):
    """Every layer with a pruning mask, descending into PQSeparableConv2d's sub-convs."""
    found = []

    def visit(layer):
        if getattr(layer, "pruning_layer", None) is not None:
            found.append(layer)
        for name in ("depthwise_conv", "pointwise_conv"):
            sub = getattr(layer, name, None)
            if sub is not None:
                visit(sub)

    for layer in model.layers:
        visit(layer)
    return found


def test_alkaid_conversion_all_layer_types(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    model = _build_all_layers_model(config)
    rng = np.random.default_rng(0)

    # Build with random input so batchnorm running stats are sane.
    model(
        [
            rng.standard_normal((4, *ALL_IMG_SHAPE)).astype("float32"),
            rng.standard_normal((4, *ALL_SEQ_SHAPE)).astype("float32"),
        ]
    )

    assert {type(layer).__name__ for layer in model.layers} >= ALL_KERAS_LAYER_TYPES

    for layer in _all_prunable_layers(model):
        layer._kernel.assign(rng.standard_normal(layer._kernel.shape).astype("float32"))
        _random_prune(layer, PRUNE_FRACTION, rng)

    apply_final_compression(model)

    inp_fv, out_fv = trace_model(model, inputs_kif=INPUT_KIF)
    comb = trace(inp_fv, out_fv, optimize=True)
    assert out_fv.shape == (OUT_FEATURES,)

    n_samples = 16
    img = rng.integers(0, 16, size=(n_samples, *ALL_IMG_SHAPE)).astype("float32") / 16.0
    seq = rng.integers(0, 16, size=(n_samples, *ALL_SEQ_SHAPE)).astype("float32") / 16.0
    reference = np.asarray(model([img, seq]), dtype=np.float64)
    emulated = _rtl_predict(comb, tmp_path, [img, seq])
    assert (tmp_path / "src" / "model.v").exists()

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)


# --- Per-layer conversion: a model that is a single layer ---------------------


def _data_quantizer(config):
    """A data Quantizer built from the config's default data settings."""
    qp = config.quantization_parameters
    return Quantizer(
        k=qp.default_data_keep_negatives,
        i=qp.default_data_integer_bits,
        f=qp.default_data_fractional_bits,
        overflow=qp.overflow_mode_data,
        round_mode=qp.round_mode,
        is_heterogeneous=qp.use_high_granularity_quantization,
        is_data=True,
        hgq_gamma=qp.hgq_gamma,
        place="datalane",
        dynamic_data=qp.dynamic_data_quantization,
    )


def _single_layer_model(input_shape, layer, tail=None):
    """A keras model that is one PQ layer, optionally followed by a Quantizer."""
    inp = keras.Input(shape=input_shape)
    x = layer(inp)
    if tail is not None:
        x = tail(x)
    return keras.Model(inp, x)


# id -> lambda(config) -> (input shape without batch, single-layer model).
# Layers with quantize_output set it; batchnorm (which has none) gets a trailing Quantizer.
_SINGLE_LAYER_CASES = {
    "conv2d": lambda c: (
        (4, 4, 2),
        _single_layer_model((4, 4, 2), PQConv2d(c, 3, KERNEL_SIZE, padding="same", quantize_output=True)),
    ),
    "conv1d": lambda c: (
        (8, 2),
        _single_layer_model((8, 2), PQConv1d(c, 3, KERNEL_SIZE, padding="same", quantize_output=True)),
    ),
    "dense": lambda c: ((6,), _single_layer_model((6,), PQDense(c, units=OUT_FEATURES, quantize_output=True))),
    "depthwise2d": lambda c: (
        (4, 4, 3),
        _single_layer_model((4, 4, 3), PQDepthwiseConv2d(c, KERNEL_SIZE, padding="same", quantize_output=True)),
    ),
    "separable2d": lambda c: (
        (4, 4, 2),
        _single_layer_model((4, 4, 2), PQSeparableConv2d(c, 3, KERNEL_SIZE, padding="same", quantize_output=True)),
    ),
    "batchnorm": lambda c: ((6,), _single_layer_model((6,), PQBatchNormalization(c, axis=-1), _data_quantizer(c))),
    "avgpool2d": lambda c: (
        (4, 4, 3),
        _single_layer_model((4, 4, 3), PQAvgPool2d(c, pool_size=2, strides=2, quantize_output=True)),
    ),
    "avgpool1d": lambda c: (
        (8, 3),
        _single_layer_model((8, 3), PQAvgPool1d(c, pool_size=2, strides=2, quantize_output=True)),
    ),
    "activation": lambda c: (
        (6,),
        _single_layer_model((6,), PQActivation(c, activation="relu", quantize_input=True, quantize_output=True)),
    ),
    "quantizer": lambda c: ((6,), _single_layer_model((6,), _data_quantizer(c))),
    "softmax": lambda c: ((6,), _single_layer_model((6,), PQSoftmax(c, axis=-1))),
}


@pytest.mark.parametrize("case_id", list(_SINGLE_LAYER_CASES))
def test_alkaid_single_layer(case_id, tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True
    input_shape, model = _SINGLE_LAYER_CASES[case_id](config)
    rng = np.random.default_rng(0)

    model(rng.standard_normal((4, *input_shape)).astype("float32"))  # build
    apply_final_compression(model)

    inp_fv, out_fv = trace_model(model, inputs_kif=INPUT_KIF)
    comb = trace(inp_fv, out_fv, optimize=True)

    n_samples = 16
    x = rng.integers(0, 16, size=(n_samples, *input_shape)).astype("float32") / 16.0
    reference = np.asarray(model(x), dtype=np.float64).reshape(n_samples, -1)
    emulated = _rtl_predict(comb, tmp_path, x)

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)


# --- Multi-head attention ------------------------------------------------------

MHA_SEQ_LEN = 4
MHA_EMBED_DIM = 4
MHA_NUM_HEADS = 2


def _build_mha_model(config, rng):
    """Self-attention PQMultiheadAttention model with every data quantizer enabled."""
    inp = keras.Input(shape=(MHA_SEQ_LEN, MHA_EMBED_DIM))
    out, _ = PQMultiheadAttention(
        config,
        embed_dim=MHA_EMBED_DIM,
        num_heads=MHA_NUM_HEADS,
        quantize_output=True,
    )(inp)
    model = keras.Model(inp, out)
    model(rng.standard_normal((4, MHA_SEQ_LEN, MHA_EMBED_DIM)).astype("float32"))  # build
    apply_final_compression(model)
    return model


def test_alkaid_multihead_attention(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model = _build_mha_model(config, rng)

    inp_fv, out_fv = trace_model(model, inputs_kif=INPUT_KIF)
    comb = trace(inp_fv, out_fv, optimize=True)
    assert out_fv.shape == (MHA_SEQ_LEN * MHA_EMBED_DIM,)

    n_samples = 16
    x = rng.integers(0, 16, size=(n_samples, MHA_SEQ_LEN, MHA_EMBED_DIM)).astype("float32") / 16.0
    reference = np.asarray(model(x), dtype=np.float64).reshape(n_samples, -1)
    emulated = _rtl_predict(comb, tmp_path, x)
    assert (tmp_path / "src" / "model.v").exists()

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)
