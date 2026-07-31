"""Tests for HGQ (high granularity quantization) per-tensor / per-channel / per-weight granularity.

The granularity controls the shape of the trainable bitwidth tensors (`i` and `f`) inside the
HGQ quantizer:
  - per_tensor:  a single shared value for the whole tensor   -> shape collapses to all-ones
  - per_channel: one value per output channel                 -> only the output-channel axis is kept
  - per_weight:  one value per element                        -> shape matches the tensor itself
                 (for data the batch axis is always shared, so it collapses to 1)

In Keras, kernels are stored output-channel-last and we assume channels_last data, so the
output-channel axis is -1 for both weights and activations.
"""

import keras
import numpy as np
import pytest
from keras import ops
from pquant.layers import (
    PQAvgPool1d,
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQMultiheadAttention,
)

from pquant import pdp_config
from pquant.core.keras.quantizer import Quantizer

BATCH_SIZE = 4
IN_FEATURES = 16
OUT_FEATURES = 32
KERNEL_SIZE = 3
STEPS = 8

# HGQ supports only per_tensor and per_weight; per_channel is rejected.
GRANULARITIES = ["per_tensor", "per_weight"]


@pytest.fixture(autouse=True)
def run_around_tests():
    keras.backend.clear_session()


def hgq_config(granularity):
    """A PQConfig with high granularity quantization enabled for the given granularity."""
    config = pdp_config()
    config.quantization_parameters.use_high_granularity_quantization = True
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.granularity = granularity
    return config


def shape_of(tensor):
    return tuple(int(d) for d in ops.shape(tensor))


def is_single_value(shape):
    """per_tensor: every axis collapsed to 1."""
    return int(np.prod(shape)) == 1


# ----------------------------------------------------------------------------- weights


def assert_weight_granularity(layer, granularity, kernel_shape):
    """The weight quantizer's i/f bitwidth tensors must match the expected shape for `granularity`."""
    _, i, f = layer.get_weight_quantization_bits()
    for name, t in (("i", i), ("f", f)):
        shape = shape_of(t)
        if granularity == "per_tensor":
            assert is_single_value(shape), f"weight {name}: expected single value, got {shape}"
        else:  # per_weight
            assert shape == kernel_shape, f"weight {name}: expected {kernel_shape}, got {shape}"


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_dense_weight_granularity(granularity):
    layer = PQDense(hgq_config(granularity), units=OUT_FEATURES)
    layer.build((BATCH_SIZE, IN_FEATURES))
    assert_weight_granularity(layer, granularity, kernel_shape=(IN_FEATURES, OUT_FEATURES))


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_conv1d_weight_granularity(granularity):
    layer = PQConv1d(hgq_config(granularity), filters=OUT_FEATURES, kernel_size=KERNEL_SIZE, data_format="channels_last")
    layer.build((BATCH_SIZE, STEPS, IN_FEATURES))
    assert_weight_granularity(layer, granularity, kernel_shape=(KERNEL_SIZE, IN_FEATURES, OUT_FEATURES))


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_conv2d_weight_granularity(granularity):
    layer = PQConv2d(hgq_config(granularity), filters=OUT_FEATURES, kernel_size=KERNEL_SIZE, data_format="channels_last")
    layer.build((BATCH_SIZE, STEPS, STEPS, IN_FEATURES))
    assert_weight_granularity(layer, granularity, kernel_shape=(KERNEL_SIZE, KERNEL_SIZE, IN_FEATURES, OUT_FEATURES))


def test_per_channel_rejected_for_hgq():
    # per_channel is not a valid HGQ granularity; constructing the HGQ weight quantizer must raise.
    with pytest.raises(ValueError, match="per_channel"):
        PQDense(hgq_config("per_channel"), units=OUT_FEATURES)


# ------------------------------------------------------------------------------- data
# A data quantizer built on a tensor shaped like a layer's output. The batch axis is always
# shared, so per_weight keeps every non-batch axis and collapses only the batch axis to 1.


def assert_data_granularity(output_shape, granularity):
    quantizer = Quantizer(k=0.0, i=0.0, f=7.0, is_heterogeneous=True, is_data=True, granularity=granularity)
    quantizer.build(output_shape)
    _, i, f = quantizer.get_quantization_bits()
    for name, t in (("i", i), ("f", f)):
        shape = shape_of(t)
        if granularity == "per_tensor":
            assert is_single_value(shape), f"data {name}: expected single value, got {shape}"
        else:  # per_weight: batch axis shared, rest per-element
            assert shape == (1,) + tuple(output_shape[1:]), f"data {name}: expected batch-collapsed, got {shape}"


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_dense_data_granularity(granularity):
    assert_data_granularity((BATCH_SIZE, OUT_FEATURES), granularity)


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_conv1d_data_granularity(granularity):
    # channels_last, "valid" padding output length
    assert_data_granularity((BATCH_SIZE, STEPS - KERNEL_SIZE + 1, OUT_FEATURES), granularity)


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_conv2d_data_granularity(granularity):
    out_len = STEPS - KERNEL_SIZE + 1
    assert_data_granularity((BATCH_SIZE, out_len, out_len, OUT_FEATURES), granularity)


# --------------------------------------------------- per-quantizer granularity override
# Each quantizer follows the config granularity unless its per-quantizer override is set.


def test_per_quantizer_granularity_override():
    # input per_tensor, everything else per_weight.
    config = hgq_config("per_weight")
    layer = PQDense(config, units=OUT_FEATURES, quantize_output=True, in_quant_granularity="per_tensor")
    layer.build((BATCH_SIZE, IN_FEATURES))

    # weight follows config (per_weight) -> full kernel shape
    _, wi, _ = layer.get_weight_quantization_bits()
    assert shape_of(wi) == (IN_FEATURES, OUT_FEATURES)
    # input overridden to per_tensor -> single value
    _, ii, _ = layer.get_input_quantization_bits()
    assert is_single_value(shape_of(ii))
    # output keeps config per_weight -> batch-collapsed data shape
    _, oi, _ = layer.get_output_quantization_bits()
    assert shape_of(oi) == (1, OUT_FEATURES)


def test_per_quantizer_granularity_defaults_to_config():
    config = hgq_config("per_tensor")
    layer = PQDense(config, units=OUT_FEATURES, weight_quant_granularity="per_weight")
    layer.build((BATCH_SIZE, IN_FEATURES))
    # weight overridden to per_weight; input uses config per_tensor
    _, wi, _ = layer.get_weight_quantization_bits()
    assert shape_of(wi) == (IN_FEATURES, OUT_FEATURES)
    _, ii, _ = layer.get_input_quantization_bits()
    assert is_single_value(shape_of(ii))


# --------------------------------------------------- granularity override on boundary layers
# A model can start/end with a BatchNorm / AvgPool whose input/output quantizer is effectively
# the model's I/O quantizer, so that must be overridable independently of the config granularity.


def test_batchnorm_input_granularity_override():
    config = hgq_config("per_weight")
    layer = PQBatchNormalization(config, in_quant_granularity="per_tensor")
    layer.build((BATCH_SIZE, IN_FEATURES))
    # input overridden to per_tensor -> single value
    _, ii, _ = layer.get_input_quantization_bits()
    assert is_single_value(shape_of(ii))
    # weight follows config per_weight -> one value per feature
    _, wi, _ = layer.get_weight_quantization_bits()
    assert shape_of(wi) == (IN_FEATURES,)


def test_avgpool_io_granularity_override():
    config = hgq_config("per_weight")
    layer = PQAvgPool1d(
        config,
        pool_size=2,
        quantize_output=True,
        in_quant_granularity="per_tensor",
        out_quant_granularity="per_tensor",
    )
    layer.build((BATCH_SIZE, STEPS, OUT_FEATURES))
    _, ii, _ = layer.get_input_quantization_bits()
    assert is_single_value(shape_of(ii))  # input overridden to per_tensor
    _, oi, _ = layer.get_output_quantization_bits()
    assert is_single_value(shape_of(oi))  # output overridden to per_tensor


def test_mha_io_param_granularity_override():
    config = hgq_config("per_weight")
    # in/out are the model-boundary granularities; param (weight+bias) is uniform across projections.
    layer = PQMultiheadAttention(
        config,
        embed_dim=IN_FEATURES,
        num_heads=4,
        quantize_output=True,
        in_quant_granularity="per_tensor",
        out_quant_granularity="per_tensor",
    )
    # Build the projections directly (a full forward would hit an unrelated dtype issue in
    # PQDense.ebops on 3-D input); granularity shapes are fixed at build time.
    for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
        proj.build((2, 5, IN_FEATURES))
    # Q/K/V projection inputs (boundary) overridden to per_tensor
    for proj in (layer.q_proj, layer.k_proj, layer.v_proj):
        _, ii, _ = proj.get_input_quantization_bits()
        assert is_single_value(shape_of(ii))
    # out_proj output (boundary) overridden to per_tensor
    _, oi, _ = layer.out_proj.get_output_quantization_bits()
    assert is_single_value(shape_of(oi))
    # weights follow param granularity = config per_weight
    _, wi, _ = layer.q_proj.get_weight_quantization_bits()
    assert shape_of(wi) == (IN_FEATURES, IN_FEATURES)
    # out_proj input is internal -> stays config per_weight (batch-collapsed, not single)
    _, oii, _ = layer.out_proj.get_input_quantization_bits()
    assert not is_single_value(shape_of(oii))


def test_mha_qkv_always_output_quantized():
    config = hgq_config("per_tensor")
    # MHA-level quantize_output=False (default): Q/K/V outputs are matmul operands and stay
    # output-quantized; only out_proj follows the MHA-level flag.
    layer = PQMultiheadAttention(config, embed_dim=IN_FEATURES, num_heads=4)
    assert layer.q_proj.quantize_output is True
    assert layer.k_proj.quantize_output is True
    assert layer.v_proj.quantize_output is True
    assert layer.out_proj.quantize_output is False
