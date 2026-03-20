import pytest
from keras import ops
from keras.random import shuffle

from pquant.pruning_methods.activation_pruning import ActivationPruning


@pytest.fixture
def config():
    return {
        "pruning_parameters": {
            "pruning_method": "activation_pruning",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "threshold": 0.3,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 2,
        }
    }


def test_linear(config):
    in_features = 8
    out_features = 16
    batch_size = 16

    weight = ops.linspace(-1, 1, num=out_features * in_features)
    weight = shuffle(weight)
    weight = ops.reshape(weight, (out_features, in_features))

    ap = ActivationPruning(config, "linear")
    ap.build(weight.shape)
    ap.post_pre_train_function()
    layer_output = ops.linspace(-0.5, 1, num=out_features)
    layer_output = shuffle(layer_output)
    layer_output = ops.tile(layer_output, (batch_size, 1))

    # Each item in batch has same output, find out where the 0 outputs are
    mask = ops.average(ops.cast(layer_output > 0, layer_output.dtype), axis=0)
    mask = ops.expand_dims(mask, -1)
    for _ in range(config["pruning_parameters"]["t_delta"]):
        ap.collect_output(layer_output, training=True)
    ap.post_epoch_function(0, 1)
    result = ap(weight)
    result_masked = mask * result
    # Multiplying by mask should not change the result at all
    assert ops.all(ops.equal(result, result_masked))


def test_conv(config):
    in_features = 8
    out_features = 16
    batch_size = 16
    kernel_size = 3

    weight = ops.linspace(-1, 1, num=out_features * in_features * kernel_size * kernel_size)
    weight = shuffle(weight)
    weight = ops.reshape(weight, (out_features, in_features, kernel_size, kernel_size))

    ap = ActivationPruning(config, "conv")
    ap.build(weight.shape)
    ap.post_pre_train_function()
    layer_output = ops.linspace(-0.5, 1, num=out_features)
    layer_output = shuffle(layer_output)
    layer_output = ops.tile(layer_output, (batch_size, 1))

    # Each item in batch has same output, find out where the 0 outputs are
    mask = ops.average(ops.cast(layer_output > 0, layer_output.dtype), axis=0)
    mask = ops.reshape(mask, list(mask.shape) + [1, 1, 1])
    # mask = ops.expand_dims(ops.expand_dims(ops.expand_dims(mask, -1), -1), -1)
    for _ in range(config["pruning_parameters"]["t_delta"]):
        ap.collect_output(layer_output, training=True)
    ap.post_epoch_function(0, 1)
    result = ap(weight)
    result_masked = mask * result
    # Multiplying by mask should not change the result at all
    assert ops.all(ops.equal(result, result_masked))
