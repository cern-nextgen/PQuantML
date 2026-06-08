import numpy as np
import pytest
from keras import ops

from pquant.core.keras.pruning_methods.wanda import Wanda


@pytest.fixture
def config():
    return {
        "pruning_parameters": {
            "pruning_method": "wanda",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "sparsity": 0.75,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 2,
            "N": None,
            "M": None,
        }
    }


def test_linear_unstructured(config):
    in_features = 8
    out_features = 16
    batch_size = 256
    sparsity = config["pruning_parameters"]["sparsity"]
    wanda = Wanda(config, "linear")
    wanda.build((out_features, in_features))
    wanda.post_pre_train_function()

    weight = ops.ones((out_features, in_features))
    layer_input = ops.expand_dims(ops.linspace(-1, 1, num=in_features, dtype=weight.dtype), 0)
    layer_input = ops.tile(layer_input, (batch_size, 1))
    # Weights are all 1, inputs largest magnitude at the beginning and the end of the tensor
    mask = np.zeros(in_features)
    ones_at_edge = int((1 - sparsity) * in_features / 2)
    mask[:ones_at_edge] = 1.0
    mask[-ones_at_edge:] = 1.0
    mask = ops.cast(mask, weight.dtype)
    mask = ops.tile(ops.expand_dims(mask, axis=0), (out_features, 1))
    for _ in range(config["pruning_parameters"]["t_delta"]):
        wanda.collect_input(layer_input, weight, training=True)
    result = wanda(weight)
    result_masked = mask * result
    # Multiplying by mask should not change the result at all
    assert ops.all(ops.equal(result, result_masked))
    assert ops.equal(ops.count_nonzero(result), ops.convert_to_tensor((1 - sparsity) * out_features * in_features))


def test_conv_unstructured(config):
    # TODO: figure out a proper test
    in_features = 8
    out_features = 16
    batch_size = 16
    kernel_size = 3
    sparsity = config["pruning_parameters"]["sparsity"]
    wanda = Wanda(config, "conv")
    wanda.build((out_features, in_features, kernel_size, kernel_size))
    wanda.post_pre_train_function()

    weight = ops.convert_to_tensor(np.random.rand(out_features, in_features, kernel_size, kernel_size), "float32")
    layer_input = ops.convert_to_tensor(np.random.rand(batch_size, in_features, kernel_size, kernel_size), "float32")

    for _ in range(config["pruning_parameters"]["t_delta"]):
        wanda.collect_input(layer_input, weight, training=True)
    result = wanda(weight)
    assert ops.equal(
        ops.count_nonzero(result),
        ops.convert_to_tensor((1 - sparsity) * out_features * in_features * kernel_size * kernel_size),
    )


def test_linear_nm(config):
    in_features = 8
    out_features = 4
    batch_size = 4
    N = 4
    M = 8
    config["pruning_parameters"]["N"] = N
    config["pruning_parameters"]["M"] = M
    sparsity = N / M
    config["pruning_parameters"]["sparsity"] = sparsity
    wanda = Wanda(config, "linear")
    wanda.build((out_features, in_features))
    wanda.post_pre_train_function()

    weight = ops.convert_to_tensor(np.random.rand(out_features, in_features), "float32")
    layer_input = ops.convert_to_tensor(np.random.rand(batch_size, in_features), "float32")

    for _ in range(config["pruning_parameters"]["t_delta"]):
        wanda.collect_input(layer_input, weight, training=True)
    result = wanda(weight)

    assert ops.equal(ops.count_nonzero(result), ops.convert_to_tensor((1 - sparsity) * out_features * in_features))
    result_reshaped = ops.reshape(result, (-1, M))
    non_zero_per_M = ops.count_nonzero(result_reshaped, axis=1)
    assert ops.all(non_zero_per_M == N)


def test_conv_nm(config):
    in_features = 8
    out_features = 4
    batch_size = 4
    kernel_size = 3
    N = 4
    M = 8
    config["pruning_parameters"]["N"] = N
    config["pruning_parameters"]["M"] = M
    sparsity = N / M
    config["pruning_parameters"]["sparsity"] = sparsity

    wanda = Wanda(config, "conv")
    wanda.build((out_features, in_features, kernel_size, kernel_size))
    wanda.post_pre_train_function()

    weight = ops.convert_to_tensor(np.random.rand(out_features, in_features, kernel_size, kernel_size), "float32")
    layer_input = ops.convert_to_tensor(np.random.rand(batch_size, in_features, kernel_size, kernel_size), "float32")

    for _ in range(config["pruning_parameters"]["t_delta"]):
        wanda.collect_input(layer_input, weight, training=True)
    result = wanda(weight)

    assert ops.equal(
        ops.count_nonzero(result),
        ops.convert_to_tensor((1 - sparsity) * out_features * in_features * kernel_size * kernel_size),
    )
    result_reshaped = ops.reshape(result, (-1, M))
    non_zero_per_M = ops.count_nonzero(result_reshaped, axis=1)
    assert ops.all(non_zero_per_M == N)
