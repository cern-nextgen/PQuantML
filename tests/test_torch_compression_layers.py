import keras
import numpy as np
import pytest
import torch
from keras import ops
from torch import nn
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Linear,
    ReLU,
    Tanh,
)

from pquant import post_training_prune
from pquant.activations import PQActivation
from pquant.core.hyperparameter_optimization import PQConfig
from pquant.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQWeightBiasBase,
    add_compression_layers,
    apply_final_compression,
    get_layer_keep_ratio,
    get_model_losses,
    post_pretrain_functions,
    pre_finetune_functions,
)

BATCH_SIZE = 4
OUT_FEATURES = 32
IN_FEATURES = 16
KERNEL_SIZE = 3
STEPS = 16


@pytest.fixture
def config_pdp():
    cfg = {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "epsilon": 1.0,
            "pruning_method": "pdp",
            "sparsity": 0.75,
            "temperature": 1e-5,
            "threshold_decay": 0.0,
            "structured_pruning": False,
        },
        "quantization_parameters": {
            "default_weight_integer_bits": 0.0,
            "default_weight_fractional_bits": 7.0,
            "default_data_integer_bits": 0.0,
            "default_data_fractional_bits": 7.0,
            "default_data_keep_negatives": 0.0,
            "default_weight_keep_negatives": 1.0,
            "quantize_input": True,
            "quantize_output": False,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "hgq_beta": 1e-5,
            "hgq_heterogeneous": True,
            "layer_specific": {},
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
            "round_mode": "RND",
            "overflow_mode_parameters": "SAT",
            "overflow_mode_data": "SAT",
            "granularity": "per_tensor",
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }
    return PQConfig.load_from_config(cfg)


@pytest.fixture
def config_ap():
    cfg = {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "pruning_method": "activation_pruning",
            "threshold": 0.3,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 1,
        },
        "quantization_parameters": {
            "default_weight_integer_bits": 0.0,
            "default_weight_fractional_bits": 7.0,
            "default_data_integer_bits": 0.0,
            "default_data_fractional_bits": 7.0,
            "default_data_keep_negatives": 0.0,
            "default_weight_keep_negatives": 1.0,
            "quantize_input": True,
            "quantize_output": False,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "hgq_beta": 1e-5,
            "hgq_heterogeneous": True,
            "layer_specific": {},
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
            "round_mode": "RND",
            "overflow_mode_parameters": "SAT",
            "overflow_mode_data": "SAT",
            "granularity": "per_tensor",
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }
    return PQConfig.load_from_config(cfg)


@pytest.fixture
def config_wanda():
    cfg = {
        "pruning_parameters": {
            "calculate_pruning_budget": False,
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "pruning_method": "wanda",
            "sparsity": 0.75,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 1,
            "N": None,
            "M": None,
        },
        "quantization_parameters": {
            "default_weight_integer_bits": 0.0,
            "default_weight_fractional_bits": 7.0,
            "default_data_integer_bits": 0.0,
            "default_data_fractional_bits": 7.0,
            "default_data_keep_negatives": 0.0,
            "default_weight_keep_negatives": 1.0,
            "quantize_input": True,
            "quantize_output": False,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "hgq_beta": 1e-5,
            "hgq_heterogeneous": True,
            "layer_specific": {},
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
            "round_mode": "RND",
            "overflow_mode_parameters": "SAT",
            "overflow_mode_data": "SAT",
            "granularity": "per_tensor",
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }
    return PQConfig.load_from_config(cfg)


@pytest.fixture
def config_cs():
    cfg = {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "final_temp": 200,
            "pruning_method": "cs",
            "threshold_decay": 0.0,
            "threshold_init": 0.1,
        },
        "quantization_parameters": {
            "default_weight_integer_bits": 0.0,
            "default_weight_fractional_bits": 7.0,
            "default_data_integer_bits": 0.0,
            "default_data_fractional_bits": 7.0,
            "default_data_keep_negatives": 0.0,
            "default_weight_keep_negatives": 1.0,
            "quantize_input": True,
            "quantize_output": False,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "hgq_beta": 1e-5,
            "hgq_heterogeneous": True,
            "layer_specific": {},
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
            "round_mode": "RND",
            "overflow_mode_parameters": "SAT",
            "overflow_mode_data": "SAT",
            "granularity": "per_tensor",
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }
    return PQConfig.load_from_config(cfg)


@pytest.fixture
def conv2d_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, 32, 32).astype(np.float32))


@pytest.fixture
def conv1d_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, 32).astype(np.float32))


@pytest.fixture
def dense_input():
    return torch.tensor(np.random.rand(BATCH_SIZE, IN_FEATURES).astype(np.float32))


class TestModel(nn.Module):
    __test__ = False

    def __init__(self, submodule, activation=None):
        super().__init__()
        self.submodule = submodule
        if activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            self.activation = activation

    def forward(self, x):
        x = self.submodule(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def test_dense_call(config_pdp, dense_input):
    layer_to_replace = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    out = layer_to_replace(dense_input)
    layer = PQDense(
        config_pdp,
        layer_to_replace.in_features,
        layer_to_replace.out_features,
        layer_to_replace.bias is not None,
        quantize_input=False,
    )
    layer._weight.data = layer_to_replace.weight.data
    out2 = layer(dense_input)
    assert ops.all(ops.equal(out, out2))


def test_conv2d_call(config_pdp, conv2d_input):
    layer_to_replace = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False, padding="same")
    out = layer_to_replace(conv2d_input)
    layer = PQConv2d(
        config_pdp,
        layer_to_replace.in_channels,
        layer_to_replace.out_channels,
        layer_to_replace.kernel_size,
        layer_to_replace.stride,
        layer_to_replace.padding,
        layer_to_replace.dilation,
        layer_to_replace.groups,
        layer_to_replace.bias is not None,
        layer_to_replace.padding_mode,
        layer_to_replace.weight.device,
        layer_to_replace.weight.dtype,
        quantize_input=False,
    )
    layer._weight.data = layer_to_replace.weight.data
    out2 = layer(conv2d_input)
    assert ops.all(ops.equal(out, out2))


def test_conv1d_call(config_pdp, conv1d_input):
    layer_to_replace = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, stride=2, bias=False)
    out = layer_to_replace(conv1d_input)
    layer = PQConv1d(
        config_pdp,
        layer_to_replace.in_channels,
        layer_to_replace.out_channels,
        layer_to_replace.kernel_size,
        layer_to_replace.stride,
        layer_to_replace.padding,
        layer_to_replace.dilation,
        layer_to_replace.groups,
        layer_to_replace.bias is not None,
        layer_to_replace.padding_mode,
        layer_to_replace.weight.device,
        layer_to_replace.weight.dtype,
        quantize_input=False,
    )
    layer._weight.data = layer_to_replace.weight.data
    out2 = layer(conv1d_input)
    assert ops.all(ops.equal(out, out2))


def test_dense_add_remove_layers(config_pdp, dense_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    orig_weight = layer.weight.data

    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)
    assert torch.all(orig_weight == model.submodule._weight.data)
    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    output1 = model(dense_input)
    model = apply_final_compression(model)
    output2 = model(dense_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.submodule.weight)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    orig_weight = layer.weight.data
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)
    assert torch.all(orig_weight == model.submodule._weight.data)
    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    output1 = model(conv2d_input)
    model = apply_final_compression(model)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.submodule.weight)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv1d_add_remove_layers(config_pdp, conv1d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.submodule.weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    output1 = model(conv1d_input)
    model = apply_final_compression(model)
    output2 = model(conv1d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.submodule.weight)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_dense_get_layer_keep_ratio(config_pdp, dense_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    model(dense_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv1d_get_layer_keep_ratio(config_pdp, conv1d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.submodule.weight)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.submodule.pruning_layer.mask.shape)
    model.submodule.pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_check_activation(config_pdp, dense_input):
    # ReLU
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, ReLU)

    config_pdp.quantization_parameters.enable_quantization = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, PQActivation)

    # Tanh
    config_pdp.quantization_parameters.enable_quantization = False
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "tanh")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, Tanh)

    config_pdp.quantization_parameters.enable_quantization = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "tanh")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert isinstance(model.activation, PQActivation)


def check_keras_layer_is_built(module, is_built):
    for m in module.modules():
        if hasattr(m, "built"):
            is_built.append(m.built)
    return is_built


class TestModelWithAvgPool(nn.Module):
    __test__ = False

    def __init__(self, submodule, activation=None):
        super().__init__()
        self.submodule = submodule
        if activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            self.activation = activation
        self.avg = AvgPool2d(2)

    def forward(self, x):
        x = self.submodule(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.avg(x)
        return x


def test_hgq_activation_built(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.quantize_output = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModelWithAvgPool(layer, "relu")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    is_built = check_keras_layer_is_built(model, [])
    torch.save(model.state_dict(), "test_model.pt")
    assert all(is_built)

    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModelWithAvgPool(layer, "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    is_built = check_keras_layer_is_built(model, [])
    assert all(is_built)


def test_post_training_wanda(config_wanda, conv2d_input):
    config_wanda.pruning_parameters.calculate_pruning_budget = False
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer, "relu")
    calibration_dataset = [conv2d_input, conv2d_input]
    model = post_training_prune(model, config_wanda, calibration_dataset)
    assert get_layer_keep_ratio(model) == 1 - config_wanda.pruning_parameters.sparsity


class TestModel2(nn.Module):
    __test__ = False

    def __init__(self, submodule, submodule2, activation=None, activation2=None):
        super().__init__()
        self.submodule = submodule
        self.submodule2 = submodule2
        if activation == "relu":
            self.activation = ReLU()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            self.activation = activation

        if activation2 == "relu":
            self.activation2 = ReLU()
        elif activation2 == "tanh":
            self.activation2 = Tanh()
        else:
            self.activation2 = activation2

    def forward(self, x):
        x = self.submodule(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.submodule2(x)
        if self.activation2 is not None:
            x = self.activation2(x)
        return x


def test_calculate_pruning_budget(config_wanda, dense_input):
    sparsity = 0.75
    config_wanda.pruning_parameters.calculate_pruning_budget = True
    config_wanda.pruning_parameters.sparsity = sparsity

    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu")

    # First layer will have 50% sparsity
    weight = np.ones(IN_FEATURES * OUT_FEATURES).astype(np.float32)
    weight[: IN_FEATURES * OUT_FEATURES // 2] = 0.001
    weight = ops.convert_to_tensor(weight)
    weight2 = ops.linspace(0.01, 0.99, OUT_FEATURES * OUT_FEATURES)

    model = add_compression_layers(model, config_wanda, dense_input.shape)
    model.submodule._weight.data = ops.reshape(weight, model.submodule.weight.shape)
    model.submodule2._weight.data = ops.reshape(weight2, model.submodule2.weight.shape)

    # Triggers calculation of pruning budget for PDP and Wanda
    post_pretrain_functions(model, config_wanda)
    total_weights = IN_FEATURES * OUT_FEATURES + OUT_FEATURES * OUT_FEATURES
    remaining_weights = 0
    for layer in model.modules():
        if hasattr(layer, "pruning_layer"):
            calculated_sparsity = layer.pruning_layer.sparsity.cpu()
            remaining_weights += np.float32(1 - calculated_sparsity) * layer.weight.numel()
    # First layer should have 50% sparsity, total sparsity should be around 75%
    assert model.submodule.pruning_layer.sparsity == 0.5
    np.testing.assert_allclose(remaining_weights / total_weights, 1 - sparsity, atol=1e-3, rtol=0)


def test_trigger_post_pretraining(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu", "tanh")

    model = add_compression_layers(model, config_pdp, dense_input.shape)

    def _to_bool(val):
        return val.numpy() if hasattr(val, "numpy") else bool(val)

    assert _to_bool(model.submodule.pruning_layer.is_pretraining)
    assert _to_bool(model.activation.is_pretraining)
    assert _to_bool(model.submodule2.pruning_layer.is_pretraining)
    assert _to_bool(model.activation2.is_pretraining)

    post_pretrain_functions(model, config_pdp)

    assert not _to_bool(model.submodule.pruning_layer.is_pretraining)
    assert not _to_bool(model.activation.is_pretraining)
    assert not _to_bool(model.submodule2.pruning_layer.is_pretraining)
    assert not _to_bool(model.activation2.is_pretraining)


def test_hgq_weight_shape(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    layer2 = Linear(OUT_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel2(layer, layer2, "relu", "tanh")

    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)

    assert model.submodule.weight_quantizer.quantizer.quantizer._i.shape == model.submodule.weight.shape
    assert model.activation.input_quantizer.quantizer.quantizer._i.shape == (1, OUT_FEATURES)


def test_qbn_build(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    layer2 = BatchNorm2d(OUT_FEATURES)
    model = TestModel2(layer, layer2, None, "tanh")

    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    assert model.submodule.weight_quantizer.quantizer.quantizer._i.shape == model.submodule.weight.shape


def test_set_activation_custom_bits_hgq(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    layer2 = AvgPool2d(2)
    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            assert m.i_weight == 0.0
            assert m.i_bias == 0.0
            assert torch.all(m.weight_quantizer.quantizer.quantizer.i == 0.0)
            assert torch.all(m.weight_quantizer.quantizer.quantizer.i == 0.0)

            assert m.f_weight == 7.0
            assert m.f_bias == 7.0
            assert torch.all(m.weight_quantizer.quantizer.quantizer.f == 7.0)
            assert torch.all(m.weight_quantizer.quantizer.quantizer.f == 7.0)
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            k_input, i_input, f_input = m.get_input_quantization_bits()

            assert torch.all(i_input == 0.0)
            assert torch.all(f_input == 7.0)
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            k_input, i_input, f_input = m.get_input_quantization_bits()

            assert torch.all(i_input == 0.0)
            assert torch.all(f_input == 8.0)

        elif isinstance(m, PQAvgPool2d):
            assert m.i_input == 0.0
            assert m.f_input == 7.0
            assert torch.all(m.input_quantizer.quantizer.quantizer.i == 0.0)
            assert torch.all(m.input_quantizer.quantizer.quantizer.f == 7.0)

    config_pdp.quantization_parameters.layer_specific = {
        'submodule': {
            'weight': {'integer_bits': 1, 'fractional_bits': 3},
            'bias': {'integer_bits': 2, 'fractional_bits': 4},
        },
        'submodule2': {"input": {'integer_bits': 1, 'fractional_bits': 3}},
        'activation': {"input": {'integer_bits': 1, 'fractional_bits': 4}},
        'activation2': {"input": {'integer_bits': 0, 'fractional_bits': 3}},
    }

    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            assert m.i_weight == 1.0
            assert m.i_bias == 2.0
            assert torch.all(m.weight_quantizer.quantizer.quantizer.i == 1.0)
            assert torch.all(m.bias_quantizer.quantizer.quantizer.i == 2.0)

            assert m.f_weight == 3.0
            assert m.f_bias == 4.0
            assert torch.all(m.weight_quantizer.quantizer.quantizer.f == 3.0)
            assert torch.all(m.bias_quantizer.quantizer.quantizer.f == 4.0)
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            k_input, i_input, f_input = m.get_input_quantization_bits()

            assert torch.all(i_input == 0.0)
            assert torch.all(f_input == 3.0)
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            k_input, i_input, f_input = m.get_input_quantization_bits()

            assert torch.all(i_input == 1.0)
            assert torch.all(f_input == 4.0)
        elif isinstance(m, PQAvgPool2d):
            assert m.i_input == 1.0
            assert m.f_input == 3.0
            assert torch.all(m.input_quantizer.quantizer.quantizer.i == 1.0)
            assert torch.all(m.input_quantizer.quantizer.quantizer.f == 3.0)


def test_disable_pruning_from_single_layer(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.pruning_parameters.enable_pruning = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    layer2 = Conv2d(OUT_FEATURES, OUT_FEATURES, KERNEL_SIZE)
    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    assert model.submodule.enable_pruning
    assert model.submodule2.enable_pruning

    config_pdp.pruning_parameters.disable_pruning_for_layers = ["submodule2"]
    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    assert model.submodule.enable_pruning
    assert not model.submodule2.enable_pruning


def test_set_activation_custom_bits_quantizer(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = False
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    layer2 = AvgPool2d(2)
    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            assert m.i_weight == 0.0
            assert m.f_bias == 7.0
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            assert m.i_input == 0.0
            assert m.f_input == 7.0
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            assert m.i_input == 0.0
            assert m.f_input == 8.0

    config_pdp.quantization_parameters.layer_specific = {
        'submodule': {
            'weight': {'integer_bits': 1.0, 'fractional_bits': 3.0},
            'bias': {'integer_bits': 1.0, 'fractional_bits': 3.0},
        },
        'submodule2': {"input": {'integer_bits': 1.0, 'fractional_bits': 3.0}},
        'activation': {"input": {'integer_bits': 0.0, 'fractional_bits': 4.0}},
        'activation2': {"input": {'integer_bits': 0.0, 'fractional_bits': 3.0}},
    }

    model = TestModel2(layer, layer2, "relu", "tanh")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            assert m.i_weight == 1.0
            assert m.f_bias == 3.0
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            assert m.i_input == 0.0
            assert m.f_input == 3.0
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            assert m.i_input == 0.0
            assert m.f_input == 4.0
        elif isinstance(m, PQAvgPool2d):
            assert m.i_input == 1.0
            assert m.f_input == 3.0


def test_ebops_dense(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.hgq_loss()

    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.hgq_loss()


def test_ebops_conv2d(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.hgq_loss()

    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.hgq_loss()


def test_ebops_conv1d(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.hgq_loss()

    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.hgq_loss()


def test_ebops_bn(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=False)
    layer2 = BatchNorm2d(OUT_FEATURES)
    model = TestModel2(layer, layer2, None, "relu")
    shape = [1] + list(conv2d_input.shape[1:])
    model = add_compression_layers(model, config_pdp, shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule2.hgq_loss()


def test_linear_direct(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_output=True)
    layer(dense_input)
    assert layer.get_input_quantization_bits() == (0, 0, 7)
    assert layer.get_weight_quantization_bits() == (1, 0, 7)
    assert layer.get_bias_quantization_bits() == (1, 0, 7)
    assert layer.get_output_quantization_bits() == (0, 0, 7)

    layer = PQDense(
        config_pdp,
        IN_FEATURES,
        OUT_FEATURES,
        quantize_output=True,
        in_quant_bits=(1, 2, 5),
        weight_quant_bits=(1, 0, 3),
        bias_quant_bits=(1, 0, 3),
        out_quant_bits=(1, 2, 5),
    )
    layer(dense_input)
    assert layer.get_input_quantization_bits() == (1, 2, 5)
    assert layer.get_weight_quantization_bits() == (1, 0, 3)
    assert layer.get_bias_quantization_bits() == (1, 0, 3)
    assert layer.get_output_quantization_bits() == (1, 2, 5)


def test_linear_direct_hgq(config_pdp, dense_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_output=True)
    layer(dense_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_weight_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_bias_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    layer = PQDense(
        config_pdp,
        IN_FEATURES,
        OUT_FEATURES,
        quantize_output=True,
        in_quant_bits=(1, 2, 5),
        weight_quant_bits=(1, 0, 3),
        bias_quant_bits=(1, 0, 3),
        out_quant_bits=(1, 2, 5),
    )
    layer(dense_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)
    k, i, f = layer.get_weight_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 3)
    k, i, f = layer.get_bias_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 3)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)


def test_conv2d_direct(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv2d(config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, quantize_output=True)
    layer(conv2d_input)
    assert layer.get_input_quantization_bits() == (0, 0, 7)
    assert layer.get_weight_quantization_bits() == (1, 0, 7)
    assert layer.get_bias_quantization_bits() == (1, 0, 7)
    assert layer.get_output_quantization_bits() == (0, 0, 7)
    layer = PQConv2d(
        config_pdp,
        IN_FEATURES,
        OUT_FEATURES,
        KERNEL_SIZE,
        quantize_output=True,
        in_quant_bits=(1, 2, 5),
        weight_quant_bits=(1, 0, 3),
        bias_quant_bits=(1, 0, 3),
        out_quant_bits=(1, 2, 5),
    )
    layer(conv2d_input)
    assert layer.get_input_quantization_bits() == (1, 2, 5)
    assert layer.get_weight_quantization_bits() == (1, 0, 3)
    assert layer.get_bias_quantization_bits() == (1, 0, 3)
    assert layer.get_output_quantization_bits() == (1, 2, 5)


def test_conv2d_direct_hgq(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv2d(config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, quantize_output=True)
    layer(conv2d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_weight_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_bias_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    layer = PQConv2d(
        config_pdp,
        IN_FEATURES,
        OUT_FEATURES,
        KERNEL_SIZE,
        quantize_output=True,
        in_quant_bits=(1, 2, 5),
        weight_quant_bits=(1, 0, 3),
        bias_quant_bits=(1, 0, 3),
        out_quant_bits=(1, 2, 5),
    )
    layer(conv2d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)
    k, i, f = layer.get_weight_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 3)
    k, i, f = layer.get_bias_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 3)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)


def test_conv1d_direct(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv1d(config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, quantize_output=True)
    layer(conv1d_input)
    assert layer.get_input_quantization_bits() == (0, 0, 7)
    assert layer.get_weight_quantization_bits() == (1, 0, 7)
    assert layer.get_bias_quantization_bits() == (1, 0, 7)
    assert layer.get_output_quantization_bits() == (0, 0, 7)
    layer = PQConv1d(
        config_pdp,
        IN_FEATURES,
        OUT_FEATURES,
        KERNEL_SIZE,
        quantize_output=True,
        in_quant_bits=(1, 2, 5),
        weight_quant_bits=(1, 0, 3),
        bias_quant_bits=(1, 0, 3),
        out_quant_bits=(1, 2, 5),
    )
    layer(conv1d_input)
    assert layer.get_input_quantization_bits() == (1, 2, 5)
    assert layer.get_weight_quantization_bits() == (1, 0, 3)
    assert layer.get_bias_quantization_bits() == (1, 0, 3)
    assert layer.get_output_quantization_bits() == (1, 2, 5)


def test_conv1d_direct_hgq(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv1d(config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, quantize_output=True)
    layer(conv1d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_weight_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_bias_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    layer = PQConv1d(
        config_pdp,
        IN_FEATURES,
        OUT_FEATURES,
        KERNEL_SIZE,
        quantize_output=True,
        in_quant_bits=(1, 2, 5),
        weight_quant_bits=(1, 0, 3),
        bias_quant_bits=(1, 0, 3),
        out_quant_bits=(1, 2, 5),
    )
    layer(conv1d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)
    k, i, f = layer.get_weight_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 3)
    k, i, f = layer.get_bias_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 0)
    assert torch.all(f == 3)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)


def test_avgpool_direct(config_pdp, conv1d_input, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQAvgPool1d(config_pdp, kernel_size=3)
    layer(conv1d_input)
    assert layer.get_input_quantization_bits() == (0, 0, 7)
    assert layer.get_output_quantization_bits() == (0, 0, 7)
    layer = PQAvgPool1d(config_pdp, KERNEL_SIZE, quantize_output=True, in_quant_bits=(1, 2, 5), out_quant_bits=(1, 2, 5))
    layer(conv1d_input)
    assert layer.get_input_quantization_bits() == (1, 2, 5)
    assert layer.get_output_quantization_bits() == (1, 2, 5)

    layer = PQAvgPool2d(config_pdp, kernel_size=3)
    layer(conv2d_input)
    assert layer.get_input_quantization_bits() == (0, 0, 7)
    assert layer.get_output_quantization_bits() == (0, 0, 7)

    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE, quantize_output=True, in_quant_bits=(1, 2, 5), out_quant_bits=(1, 2, 5))
    layer(conv2d_input)
    assert layer.get_input_quantization_bits() == (1, 2, 5)
    assert layer.get_output_quantization_bits() == (1, 2, 5)


def test_avgpool_direct_hgq(config_pdp, conv1d_input, conv2d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQAvgPool1d(config_pdp, kernel_size=3, quantize_output=True)
    layer(conv1d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    layer = PQAvgPool1d(config_pdp, KERNEL_SIZE, quantize_output=True, in_quant_bits=(1, 2, 5), out_quant_bits=(1, 2, 5))
    layer(conv1d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)

    # 2D
    layer = PQAvgPool2d(config_pdp, kernel_size=3, quantize_output=True)
    layer(conv2d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)

    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE, quantize_output=True, in_quant_bits=(1, 2, 5), out_quant_bits=(1, 2, 5))
    layer(conv2d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)

    k, i, f = layer.get_output_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)


def test_batchnorm2d_direct(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES)
    layer(conv2d_input)
    assert layer.get_input_quantization_bits() == (0, 0, 7)
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES, in_quant_bits=(1, 2, 5))
    layer(conv2d_input)
    assert layer.get_input_quantization_bits() == (1, 2, 5)


def test_batchnorm2d_direct_hgq(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES)
    layer(conv2d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 0)
    assert torch.all(i == 0)
    assert torch.all(f == 7)
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES, in_quant_bits=(1, 2, 5))
    layer(conv2d_input)
    k, i, f = layer.get_input_quantization_bits()
    assert torch.all(k == 1)
    assert torch.all(i == 2)
    assert torch.all(f == 5)


class DummyLayer(nn.Module):

    def __init__(self, is_pretraining=False):
        super().__init__()
        self.built = True
        self.layer_called = 0
        self.is_pretraining = is_pretraining

    def forward(self, x):
        self.layer_called += 1
        return x

    def extra_repr(self):
        return f"Layer called = {self.layer_called} times."


def test_dense_input_parameters_quantizers_called(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, bias=True)

    layer(dense_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(dense_input)

    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 0


def test_dense_output_parameters_quantizers_called(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, bias=True, quantize_input=False, quantize_output=True)

    layer(dense_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(dense_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def test_dense_quantizers_not_called_when_global_disabled(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = False
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, bias=True, quantize_input=True, quantize_output=True)

    layer(dense_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(dense_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0


def test_dense_quantizers_not_called_when_fitcompress_pretraining(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.fitcompress_parameters.enable_fitcompress = True
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, bias=True, quantize_input=True, quantize_output=True)

    layer(dense_input)
    layer.is_pretraining = True
    layer.pruning_layer = DummyLayer()
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()

    assert layer.pruning_layer.layer_called == 0
    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0

    layer.is_pretraining = False
    layer(dense_input)
    assert layer.pruning_layer.layer_called == 1
    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def test_dense_pruning_layer_not_called_when_global_disabled(config_pdp, dense_input):
    config_pdp.pruning_parameters.enable_pruning = False
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, bias=True, quantize_input=False, quantize_output=False)
    layer(dense_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer.pruning_layer = DummyLayer()

    layer(dense_input)

    assert layer.pruning_layer.layer_called == 0


def test_dense_pruning_layer_called_when_global_enabled(config_pdp, dense_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, bias=True, quantize_input=False, quantize_output=False)
    layer(dense_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer.pruning_layer = DummyLayer()

    layer(dense_input)

    assert layer.pruning_layer.layer_called == 1


# Conv1d


def test_conv1d_input_parameters_quantizers_called(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv1d(config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)

    layer(conv1d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv1d_input)

    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 0


def test_conv1d_output_parameters_quantizers_called(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv1d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=False, quantize_output=True
    )

    layer(conv1d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv1d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def test_conv1d_quantizers_not_called_when_global_disabled(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = False
    layer = PQConv1d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )

    layer(conv1d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv1d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0


def test_conv1d_quantizers_not_called_when_fitcompress_pretraining(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.fitcompress_parameters.enable_fitcompress = True
    layer = PQConv1d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )

    layer(conv1d_input)
    layer.is_pretraining = True
    layer.pruning_layer = DummyLayer()
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()

    assert layer.pruning_layer.layer_called == 0
    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0

    layer.is_pretraining = False
    layer(conv1d_input)
    assert layer.pruning_layer.layer_called == 1
    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def test_conv1d_pruning_layer_not_called_when_global_disabled(config_pdp, conv1d_input):
    config_pdp.pruning_parameters.enable_pruning = False
    layer = PQConv1d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )
    layer(conv1d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer.pruning_layer = DummyLayer()

    layer(conv1d_input)

    assert layer.pruning_layer.layer_called == 0


def test_conv1d_pruning_layer_called_when_global_enabled(config_pdp, conv1d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = PQConv1d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )
    layer(conv1d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer.pruning_layer = DummyLayer()

    layer(conv1d_input)

    assert layer.pruning_layer.layer_called == 1


# Conv2d


def test_conv2d_input_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv2d(config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 0


def test_conv2d_output_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv2d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=False, quantize_output=True
    )

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def test_conv2d_quantizers_not_called_when_global_disabled(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = False
    layer = PQConv2d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0


def test_conv2d_quantizers_not_called_when_fitcompress_pretraining(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.fitcompress_parameters.enable_fitcompress = True
    layer = PQConv2d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )

    layer(conv2d_input)
    layer.is_pretraining = True
    layer.pruning_layer = DummyLayer()
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()

    assert layer.pruning_layer.layer_called == 0
    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0

    layer.is_pretraining = False
    layer(conv2d_input)
    assert layer.pruning_layer.layer_called == 1
    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def test_conv2d_pruning_layer_not_called_when_global_disabled(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = False
    layer = PQConv2d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )
    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer.pruning_layer = DummyLayer()

    layer(conv2d_input)

    assert layer.pruning_layer.layer_called == 0


def test_conv2d_pruning_layer_called_when_global_enabled(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    layer = PQConv2d(
        config_pdp, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True, quantize_input=True, quantize_output=True
    )
    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer.pruning_layer = DummyLayer()

    layer(conv2d_input)

    assert layer.pruning_layer.layer_called == 1


# AvgPool


def test_avgpool2d_input_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 0


def test_avgpool2d_output_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE, quantize_input=False, quantize_output=True)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 1


def test_avgpool2d_quantizers_not_called_when_global_disabled(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = False
    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE, quantize_input=True, quantize_output=True)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0


def test_avgpool2d_quantizers_not_called_when_fitcompress_pretraining(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.fitcompress_parameters.enable_fitcompress = True
    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE, quantize_input=True, quantize_output=True)

    layer(conv2d_input)
    layer.is_pretraining = True
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()

    assert layer.input_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0

    layer.is_pretraining = False
    layer(conv2d_input)
    assert layer.input_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


# BatchNorm


def test_batchnorm2d_input_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1


def test_batchnorm2d_input_quantizer_disabled_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES, quantize_input=False)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1


def test_batchnorm2d_quantizers_not_called_when_global_disabled(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = False
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0


def test_batchnorm2d_quantizers_not_called_when_fitcompress_pretraining(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.fitcompress_parameters.enable_fitcompress = True
    layer = PQBatchNorm2d(config_pdp, IN_FEATURES)

    layer(conv2d_input)
    layer.is_pretraining = True
    layer.input_quantizer = DummyLayer()
    layer.weight_quantizer = DummyLayer()
    layer.bias_quantizer = DummyLayer()

    assert layer.input_quantizer.layer_called == 0
    assert layer.weight_quantizer.layer_called == 0
    assert layer.bias_quantizer.layer_called == 0

    layer.is_pretraining = False
    layer(conv2d_input)
    assert layer.input_quantizer.layer_called == 1
    assert layer.weight_quantizer.layer_called == 1
    assert layer.bias_quantizer.layer_called == 1


# Activations


def test_activation_input_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQActivation(config_pdp, "relu")

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 0


def test_activation_output_parameters_quantizers_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQActivation(config_pdp, "relu", quantize_input=False, quantize_output=True)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 1


def test_activation_quantizers_not_called_when_global_disabled(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = False

    layer = PQActivation(config_pdp, "relu", quantize_input=True, quantize_output=True)

    layer(conv2d_input)  # Builds quantizers
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()
    layer(conv2d_input)

    assert layer.input_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0


def test_activation_quantizers_not_called_when_fitcompress_pretraining(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.fitcompress_parameters.enable_fitcompress = True
    layer = PQActivation(config_pdp, "relu", quantize_input=True, quantize_output=True)

    layer(conv2d_input)
    layer.is_pretraining = True
    layer.input_quantizer = DummyLayer()
    layer.output_quantizer = DummyLayer()

    assert layer.input_quantizer.layer_called == 0
    assert layer.output_quantizer.layer_called == 0

    layer.is_pretraining = False
    layer(conv2d_input)
    assert layer.input_quantizer.layer_called == 1
    assert layer.output_quantizer.layer_called == 1


def dummy_ebops():
    return 0.0


def dummy_hgq_loss():
    return 1.0


class ModelWithAllLayers(nn.Module):

    def __init__(self, use_bias=True):
        super().__init__()
        self.conv = Conv2d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=use_bias)
        self.bn = BatchNorm2d(OUT_FEATURES)
        self.relu = ReLU()
        self.avgpool2d = nn.AvgPool2d(2)
        self.conv1d = Conv1d(OUT_FEATURES, 4, KERNEL_SIZE, bias=use_bias)
        self.avgpool1d = nn.AvgPool1d(2)
        self.tanh = Tanh()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(444, 2, bias=use_bias)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.avgpool2d(x)
        x = self.tanh(x)
        x = torch.reshape(x, list(x.shape[:-2]) + [x.shape[-2] * x.shape[-1]])
        x = self.conv1d(x)
        x = self.avgpool1d(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


def test_hgq_loss_calc_no_qoutput(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.hgq_beta = 0.0

    # Bias in weight layers, don't quantize output
    model = ModelWithAllLayers()
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    expected_loss = 0.0
    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            if hasattr(m, "output_quantizer"):
                m.output_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            expected_loss += 3.0
        elif isinstance(m, (PQAvgPool1d, PQAvgPool2d, PQActivation)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss
            if hasattr(m, "output_quantizer"):
                m.output_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            expected_loss += 1.0
        elif isinstance(m, (PQBatchNorm2d)):
            m.ebops = dummy_ebops
            m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 3.0

    losses = get_model_losses(model, torch.tensor(0.0))
    assert losses == expected_loss


def test_hgq_loss_calc_no_bias_no_qoutput(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.hgq_beta = 0.0

    # No bias in weight layers, don't quantize output
    model = ModelWithAllLayers(use_bias=False)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)

    expected_loss = 0.0
    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            if hasattr(m, "bias_quantizer"):
                m.bias_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            if hasattr(m, "output_quantizer"):
                m.output_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            expected_loss += 2.0
        elif isinstance(m, (PQAvgPool1d, PQAvgPool2d, PQActivation)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss
            if hasattr(m, "output_quantizer"):
                m.output_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            expected_loss += 1.0
        elif isinstance(m, (PQBatchNorm2d)):
            m.ebops = dummy_ebops
            m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 3.0

    losses = get_model_losses(model, torch.tensor(0.0))
    assert losses == expected_loss


def test_hgq_loss_calc_qoutput(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.hgq_beta = 0.0

    # Bias in weight layers, quantize output
    config_pdp.quantization_parameters.quantize_output = True
    model = ModelWithAllLayers()
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    expected_loss = 0.0
    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            m.ebops = dummy_ebops
            m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            m.output_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 4.0
        elif isinstance(m, (PQAvgPool1d, PQAvgPool2d, PQActivation)):
            m.ebops = dummy_ebops
            m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.output_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 2.0
        elif isinstance(m, (PQBatchNorm2d)):
            m.ebops = dummy_ebops
            m.input_quantizer.hgq_loss = dummy_hgq_loss
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 3.0

    losses = get_model_losses(model, torch.tensor(0.0))
    assert losses == expected_loss


def test_hgq_loss_calc_no_qinput(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.hgq_beta = 0.0

    config_pdp.quantization_parameters.quantize_output = True
    config_pdp.quantization_parameters.quantize_input = False
    # Bias in weight layers, don't quantize input
    model = ModelWithAllLayers()
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    expected_loss = 0.0
    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            if hasattr(m, "output_quantizer"):
                m.output_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 3.0
        elif isinstance(m, (PQAvgPool1d, PQAvgPool2d, PQActivation)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            if hasattr(m, "output_quantizer"):
                m.output_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 1.0
        elif isinstance(m, (PQBatchNorm2d)):
            m.ebops = dummy_ebops
            if hasattr(m, "input_quantizer"):
                m.input_quantizer.hgq_loss = dummy_hgq_loss  # Won't be called
            m.weight_quantizer.hgq_loss = dummy_hgq_loss
            m.bias_quantizer.hgq_loss = dummy_hgq_loss
            expected_loss += 2.0

    losses = get_model_losses(model, torch.tensor(0.0))
    assert losses == expected_loss


# After final compression done


def test_conv1d_parameter_quantizers_not_called_when_final_compression_done(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.quantize_output = True
    layer = Conv1d(IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, bias=True)
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    model = apply_final_compression(model)
    model.submodule.input_quantizer = DummyLayer()
    model.submodule.weight_quantizer = DummyLayer()
    model.submodule.bias_quantizer = DummyLayer()
    model.submodule.output_quantizer = DummyLayer()
    model(conv1d_input)

    assert model.submodule.input_quantizer.layer_called == 1
    assert model.submodule.weight_quantizer.layer_called == 0
    assert model.submodule.bias_quantizer.layer_called == 0
    assert model.submodule.output_quantizer.layer_called == 1


def test_batchnorm2d_parameter_quantizers_not_called_when_final_compression_done(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.quantize_output = True
    layer = BatchNorm2d(IN_FEATURES)
    model = TestModel(layer)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model = apply_final_compression(model)
    model.submodule.input_quantizer = DummyLayer()
    model.submodule.weight_quantizer = DummyLayer()
    model.submodule.bias_quantizer = DummyLayer()
    model(conv2d_input)

    assert model.submodule.input_quantizer.layer_called == 1
    assert model.submodule.weight_quantizer.layer_called == 0
    assert model.submodule.bias_quantizer.layer_called == 0


def test_ebops_dense_nonhgq(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    # config_pdp.quantization_parameters.use_high_granularity_quantization = True
    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.ebops(include_mask=True)

    layer = Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    model = TestModel(layer, "relu")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.submodule.ebops(include_mask=True)
