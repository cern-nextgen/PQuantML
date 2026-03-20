from unittest.mock import patch

import keras
import numpy as np
import pytest
from keras import ops
from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    ReLU,
    SeparableConv2D,
)

from pquant import (
    ap_config,
    autosparse_config,
    cs_config,
    dst_config,
    mdmm_config,
    pdp_config,
    wanda_config,
)
from pquant.activations import PQActivation
from pquant.core.hyperparameter_optimization import PQConfig
from pquant.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQSeparableConv2d,
    add_compression_layers,
    apply_final_compression,
    get_layer_keep_ratio,
    post_pretrain_functions,
    pre_finetune_functions,
)

BATCH_SIZE = 4
OUT_FEATURES = 32
IN_FEATURES = 16
KERNEL_SIZE = 3
STEPS = 16


@pytest.fixture(autouse=True)
def run_around_tests():
    keras.backend.clear_session()


@pytest.fixture
def config_pdp():
    return PQConfig.load_from_config(
        {
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
    )


@pytest.fixture
def config_ap():
    return PQConfig.load_from_config(
        {
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
            },
            "training_parameters": {"pruning_first": False},
            "fitcompress_parameters": {"enable_fitcompress": False},
        }
    )


@pytest.fixture
def config_wanda():
    return PQConfig.load_from_config(
        {
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
            },
            "training_parameters": {"pruning_first": False},
            "fitcompress_parameters": {"enable_fitcompress": False},
        }
    )


@pytest.fixture
def config_cs():
    return PQConfig.load_from_config(
        {
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
            },
            "training_parameters": {"pruning_first": False},
            "fitcompress_parameters": {"enable_fitcompress": False},
        }
    )


np.random.seed(42)


@pytest.fixture(scope="function", autouse=True)
def conv2d_input():
    if keras.backend.image_data_format() == "channels_first":
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, 32, 32))
    else:
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, 32, 32, IN_FEATURES))
    return inp


@pytest.fixture(scope="function", autouse=True)
def conv1d_input():
    if keras.backend.image_data_format() == "channels_first":
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, IN_FEATURES, 32))
    else:
        inp = ops.convert_to_tensor(np.random.rand(BATCH_SIZE, 32, IN_FEATURES))
    return inp


@pytest.fixture(scope="function", autouse=True)
def dense_input():
    return ops.convert_to_tensor(np.random.rand(BATCH_SIZE, IN_FEATURES))


def test_dense_call(config_pdp, dense_input):
    layer_to_replace = Dense(OUT_FEATURES, use_bias=False)
    layer_to_replace.build((BATCH_SIZE, IN_FEATURES))
    out = layer_to_replace(dense_input)
    layer = PQDense(
        config_pdp,
        units=OUT_FEATURES,
        use_bias=False,
        quantize_input=False,
        quantize_output=False,
    )
    layer.build(dense_input.shape)
    layer._kernel.assign(layer_to_replace.kernel)
    out2 = layer(dense_input)
    assert ops.all(ops.equal(out, out2))


def test_conv2d_call(config_pdp, conv2d_input):
    layer_to_replace = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")
    layer_to_replace.build(conv2d_input.shape)
    out = layer_to_replace(conv2d_input)
    layer = PQConv2d(config_pdp, OUT_FEATURES, KERNEL_SIZE, padding="same", quantize_output=True, use_bias=True)
    layer.build(conv2d_input.shape)
    layer._kernel.assign(layer_to_replace.kernel)
    out2 = layer(conv2d_input)
    assert ops.all(ops.equal(out, out2))


def test_separable_conv2d_call(config_pdp, conv2d_input):
    layer_to_replace = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")
    layer_to_replace.build(conv2d_input.shape)
    out = layer_to_replace(conv2d_input)
    layer = PQSeparableConv2d(
        config_pdp,
        layer_to_replace.filters,
        layer_to_replace.kernel_size,
        layer_to_replace.strides,
        layer_to_replace.padding,
        layer_to_replace.data_format,
        layer_to_replace.dilation_rate,
        layer_to_replace.depth_multiplier,
        layer_to_replace.use_bias,
        layer_to_replace.depthwise_initializer,
        layer_to_replace.pointwise_initializer,
        layer_to_replace.bias_initializer,
        layer_to_replace.depthwise_regularizer,
        layer_to_replace.pointwise_regularizer,
        layer_to_replace.bias_regularizer,
        layer_to_replace.depthwise_constraint,
        layer_to_replace.pointwise_constraint,
        layer_to_replace.bias_constraint,
    )
    layer.depthwise_conv.build(conv2d_input.shape)
    layer.pointwise_conv.build(conv2d_input.shape)
    layer.depthwise_conv._kernel.assign(layer_to_replace.depthwise_kernel)
    layer.pointwise_conv._kernel.assign(layer_to_replace.pointwise_kernel)

    out2 = layer(conv2d_input)
    assert ops.all(ops.equal(out, out2))


def test_separable_conv2d_add_remove_layers(config_pdp, conv2d_input):
    # Case pruning not quantizing
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)

    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    # Set Depthwise mask to 50% 0s
    mask_50pct_dw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].depthwise_conv.kernel)) < 0.5, "float32")
    mask_50pct_dw = ops.reshape(keras.random.shuffle(mask_50pct_dw), model.layers[1].depthwise_conv.pruning_layer.mask.shape)
    model.layers[1].depthwise_conv.pruning_layer.mask = mask_50pct_dw
    # Set Pointwise mask to 50% 0s
    mask_50pct_pw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].pointwise_conv.kernel)) < 0.5, "float32")
    mask_50pct_pw = ops.reshape(keras.random.shuffle(mask_50pct_pw), model.layers[1].pointwise_conv.pruning_layer.mask.shape)
    model.layers[1].pointwise_conv.pruning_layer.mask = mask_50pct_pw

    output1 = model(conv2d_input)

    model = apply_final_compression(model)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))

    expected_nonzero_count_depthwise = ops.count_nonzero(mask_50pct_dw)
    nonzero_count_depthwise = ops.count_nonzero(model.layers[1].depthwise_conv.kernel)
    assert ops.equal(expected_nonzero_count_depthwise, nonzero_count_depthwise)

    expected_nonzero_count_pointwise = ops.count_nonzero(mask_50pct_pw)
    nonzero_count_pointwise = ops.count_nonzero(model.layers[1].pointwise_conv.kernel)
    assert ops.equal(expected_nonzero_count_pointwise, nonzero_count_pointwise)


def test_separable_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    # Set Depthwise mask to 50% 0s
    mask_50pct_dw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].depthwise_conv.kernel)) < 0.5, "float32")
    mask_50pct_dw = ops.reshape(keras.random.shuffle(mask_50pct_dw), model.layers[1].depthwise_conv.pruning_layer.mask.shape)
    model.layers[1].depthwise_conv.pruning_layer.mask = mask_50pct_dw
    # Set Pointwise mask to 50% 0s
    mask_50pct_pw = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].pointwise_conv.kernel)) < 0.5, "float32")
    mask_50pct_pw = ops.reshape(keras.random.shuffle(mask_50pct_pw), model.layers[1].pointwise_conv.pruning_layer.mask.shape)
    model.layers[1].pointwise_conv.pruning_layer.mask = mask_50pct_pw

    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)

    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct_dw) / ops.size(mask_50pct_dw), ratio1)


def test_separable_conv2d_trigger_post_pretraining(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = SeparableConv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    act1 = Activation("tanh")(out)
    flat = keras.layers.Flatten()(act1)
    out2 = Dense(OUT_FEATURES, use_bias=False)(flat)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    assert model.layers[1].depthwise_conv.pruning_layer.is_pretraining == True  # noqa: E712
    assert model.layers[1].pointwise_conv.pruning_layer.is_pretraining == True  # noqa: E712
    assert model.layers[2].is_pretraining == True  # noqa: E712
    assert model.layers[4].pruning_layer.is_pretraining == True  # noqa: E712
    assert model.layers[5].is_pretraining == True  # noqa: E712

    post_pretrain_functions(model, config_pdp)

    assert model.layers[1].depthwise_conv.pruning_layer.is_pretraining == False  # noqa: E712
    assert model.layers[1].pointwise_conv.pruning_layer.is_pretraining == False  # noqa: E712
    assert model.layers[2].is_pretraining == False  # noqa: E712
    assert model.layers[4].pruning_layer.is_pretraining == False  # noqa: E712
    assert model.layers[5].is_pretraining == False  # noqa: E712


def test_conv1d_call(config_pdp, conv1d_input):
    layer_to_replace = Conv1D(OUT_FEATURES, KERNEL_SIZE, strides=2, use_bias=False)
    layer_to_replace.build(conv1d_input.shape)
    out = layer_to_replace(conv1d_input)
    layer = PQConv1d(config_pdp, OUT_FEATURES, KERNEL_SIZE, strides=2)
    layer.build(conv1d_input.shape)
    layer._kernel.assign(layer_to_replace.kernel)
    out2 = layer(conv1d_input)
    assert ops.all(ops.equal(out, out2))


def test_dense_add_remove_layers(config_pdp, dense_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=(dense_input.shape[1:]))
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    model(dense_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(dense_input)
    model = apply_final_compression(model)
    output2 = model(dense_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(conv2d_input)
    model = apply_final_compression(model)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_depthwise_conv2d_add_remove_layers(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].kernel)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(conv2d_input)
    model = apply_final_compression(model)
    output2 = model(conv2d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_conv1d_add_remove_layers(config_pdp, conv1d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].kernel)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    output1 = model(conv1d_input)
    model = apply_final_compression(model)
    output2 = model(conv1d_input)
    assert ops.all(ops.equal(output1, output2))
    expected_nonzero_count = ops.count_nonzero(mask_50pct)
    nonzero_count = ops.count_nonzero(model.layers[1].kernel)
    assert ops.equal(expected_nonzero_count, nonzero_count)


def test_dense_get_layer_keep_ratio(config_pdp, dense_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=(dense_input.shape[1:]))
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    model(dense_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=OUT_FEATURES * IN_FEATURES * KERNEL_SIZE * KERNEL_SIZE) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_depthwise_conv2d_get_layer_keep_ratio(config_pdp, conv2d_input):
    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    model(conv2d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].kernel)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_conv1d_get_layer_keep_ratio(config_pdp, conv1d_input):

    config_pdp.pruning_parameters.enable_pruning = True
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False)(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    model(conv1d_input)
    post_pretrain_functions(model, config_pdp)
    pre_finetune_functions(model)

    mask_50pct = ops.cast(ops.linspace(0, 1, num=ops.size(model.layers[1].kernel)) < 0.5, "float32")
    mask_50pct = ops.reshape(keras.random.shuffle(mask_50pct), model.layers[1].pruning_layer.mask.shape)
    model.layers[1].pruning_layer.mask = mask_50pct
    ratio1 = get_layer_keep_ratio(model)
    model = apply_final_compression(model)
    ratio2 = get_layer_keep_ratio(model)
    assert ops.equal(ratio1, ratio2)
    assert ops.equal(ops.count_nonzero(mask_50pct) / ops.size(mask_50pct), ratio1)


def test_check_activation(config_pdp, dense_input):
    # ReLU
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="relu")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)

    assert isinstance(model.layers[2], ReLU)

    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="relu")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert isinstance(model.layers[2], PQActivation)
    assert model.layers[2].activation_name == "relu"

    # Tanh
    config_pdp.quantization_parameters.enable_quantization = False
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="tanh")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)

    assert isinstance(model.layers[2], Activation)
    assert model.layers[2].activation.__name__ == "tanh"

    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_real_tanh = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False, activation="tanh")(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert isinstance(model.layers[2], PQActivation)
    assert model.layers[2].activation_name == "tanh"


def test_hgq_activation_built(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=True, padding="same")(inputs)
    act = ReLU()(out)
    avg = AveragePooling2D(2)(act)
    model = keras.Model(inputs=inputs, outputs=avg, name="test_conv2d_hgq")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    is_built = []
    for layer in model.layers:
        is_built.append(layer.built)
        if hasattr(layer, "hgq"):  # Activation layers
            is_built.append(layer.hgq.built)
        if hasattr(layer, "hgq_weight"):  # Compression layers
            is_built.append(layer.hgq_weight.built)
        if hasattr(layer, "hgq_bias"):  # Compression layers
            is_built.append(layer.hgq_bias.built)
    assert all(is_built)
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=True)(inputs)
    act = Activation("tanh")(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_conv2d_hgq")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    is_built = []
    for layer in model.layers:
        is_built.append(layer.built)
        if hasattr(layer, "hgq"):  # Activation layers
            is_built.append(layer.hgq.built)
        if hasattr(layer, "hgq_weight"):  # Compression layers
            is_built.append(layer.hgq_weight.built)
        if hasattr(layer, "hgq_bias"):  # Compression layers
            is_built.append(layer.hgq_bias.built)
    assert all(is_built)


# Activation Pruning


def test_ap_conv2d_channels_last_transpose(config_ap, conv2d_input):
    if keras.backend.image_data_format() == "channels_last":
        conv2d_input = ops.transpose(conv2d_input, (0, 3, 1, 2))
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same", data_format="channels_first")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf(conv2d_input)

    model_cf = add_compression_layers(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1].kernel

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers(model_cl, config_ap, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-4)


def test_ap_conv1d_channels_last_transpose(config_ap, conv1d_input):
    if keras.backend.image_data_format() == "channels_last":
        conv1d_input = ops.transpose(conv1d_input, (0, 2, 1))
    keras.backend.set_image_data_format("channels_first")

    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same", data_format="channels_first")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers(model_cl, config_ap, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_ap_depthwiseconv2d_channels_last_transpose(config_ap, conv2d_input):
    if keras.backend.image_data_format() == "channels_last":
        conv2d_input = ops.transpose(conv2d_input, (0, 3, 1, 2))
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same", data_format="channels_first")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers(model_cf, config_ap, inp.shape)

    weight_cf = model_cf.layers[1]._kernel
    model_cf.summary()
    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers(model_cl, config_ap, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_ap_dense_channels_last_transpose(config_ap, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers(model_cf, config_ap, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_ap)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers(model_cl, config_ap, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_ap)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


# Wanda


def test_wanda_conv2d_channels_last_transpose(config_wanda, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers(model_cl, config_wanda, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_wanda_conv1d_channels_last_transpose(config_wanda, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers(model_cl, config_wanda, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_wanda_depthwiseconv2d_channels_last_transpose(config_wanda, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers(model_cl, config_wanda, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_wanda_dense_channels_last_transpose(config_wanda, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers(model_cf, config_wanda, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_wanda)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers(model_cl, config_wanda, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_wanda)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


# PDP


def test_pdp_conv2d_channels_last_transpose(config_pdp, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers(model_cl, config_pdp, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_pdp_conv1d_channels_last_transpose(config_pdp, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers(model_cl, config_pdp, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_pdp_depthwiseconv2d_channels_last_transpose(config_pdp, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers(model_cl, config_pdp, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_pdp_dense_channels_last_transpose(config_pdp, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)
    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers(model_cf, config_pdp, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_pdp)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers(model_cl, config_pdp, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_pdp)
    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)

    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1].kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1].kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))
    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


# CS


def test_cs_conv2d_channels_last_transpose(config_cs, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv2d")
    model_cf = add_compression_layers(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1]._kernel
    s = model_cf.layers[1].pruning_layer.s.value
    new_s = np.zeros_like(s) + 0.1
    new_s = np.reshape(new_s, -1)
    new_s[: ops.size(s) // 2] = -1.0
    new_s = ops.reshape(new_s, s.shape)
    model_cf.layers[1].pruning_layer.s.assign(new_s)

    post_pretrain_functions(model_cf, config_cs)
    out_cf = model_cf(inp, training=True)
    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv2d1")
    model_cl = add_compression_layers(model_cl, config_cs, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    model_cl.layers[1].pruning_layer.s.assign(new_s)

    post_pretrain_functions(model_cl, config_cs)

    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.get_hard_mask(None)

    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.get_hard_mask(None)
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))
    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_cs_conv1d_channels_last_transpose(config_cs, conv1d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv1d_input)), conv1d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_conv1d")
    model_cf = add_compression_layers(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_cs)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = Conv1D(OUT_FEATURES, KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_conv1d1")
    model_cl = add_compression_layers(model_cl, config_cs, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_cs)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_cs_depthwiseconv2d_channels_last_transpose(config_cs, conv2d_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(conv2d_input)), conv2d_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d")
    model_cf = add_compression_layers(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_cs)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")
    inp = ops.transpose(inp, (0, 2, 3, 1))

    inputs = keras.Input(shape=inp.shape[1:])
    out = DepthwiseConv2D(KERNEL_SIZE, use_bias=False, padding="same")(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dwconv2d1")
    model_cl = add_compression_layers(model_cl, config_cs, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_cs)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))

    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_cs_dense_channels_last_transpose(config_cs, dense_input):
    keras.backend.set_image_data_format("channels_first")
    inp = ops.reshape(ops.linspace(0, 1, ops.size(dense_input)), dense_input.shape)

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cf = keras.Model(inputs=inputs, outputs=out, name="test_dense")
    model_cf = add_compression_layers(model_cf, config_cs, inp.shape)
    weight_cf = model_cf.layers[1]._kernel

    post_pretrain_functions(model_cf, config_cs)
    model_cf(inp, training=True)
    model_cf(inp, training=True)
    out_cf = model_cf(inp, training=True)

    keras.backend.set_image_data_format("channels_last")

    inputs = keras.Input(shape=inp.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    model_cl = keras.Model(inputs=inputs, outputs=out, name="test_dense1")
    model_cl = add_compression_layers(model_cl, config_cs, inp.shape)
    model_cl.layers[1]._kernel.assign(weight_cf)
    post_pretrain_functions(model_cl, config_cs)

    model_cl(inp, training=True)
    model_cl(inp, training=True)
    out_cl = model_cl(inp, training=True)
    cf_mask = model_cf.layers[1].pruning_layer.mask
    cf_weight = ops.transpose(model_cf.layers[1]._kernel, model_cf.layers[1].weight_transpose)
    cf_masked_weight = cf_mask * cf_weight
    cl_mask = model_cl.layers[1].pruning_layer.mask
    cl_weight = ops.transpose(model_cl.layers[1]._kernel, model_cl.layers[1].weight_transpose)
    cl_masked_weight = cl_mask * cl_weight
    out_cl_transposed = ops.transpose(out_cl, (model_cl.layers[1].data_transpose))
    assert ops.all(ops.equal(ops.ravel(cf_mask), ops.ravel(cl_mask)))
    assert ops.all(ops.equal(cf_masked_weight, cl_masked_weight))
    np.testing.assert_allclose(out_cf, out_cl_transposed, rtol=0, atol=5e-6)


def test_calculate_pruning_budget(config_wanda, dense_input):
    sparsity = 0.75
    config_wanda.pruning_parameters.calculate_pruning_budget = True
    config_wanda.pruning_parameters.sparsity = sparsity

    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    out2 = Dense(OUT_FEATURES, use_bias=False)(out)
    model = keras.Model(inputs=inputs, outputs=out2, name="test_conv2d")

    # First layer will have 50% sparsity
    weight = np.ones(IN_FEATURES * OUT_FEATURES).astype(np.float32)
    weight[: IN_FEATURES * OUT_FEATURES // 2] = 0.001
    weight = ops.reshape(ops.convert_to_tensor(weight), (IN_FEATURES, OUT_FEATURES))
    weight2 = ops.reshape(ops.linspace(0.01, 0.99, OUT_FEATURES * OUT_FEATURES), (OUT_FEATURES, OUT_FEATURES))

    model = add_compression_layers(model, config_wanda, dense_input.shape)
    model.layers[1]._kernel.assign(weight)
    model.layers[2]._kernel.assign(weight2)
    # Triggers calculation of pruning budget for PDP and Wanda
    post_pretrain_functions(model, config_wanda)
    total_weights = IN_FEATURES * OUT_FEATURES + OUT_FEATURES * OUT_FEATURES
    remaining_weights = 0
    for layer in model.layers:
        if hasattr(layer, "pruning_layer"):
            calculated_sparsity = layer.pruning_layer.sparsity
            remaining_weights += (1 - calculated_sparsity) * ops.cast(ops.size(layer.kernel), "float32")
    # First layer should have 50% sparsity, total sparsity should be around 75%
    assert model.layers[1].pruning_layer.sparsity == 0.5
    np.testing.assert_allclose(remaining_weights / total_weights, 1 - sparsity, atol=1e-3, rtol=0)


def test_trigger_post_pretraining(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    act1 = Activation("tanh")(out)
    out2 = Dense(OUT_FEATURES, use_bias=False)(act1)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    assert model.layers[1].pruning_layer.is_pretraining == True  # noqa: E712
    assert model.layers[2].is_pretraining == True  # noqa: E712
    assert model.layers[3].pruning_layer.is_pretraining == True  # noqa: E712
    assert model.layers[4].is_pretraining == True  # noqa: E712

    post_pretrain_functions(model, config_pdp)

    assert model.layers[1].pruning_layer.is_pretraining == False  # noqa: E712
    assert model.layers[2].is_pretraining == False  # noqa: E712
    assert model.layers[3].pruning_layer.is_pretraining == False  # noqa: E712
    assert model.layers[4].is_pretraining == False  # noqa: E712


def test_hgq_weight_shape(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    act1 = Activation("tanh")(out)
    out2 = Dense(OUT_FEATURES, use_bias=False)(act1)
    act2 = ReLU()(out2)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_conv2d")

    model = add_compression_layers(model, config_pdp, dense_input.shape)
    assert model.layers[1].weight_quantizer.quantizer.quantizer._i.shape == model.layers[1].kernel.shape
    layer_2_input_shape = [1] + list(model.layers[2].input.shape[1:])
    assert model.layers[2].input_quantizer.quantizer.quantizer._i.shape == layer_2_input_shape


def test_replace_weight_with_original_value(config_pdp, conv2d_input, conv1d_input, dense_input):
    config_pdp.quantization_parameters.enable_quantization = False
    config_pdp.pruning_parameters.enable_pruning = False
    # Case Dense
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=True)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    orig_output = model(dense_input)
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    output = model(dense_input)
    assert ops.all(ops.equal(orig_output, output))

    # Case Conv2D
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    orig_output = model(conv2d_input)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    output = model(conv2d_input)
    assert ops.all(ops.equal(orig_output, output))
    # Case Conv1D
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    orig_output = model(conv1d_input)
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    output = model(conv1d_input)
    assert ops.all(ops.equal(orig_output, output))


def test_set_activation_custom_bits_hgq(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    out = ReLU()(out)
    out = AveragePooling2D(2)(out)
    out = Activation("tanh")(out)
    model = keras.Model(inputs=inputs, outputs=out)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    for m in model.layers:
        if isinstance(m, (PQConv2d)):
            _, iw, fw = m.get_weight_quantization_bits()
            _, ib, fb = m.get_bias_quantization_bits()
            assert ops.all(iw == 0.0)
            assert ops.all(ib == 0.0)
            assert ops.all(fw == 7.0)
            assert ops.all(fb == 7.0)
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            k_input, i_input, f_input = m.get_input_quantization_bits()
            assert ops.all(i_input == 0.0)
            assert ops.all(f_input == 7.0)
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            k_input, i_input, f_input = m.get_input_quantization_bits()
            assert ops.all(i_input == 0.0)
            assert ops.all(f_input == 8.0)
        elif isinstance(m, (PQAvgPool2d)):
            _, i_input, f_input = m.get_input_quantization_bits()
            assert ops.all(i_input == 0.0)
            assert ops.all(f_input == 7.0)

    config_pdp.quantization_parameters.layer_specific = {
        'conv2d': {
            'weight': {'integer_bits': 1.0, 'fractional_bits': 3.0},
            'bias': {'integer_bits': 2.0, 'fractional_bits': 4.0},
        },
        're_lu': {"input": {'integer_bits': 1.0, 'fractional_bits': 3.0}},
        'average_pooling2d': {"input": {'integer_bits': 1.0, 'fractional_bits': 3.0}},
        'activation': {"input": {'integer_bits': 0.0, 'fractional_bits': 3.0}},
    }
    keras.backend.clear_session()
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    out = ReLU()(out)
    out = AveragePooling2D(2)(out)
    out = Activation("tanh")(out)
    model = keras.Model(inputs=inputs, outputs=out)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    for m in model.layers:
        if isinstance(m, (PQConv2d)):
            _, iw, fw = m.get_weight_quantization_bits()
            _, ib, fb = m.get_bias_quantization_bits()
            assert ops.all(iw == 1.0)
            assert ops.all(ib == 2.0)
            assert ops.all(fw == 3.0)
            assert ops.all(fb == 4.0)
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            k_input, i_input, f_input = m.get_input_quantization_bits()
            assert ops.all(i_input == 0.0)
            assert ops.all(f_input == 3.0)
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            k_input, i_input, f_input = m.get_input_quantization_bits()
            assert ops.all(i_input == 1.0)
            assert ops.all(f_input == 3.0)
        elif isinstance(m, (PQAvgPool2d)):
            _, i_input, f_input = m.get_input_quantization_bits()
            assert ops.all(i_input == 1.0)
            assert ops.all(f_input == 3.0)


def test_set_activation_custom_bits_quantizer(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = False
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    out = ReLU()(out)
    out = AveragePooling2D(2)(out)
    out = Activation("tanh")(out)
    model = keras.Model(inputs=inputs, outputs=out)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)

    for m in model.layers:
        if isinstance(m, (PQConv2d)):
            assert m.i_weight == 0.0
            assert m.i_bias == 0.0

            assert m.f_weight == 7.0
            assert m.f_bias == 7.0
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            assert m.i_input == 0.0
            assert m.f_input == 7.0
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            assert m.i_input == 0.0
            assert m.f_input == 8.0
        elif isinstance(m, (PQAvgPool2d)):
            assert m.i_input == 0.0
            assert m.f_input == 7.0

    config_pdp.quantization_parameters.layer_specific = {
        'conv2d': {
            'weight': {'integer_bits': 1.0, 'fractional_bits': 3.0},
            'bias': {'integer_bits': 2.0, 'fractional_bits': 4.0},
        },
        're_lu': {"input": {'integer_bits': 1.0, 'fractional_bits': 3.0}},
        'average_pooling2d': {"input": {'integer_bits': 1.0, 'fractional_bits': 3.0}},
        'activation': {"input": {'integer_bits': 0.0, 'fractional_bits': 3.0}},
    }
    keras.backend.clear_session()
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    out = ReLU()(out)
    out = AveragePooling2D(2)(out)
    out = Activation("tanh")(out)
    model = keras.Model(inputs=inputs, outputs=out)
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    for m in model.layers:
        if isinstance(m, (PQConv2d)):
            assert m.i_weight == 1.0
            assert m.i_bias == 2.0

            assert m.f_weight == 3.0
            assert m.f_bias == 4.0
        elif isinstance(m, PQActivation) and m.activation_name == "tanh":
            assert m.i_input == 0.0
            assert m.f_input == 3.0
        elif isinstance(m, PQActivation) and m.activation_name == "relu":
            assert m.i_input == 1.0
            assert m.f_input == 3.0
        elif isinstance(m, (PQAvgPool2d)):
            assert m.i_input == 1.0
            assert m.f_input == 3.0


def test_ebops_dense(config_pdp, dense_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=False)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[1].hgq_loss()

    inputs = keras.Input(shape=dense_input.shape[1:])
    out = Dense(OUT_FEATURES, use_bias=True)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_dense")
    model = add_compression_layers(model, config_pdp, dense_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[1].hgq_loss()


def test_ebops_conv2d(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=False)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[1].hgq_loss()

    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_conv2d")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[1].hgq_loss()


def test_ebops_conv1d(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=False)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_dense")
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[1].hgq_loss()

    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv1d_input.shape[1:])
    out = Conv1D(OUT_FEATURES, kernel_size=KERNEL_SIZE, use_bias=True)(inputs)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_dense")
    model = add_compression_layers(model, config_pdp, conv1d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[1].hgq_loss()


def test_ebops_bn(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    config_pdp.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=conv2d_input.shape[1:])
    out = Conv2D(OUT_FEATURES, KERNEL_SIZE)(inputs)
    axis = 1 if keras.backend.image_data_format() == "channels_first" else -1
    out = BatchNormalization(axis=axis)(out)
    act = ReLU()(out)
    model = keras.Model(inputs=inputs, outputs=act, name="test_bn")
    model = add_compression_layers(model, config_pdp, conv2d_input.shape)
    post_pretrain_functions(model, config_pdp)
    model.layers[2].hgq_loss()


def test_ebops_activations(config_cs, dense_input):
    config_cs.quantization_parameters.use_high_granularity_quantization = True
    config_cs.quantization_parameters.enable_quantization = True
    inputs = keras.Input(shape=dense_input.shape[1:])
    act = ReLU()(inputs)
    act2 = Activation("tanh")(act)
    model = keras.Model(inputs=inputs, outputs=act2, name="test_activations")
    model = add_compression_layers(model, config_cs, dense_input.shape)
    post_pretrain_functions(model, config_cs)
    model.layers[1].hgq_loss()


def test_linear_direct(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQDense(config_pdp, OUT_FEATURES, quantize_output=True, use_bias=True)
    layer(dense_input)
    assert True


def test_1dconv_direct(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv1d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_output=True, use_bias=True)
    layer(conv1d_input)
    assert True


def test_2dconv_direct(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQConv2d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_output=True, use_bias=True)
    layer(conv2d_input)
    assert True


def test_batch_normalization(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQBatchNormalization(config_pdp)
    layer(conv2d_input)
    assert True


def test_2dconv_depth(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQDepthwiseConv2d(config_pdp, KERNEL_SIZE)
    layer(conv2d_input)
    assert True


def test_avg_pool2d(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQAvgPool2d(config_pdp, KERNEL_SIZE)
    layer(conv2d_input)
    assert True


def test_avg_pool1d(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    layer = PQAvgPool1d(config_pdp, KERNEL_SIZE)
    layer(conv1d_input)
    assert True


class DummyLayer(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.built = True
        self.layer_called = 0

    def call(self, x, *args, **kwargs):
        self.layer_called += 1
        return x

    def post_pre_train_function(self):
        pass

    def get_total_bits(self, shape):
        return keras.ops.ones(shape)

    def hgq_loss(self):
        return 0.0

    def extra_repr(self):
        return f"Layer called = {self.layer_called} times."


def test_avgpool_quant_called(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    with patch('pquant.layers.Quantizer', DummyLayer):
        layer = PQAvgPool1d(config_pdp, KERNEL_SIZE, quantize_input=True)
        layer(conv1d_input)
        assert layer.input_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 0

        layer = PQAvgPool1d(config_pdp, KERNEL_SIZE, quantize_input=False, quantize_output=True)
        layer(conv1d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.output_quantizer.layer_called == 1

        config_pdp.quantization_parameters.enable_quantization = False
        layer = PQAvgPool1d(config_pdp, KERNEL_SIZE, quantize_input=True, quantize_output=True)
        layer(conv1d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.output_quantizer.layer_called == 0
    assert True


def test_batchnorm_quant_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    axis = -1 if keras.backend.image_data_format() == "channels_last" else 1
    with patch('pquant.layers.Quantizer', DummyLayer):
        layer = PQBatchNormalization(config_pdp, axis=axis, quantize_input=True)
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 1
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1

        layer = PQBatchNormalization(config_pdp, axis=axis, quantize_input=False)
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1

        config_pdp.quantization_parameters.enable_quantization = False
        layer = PQBatchNormalization(config_pdp, axis=axis, quantize_input=True)
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 0
        assert layer.bias_quantizer.layer_called == 0
    assert True


def test_pqconv2d_quant_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    with patch('pquant.layers.Quantizer', DummyLayer):
        layer = PQConv2d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_input=True, use_bias=True)
        layer.post_pre_train_function()
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 1
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 0

        layer = PQConv2d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_input=False, quantize_output=True, use_bias=True)
        layer.post_pre_train_function()
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 1

        config_pdp.quantization_parameters.enable_quantization = False
        layer.post_pre_train_function()
        layer = PQConv2d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_input=True, quantize_output=True, use_bias=True)
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 0
        assert layer.bias_quantizer.layer_called == 0
        assert layer.output_quantizer.layer_called == 0
    assert True


def test_pqdepthwiseconv2d_quant_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True

    with patch('pquant.layers.Quantizer', DummyLayer):
        layer = PQDepthwiseConv2d(config_pdp, KERNEL_SIZE, quantize_input=True, use_bias=True)
        layer.post_pre_train_function()
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 1
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 0

        layer = PQDepthwiseConv2d(config_pdp, KERNEL_SIZE, quantize_input=False, quantize_output=True, use_bias=True)
        layer.post_pre_train_function()
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 1

        config_pdp.quantization_parameters.enable_quantization = False
        layer.post_pre_train_function()
        layer = PQDepthwiseConv2d(config_pdp, KERNEL_SIZE, quantize_input=True, quantize_output=True, use_bias=True)
        layer(conv2d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 0
        assert layer.bias_quantizer.layer_called == 0
        assert layer.output_quantizer.layer_called == 0
    assert True


def test_pqconv1d_quant_called(config_pdp, conv1d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    with patch('pquant.layers.Quantizer', DummyLayer):
        layer = PQConv1d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_input=True, use_bias=True)
        layer.post_pre_train_function()
        layer(conv1d_input)
        assert layer.input_quantizer.layer_called == 1
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 0

        layer = PQConv1d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_input=False, quantize_output=True, use_bias=True)
        layer.post_pre_train_function()
        layer(conv1d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 1

        config_pdp.quantization_parameters.enable_quantization = False
        layer.post_pre_train_function()
        layer = PQConv1d(config_pdp, OUT_FEATURES, KERNEL_SIZE, quantize_input=True, quantize_output=True, use_bias=True)
        layer(conv1d_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 0
        assert layer.bias_quantizer.layer_called == 0
        assert layer.output_quantizer.layer_called == 0
    assert True


def test_dense_quant_called(config_pdp, dense_input):
    config_pdp.quantization_parameters.enable_quantization = True
    with patch('pquant.layers.Quantizer', DummyLayer):
        layer = PQDense(config_pdp, OUT_FEATURES, quantize_input=True, use_bias=True)
        layer.post_pre_train_function()
        layer(dense_input)
        assert layer.input_quantizer.layer_called == 1
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 0

        layer = PQDense(config_pdp, OUT_FEATURES, quantize_input=False, quantize_output=True, use_bias=True)
        layer.post_pre_train_function()
        layer(dense_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 1
        assert layer.bias_quantizer.layer_called == 1
        assert layer.output_quantizer.layer_called == 1

        config_pdp.quantization_parameters.enable_quantization = False
        layer.post_pre_train_function()
        layer = PQDense(config_pdp, OUT_FEATURES, quantize_input=True, quantize_output=True, use_bias=True)
        layer(dense_input)
        assert layer.input_quantizer.layer_called == 0
        assert layer.weight_quantizer.layer_called == 0
        assert layer.bias_quantizer.layer_called == 0
        assert layer.output_quantizer.layer_called == 0
    assert True


def test_layer_replacement_quant_called(config_pdp, conv2d_input):
    config_pdp.quantization_parameters.enable_quantization = True
    config_pdp.quantization_parameters.quantize_input = True
    config_pdp.quantization_parameters.quantize_output = True
    config_pdp.quantization_parameters.use_high_granularity_quantization = True
    with patch('pquant.layers.Quantizer', DummyLayer):
        inp = keras.Input(shape=conv2d_input.shape[1:])
        x = Conv2D(OUT_FEATURES, KERNEL_SIZE)(inp)

        x = ReLU()(x)
        x = keras.layers.Flatten()(x)
        out = Dense(4)(x)

        model = keras.Model(inputs=inp, outputs=out)
        model.summary()

        model = add_compression_layers(model, config_pdp, conv2d_input.shape)
        model(conv2d_input, training=True)
        assert model.layers[-1].output_quantizer.layer_called == 1
        model(conv2d_input, training=False)
        assert model.layers[-1].output_quantizer.layer_called == 2


def build_model(config):
    inp = keras.layers.Input(shape=((16,)))
    x = PQDense(
        config,
        64,
        in_quant_bits=(
            1.0,
            3.0,
            3.0,
        ),
    )(inp)
    x = keras.layers.ReLU()(x)
    x = PQDense(config, 32)(x)
    x = keras.layers.ReLU()(x)
    x = PQDense(config, 32)(x)
    x = keras.layers.ReLU()(x)
    out = PQDense(config, 5, out_quant_bits=(1.0, 3.0, 3.0), quantize_output=True)(x)
    return keras.Model(inputs=inp, outputs=out)


def test_ebops_dense_varied(config_pdp, dense_input):
    config = config_pdp
    config.pruning_parameters.enable_pruning = False
    config.quantization_parameters.use_high_granularity_quantization = True
    config.quantization_parameters.hgq_gamma = 1.0
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.overflow_mode_data = "WRAP"
    config.quantization_parameters.overflow_mode_parameters = "SAT_SYM"
    model = build_model(config)
    model(dense_input, training=True)
    post_pretrain_functions(model, config)


@pytest.mark.parametrize(
    "config_fn",
    [pdp_config, ap_config, autosparse_config, cs_config, dst_config, mdmm_config, wanda_config],
    ids=["pdp", "ap", "autosparse", "cs", "dst", "mdmm", "wanda"],
)
def test_model_serialization(tmp_path, config_fn):
    config = config_fn()
    config.quantization_parameters.enable_quantization = True
    channels_first = keras.backend.image_data_format() == "channels_first"
    if channels_first:
        input_shape = (IN_FEATURES, 32, 32)
        conv1d_shape = (OUT_FEATURES, 16 * 16)
        dummy = np.zeros((1, IN_FEATURES, 32, 32), dtype=np.float32)
        bn_axis = 1
    else:
        input_shape = (32, 32, IN_FEATURES)
        conv1d_shape = (16 * 16, OUT_FEATURES)
        dummy = np.zeros((1, 32, 32, IN_FEATURES), dtype=np.float32)
        bn_axis = -1

    inputs = keras.Input(shape=input_shape)
    x = PQConv2d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(inputs)
    x = PQBatchNormalization(config, axis=bn_axis)(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    x = PQAvgPool2d(config, pool_size=2, strides=2, padding="same")(x)
    x = keras.layers.Reshape(conv1d_shape)(x)
    x = PQConv1d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    x = PQAvgPool1d(config, pool_size=2, strides=2, padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = PQDense(config, units=OUT_FEATURES)(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    model = keras.Model(inputs, x)

    # Call the model once to trigger build of all sublayers (pruning masks, etc.)
    model(dummy)

    # Randomize weights; skip boolean flags which must stay at their 0/1 values
    rng = np.random.default_rng(42)
    for w in model.weights:
        if w.name.endswith(("is_pretraining", "is_finetuning")):
            continue
        w.assign(rng.standard_normal(w.shape).astype(w.dtype))

    pq_types = (PQConv2d, PQConv1d, PQDense)

    def assert_pq_state_survives_roundtrip(m, label):
        """Save m, reload it, and verify all state round-trips correctly."""
        save_path = tmp_path / f"{label}.keras"
        m.save(save_path)
        reloaded = keras.models.load_model(save_path)

        orig_pq = [layer for layer in m.layers if isinstance(layer, pq_types)]
        loaded_pq = [layer for layer in reloaded.layers if isinstance(layer, pq_types)]
        assert len(loaded_pq) == len(orig_pq)

        for orig_l, loaded_l in zip(orig_pq, loaded_pq):
            for attr in ("final_compression_done", "is_pretraining", "is_finetuning"):
                np.testing.assert_equal(
                    getattr(loaded_l, attr),
                    getattr(orig_l, attr),
                    err_msg=f"[{label}] {orig_l.name}.{attr} mismatch",
                )

        assert len(m.weights) == len(reloaded.weights)
        for orig_w, loaded_w in zip(m.weights, reloaded.weights):
            np.testing.assert_array_equal(
                np.array(orig_w),
                np.array(loaded_w),
                err_msg=f"[{label}] Weight mismatch: {orig_w.name}",
            )

    # Stage 1: initial state (is_pretraining=True, is_finetuning=False)
    assert_pq_state_survives_roundtrip(model, "initial")

    # Stage 2: after post_pretrain_functions (is_pretraining=False, is_finetuning=False)
    post_pretrain_functions(model, config)
    assert_pq_state_survives_roundtrip(model, "post_pretrain")

    # Stage 3: after pre_finetune_functions (is_pretraining=False, is_finetuning=True)
    pre_finetune_functions(model)
    assert_pq_state_survives_roundtrip(model, "pre_finetune")

    # Stage 4: after apply_final_compression (final_compression_done=True on all layers)
    apply_final_compression(model)
    assert_pq_state_survives_roundtrip(model, "final_compression")


@pytest.mark.parametrize(
    "config_fn",
    [pdp_config, ap_config, autosparse_config, cs_config, dst_config, mdmm_config, wanda_config],
    ids=["pdp", "ap", "autosparse", "cs", "dst", "mdmm", "wanda"],
)
def test_checkpoint_save_load(tmp_path, config_fn):
    """Verify that save_weights/load_weights preserves non-trainable pruning state (e.g. mask)."""
    config = config_fn()
    config.quantization_parameters.enable_quantization = True
    channels_first = keras.backend.image_data_format() == "channels_first"
    if channels_first:
        input_shape = (IN_FEATURES, 32, 32)
        dummy = np.zeros((1, IN_FEATURES, 32, 32), dtype=np.float32)
    else:
        input_shape = (32, 32, IN_FEATURES)
        dummy = np.zeros((1, 32, 32, IN_FEATURES), dtype=np.float32)

    inputs = keras.Input(shape=input_shape)
    x = PQConv2d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(inputs)
    x = keras.layers.Flatten()(x)
    x = PQDense(config, units=OUT_FEATURES)(x)
    model = keras.Model(inputs, x)
    model(dummy)

    # Randomize all weights including non-trainable pruning state (mask, etc.)
    rng = np.random.default_rng(0)
    for w in model.weights:
        if w.name.endswith(("is_pretraining", "is_finetuning")):
            continue
        w.assign(rng.standard_normal(w.shape).astype(w.dtype))

    original_weights = [np.array(w) for w in model.weights]

    path = str(tmp_path / "ckpt.weights.h5")
    model.save_weights(path)

    # Overwrite all weights with zeros
    for w in model.weights:
        w.assign(np.zeros(w.shape, dtype=w.dtype))

    model.load_weights(path)

    for orig, w in zip(original_weights, model.weights):
        np.testing.assert_array_equal(orig, np.array(w), err_msg=f"Checkpoint weight mismatch: {w.name}")


@pytest.mark.parametrize(
    "config_fn",
    [pdp_config, ap_config, autosparse_config, cs_config, dst_config, mdmm_config, wanda_config],
    ids=["pdp", "ap", "autosparse", "cs", "dst", "mdmm", "wanda"],
)
def test_model_fit(config_fn):
    from pquant.core.keras.train import PQuantCallback

    config = config_fn()
    config.quantization_parameters.enable_quantization = True
    config.training_parameters.pretraining_epochs = 1
    config.training_parameters.epochs = 1
    config.training_parameters.fine_tuning_epochs = 1
    config.training_parameters.rounds = 1
    config.training_parameters.save_weights_epoch = 0

    channels_first = keras.backend.image_data_format() == "channels_first"
    if channels_first:
        input_shape = (IN_FEATURES, 32, 32)
        conv1d_shape = (OUT_FEATURES, 16 * 16)
        dummy_x = np.zeros((BATCH_SIZE, IN_FEATURES, 32, 32), dtype=np.float32)
        bn_axis = 1
    else:
        input_shape = (32, 32, IN_FEATURES)
        conv1d_shape = (16 * 16, OUT_FEATURES)
        dummy_x = np.zeros((BATCH_SIZE, 32, 32, IN_FEATURES), dtype=np.float32)
        bn_axis = -1

    inputs = keras.Input(shape=input_shape)
    x = PQConv2d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(inputs)
    x = PQBatchNormalization(config, axis=bn_axis)(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    x = PQAvgPool2d(config, pool_size=2, strides=2, padding="same")(x)
    x = keras.layers.Reshape(conv1d_shape)(x)
    x = PQConv1d(config, OUT_FEATURES, KERNEL_SIZE, padding="same")(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    x = PQAvgPool1d(config, pool_size=2, strides=2, padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = PQDense(config, units=OUT_FEATURES)(x)
    x = PQActivation(config, activation="relu", quantize_input=True, quantize_output=True)(x)
    model = keras.Model(inputs, x)

    dummy_y = np.zeros((BATCH_SIZE, model.output_shape[-1]), dtype=np.float32)

    model.compile(optimizer="adam", loss="mse", jit_compile=False)
    callback = PQuantCallback(config, log_ebops=False, log_keep_ratio=False)
    model.fit(dummy_x, dummy_y, epochs=callback.total_epochs, callbacks=[callback], verbose=0)
