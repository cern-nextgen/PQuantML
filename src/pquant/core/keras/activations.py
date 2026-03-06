from typing import Tuple
from typing import TypeVar as T

import keras
from keras.ops import maximum, minimum, relu, tanh

from pquant.core.keras.quantizer import Quantizer


def hard_sigmoid(x):
    """Computes hard_sigmoid function that saturates between 0 and 1."""
    x = 0.5 * x + 0.5
    x = maximum(x, 0.0)
    x = minimum(x, 1.0)
    return x


def hard_tanh(x):
    """Computes hard_tanh function that saturates between -1 and 1."""
    return 2.0 * hard_sigmoid(x) - 1.0


activation_registry = {"relu": relu, "tanh": tanh, "hard_tanh": hard_tanh}


@keras.saving.register_keras_serializable(package="PQuantML")
class PQActivation(keras.layers.Layer):
    def __init__(
        self,
        config,
        activation="relu",
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        quantize_input=True,
        quantize_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        if in_quant_bits is None:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits
        else:
            self.k_input, self.i_input, self.f_input = in_quant_bits

        if out_quant_bits is None:
            self.k_output = config.quantization_parameters.default_data_keep_negatives
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits
        else:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        self.in_quant_bits = in_quant_bits
        self.out_quant_bits = out_quant_bits
        self.activation_name = activation.lower()
        self.activation_function = activation_registry.get(self.activation_name)
        self.config = config
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.use_multiplier = config.quantization_parameters.use_relu_multiplier
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress

        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        self.built = False

    def build(self, input_shape):
        self.input_shape = (1,) + tuple(input_shape[1:])

        if self.quantize_input:
            self.input_quantizer = Quantizer(
                k=self.k_input,
                i=self.i_input,
                f=self.f_input,
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_data=True,
                is_heterogeneous=self.use_hgq,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
            )
        if self.quantize_output:
            self.output_quantizer = Quantizer(
                k=self.k_output,
                i=self.i_output,
                f=self.f_output,
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_data=True,
                is_heterogeneous=self.use_hgq,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
            )

        if self.use_multiplier:
            self.multiplier = self.add_weight(shape=(1,), trainable=True, initializer=keras.initializers.Constant(-1.0))
        super().build(input_shape)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def set_input_quantization_bits(self, i, f):
        self.input_quantizer.set_quantization_bits(i, f)

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def set_output_quantization_bits(self, i, f):
        self.output_quantizer.set_quantization_bits(i, f)

    def post_pre_train_function(self):
        self.is_pretraining = False
        if self.quantize_input:
            self.input_quantizer.post_pre_train_function()
        if self.quantize_output:
            self.output_quantizer.post_pre_train_function()

    def ebops(self):
        if self.quantize_input and self.quantize_output:
            bw_inp = self.input_quantizer.quantizer.bits_(self.input_shape)
            bw_out = self.output_quantizer.quantizer.bits_(self.input_shape)
            return keras.ops.sum((2.0**bw_inp) * bw_out) * 1e-4  # type: ignore
        return 0.0

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def pre_activation(self, x):
        if not self.use_hgq and self.use_multiplier and self.activation_name == "relu":
            x = x * 2 ** (keras.ops.stop_gradient(keras.ops.round(self.multiplier) - self.multiplier) + self.multiplier)
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x)
        return x

    def post_activation(self, x):
        if self.quantize_output and self.enable_quantization:
            return self.output_quantizer(x)
        return x

    def call(self, x):
        if self.use_fitcompress and self.is_pretraining and self.activation_name == "relu":
            if self.post_fitcompress_calibration:
                # Save quantized input into ReLU
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized activations
            return relu(x)
        # Multiplier after fitcompress if condition, such that we don't use any relu multiplier during FITcompress search
        x = self.pre_activation(x)
        x = self.activation_function(x)
        x = self.post_activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "quantize_input": self.quantize_input,
                "quantize_output": self.quantize_output,
                "activation": self.activation_name,
                "in_quant_bits": self.in_quant_bits,
                "out_quant_bits": self.out_quant_bits,
            }
        )
        return config

    def extra_repr(self):
        return f"quantize_input = {self.quantize_input}, quantize_output = {self.quantize_output}"
