from math import prod
from typing import Tuple
from typing import TypeVar as T
from typing import Union

import keras
from keras import ops
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
        enable_ebops=True,
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
        if isinstance(activation, str):
            self.activation_name = activation.lower()
            self.activation_function = activation_registry.get(self.activation_name)
        else:
            # A callable was passed directly instead of a registry key (e.g. the exp/inv
            # lookup-table functions used by QSoftmax).
            self.activation_function = activation
            self.activation_name = getattr(activation, "__name__", activation.__class__.__name__).lower()
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
        self.dynamic_data = config.quantization_parameters.dynamic_data_quantization

        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        self.enable_ebops = enable_ebops
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
                dynamic_data=self.dynamic_data,
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
                dynamic_data=self.dynamic_data,
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
        if not self.enable_ebops:
            return 0.0
        if self.quantize_input and self.quantize_output:
            bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
            bw_out = self.output_quantizer.get_total_bits(self.input_shape)
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
                "enable_ebops": self.enable_ebops,
                "activation": self.activation_name,
                "in_quant_bits": self.in_quant_bits,
                "out_quant_bits": self.out_quant_bits,
            }
        )
        return config

    def extra_repr(self):
        return f"quantize_input = {self.quantize_input}, quantize_output = {self.quantize_output}"


@keras.saving.register_keras_serializable(package="PQuantML")
class PQSoftmax(keras.layers.Layer):
    """Quantized softmax that mirrors HGQ's ``QSoftmax``.

    Args:
        config: PQuant configuration object (or a serialized dict).
        axis: Axis (or axes) the softmax normalizes over.
        stable: If True, subtract the max before exponentiating for numerical stability.
        input_scaler: Scalar multiplied with the logits before exponentiating.
        parallelization_factor: hls4ml parallelization factor used in the ebops cost model.
        quantize_input: Whether to quantize the softmax logits before the exp table.
        quantize_output: Whether to quantize the exp * inv product (the softmax output).
        in_quant_bits: (k, i, f) bits for the softmax input quantizer.
        out_quant_bits: (k, i, f) bits for the softmax output quantizer.
        exp_in_quant_bits / exp_out_quant_bits: (k, i, f) bits for the exp table.
        inv_in_quant_bits / inv_out_quant_bits: (k, i, f) bits for the inv table.
    """

    def __init__(
        self,
        config,
        axis: Union[int, Tuple[int, ...]] = -1,
        stable: bool = True,
        input_scaler: float = 1.0,
        parallelization_factor: int = -1,
        quantize_input: bool = True,
        quantize_output: bool = False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        exp_in_quant_bits: Tuple[T, T, T] = None,
        exp_out_quant_bits: Tuple[T, T, T] = None,
        inv_in_quant_bits: Tuple[T, T, T] = None,
        inv_out_quant_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config

        self.supports_masking = True
        self._axis = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
        self.axes = self._axis
        self.stable = stable
        self.input_scaler = input_scaler
        self.parallelization_factor = parallelization_factor
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        self.epsilon = keras.backend.epsilon()

        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if out_quant_bits is not None:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        else:
            # The softmax output is in [0, 1] -> unsigned by default.
            self.k_output = 0
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits

        self.in_quant_bits = in_quant_bits
        self.out_quant_bits = out_quant_bits
        self.exp_in_quant_bits = exp_in_quant_bits
        self.exp_out_quant_bits = exp_out_quant_bits
        self.inv_in_quant_bits = inv_in_quant_bits
        self.inv_out_quant_bits = inv_out_quant_bits

        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.dynamic_data = config.quantization_parameters.dynamic_data_quantization
        self.is_pretraining = True

        i_data = config.quantization_parameters.default_data_integer_bits
        f_data = config.quantization_parameters.default_data_fractional_bits
        k_data = config.quantization_parameters.default_data_keep_negatives
        # Table outputs (exp, 1/x) are non-negative, so default them to unsigned (k=0).
        exp_in_quant_bits = exp_in_quant_bits if exp_in_quant_bits is not None else (k_data, i_data, f_data)
        exp_out_quant_bits = exp_out_quant_bits if exp_out_quant_bits is not None else (0, i_data, f_data)
        inv_in_quant_bits = inv_in_quant_bits if inv_in_quant_bits is not None else (k_data, i_data, f_data)
        inv_out_quant_bits = inv_out_quant_bits if inv_out_quant_bits is not None else (0, i_data, f_data)

        def _exp(x):
            if self.stable:
                return ops.exp(-x * self.input_scaler)
            return ops.exp(x * self.input_scaler)

        def _inv(x):
            return 1.0 / (x + self.epsilon)

        # exp/inv tables are PQActivations whose activation is the callable above.
        self.exp_table = PQActivation(
            config,
            _exp,
            in_quant_bits=exp_in_quant_bits,
            out_quant_bits=exp_out_quant_bits,
            quantize_input=stable,
            quantize_output=True,
            # When not stable the exp table has no input quantizer; its LUT cost is then
            # folded into this layer's ebops (mirrors HGQ's enable_ebops=stable).
            enable_ebops=stable,
        )
        self.inv_table = PQActivation(
            config,
            _inv,
            in_quant_bits=inv_in_quant_bits,
            out_quant_bits=inv_out_quant_bits,
            quantize_input=True,
            quantize_output=True,
        )

    def build(self, input_shape):
        ndim = len(input_shape)
        self.axes = tuple(sorted(a if a >= 0 else a + ndim for a in self._axis))
        self.input_shape = (1,) + tuple(input_shape[1:])

        def _data_quantizer(k, i, f):
            return Quantizer(
                k=k,
                i=i,
                f=f,
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_data=True,
                is_heterogeneous=self.use_hgq,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
                dynamic_data=self.dynamic_data,
            )

        self.input_quantizer = _data_quantizer(self.k_input, self.i_input, self.f_input)
        self.output_quantizer = _data_quantizer(self.k_output, self.i_output, self.f_output)
        if self.use_hgq:
            self.input_quantizer.build(input_shape)
            self.output_quantizer.build(input_shape)
        super().build(input_shape)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def post_pre_train_function(self):
        self.is_pretraining = False
        if self.quantize_input:
            self.input_quantizer.post_pre_train_function()
        if self.quantize_output:
            self.output_quantizer.post_pre_train_function()
        self.exp_table.post_pre_train_function()
        self.inv_table.post_pre_train_function()

    def ebops(self):
        # Cost of the softmax-specific arithmetic (subtraction, accumulation, multiplication)
        # plus the exp/inv lookup-table costs. Unlike the torch backend (whose get_ebops walks
        # model.modules() recursively), keras get_ebops iterates only top-level model.layers,
        # so the nested table costs are accounted for here to avoid undercounting.
        shape = self.input_shape
        accum_shape = tuple(1 if i in self.axes else s for i, s in enumerate(shape))
        max_instance = prod(accum_shape)
        n_instance = self.parallelization_factor if self.parallelization_factor > 0 else max_instance
        factor = n_instance / max_instance

        inp_bits = self.input_quantizer.get_total_bits(shape)
        exp_bits = self.exp_table.output_quantizer.get_total_bits(shape)
        inv_bits = self.inv_table.output_quantizer.get_total_bits(accum_shape)

        substract_ebops = ops.sum(inp_bits) if self.stable else 0.0
        accum_ebops = ops.sum(exp_bits) - ops.sum(ops.min(exp_bits, axis=self.axes))
        mult_ebops = ops.sum(exp_bits * inv_bits)

        ebops = substract_ebops + accum_ebops + mult_ebops
        if not self.stable:
            # exp table input quantization is disabled (enable_ebops=False there), so its
            # lookup-table input cost is accounted for here instead, avoiding double counting.
            ebops = ebops + ops.sum((2.0**inp_bits) * exp_bits) * 1e-4
        ebops = ebops * factor
        # Table LUT costs are not scaled by the parallelization factor (matches torch get_ebops,
        # which sums the table PQActivation ebops separately from this layer's softmax cost).
        return ebops + self.exp_table.ebops() + self.inv_table.ebops()

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def call(self, inputs, training=None, mask=None):
        if self.quantize_input and self.enable_quantization:
            inputs = self.input_quantizer(inputs, training=training)

        if self.stable:
            inputs = ops.max(inputs, axis=self.axes, keepdims=True) - inputs

        exp_inp = self.exp_table(inputs)

        if mask is not None:
            exp_inp = ops.cast(mask, exp_inp.dtype) * exp_inp

        sums = ops.sum(exp_inp, axis=self.axes, keepdims=True)
        divisor = self.inv_table(sums)

        out = exp_inp * divisor
        if self.quantize_output and self.enable_quantization:
            out = self.output_quantizer(out, training=training)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "axis": self.axes,
                "stable": self.stable,
                "input_scaler": self.input_scaler,
                "parallelization_factor": self.parallelization_factor,
                "quantize_input": self.quantize_input,
                "quantize_output": self.quantize_output,
                "in_quant_bits": self.in_quant_bits,
                "out_quant_bits": self.out_quant_bits,
                "exp_in_quant_bits": self.exp_in_quant_bits,
                "exp_out_quant_bits": self.exp_out_quant_bits,
                "inv_in_quant_bits": self.inv_in_quant_bits,
                "inv_out_quant_bits": self.inv_out_quant_bits,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("exp_table", None)
        config.pop("inv_table", None)
        config.pop("input_quantizer", None)
        config.pop("output_quantizer", None)
        return cls(**config)
