from math import prod
from typing import TypeVar

import torch
import torch.nn as nn
from torch import maximum, minimum, relu, tanh

from pquant.core.torch.quantizer import Quantizer

T = TypeVar("T")


def hard_sigmoid(x):
    """Computes hard_sigmoid function that saturates between 0 and 1."""
    x = torch.tensor(0.5) * x + torch.tensor(0.5)
    x = maximum(x, torch.tensor(0.0))
    x = minimum(x, torch.tensor(1.0))
    return x


def hard_tanh(x):
    """Computes hard_tanh function that saturates between -1 and 1."""
    return 2.0 * hard_sigmoid(x) - 1.0


activation_registry = {
    "relu": relu,
    "tanh": tanh,
    "hard_tanh": hard_tanh,
    "leaky_relu": nn.LeakyReLU(negative_slope=0.1015625),
    "gelu": nn.GELU(),
}


class PQActivation(nn.Module):
    def __init__(
        self,
        config,
        activation="relu",
        in_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        quantize_input=True,
        quantize_output=False,
        enable_ebops=True,
    ):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
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

        if isinstance(activation, str):
            self.activation_name = activation.lower()
            self.activation_function = activation_registry.get(self.activation_name)
        else:
            # An activation function/callable was passed directly instead of a registry key.
            self.activation_function = activation
            self.activation_name = getattr(activation, "__name__", activation.__class__.__name__).lower()

        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow_mode_parameters = config.quantization_parameters.overflow_mode_parameters
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

    def check_is_built(self, input_shape):
        if self.built:
            return
        self.built = True
        self.input_shape = (1,) + input_shape[1:]
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
            granularity=self.config.quantization_parameters.granularity,
        )
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
            granularity=self.config.quantization_parameters.granularity,
        )
        if self.use_hgq:
            self.input_quantizer.quantizer.build(input_shape)
            self.output_quantizer.quantizer.build(input_shape)

        if self.use_multiplier:
            self.multiplier = nn.Parameter(torch.tensor(-1.0), requires_grad=True)

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

    def ebops(self):
        if not self.enable_ebops:
            return torch.tensor(0.0)
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_out = self.output_quantizer.get_total_bits(self.input_shape)
        return torch.sum((2.0**bw_inp) * bw_out) * 1e-4  # type: ignore

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def pre_activation(self, x):
        if not self.use_hgq and self.use_multiplier and self.activation_name == "relu":
            x = x * 2 ** ((torch.round(self.multiplier) - self.multiplier).detach() + self.multiplier)
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x)
        return x

    def post_activation(self, x):
        if self.quantize_output and self.enable_quantization:
            return self.output_quantizer(x)
        return x

    def forward(self, x):
        self.check_is_built(x.shape)
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
                "i_input": float(self.i_input),
                "f_input": float(self.f_input),
                "i_output": float(self.i_output),
                "f_output": float(self.f_output),
            }
        )
        return config

    def extra_repr(self):
        return f"quantize_input = {self.quantize_input}, quantize_output = {self.quantize_output}"


class PQSoftmax(nn.Module):
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
        axis: int | tuple[int, ...] = -1,
        stable: bool = True,
        input_scaler: float = 1.0,
        parallelization_factor: int = -1,
        quantize_input: bool = True,
        quantize_output: bool = False,
        in_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        exp_in_quant_bits: tuple[T, T, T] = None,
        exp_out_quant_bits: tuple[T, T, T] = None,
        inv_in_quant_bits: tuple[T, T, T] = None,
        inv_out_quant_bits: tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config

        self._axis = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
        self.axes = self._axis
        self.stable = stable
        self.input_scaler = input_scaler
        self.parallelization_factor = parallelization_factor
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        self.epsilon = 1e-7

        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if out_quant_bits is not None:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        else:
            self.k_output = 0
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits

        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.is_pretraining = True
        self.built = False

        i_data = config.quantization_parameters.default_data_integer_bits
        f_data = config.quantization_parameters.default_data_fractional_bits
        k_data = config.quantization_parameters.default_data_keep_negatives
        exp_in_quant_bits = exp_in_quant_bits if exp_in_quant_bits is not None else (k_data, i_data, f_data)
        exp_out_quant_bits = exp_out_quant_bits if exp_out_quant_bits is not None else (0, i_data, f_data)
        inv_in_quant_bits = inv_in_quant_bits if inv_in_quant_bits is not None else (k_data, i_data, f_data)
        inv_out_quant_bits = inv_out_quant_bits if inv_out_quant_bits is not None else (0, i_data, f_data)

        def _exp(x):
            if self.stable:
                return torch.exp(-x * self.input_scaler)
            return torch.exp(x * self.input_scaler)

        def _inv(x):
            return 1.0 / (x + self.epsilon)

        self.exp_table = PQActivation(
            config,
            _exp,
            in_quant_bits=exp_in_quant_bits,
            out_quant_bits=exp_out_quant_bits,
            quantize_input=stable,
            quantize_output=True,
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

    def check_is_built(self, input_shape):
        if self.built:
            return
        self.built = True
        ndim = len(input_shape)
        self.axes = tuple(sorted(a if a >= 0 else a + ndim for a in self._axis))
        self.input_shape = (1,) + tuple(input_shape[1:])

        def _data_quantizer(k, i, f):
            return Quantizer(
                k=torch.tensor(k),
                i=torch.tensor(i),
                f=torch.tensor(f),
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
                dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
            )

        self.input_quantizer = _data_quantizer(self.k_input, self.i_input, self.f_input)
        self.output_quantizer = _data_quantizer(self.k_output, self.i_output, self.f_output)
        if self.use_hgq:
            self.input_quantizer.quantizer.build(input_shape)
            self.output_quantizer.quantizer.build(input_shape)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def post_pre_train_function(self):
        self.is_pretraining = False

    def ebops(self):
        shape = self.input_shape
        accum_shape = tuple(1 if i in self.axes else s for i, s in enumerate(shape))
        max_instance = prod(accum_shape)
        n_instance = self.parallelization_factor if self.parallelization_factor > 0 else max_instance
        factor = n_instance / max_instance

        inp_bits = self.input_quantizer.get_total_bits(shape)
        exp_bits = self.exp_table.output_quantizer.get_total_bits(shape)
        inv_bits = self.inv_table.output_quantizer.get_total_bits(accum_shape)

        substract_ebops = torch.sum(inp_bits) if self.stable else 0.0
        accum_ebops = torch.sum(exp_bits) - torch.sum(torch.amin(exp_bits, dim=self.axes))
        mult_ebops = torch.sum(exp_bits * inv_bits)

        ebops = substract_ebops + accum_ebops + mult_ebops
        if not self.stable:
            ebops = ebops + torch.sum((2.0**inp_bits) * exp_bits) * 1e-4
        return ebops * factor

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def forward(self, inputs, mask=None):
        self.check_is_built(inputs.shape)
        if self.quantize_input and self.enable_quantization:
            inputs = self.input_quantizer(inputs)

        if self.stable:
            inputs = torch.amax(inputs, dim=self.axes, keepdim=True) - inputs

        exp_inp = self.exp_table(inputs)

        if mask is not None:
            exp_inp = mask.to(exp_inp.dtype) * exp_inp

        sums = torch.sum(exp_inp, dim=self.axes, keepdim=True)
        divisor = self.inv_table(sums)

        out = exp_inp * divisor
        if self.quantize_output and self.enable_quantization:
            out = self.output_quantizer(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"axis={self.axes}, stable={self.stable}, input_scaler={self.input_scaler}, "
            f"quantize_input={self.quantize_input}, quantize_output={self.quantize_output}"
        )
