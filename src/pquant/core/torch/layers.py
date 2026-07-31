import math
import typing
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t

from pquant.core.torch.activations import PQActivation, PQSoftmax
from pquant.core.torch.quantizer import Quantizer
from pquant.core.torch.utils import get_pruning_layer

if typing.TYPE_CHECKING:
    pass  # noqa: 401

T = TypeVar("T")


def _resolve_data_quant_bits(quant_bits, config):
    """Return (k, i, f) from an explicit tuple, or the config's data-lane defaults."""
    if quant_bits is not None:
        return quant_bits
    parameters = config.quantization_parameters
    return (
        parameters.default_data_keep_negatives,
        parameters.default_data_integer_bits,
        parameters.default_data_fractional_bits,
    )


def _resolve_weight_quant_bits(quant_bits, config):
    """Return (k, i, f) from an explicit tuple, or the config's weight defaults."""
    if quant_bits is not None:
        return quant_bits
    parameters = config.quantization_parameters
    return (
        parameters.default_weight_keep_negatives,
        parameters.default_weight_integer_bits,
        parameters.default_weight_fractional_bits,
    )


class PQWeightBiasBase(nn.Module):
    def __init__(
        self,
        config,
        layer_type,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.k_input, self.i_input, self.f_input = _resolve_data_quant_bits(in_quant_bits, config)
        self.k_weight, self.i_weight, self.f_weight = _resolve_weight_quant_bits(weight_quant_bits, config)
        self.k_bias, self.i_bias, self.f_bias = _resolve_weight_quant_bits(bias_quant_bits, config)
        self.k_output, self.i_output, self.f_output = _resolve_data_quant_bits(out_quant_bits, config)

        self.pruning_layer = get_pruning_layer(config=config, layer_type=layer_type)
        self.pruning_method = config.pruning_parameters.pruning_method
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

        self.pruning_first = config.training_parameters.pruning_first
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.round_mode = config.quantization_parameters.round_mode
        self.overflow_mode_parameters = config.quantization_parameters.overflow_mode_parameters
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_pruning = enable_pruning if enable_pruning is not None else config.pruning_parameters.enable_pruning
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.granularity = config.quantization_parameters.granularity
        self.weight_quant_granularity = (
            weight_quant_granularity if weight_quant_granularity is not None else self.granularity
        )
        self.in_quant_granularity = in_quant_granularity if in_quant_granularity is not None else self.granularity
        self.bias_quant_granularity = bias_quant_granularity if bias_quant_granularity is not None else self.granularity
        self.out_quant_granularity = out_quant_granularity if out_quant_granularity is not None else self.granularity
        self.register_buffer("final_compression_done", torch.tensor(False))
        self.built = False
        self.parallelization_factor = -1
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.input_shape = None
        self.is_pretraining = True
        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.saved_outputs = []
        self.config = config

    def _check_is_built(self, input_shape):
        if self.built:
            return
        # Quantizer creation is delayed until the first forward so custom i/f bits set
        # after __init__ are picked up.
        if self.quantize_input:
            self.input_quantizer = Quantizer(
                k=self.k_input,
                i=self.i_input,
                f=self.f_input,
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
                granularity=self.in_quant_granularity,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
                dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
            )
        self.weight_quantizer = Quantizer(
            k=self.k_weight,
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=self.weight_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="weight",
            shape=self._weight.shape,
        )
        self.bias_quantizer = Quantizer(
            k=self.k_bias,
            i=self.i_bias,
            f=self.f_bias,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=self.bias_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="bias",
            shape=None if self._bias is None else self._bias.shape,
        )
        if self.quantize_output:
            self.output_quantizer = Quantizer(
                k=self.k_output,
                i=self.i_output,
                f=self.f_output,
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
                granularity=self.out_quant_granularity,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
                dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
            )

        self.n_parallel = math.prod(tuple(input_shape)[1:-1])
        self.parallelization_factor = self.parallelization_factor if self.parallelization_factor > 0 else self.n_parallel
        self.built = True
        self.input_shape = (1,) + input_shape[1:]

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def apply_final_compression(self):
        pass

    def _register_compressed_parameters(self, bias):
        """Store the wrapped layer's weight/bias as `_weight`/`_bias`; the `weight`/`bias`
        properties then return their pruned and quantized views."""
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.pruning_layer.build(self._weight.shape)

    def post_pre_train_function(self):
        self.is_pretraining = False
        if self.pruning_layer is not None:
            self.pruning_layer.post_pre_train_function()

    def _save_weights(self):
        self.init_weight = self._weight.clone()

    def _rewind_weights(self):
        if not hasattr(self, "init_weight"):
            return
        self._weight.data = self.init_weight.clone()

    def ebops(self):
        return 0.0

    def _masked_weight_bits(self, bw_ker):
        """Zero the bit counts of weights that are pruned away or below the quantization step size."""
        bw_ker = bw_ker * self.pruning_layer.get_hard_mask()
        _, _, f = self.get_weight_quantization_bits()
        quantization_step_size = 2 ** (-f - 1)
        step_size_mask = (torch.abs(self._weight) > quantization_step_size).float()
        return bw_ker * step_size_mask

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        if self._bias is not None:
            loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def quantize(self, x, quantizer):
        if self.enable_quantization and not self._is_fitcompress_pretraining():
            return quantizer(x) if x is not None else x
        return x

    def _prune(self, weight):
        if self.enable_pruning:
            weight = self.pruning_layer(weight)
        return weight

    def _is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    def pre_forward(self, x):
        self._check_is_built(x.shape)
        if self.post_fitcompress_calibration:
            self.saved_inputs.append(x)
            return x
        if self.quantize_input:
            x = self.quantize(x, self.input_quantizer)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        return x

    def _post_forward(self, x):
        if self.post_fitcompress_calibration:
            self.saved_outputs.append(x)
            return x
        if self.quantize_output:
            x = self.quantize(x, self.output_quantizer)
        if self.pruning_method == "activation_pruning":
            self.pruning_layer.collect_output(x, self.training)
        return x


class PQDense(PQWeightBiasBase, nn.Linear):
    def __init__(
        self,
        config,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        device=None,
        dtype=None,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            config=config,
            layer_type="linear",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            enable_pruning=enable_pruning,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            **kwargs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self._register_compressed_parameters(bias)

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(self._weight.shape)
        if include_mask:
            bw_ker = self._masked_weight_bits(bw_ker)
        ebops = torch.sum(F.linear(bw_inp, bw_ker))
        if self._bias is not None:
            bw_bias = self.bias_quantizer.get_total_bits(self._bias.shape)
            size = float(math.prod(self.input_shape[:-1]) * self.out_features)
            ebops += torch.mean(bw_bias) * size
        ebops = ebops * self.parallelization_factor / self.n_parallel
        return ebops

    @property
    def weight(self):
        if self.final_compression_done or self._is_fitcompress_pretraining():
            return self._weight
        if self.pruning_first:
            weight = self._prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self._prune(weight)

    @property
    def bias(self):
        if self.final_compression_done or self._is_fitcompress_pretraining():
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done.fill_(True)

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self._post_forward(x)
        return x

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            f"in_features={self.in_features} "
            f"out_features={self.out_features} "
            f"bias={self._bias is not None} "
            f"quantize_input={self.quantize_input} "
            f"quantize_output={self.quantize_output} "
        )


class PQConvBase(PQWeightBiasBase):
    """Pruning/quantization behavior shared by PQConv1d and PQConv2d."""

    conv_bits_fn = None  # F.conv1d / F.conv2d, set by the subclasses

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(self._weight.shape)
        if include_mask:
            bw_ker = self._masked_weight_bits(bw_ker)
        if self.parallelization_factor < 0:
            ebops = torch.sum(
                self.conv_bits_fn(bw_inp, bw_ker, stride=self.stride, padding=self.padding, dilation=self.dilation)
            )
        else:
            spatial_axes = tuple(range(2, 2 + len(self.kernel_size)))
            bw_inp = torch.amax(bw_inp, dim=(0,) + spatial_axes)
            bw_ker = torch.sum(bw_ker, dim=spatial_axes)
            ebops = torch.sum(bw_inp[None, :] * bw_ker)
        if self._bias is not None:
            size = float(math.prod(self.input_shape))
            bw_bias = self.bias_quantizer.get_total_bits(self._bias.shape)
            ebops += torch.mean(bw_bias) * size
        return ebops

    @property
    def weight(self):
        if self.final_compression_done:
            return self._weight
        if self.pruning_first:
            weight = self._prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        weight = self.quantize(self._weight, self.weight_quantizer)
        return self._prune(weight)

    @property
    def bias(self):
        if self.final_compression_done:
            return self._bias
        return self.quantize(self._bias, self.bias_quantizer)

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done.fill_(True)

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self._post_forward(x)
        return x

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self._bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += ", quantize_input={quantize_input}"
        s += ", quantize_output={quantize_output}"
        return s.format(**self.__dict__)


class PQConv2d(PQConvBase, nn.Conv2d):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            enable_pruning=enable_pruning,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            **kwargs,
        )
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self._register_compressed_parameters(bias)

    conv_bits_fn = staticmethod(F.conv2d)


class PQConv1d(PQConvBase, nn.Conv1d):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            enable_pruning=enable_pruning,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            **kwargs,
        )
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self._register_compressed_parameters(bias)

    conv_bits_fn = staticmethod(F.conv1d)


def add_compression_layers(model, config, input_shape=None, add_missing_quantizers=False):
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    model = _add_quantized_activations_to_model_layer(model, config)
    model = _add_pruning_to_model(model, config)
    if add_missing_quantizers:
        # Imported here (not at module top) to avoid a circular import: tracing.py
        # imports the layer classes defined in this module.
        from pquant.core.torch.tracing import check_quantization

        model = check_quantization(model, add_missing_quantizers=True, config=config)
    model.to(device)
    if input_shape is not None:
        model(torch.rand(input_shape).to(device))
    return model


class PQAvgPoolBase(nn.Module):
    def __init__(
        self,
        config,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k_input, self.i_input, self.f_input = _resolve_data_quant_bits(in_quant_bits, config)
        self.k_output, self.i_output, self.f_output = _resolve_data_quant_bits(out_quant_bits, config)
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.config = config
        self.is_pretraining = True
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        # Optional per-quantizer granularity override; None → inherit config granularity.
        granularity = config.quantization_parameters.granularity
        self.in_quant_granularity = in_quant_granularity if in_quant_granularity is not None else granularity
        self.out_quant_granularity = out_quant_granularity if out_quant_granularity is not None else granularity

    def build(self, input_shape):
        self.input_quantizer = Quantizer(
            k=self.k_input,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=self.in_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
        )
        self.output_quantizer = Quantizer(
            k=self.k_output,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=self.out_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
        )
        self.input_shape = (1,) + input_shape[1:]

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def post_pre_train_function(self):
        self.is_pretraining = False

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        return torch.sum(bw_inp)

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def _is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    def _pre_pooling(self, x):
        if not hasattr(self, "input_quantizer"):
            self.build(x.shape)
        if self._is_fitcompress_pretraining():
            if self.post_fitcompress_calibration:
                # Save inputs
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized pooling
            return x
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x)
        return x

    def _post_pooling(self, x):
        if self.quantize_output and self.enable_quantization and not self._is_fitcompress_pretraining():
            x = self.output_quantizer(x)
        return x

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, quantize_input={self.quantize_input}, quantize_output={self.quantize_output}"  # noqa: 501


class PQAvgPool1d(PQAvgPoolBase, nn.AvgPool1d):
    def __init__(
        self,
        config,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            out_quant_bits=out_quant_bits,
            in_quant_granularity=in_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            **kwargs,
        )

    def forward(self, x):
        x = self._pre_pooling(x)
        x = super().forward(x)
        x = self._post_pooling(x)
        return x


class PQAvgPool2d(PQAvgPoolBase, nn.AvgPool2d):
    def __init__(
        self,
        config,
        kernel_size: _size_2_t,
        stride: _size_2_t = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            out_quant_bits=out_quant_bits,
            in_quant_granularity=in_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            **kwargs,
        )

    def forward(self, x):
        x = self._pre_pooling(x)
        x = super().forward(x)
        x = self._post_pooling(x)
        return x


class PQBatchNormBase:
    """Quantization behavior shared by PQBatchNorm1d and PQBatchNorm2d."""

    def _init_quantization(
        self,
        config,
        quantize_input,
        in_quant_bits,
        weight_quant_bits,
        bias_quant_bits,
        in_quant_granularity,
        weight_quant_granularity,
        bias_quant_granularity,
    ):
        self.k_input, self.i_input, self.f_input = _resolve_data_quant_bits(in_quant_bits, config)
        self.k_weight, self.i_weight, self.f_weight = _resolve_weight_quant_bits(weight_quant_bits, config)
        self.k_bias, self.i_bias, self.f_bias = _resolve_weight_quant_bits(bias_quant_bits, config)
        self.overflow_mode_parameters = config.quantization_parameters.overflow_mode_parameters
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.config = config
        self.quantize_input = quantize_input
        granularity = config.quantization_parameters.granularity
        self.in_quant_granularity = in_quant_granularity if in_quant_granularity is not None else granularity
        self.weight_quant_granularity = weight_quant_granularity if weight_quant_granularity is not None else granularity
        self.bias_quant_granularity = bias_quant_granularity if bias_quant_granularity is not None else granularity
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if self.bias is not None:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.built = False
        self.register_buffer("final_compression_done", torch.tensor(False))
        self.is_pretraining = True
        self.post_fitcompress_calibration = False
        self.saved_inputs = []

    def _check_is_built(self, input_shape):
        if self.built:
            return
        self.built = True
        self.input_quantizer = Quantizer(
            k=self.k_input,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=self.in_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
        )
        self.weight_quantizer = Quantizer(
            k=self.k_weight,
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=self.weight_quant_granularity,
            place="weight",
            shape=self._weight.shape,
        )
        self.bias_quantizer = Quantizer(
            k=self.k_bias,
            i=self.i_bias,
            f=self.f_bias,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=self.bias_quant_granularity,
            place="bias",
            shape=None if self._bias is None else self._bias.shape,
        )
        if self.use_hgq:
            self.input_quantizer.quantizer.build(input_shape)
        shape = [1] * len(input_shape)
        shape[1] = input_shape[1]
        self._shape = tuple(shape)
        self.input_shape = (1,) + input_shape[1:]

    def apply_final_compression(self):
        self._weight.data = self.weight
        self._bias.data = self.bias
        self.final_compression_done.fill_(True)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def _is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    @property
    def weight(self):
        if self.enable_quantization and not self.final_compression_done and not self._is_fitcompress_pretraining():
            return self.weight_quantizer(self._weight)
        return self._weight

    @property
    def bias(self):
        if self.enable_quantization and not self.final_compression_done and not self._is_fitcompress_pretraining():
            return self.bias_quantizer(self._bias)
        return self._bias

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = torch.reshape(self.weight_quantizer.get_total_bits(self.running_mean.shape), self._shape)
        bw_bias = torch.reshape(self.bias_quantizer.get_total_bits(self.running_mean.shape), self._shape)
        size = float(math.prod(self.input_shape))
        ebops = torch.sum(bw_inp * bw_ker) + torch.mean(bw_bias) * size
        return ebops

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        return loss

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_is_built(input.shape)
        if self.quantize_input and self.enable_quantization:
            if not self._is_fitcompress_pretraining():
                input = self.input_quantizer(input)
            elif self.post_fitcompress_calibration:
                self.saved_inputs.append(input)
        return super().forward(input)


class PQBatchNorm2d(PQBatchNormBase, nn.BatchNorm2d):
    def __init__(
        self,
        config,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        quantize_input=True,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        weight_quant_granularity=None,
        bias_quant_granularity=None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        self._init_quantization(
            config,
            quantize_input,
            in_quant_bits,
            weight_quant_bits,
            bias_quant_bits,
            in_quant_granularity,
            weight_quant_granularity,
            bias_quant_granularity,
        )


class PQBatchNorm1d(PQBatchNormBase, nn.BatchNorm1d):
    def __init__(
        self,
        config,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        quantize_input=True,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        weight_quant_granularity=None,
        bias_quant_granularity=None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        self._init_quantization(
            config,
            quantize_input,
            in_quant_bits,
            weight_quant_bits,
            bias_quant_bits,
            in_quant_granularity,
            weight_quant_granularity,
            bias_quant_granularity,
        )


class PQLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        config,
        normalized_shape: int | tuple[int, ...] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        weight_quant_granularity=None,
        bias_quant_granularity=None,
    ):
        try:
            super().__init__(normalized_shape, eps, elementwise_affine, bias, device=device, dtype=dtype)
        except TypeError:
            # Older torch versions don't accept the bias kwarg
            super().__init__(normalized_shape, eps, elementwise_affine, device=device, dtype=dtype)
        self.k_input, self.i_input, self.f_input = _resolve_data_quant_bits(in_quant_bits, config)
        self.k_output, self.i_output, self.f_output = _resolve_data_quant_bits(out_quant_bits, config)
        self.k_weight, self.i_weight, self.f_weight = _resolve_weight_quant_bits(weight_quant_bits, config)
        self.k_bias, self.i_bias, self.f_bias = _resolve_weight_quant_bits(bias_quant_bits, config)
        self.overflow_mode_parameters = config.quantization_parameters.overflow_mode_parameters
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.round_mode = config.quantization_parameters.round_mode
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self.config = config
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        granularity = config.quantization_parameters.granularity
        self.in_quant_granularity = in_quant_granularity if in_quant_granularity is not None else granularity
        self.out_quant_granularity = out_quant_granularity if out_quant_granularity is not None else granularity
        self.weight_quant_granularity = weight_quant_granularity if weight_quant_granularity is not None else granularity
        self.bias_quant_granularity = bias_quant_granularity if bias_quant_granularity is not None else granularity
        if self.weight is not None:
            self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
            self.register_parameter("_weight", self._weight)
        else:
            self.register_parameter("_weight", None)
        if self.bias is not None:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.built = False
        self.register_buffer("final_compression_done", torch.tensor(False))
        self.is_pretraining = True
        self.post_fitcompress_calibration = False
        self.saved_inputs = []

    def _check_is_built(self, input_shape):
        if self.built:
            return
        self.built = True
        self.input_quantizer = Quantizer(
            k=self.k_input,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=self.in_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
        )
        self.output_quantizer = Quantizer(
            k=self.k_output,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=self.out_quant_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.config.quantization_parameters.dynamic_data_quantization,
        )
        self.weight_quantizer = Quantizer(
            k=self.k_weight,
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=self.weight_quant_granularity,
            place="weight",
            shape=self._weight.shape,
        )
        self.bias_quantizer = Quantizer(
            k=self.k_bias,
            i=self.i_bias,
            f=self.f_bias,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=self.bias_quant_granularity,
            place="bias",
            shape=None if self._bias is None else self._bias.shape,
        )
        if self.use_hgq:
            self.input_quantizer.quantizer.build(input_shape)
            self.output_quantizer.quantizer.build(input_shape)
        self.input_shape = (1,) + tuple(input_shape[1:])

    def apply_final_compression(self):
        if self._weight is not None:
            self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done.fill_(True)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def _is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    @property
    def weight(self):
        if self._weight is None:
            return None
        if self.enable_quantization and not self.final_compression_done and not self._is_fitcompress_pretraining():
            return self.weight_quantizer(self._weight)
        return self._weight

    @property
    def bias(self):
        if self._bias is None:
            return None
        if self.enable_quantization and not self.final_compression_done and not self._is_fitcompress_pretraining():
            return self.bias_quantizer(self._bias)
        return self._bias

    def ebops(self):
        return 0.0

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self._weight is not None:
            loss += self.weight_quantizer.hgq_loss()
        if self._bias is not None:
            loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return loss

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_is_built(input.shape)
        if self.quantize_input and self.enable_quantization:
            if not self._is_fitcompress_pretraining():
                input = self.input_quantizer(input)
            elif self.post_fitcompress_calibration:
                self.saved_inputs.append(input)
        out = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.quantize_output and self.enable_quantization and not self._is_fitcompress_pretraining():
            out = self.output_quantizer(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={tuple(self.normalized_shape)}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"quantize_input={self.quantize_input}, quantize_output={self.quantize_output}"
        )


class PQMultiheadAttention(nn.Module):
    """Multi-head attention with quantization support, implemented without F.multihead_attention.

    Uses separate PQDense projections for Q, K, V, and output, and computes
    scaled dot-product attention manually.

    Args:
        config: PQuant configuration object.
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
        bias: Whether to add bias to projection layers.
        kdim: Key feature dimension (defaults to embed_dim).
        vdim: Value feature dimension (defaults to embed_dim).
        batch_first: If True, input/output tensors are (batch, seq, feature).
            If False (default), tensors are (seq, batch, feature).
        quantize_input: Whether to quantize the Q/K/V projection inputs (the MHA inputs).
        quantize_output: Whether to quantize the output projection's output (the MHA output).
            The q/k/v projection outputs and the out_proj input (the context) are always
            quantized, mirroring HGQ's QMultiHeadAttention.
        in_quant_bits: (k, i, f) bits for input quantization.
        weight_quant_bits: (k, i, f) bits for weight quantization.
        bias_quant_bits: (k, i, f) bits for bias quantization.
        out_quant_bits: (k, i, f) bits for output quantization.
        attn_quant_bits: (k, i, f) bits for the softmax output quantizer (the attention
            weights). The scores and context need no dedicated quantizers: the softmax's
            input quantizer and the output projection's input quantizer cover them.
    """

    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = False,
        quantize_input: bool = True,
        quantize_output: bool = False,
        approximate_softmax: bool = False,
        in_quant_bits: tuple[T, T, T] = None,
        weight_quant_bits: tuple[T, T, T] = None,
        bias_quant_bits: tuple[T, T, T] = None,
        out_quant_bits: tuple[T, T, T] = None,
        attn_quant_bits: tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        param_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout
        self.approximate_softmax = approximate_softmax
        self.scale = float(torch.tensor(self.head_dim**-0.5, dtype=torch.float32).item())
        self.softmax = PQSoftmax(config, -1, quantize_input=True, quantize_output=True, out_quant_bits=attn_quant_bits)

        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim

        self.in_quant_granularity = in_quant_granularity
        self.out_quant_granularity = out_quant_granularity
        self.param_quant_granularity = param_quant_granularity
        proj_kwargs = dict(
            bias=bias,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=param_quant_granularity,
            bias_quant_granularity=param_quant_granularity,
        )

        qkv_kwargs = dict(
            quantize_input=quantize_input, quantize_output=True, in_quant_granularity=in_quant_granularity, **proj_kwargs
        )
        self.q_proj = PQDense(config, embed_dim, embed_dim, enable_pruning=False, **qkv_kwargs)
        self.k_proj = PQDense(config, kdim, embed_dim, enable_pruning=False, **qkv_kwargs)
        self.v_proj = PQDense(config, vdim, embed_dim, enable_pruning=False, **qkv_kwargs)
        self.out_proj = PQDense(
            config,
            embed_dim,
            embed_dim,
            quantize_input=True,
            quantize_output=quantize_output,
            out_quant_granularity=out_quant_granularity,
            **proj_kwargs,
        )

        self.attn_dropout = None
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.is_pretraining = True

    def post_pre_train_function(self):
        # The projections, softmax and data quantizers are handled separately by the
        # recursive modules() walk in post_pretrain_functions.
        self.is_pretraining = False

    def _head_bits(self, proj, seq_len):
        """Bitwidths of a projection's output, in per-head layout (1, H, seq, head_dim)."""
        bw = proj.output_quantizer.get_total_bits((1, seq_len, self.embed_dim))
        return bw.reshape(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def attention_ebops(self):
        """EBOPs of the q @ k^T and attn @ v einsums (mirrors HGQ's QMultiHeadAttention)."""
        attn_shape = self.softmax.input_shape  # (1, H, T, S), stored when the softmax was built
        query_len, key_len = attn_shape[2], attn_shape[3]
        bw_q = self._head_bits(self.q_proj, query_len)
        bw_k = self._head_bits(self.k_proj, key_len)
        bw_v = self._head_bits(self.v_proj, key_len)
        bw_attn = self.softmax.output_quantizer.get_total_bits(attn_shape)
        ebops_qk = torch.einsum("bhtd,bhsd->", bw_q, bw_k)
        ebops_av = torch.einsum("bhts,bhsd->", bw_attn, bw_v)
        return ebops_qk + ebops_av

    def ebops(self):
        # Only the attention einsum costs are this module's own: the projections,
        # softmax and lookup tables are counted by get_ebops's recursive modules() walk.
        return self.attention_ebops()

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return torch.tensor(0.0)
        return self.hgq_beta * self.attention_ebops()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.batch_first:
            # (T, B, E) -> (B, T, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T = query.shape[0], query.shape[1]
        S = key.shape[1]

        q = self.q_proj(query)  # (B, T, E)
        k = self.k_proj(key)  # (B, S, E)
        v = self.v_proj(value)  # (B, S, E)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.view(B, self.num_heads, T, S)
            attn_scores = attn_scores + attn_mask

        mask = None
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        # The softmax's own input/output quantizers handle the scores and the attention weights;
        attn_weights = self.softmax(attn_scores, mask=mask)

        if self.attn_dropout is not None and self.training:
            attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)

        # Merge heads: (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        if need_weights:
            # Average attention weights over heads: (B, T, S)
            return out, attn_weights.mean(dim=1)
        return out, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}, batch_first={self.batch_first}"
        )


# Prunable leaf layers, larger layers like MHA not included as they consist of these layers
LAYERS_WITH_PRUNING_LAYER = (PQConv2d, PQConv1d, PQDense)
PQ_MODULES = (
    PQConv2d,
    PQConv1d,
    PQDense,
    PQActivation,
    PQBatchNorm2d,
    PQBatchNorm1d,
    PQLayerNorm,
    PQAvgPoolBase,
    PQSoftmax,
    PQMultiheadAttention,
    Quantizer,
)


def _apply_quant_bits(layer, section_config, suffix):
    """Copy keep_negatives/integer/fractional bits from one config section onto layer.{k,i,f}_{suffix}."""
    if "keep_negatives" in section_config:
        setattr(layer, f"k_{suffix}", torch.tensor(section_config["keep_negatives"]))
    if "integer_bits" in section_config:
        setattr(layer, f"i_{suffix}", torch.tensor(section_config["integer_bits"]))
    if "fractional_bits" in section_config:
        setattr(layer, f"f_{suffix}", torch.tensor(section_config["fractional_bits"]))


def _add_layer_specific_quantization_to_model(name, layer, config):
    layer_config = config.quantization_parameters.layer_specific.get(name)
    if layer_config is None:
        return layer
    if isinstance(layer, (PQWeightBiasBase, PQLayerNorm)):
        sections = ("weight", "bias", "input", "output")
    elif isinstance(layer, (PQBatchNormBase)):
        sections = ("weight", "bias", "input")
    elif isinstance(layer, (PQAvgPoolBase, PQActivation)):
        sections = ("input", "output")
    else:
        return layer
    for section in sections:
        if section not in layer_config:
            continue
        _apply_quant_bits(layer, layer_config[section], section)
        if section in ("input", "output") and "quantize" in layer_config[section]:
            setattr(layer, f"quantize_{section}", layer_config[section]["quantize"])
    return layer


def _add_quantized_activations_to_model_layer(module, config, prefix=""):
    if not config.quantization_parameters.enable_quantization:
        return module
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    # Replaces ReLU and Tanh layers with quantized versions
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        i = config.quantization_parameters.default_data_integer_bits
        f = config.quantization_parameters.default_data_fractional_bits
        if layer.__class__ in [nn.ReLU]:
            # For ReLU, if using default values, add 1 bit since values are unsigned.
            # Otherwise user provides bits. TODO: Find better way to do this
            f = config.quantization_parameters.default_data_fractional_bits + 1
            relu = PQActivation(
                config,
                "relu",
                in_quant_bits=(0, i, f),
                out_quant_bits=(0, i, f),
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            relu = _add_layer_specific_quantization_to_model(full_name, relu, config)
            setattr(module, name, relu)
        elif layer.__class__ in [nn.Tanh]:
            type_of_tanh = "tanh" if config.quantization_parameters.use_real_tanh else "hard_tanh"
            tanh = PQActivation(
                config,
                type_of_tanh,
                in_quant_bits=(0, i, f),
                out_quant_bits=(0, i, f),
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            tanh = _add_layer_specific_quantization_to_model(full_name, tanh, config)
            setattr(module, name, tanh)
        elif layer.__class__ == nn.AvgPool1d:
            new_layer = PQAvgPool1d(
                config,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.ceil_mode,
                layer.count_include_pad,
                quantize_input,
                quantize_output,
            )
            new_layer = _add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.AvgPool2d:
            new_layer = PQAvgPool2d(
                config,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.ceil_mode,
                layer.count_include_pad,
                layer.divisor_override,
                quantize_input,
                quantize_output,
            )
            new_layer = _add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        elif layer.__class__ in (nn.BatchNorm1d, nn.BatchNorm2d):
            pq_batchnorm = PQBatchNorm1d if layer.__class__ is nn.BatchNorm1d else PQBatchNorm2d
            new_layer = pq_batchnorm(
                config,
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                quantize_input=quantize_input,
            )
            new_layer = _add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.LayerNorm:
            ln_kwargs = dict(
                normalized_shape=layer.normalized_shape,
                eps=layer.eps,
                elementwise_affine=layer.elementwise_affine,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            new_layer = PQLayerNorm(config, **ln_kwargs)
            if layer.elementwise_affine:
                if layer.weight is not None and new_layer._weight is not None:
                    new_layer._weight.data.copy_(layer.weight.data)
                if layer.bias is not None and new_layer._bias is not None:
                    new_layer._bias.data.copy_(layer.bias.data)
            new_layer = _add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        else:
            layer = _add_quantized_activations_to_model_layer(layer, config, full_name)
    return module


def _disable_pruning_from_layers(name, layer, config):
    if isinstance(layer, LAYERS_WITH_PRUNING_LAYER) and name in config.pruning_parameters.disable_pruning_for_layers:
        layer.enable_pruning = False
    return layer


def _replace_layer_with_pq_layer(module, name, full_name, layer, sparse_layer, config):
    sparse_layer._weight.data = layer.weight.data
    if layer.bias is not None:
        sparse_layer._bias.data = layer.bias.data
    sparse_layer = _add_layer_specific_quantization_to_model(full_name, sparse_layer, config)
    sparse_layer = _disable_pruning_from_layers(full_name, sparse_layer, config)
    setattr(module, name, sparse_layer)


def _add_pruning_to_model(module, config, prefix=""):
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if layer.__class__ is nn.Linear:
            sparse_layer = PQDense(
                config, layer.in_features, layer.out_features, layer.bias is not None, quantize_input, quantize_output
            )
            _replace_layer_with_pq_layer(module, name, full_name, layer, sparse_layer, config)
        elif layer.__class__ in (nn.Conv1d, nn.Conv2d):
            pq_conv = PQConv1d if layer.__class__ is nn.Conv1d else PQConv2d
            sparse_layer = pq_conv(
                config,
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                layer.bias is not None,
                layer.padding_mode,
                layer.weight.device,
                layer.weight.dtype,
                quantize_input,
                quantize_output,
            )
            _replace_layer_with_pq_layer(module, name, full_name, layer, sparse_layer, config)
        elif layer.__class__ is nn.MultiheadAttention:
            if layer.bias_k is not None or layer.add_zero_attn:
                raise ValueError(f"add_bias_kv/add_zero_attn are not supported by PQMultiheadAttention ({full_name})")
            sparse_layer = PQMultiheadAttention(
                config,
                embed_dim=layer.embed_dim,
                num_heads=layer.num_heads,
                dropout=layer.dropout,
                bias=layer.in_proj_bias is not None,
                kdim=layer.kdim,
                vdim=layer.vdim,
                batch_first=layer.batch_first,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            if layer._qkv_same_embed_dim:
                q_w, k_w, v_w = layer.in_proj_weight.chunk(3)
            else:
                q_w, k_w, v_w = layer.q_proj_weight, layer.k_proj_weight, layer.v_proj_weight
            if layer.in_proj_bias is not None:
                biases = (*layer.in_proj_bias.chunk(3), layer.out_proj.bias)
            else:
                biases = (None, None, None, None)
            projs = (sparse_layer.q_proj, sparse_layer.k_proj, sparse_layer.v_proj, sparse_layer.out_proj)
            for proj, weight, bias in zip(projs, (q_w, k_w, v_w, layer.out_proj.weight), biases):
                proj._weight.data = weight.data.clone()
                if bias is not None:
                    proj._bias.data = bias.data.clone()
            sparse_layer = _add_layer_specific_quantization_to_model(full_name, sparse_layer, config)
            setattr(module, name, sparse_layer)
        else:
            _add_pruning_to_model(layer, config, full_name)
    return module


def apply_final_compression(module):
    for layer in module.modules():
        if isinstance(layer, (PQWeightBiasBase, PQBatchNorm2d, PQBatchNorm1d, PQLayerNorm, Quantizer)):
            layer.apply_final_compression()
    return module


def call_post_round_functions(model, rewind, rounds, r):
    last_round = r == rounds - 1
    if rewind == "every-round":
        rewind_weights_functions(model)
    elif rewind == "post-training-stage" and last_round:
        rewind_weights_functions(model)
    elif not last_round:
        post_round_functions(model)


def _update_pruning_mask(layer):
    if layer.enable_pruning and hasattr(layer.pruning_layer, "update_mask"):
        layer.pruning_layer.update_mask(layer._weight)


def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
            _update_pruning_mask(layer)
        elif isinstance(layer, Quantizer):
            layer.post_epoch_function()


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer._save_weights()


def rewind_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer._rewind_weights()


def pre_finetune_functions(model):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config, train_loader=None, loss_function=None, input_shape=None):
    for layer in model.modules():
        if isinstance(layer, PQ_MODULES):
            # For FITCompress this must happen before the compression path search,
            # so quantization is already enabled during it.
            layer.post_pre_train_function()

    if config.fitcompress_parameters.enable_fitcompress:
        from pquant.core.torch.fit_compress import call_fitcompress  # noqa: 811

        config, pruning_mask_importance_scores = call_fitcompress(
            config, model, train_loader, loss_function, input_shape=input_shape
        )
        idx = 0
        for layer in model.modules():
            if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
                with torch.no_grad():
                    layer.pruning_layer.mask.data = pruning_mask_importance_scores[idx]
                layer.pruning_layer.pre_finetune_function()  # So mask is not updated during training anymore
                idx += 1
        return
    if config.pruning_parameters.pruning_method == "pdp" or (
        config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
    ):
        _pdp_setup(model, config)


def _pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = torch.concat(
        [layer._weight.flatten() for layer in model.modules() if isinstance(layer, LAYERS_WITH_PRUNING_LAYER)]
    )
    abs_global_weights = torch.abs(global_weights)
    global_weight_topk, _ = torch.topk(abs_global_weights, abs_global_weights.numel())
    threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * global_weight_topk.numel())]
    global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            weight_size = layer._weight.numel()
            w = torch.sum(global_weights_below_threshold[idx : idx + weight_size])
            layer.pruning_layer.init_r = w / weight_size
            layer.pruning_layer.sparsity = w / weight_size  # Wanda
            idx += weight_size


@torch.no_grad
def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            weight = layer.weight
            total_w += weight.numel()
            remaining_weights += torch.count_nonzero(weight)
        elif layer.__class__ in (nn.Conv2d, nn.Conv1d, nn.Linear):
            total_w += layer.weight.numel()
            remaining_weights += torch.count_nonzero(layer.weight)
    if total_w != 0:
        return remaining_weights / total_w
    return 0.0


def _is_training_stage(layer):
    return not (layer.pruning_layer._is_finetuning or layer.pruning_layer._is_pretraining)


def get_model_losses(model, losses):
    for layer in model.modules():
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            if layer.enable_pruning and _is_training_stage(layer) and not layer.use_fitcompress:
                losses += layer.pruning_layer.calculate_additional_loss()
            if layer.use_hgq:
                losses += layer.hgq_loss()
        elif isinstance(
            layer,
            (
                PQAvgPool1d,
                PQAvgPool2d,
                PQBatchNorm2d,
                PQBatchNorm1d,
                PQLayerNorm,
                PQActivation,
                PQSoftmax,
                PQMultiheadAttention,
            ),
        ):
            if layer.use_hgq:
                losses += layer.hgq_loss()
    return losses


def _create_default_layer_quantization_pruning_config(model, config):
    quant_params = config.quantization_parameters

    def data_section(quantize, integer_bits, fractional_bits):
        return {
            "quantize": quantize,
            "keep_negatives": quant_params.default_data_keep_negatives,
            "integer_bits": integer_bits,
            "fractional_bits": fractional_bits,
        }

    def param_section(integer_bits, fractional_bits):
        return {
            "keep_negatives": quant_params.default_weight_keep_negatives,
            "integer_bits": integer_bits,
            "fractional_bits": fractional_bits,
        }

    for name, layer in model.named_modules():
        if layer.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            layer_config = {
                "input": data_section(quant_params.quantize_input, 0, 7),
                "weight": param_section(0, 7),
                "output": data_section(quant_params.quantize_output, 0, 7),
            }
            if layer.bias is not None:
                layer_config["bias"] = param_section(0, 7)
            quant_params.layer_specific[name] = layer_config
            config.pruning_parameters.disable_pruning_for_layers.append(name)
        elif layer.__class__ in [nn.Tanh, nn.ReLU, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            quant_params.layer_specific[name] = {
                "input": data_section(quant_params.quantize_input, 0.0, 7.0),
                "output": data_section(quant_params.quantize_output, 0.0, 7.0),
            }
        elif layer.__class__ in [nn.BatchNorm2d]:
            quant_params.layer_specific[name] = {
                "input": data_section(quant_params.quantize_input, 0.0, 7.0),
                "weight": param_section(0, 7.0),
                "bias": param_section(0, 7.0),
            }
    return config


def populate_config_with_all_layers(model, config):
    return _create_default_layer_quantization_pruning_config(model, config)


def _remove_compression_layers(module, config):
    for name, layer in module.named_children():
        if isinstance(layer, PQDense):
            new_layer = nn.Linear(
                in_features=layer.in_features, out_features=layer.out_features, bias=layer.bias is not None
            )
            new_layer.weight.data.copy_(layer.weight)
            if new_layer.bias is not None:
                new_layer.bias.data.copy_(layer.bias)
            setattr(module, name, new_layer)
        elif isinstance(layer, (PQConv1d, PQConv2d)):
            bias_values = layer.bias
            conv = nn.Conv2d if isinstance(layer, PQConv2d) else nn.Conv1d
            new_layer = conv(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
                bias_values is not None,
                layer.padding_mode,
            )
            new_layer.weight.data.copy_(layer.weight)
            if new_layer.bias is not None:
                new_layer.bias.data.copy_(bias_values.data)
            setattr(module, name, new_layer)
        else:
            _remove_compression_layers(layer, config)
    return module


def post_training_prune(model, config, calibration_data):
    t_delta = config.pruning_parameters.t_delta
    config.pruning_parameters.t_start_collecting_batch = 0
    for i in range(t_delta):
        inputs = calibration_data[i]
        if i == 0:
            model = add_compression_layers(model, config, inputs.shape)
            post_pretrain_functions(model, config)
        model(inputs)
    return _remove_compression_layers(model, config)


def get_ebops(model, **kwargs):
    ebops = 0
    for m in model.modules():
        if isinstance(m, PQWeightBiasBase):
            ebops += m.ebops(include_mask=m.enable_pruning)
        elif isinstance(
            m,
            (
                PQAvgPoolBase,
                PQBatchNorm1d,
                PQBatchNorm2d,
                PQLayerNorm,
                PQActivation,
                PQSoftmax,
                PQMultiheadAttention,
            ),
        ):
            ebops += m.ebops()
    return ebops
