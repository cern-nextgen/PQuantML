import typing
from typing import Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace
from torch.nn.common_types import _size_1_t, _size_2_t

from pquant.core.torch.activations import PQActivation
from pquant.core.torch.quantizer import Quantizer
from pquant.core.torch.utils import get_pruning_layer

if typing.TYPE_CHECKING:
    from pquant.core.torch.fit_compress import call_fitcompress  # noqa: 401

from keras import ops

T = TypeVar("T")


class PQWeightBiasBase(nn.Module):
    def __init__(
        self,
        config,
        layer_type,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if weight_quant_bits is not None:
            self.k_weight, self.i_weight, self.f_weight = weight_quant_bits
        else:
            self.k_weight = config.quantization_parameters.default_weight_keep_negatives
            self.i_weight = config.quantization_parameters.default_weight_integer_bits
            self.f_weight = config.quantization_parameters.default_weight_fractional_bits
        if bias_quant_bits is not None:
            self.k_bias, self.i_bias, self.f_bias = bias_quant_bits
        else:
            self.k_bias = config.quantization_parameters.default_weight_keep_negatives
            self.i_bias = config.quantization_parameters.default_weight_integer_bits
            self.f_bias = config.quantization_parameters.default_weight_fractional_bits

        if out_quant_bits is not None:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        else:
            self.k_output = config.quantization_parameters.default_data_keep_negatives
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits

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
        self.final_compression_done = False
        self.built = False
        self.parallelization_factor = -1
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.input_shape = None
        self.is_pretraining = True
        self.post_fitcompress_calibration = False
        self.saved_inputs = []
        self.saved_outputs = []

    def check_is_built(self, input_shape):
        if self.built:
            return
        # Build function to delay quantizer creation until after custom i,f bits have been set
        if self.quantize_input:
            self.input_quantizer = Quantizer(
                k=torch.tensor(self.k_input),
                i=torch.tensor(self.i_input),
                f=torch.tensor(self.f_input),
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
            )
        self.weight_quantizer = Quantizer(
            k=torch.tensor(self.k_weight),
            i=torch.tensor(self.i_weight),
            f=torch.tensor(self.f_weight),
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            hgq_gamma=self.hgq_gamma,
            granularity=self.granularity,
            place="weight",
        )

        self.bias_quantizer = Quantizer(
            k=torch.tensor(self.k_bias),
            i=torch.tensor(self.i_bias),
            f=torch.tensor(self.f_bias),
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            hgq_gamma=self.hgq_gamma,
            place="bias",
        )
        if self.quantize_output:
            self.output_quantizer = Quantizer(
                k=torch.tensor(self.k_output),
                i=torch.tensor(self.i_output),
                f=torch.tensor(self.f_output),
                overflow=self.overflow_mode_data,
                round_mode=self.round_mode,
                is_heterogeneous=self.use_hgq,
                is_data=True,
                hgq_gamma=self.hgq_gamma,
                place="datalane",
            )

        self.n_parallel = ops.prod(tuple(input_shape)[1:-1])
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

    def post_pre_train_function(self):
        self.is_pretraining = False
        if self.pruning_layer is not None:
            self.pruning_layer.post_pre_train_function()

    def save_weights(self):
        self.init_weight = self._weight.clone()

    def rewind_weights(self):
        if not hasattr(self, "init_weight"):
            return
        self._weight.data = self.init_weight.clone()

    def ebops(self):
        return 0.0

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
        if self.enable_quantization and not self.is_fitcompress_pretraining():
            return quantizer(x) if x is not None else x
        return x

    def prune(self, weight):
        if self.enable_pruning:
            weight = self.pruning_layer(weight)
        return weight

    def is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    def pre_forward(self, x):
        self.check_is_built(x.shape)
        if self.post_fitcompress_calibration:
            self.saved_inputs.append(x)
            return x
        if self.quantize_input:
            x = self.quantize(x, self.input_quantizer)
        if self.pruning_method == "wanda":
            self.pruning_layer.collect_input(x, self.weight, self.training)
        return x

    def post_forward(self, x):
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
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
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
            **kwargs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.pruning_layer.build(self._weight.shape)
        self.final_compression_done = nn.Parameter(torch.tensor(False), requires_grad=False)

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._weight))
        if include_mask:
            bw_ker = bw_ker * self.pruning_layer.get_hard_mask()
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = (torch.abs(self._weight) >= quantization_step_size).float()
            bw_ker = bw_ker * step_size_mask
        ebops = ops.sum(F.linear(bw_inp, bw_ker))
        if self._bias is not None:
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            size = ops.cast(ops.prod(self.input_shape[:-1]) * self.out_features, self._weight.dtype)
            ebops += ops.mean(bw_bias) * size
        ebops = ebops * self.parallelization_factor / self.n_parallel
        return ebops

    @property
    def weight(self):
        if self.final_compression_done or self.is_fitcompress_pretraining():
            return self._weight
        if self.pruning_first:
            weight = self.prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done or self.is_fitcompress_pretraining():
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done.data = torch.tensor(True)

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self.post_forward(x)
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


class PQConv2d(PQWeightBiasBase, nn.Conv2d):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
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
            **kwargs,
        )
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.pruning_layer.build(self._weight.shape)

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._weight))
        if include_mask:
            bw_ker = bw_ker * self.pruning_layer.get_hard_mask()
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = (torch.abs(self._weight) > quantization_step_size).float()
            bw_ker = bw_ker * step_size_mask
        if self.parallelization_factor < 0:
            ebops = ops.sum(F.conv2d(bw_inp, bw_ker, stride=self.stride, padding=self.padding, dilation=self.dilation))
        else:
            reduce_axis_kernel = tuple(range(2, 4))
            reduce_axis_input = (0,) + tuple(range(2, 4))

            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)
            ebops = ops.sum(bw_inp[None, :] * bw_ker)
        if self._bias is not None:
            size = ops.cast(ops.prod(list(self.input_shape)), self.weight.dtype)
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    @property
    def weight(self):
        if self.final_compression_done:
            return self._weight
        if self.pruning_first:
            weight = self.prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done:
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done = True

    def forward(self, x):
        x = self.pre_forward(x)
        weight = self.weight
        bias = self.bias
        x = F.conv2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        x = self.post_forward(x)
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
        s += ", self.quantize_input={quantize_input} "
        s += ", self.quantize_output={quantize_output}"

        return s.format(**self.__dict__)


class PQConv1d(PQWeightBiasBase, nn.Conv1d):
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        quantize_input=True,
        quantize_output=False,
        enable_pruning: bool = None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
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
            **kwargs,
        )
        self.use_fitcompress = config.fitcompress_parameters.enable_fitcompress
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if bias:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.pruning_layer.build(self._weight.shape)

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._weight))
        if include_mask:
            bw_ker = bw_ker * self.pruning_layer.get_hard_mask()
            _, _, f = self.get_weight_quantization_bits()
            quantization_step_size = 2 ** (-f - 1)
            step_size_mask = (torch.abs(self._weight) > quantization_step_size).float()
            bw_ker = bw_ker * step_size_mask
        if self.parallelization_factor < 0:
            ebops = ops.sum(F.conv1d(bw_inp, bw_ker, stride=self.stride, padding=self.padding, dilation=self.dilation))
        else:
            reduce_axis_kernel = tuple(range(2, 3))
            reduce_axis_input = (0,) + tuple(range(2, 3))

            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            bw_ker = ops.sum(bw_ker, axis=reduce_axis_kernel)
            ebops = ops.sum(bw_inp[None, :] * bw_ker)
        if self.bias is not None:
            size = ops.cast(ops.prod(list(self.input_shape)), self.weight.dtype)
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            ebops += ops.mean(bw_bias) * size
        return ebops

    @property
    def weight(self):
        if self.final_compression_done:
            return self._weight
        if self.pruning_first:
            weight = self.prune(self._weight)
            return self.quantize(weight, self.weight_quantizer)
        else:
            weight = self.quantize(self._weight, self.weight_quantizer)
            return self.prune(weight)

    @property
    def bias(self):
        if self.final_compression_done:
            return self._bias
        bias = self.quantize(self._bias, self.bias_quantizer)
        return bias

    def apply_final_compression(self):
        self._weight.data = self.weight
        if self._bias is not None:
            self._bias.data = self.bias
        self.final_compression_done = True

    def forward(self, x):
        x = self.pre_forward(x)
        x = super().forward(x)
        x = self.post_forward(x)
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
        s += ", self.quantize_input={quantize_input}"
        s += ", self.quantize_output={quantize_output}"
        return s.format(**self.__dict__)


def add_compression_layers(model, config, input_shape=None):
    model = add_quantized_activations_to_model_layer(model, config)
    model = add_pruning_to_model(model, config)
    model.to("cuda")
    if input_shape is not None:
        model(torch.rand(input_shape).to("cuda"))
        model.to("cuda")
    return model


class PQAvgPoolBase(nn.Module):
    def __init__(
        self,
        config,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if out_quant_bits is not None:
            self.k_output, self.i_output, self.f_output = out_quant_bits
        else:
            self.k_output = config.quantization_parameters.default_data_keep_negatives
            self.i_output = config.quantization_parameters.default_data_integer_bits
            self.f_output = config.quantization_parameters.default_data_fractional_bits
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

    def build(self, input_shape):
        self.input_quantizer = Quantizer(
            k=torch.tensor(self.k_input),
            i=torch.tensor(self.i_input),
            f=torch.tensor(self.f_input),
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
        )
        self.output_quantizer = Quantizer(
            k=torch.tensor(self.k_output),
            i=torch.tensor(self.i_output),
            f=torch.tensor(self.f_output),
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
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

    def is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    def pre_pooling(self, x):
        if not hasattr(self, "input_quantizer"):
            self.build(x.shape)
        if self.is_fitcompress_pretraining():
            if self.post_fitcompress_calibration:
                # Save inputs
                self.saved_inputs.append(x)
            # During FITcompress, we do not use any quantized pooling
            return x
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x)
        return x

    def post_pooling(self, x):
        if self.quantize_output and self.enable_quantization and not self.is_fitcompress_pretraining():
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
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
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
            **kwargs,
        )

    def forward(self, x):
        x = self.pre_pooling(x)
        x = super().forward(x)
        x = self.post_pooling(x)
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
        divisor_override: Optional[int] = None,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
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
            **kwargs,
        )

    def forward(self, x):
        x = self.pre_pooling(x)
        x = super().forward(x)
        x = self.post_pooling(x)
        return x


class PQBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        config,
        num_features: int,
        eps: float = 1e-5,
        momentum: typing.Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        quantize_input=True,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if weight_quant_bits is not None:
            self.k_weight, self.i_weight, self.f_weight = weight_quant_bits
        else:
            self.k_weight = config.quantization_parameters.default_weight_keep_negatives
            self.i_weight = config.quantization_parameters.default_weight_integer_bits
            self.f_weight = config.quantization_parameters.default_weight_fractional_bits
        if bias_quant_bits is not None:
            self.k_bias, self.i_bias, self.f_bias = bias_quant_bits
        else:
            self.k_bias = config.quantization_parameters.default_weight_keep_negatives
            self.i_bias = config.quantization_parameters.default_weight_integer_bits
            self.f_bias = config.quantization_parameters.default_weight_fractional_bits
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
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if self.bias is not None:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.built = False
        self.final_compression_done = False
        self.is_pretraining = True
        self.post_fitcompress_calibration = False
        self.saved_inputs = []

    def check_is_built(self, input_shape):
        if self.built:
            return
        self.built = True
        self.input_quantizer = Quantizer(
            k=torch.tensor(self.k_input),
            i=torch.tensor(self.i_input),
            f=torch.tensor(self.f_input),
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
        )
        self.weight_quantizer = Quantizer(
            k=torch.tensor(self.k_weight),
            i=torch.tensor(self.i_weight),
            f=torch.tensor(self.f_weight),
            round_mode=self.round_mode,
            overflow=self.overflow_mode_parameters,
            is_data=False,
            is_heterogeneous=self.use_hgq,
            place="weight",
        )
        self.bias_quantizer = Quantizer(
            k=torch.tensor(self.k_bias),
            i=torch.tensor(self.i_bias),
            f=torch.tensor(self.f_bias),
            round_mode=self.round_mode,
            overflow=self.overflow_mode_parameters,
            is_data=False,
            is_heterogeneous=self.use_hgq,
            place="bias",
        )
        if self.use_hgq:
            self.input_quantizer.quantizer.build(input_shape)
        shape = [1] * len(input_shape)
        shape[1] = input_shape[1]
        self._shape = tuple(shape)
        self.input_shape = (1,) + input_shape[1:]

    def apply_final_compression(self):
        self.final_compression_done = True
        self._weight.data = self.weight
        self._bias.data = self.bias

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    @property
    def weight(self):
        if self.enable_quantization and not self.final_compression_done and not self.is_fitcompress_pretraining():
            return self.weight_quantizer(self._weight)
        return self._weight

    @property
    def bias(self):
        if self.enable_quantization and not self.final_compression_done and not self.is_fitcompress_pretraining():
            return self.bias_quantizer(self._bias)
        return self._bias

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = ops.reshape(self.weight_quantizer.get_total_bits(self.running_mean.shape), self._shape)
        bw_bias = ops.reshape(self.bias_quantizer.get_total_bits(self.running_mean.shape), self._shape)
        size = ops.cast(ops.prod(list(self.input_shape)), self._weight.dtype)
        ebops = ops.sum(bw_inp * bw_ker) + ops.mean(bw_bias) * size
        return ebops

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        return loss

    def post_pre_train_function(self):
        self.is_pretraining = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.check_is_built(input.shape)
        if self.quantize_input and self.enable_quantization:
            if not self.is_fitcompress_pretraining():
                input = self.input_quantizer(input)
            else:
                if self.post_fitcompress_calibration:
                    self.saved_inputs.append(input)
        return super().forward(input)


class PQBatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        config,
        num_features: int,
        eps: float = 1e-5,
        momentum: typing.Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        quantize_input=True,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        if in_quant_bits is not None:
            self.k_input, self.i_input, self.f_input = in_quant_bits
        else:
            self.k_input = config.quantization_parameters.default_data_keep_negatives
            self.i_input = config.quantization_parameters.default_data_integer_bits
            self.f_input = config.quantization_parameters.default_data_fractional_bits

        if weight_quant_bits is not None:
            self.k_weight, self.i_weight, self.f_weight = weight_quant_bits
        else:
            self.k_weight = config.quantization_parameters.default_weight_keep_negatives
            self.i_weight = config.quantization_parameters.default_weight_integer_bits
            self.f_weight = config.quantization_parameters.default_weight_fractional_bits
        if bias_quant_bits is not None:
            self.k_bias, self.i_bias, self.f_bias = bias_quant_bits
        else:
            self.k_bias = config.quantization_parameters.default_weight_keep_negatives
            self.i_bias = config.quantization_parameters.default_weight_integer_bits
            self.f_bias = config.quantization_parameters.default_weight_fractional_bits
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
        self._weight = nn.Parameter(self.weight.clone()).to(self.weight.device)
        self.register_parameter("_weight", self._weight)
        if self.bias is not None:
            self._bias = nn.Parameter(self.bias.clone()).to(self.bias.device)
            self.register_parameter("_bias", self._bias)
        else:
            self.register_parameter("_bias", None)
        self.register_parameter("_weight", self._weight)
        self.built = False
        self.final_compression_done = False
        self.is_pretraining = True
        self.post_fitcompress_calibration = False
        self.saved_inputs = []

    def check_is_built(self, input_shape):
        if self.built:
            return
        self.built = True
        self.input_quantizer = Quantizer(
            k=torch.tensor(self.k_input),
            i=torch.tensor(self.i_input),
            f=torch.tensor(self.f_input),
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
        )
        self.weight_quantizer = Quantizer(
            k=torch.tensor(self.k_weight),
            i=torch.tensor(self.i_weight),
            f=torch.tensor(self.f_weight),
            round_mode=self.round_mode,
            overflow=self.overflow_mode_parameters,
            is_data=False,
            is_heterogeneous=self.use_hgq,
            place="weight",
        )
        self.bias_quantizer = Quantizer(
            k=torch.tensor(self.k_bias),
            i=torch.tensor(self.i_bias),
            f=torch.tensor(self.f_bias),
            round_mode=self.round_mode,
            overflow=self.overflow_mode_parameters,
            is_data=False,
            is_heterogeneous=self.use_hgq,
            place="bias",
        )
        if self.use_hgq:
            self.input_quantizer.quantizer.build(input_shape)
        shape = [1] * len(input_shape)
        shape[1] = input_shape[1]
        self._shape = tuple(shape)
        self.input_shape = (1,) + input_shape[1:]

    def apply_final_compression(self):
        self.final_compression_done = True
        self._weight.data = self.weight
        self._bias.data = self.bias

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def is_fitcompress_pretraining(self):
        return self.is_pretraining and self.use_fitcompress

    @property
    def weight(self):
        if self.enable_quantization and not self.final_compression_done and not self.is_fitcompress_pretraining():
            return self.weight_quantizer(self._weight)
        return self._weight

    @property
    def bias(self):
        if self.enable_quantization and not self.final_compression_done and not self.is_fitcompress_pretraining():
            return self.bias_quantizer(self._bias)
        return self._bias

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = ops.reshape(self.weight_quantizer.get_total_bits(self.running_mean.shape), self._shape)
        bw_bias = ops.reshape(self.bias_quantizer.get_total_bits(self.running_mean.shape), self._shape)
        size = ops.cast(ops.prod(list(self.input_shape)), self._weight.dtype)
        ebops = ops.sum(bw_inp * bw_ker) + ops.mean(bw_bias) * size
        return ebops

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        return loss

    def post_pretrain_function(self):
        self.is_pretraining = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.check_is_built(input.shape)
        if self.quantize_input and self.enable_quantization:
            if not self.is_fitcompress_pretraining():
                input = self.input_quantizer(input)
            else:
                if self.post_fitcompress_calibration:
                    self.saved_inputs.append(input)
        return super().forward(input)


def add_layer_specific_quantization_to_model(name, layer, config):
    if isinstance(layer, PQWeightBiasBase):
        if name in config.quantization_parameters.layer_specific:
            layer_config = config.quantization_parameters.layer_specific[name]
            if "weight" in layer_config:
                if "keep_negatives" in layer_config["weight"]:
                    layer.k_weight = torch.tensor(layer_config["weight"]["keep_negatives"])
                if "integer_bits" in layer_config["weight"]:
                    layer.i_weight = torch.tensor(layer_config["weight"]["integer_bits"])
                if "fractional_bits" in layer_config["weight"]:
                    layer.f_weight = torch.tensor(layer_config["weight"]["fractional_bits"])
            if "bias" in layer_config:
                if "keep_negatives" in layer_config["bias"]:
                    layer.k_bias = torch.tensor(layer_config["bias"]["keep_negatives"])
                if "integer_bits" in layer_config["bias"]:
                    layer.i_bias = torch.tensor(layer_config["bias"]["integer_bits"])
                if "fractional_bits" in layer_config["bias"]:
                    layer.f_bias = torch.tensor(layer_config["bias"]["fractional_bits"])
            if "input" in layer_config:
                if "keep_negatives" in layer_config["input"]:
                    input_keep_negatives = torch.tensor(layer_config["input"]["keep_negatives"])
                    layer.k_input = input_keep_negatives
                if "integer_bits" in layer_config["input"]:
                    input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                    layer.i_input = input_int_bits
                if "fractional_bits" in layer_config["input"]:
                    input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                    layer.f_input = input_fractional_bits
                if "quantize" in layer_config["input"]:
                    quantize = layer_config["input"]["quantize"]
                    layer.quantize_input = quantize
            if "output" in layer_config:
                if "keep_negatives" in layer_config["input"]:
                    output_keep_negatives = torch.tensor(layer_config["output"]["keep_negatives"])
                    layer.k_output = output_keep_negatives
                if "integer_bits" in layer_config["output"]:
                    output_int_bits = torch.tensor(layer_config["output"]["integer_bits"])
                    layer.i_output = input_int_bits
                if "fractional_bits" in layer_config["output"]:
                    input_fractional_bits = torch.tensor(layer_config["output"]["fractional_bits"])
                    layer.f_output = input_fractional_bits
                if "quantize" in layer_config["output"]:
                    quantize = layer_config["output"]["quantize"]
                    layer.quantize_output = quantize

    elif layer.__class__ in [PQBatchNorm2d, PQBatchNorm1d]:
        if name in config.quantization_parameters.layer_specific:
            layer_config = config.quantization_parameters.layer_specific[name]
            if "weight" in layer_config:
                i = torch.tensor(layer_config["weight"]["integer_bits"])
                f = torch.tensor(layer_config["weight"]["fractional_bits"])
                layer.i_weight = i
                layer.f_weight = f
            if "bias" in layer_config:
                i = torch.tensor(layer_config["bias"]["integer_bits"])
                f = torch.tensor(layer_config["bias"]["fractional_bits"])
                layer.i_bias = i
                layer.f_bias = f
            if "input" in layer_config:
                if "integer_bits" in layer_config["input"]:
                    input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                    layer.i_input = input_int_bits
                if "fractional_bits" in layer_config["input"]:
                    input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                    layer.f_input = input_fractional_bits
                if "quantize" in layer_config["input"]:
                    quantize = layer_config["input"]["quantize"]
                    layer.quantize_input = quantize
    elif layer.__class__ in [PQAvgPool1d, PQAvgPool2d]:
        if name in config.quantization_parameters.layer_specific:
            layer_config = config.quantization_parameters.layer_specific[name]
            if "input" in layer_config:
                if "integer_bits" in layer_config["input"]:
                    input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                    layer.i_input = input_int_bits
                if "fractional_bits" in layer_config["input"]:
                    input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                    layer.f_input = input_fractional_bits
                if "quantize" in layer_config["input"]:
                    quantize = layer_config["input"]["quantize"]
                    layer.quantize_input = quantize
            if "output" in layer_config:
                if "integer_bits" in layer_config["output"]:
                    output_int_bits = torch.tensor(layer_config["output"]["integer_bits"])
                    layer.i_output = output_int_bits
                if "fractional_bits" in layer_config["output"]:
                    output_fractional_bits = torch.tensor(layer_config["output"]["fractional_bits"])
                    layer.f_output = output_fractional_bits
                if "quantize" in layer_config["output"]:
                    quantize = layer_config["output"]["quantize"]
                    layer.quantize_output = quantize

    elif layer.__class__ == PQActivation:
        if name in config.quantization_parameters.layer_specific:
            layer_config = config.quantization_parameters.layer_specific[name]
            if "input" in layer_config:
                if "integer_bits" in layer_config["input"]:
                    input_int_bits = torch.tensor(layer_config["input"]["integer_bits"])
                    layer.i_input = input_int_bits
                if "fractional_bits" in layer_config["input"]:
                    input_fractional_bits = torch.tensor(layer_config["input"]["fractional_bits"])
                    layer.f_input = input_fractional_bits
                if "quantize" in layer_config["input"]:
                    quantize = layer_config["input"]["quantize"]
                    layer.quantize_input = quantize
            if "output" in layer_config:
                if "integer_bits" in layer_config["output"]:
                    output_int_bits = torch.tensor(layer_config["output"]["integer_bits"])
                    layer.i_output = output_int_bits
                if "fractional_bits" in layer_config["output"]:
                    output_fractional_bits = torch.tensor(layer_config["output"]["fractional_bits"])
                    layer.f_output = output_fractional_bits
                if "quantize" in layer_config["output"]:
                    quantize = layer_config["output"]["quantize"]
                    layer.quantize_output = quantize
    return layer


def add_quantized_activations_to_model_layer(module, config, prefix=""):
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
            relu = add_layer_specific_quantization_to_model(full_name, relu, config)
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
            tanh = add_layer_specific_quantization_to_model(full_name, tanh, config)
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
            new_layer = add_layer_specific_quantization_to_model(full_name, new_layer, config)
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
            new_layer = add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.BatchNorm2d:
            new_layer = PQBatchNorm2d(
                config,
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                quantize_input=quantize_input,
            )
            new_layer = add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        elif layer.__class__ == nn.BatchNorm1d:
            new_layer = PQBatchNorm1d(
                config,
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
                quantize_input=quantize_input,
            )
            new_layer = add_layer_specific_quantization_to_model(full_name, new_layer, config)
            setattr(module, name, new_layer)
        else:
            layer = add_quantized_activations_to_model_layer(layer, config, full_name)
    return module


def add_quantized_activations_to_model_functional(module, config):
    # Currently not in use. TODO: Fix this
    if config.quantization_parameters.use_high_granularity_quantization:
        return module
    # Replaces functional activation calls with quantized versions
    traced_model = symbolic_trace(module)
    for node in traced_model.graph.nodes:
        if node.op in ["call_method", "call_function"] and (node.target == "tanh" or "function relu" in str(node.target)):
            with traced_model.graph.inserting_after(node):
                if node.name in config.quantization_parameters.layer_specific:
                    bits = config.quantization_parameters.layer_specific[node.name]["bits"]
                else:
                    bits = (
                        config.quantization_parameters.default_integer_bits
                        + config.quantization_parameters.default_fractional_bits
                        + 1
                    )  # 1 sign bit
                kwargs = {"bits": bits}
                if node.target == "tanh":
                    kwargs["use_real_tanh"] = config.quantization_parameters.use_real_tanh
                    kwargs["use_symmetric"] = config.quantization_parameters.use_symmetric_quantization
                    # new_node = traced_model.graph.call_function(quantized_tanh, node.args, kwargs)
                else:
                    kwargs = {"integer_bits": config.quantization_parameters.default_integer_bits, "bits": bits}
                    # new_node = traced_model.graph.call_function(quantized_relu, node.args, kwargs)
                # node.replace_all_uses_with(new_node)
            traced_model.graph.erase_node(node)

    traced_model.graph.lint()
    traced_model.recompile()
    return traced_model


def disable_pruning_from_layers(name, layer, config):
    enable_pruning = name not in config.pruning_parameters.disable_pruning_for_layers
    if layer.__class__ in [PQDense, PQConv2d, PQConv1d] and not enable_pruning:
        layer.enable_pruning = enable_pruning
    return layer


def add_pruning_to_model(module, config, prefix=""):
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    for name, layer in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if layer.__class__ is nn.Linear:
            sparse_layer = PQDense(
                config, layer.in_features, layer.out_features, layer.bias is not None, quantize_input, quantize_output
            )
            sparse_layer._weight.data = layer.weight.data
            if layer.bias is not None:
                sparse_layer._bias.data = layer.bias.data

            sparse_layer = add_layer_specific_quantization_to_model(full_name, sparse_layer, config)
            sparse_layer = disable_pruning_from_layers(full_name, sparse_layer, config)
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv2d:
            sparse_layer = PQConv2d(
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
            sparse_layer._weight.data = layer.weight.data
            if layer.bias is not None:
                sparse_layer._bias.data = layer.bias.data
            sparse_layer = add_layer_specific_quantization_to_model(full_name, sparse_layer, config)
            sparse_layer = disable_pruning_from_layers(full_name, sparse_layer, config)
            setattr(module, name, sparse_layer)
        elif layer.__class__ is nn.Conv1d:
            sparse_layer = PQConv1d(
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
            sparse_layer._weight.data = layer.weight.data
            if layer.bias is not None:
                sparse_layer._bias.data = layer.bias.data
            sparse_layer = add_layer_specific_quantization_to_model(full_name, sparse_layer, config)
            sparse_layer = disable_pruning_from_layers(full_name, sparse_layer, config)
            setattr(module, name, sparse_layer)
        else:
            add_pruning_to_model(layer, config, full_name)
    return module


def apply_final_compression(module):
    for layer in module.modules():
        if isinstance(layer, (PQWeightBiasBase, PQBatchNorm2d, PQBatchNorm1d, Quantizer)):
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


def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
        elif isinstance(layer, Quantizer):
            layer.post_epoch_function()


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.pre_epoch_function(epoch, total_epochs)


def post_round_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.post_round_function()


def save_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.save_weights()


def rewind_weights_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.rewind_weights()


def pre_finetune_functions(model):
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            layer.pruning_layer.pre_finetune_function()


def post_pretrain_functions(model, config, train_loader=None, loss_function=None, input_shape=None):

    if config.fitcompress_parameters.enable_fitcompress:
        from pquant.core.torch.fit_compress import call_fitcompress  # noqa: 811

        for layer in model.modules():
            if isinstance(
                layer, (PQConv2d, PQConv1d, PQDense, PQActivation, PQBatchNorm2d, PQBatchNorm1d, PQAvgPoolBase, Quantizer)
            ):
                # Trigger it here to enable quantization before FITCompress
                layer.post_pre_train_function()
        config, pruning_mask_importance_scores = call_fitcompress(
            config, model, train_loader, loss_function, input_shape=input_shape
        )
        idx = 0
        for layer in model.modules():
            if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
                # layer.post_pre_train_function()
                # set_data_quantization_bits(model)
                with torch.no_grad():
                    layer.pruning_layer.mask.data = pruning_mask_importance_scores[idx]
                layer.pruning_layer.pre_finetune_function()  # So mask is not updated during training anymore
                idx += 1
        return
    else:
        for layer in model.modules():
            if isinstance(
                layer, (PQConv2d, PQConv1d, PQDense, PQActivation, PQBatchNorm2d, PQBatchNorm1d, PQAvgPoolBase, Quantizer)
            ):
                layer.post_pre_train_function()
    if config.pruning_parameters.pruning_method == "pdp" or (
        config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
    ):
        # pass
        pdp_setup(model, config)


def pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = None
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            if global_weights is None:
                global_weights = layer._weight.flatten()
            else:
                global_weights = torch.concat((global_weights, layer._weight.flatten()))

    abs_global_weights = torch.abs(global_weights)
    global_weight_topk, _ = torch.topk(abs_global_weights, abs_global_weights.numel())
    threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * global_weight_topk.numel())]
    global_weights_below_threshold = torch.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
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
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            weight = layer.weight
            total_w += ops.size(weight)
            rem = ops.count_nonzero(weight)
            remaining_weights += rem
        elif layer.__class__ in (nn.Conv2d, nn.Conv1d, nn.Linear):
            total_w += layer.weight.numel()
            remaining_weights += torch.count_nonzero(layer.weight)
    if total_w != 0:
        return remaining_weights / total_w
    return 0.0


def is_training_stage(layer):
    return False if layer.pruning_layer._is_finetuning or layer.pruning_layer._is_pretraining else True


def get_model_losses(model, losses):
    for layer in model.modules():
        loss = 0.0
        if isinstance(layer, (PQConv2d, PQConv1d, PQDense)):
            if layer.enable_pruning and is_training_stage(layer) and not layer.use_fitcompress:
                loss += layer.pruning_layer.calculate_additional_loss()
            if layer.use_hgq:
                loss += layer.hgq_loss()
            losses += loss
        elif isinstance(layer, (PQAvgPool1d, PQAvgPool2d, PQBatchNorm2d, PQBatchNorm1d, PQActivation)):
            if layer.use_hgq:
                losses += layer.hgq_loss()
    return losses


def create_default_layer_quantization_pruning_config(model, config):
    # subconfig = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for name, layer in model.named_modules():
        if layer.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            if layer.bias is None:
                config.quantization_parameters.layer_specific[name] = {
                    "input": {
                        "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                        "quantize": config.quantization_parameters.quantize_input,
                    },
                    "weight": {
                        "keep_negatives": config.quantization_parameters.default_weight_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                    },
                    "output": {
                        "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                        "quantize": config.quantization_parameters.quantize_output,
                    },
                }
            else:
                config.quantization_parameters.layer_specific[name] = {
                    "input": {
                        "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                        "quantize": config.quantization_parameters.quantize_input,
                    },
                    "weight": {
                        "keep_negatives": config.quantization_parameters.default_weight_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                    },
                    "bias": {
                        "keep_negatives": config.quantization_parameters.default_weight_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                    },
                    "output": {
                        "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                        "integer_bits": 0,
                        "fractional_bits": 7,
                        "quantize": config.quantization_parameters.quantize_output,
                    },
                }
            config.pruning_parameters.disable_pruning_for_layers.append(name)
        elif layer.__class__ in [nn.Tanh, nn.ReLU, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            config.quantization_parameters.layer_specific[name] = {
                "input": {
                    "quantize": config.quantization_parameters.quantize_input,
                    "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                    "integer_bits": 0.0,
                    "fractional_bits": 7.0,
                },
                "output": {
                    "quantize": config.quantization_parameters.quantize_output,
                    "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                    "integer_bits": 0.0,
                    "fractional_bits": 7.0,
                },
            }
        elif layer.__class__ in [nn.BatchNorm2d]:
            config.quantization_parameters.layer_specific[name] = {
                "input": {
                    "quantize": config.quantization_parameters.quantize_input,
                    "keep_negatives": config.quantization_parameters.default_data_keep_negatives,
                    "integer_bits": 0.0,
                    "fractional_bits": 7.0,
                },
                "weight": {
                    "keep_negatives": config.quantization_parameters.default_weight_keep_negatives,
                    "integer_bits": 0,
                    "fractional_bits": 7.0,
                },
                "bias": {
                    "keep_negatives": config.quantization_parameters.default_weight_keep_negatives,
                    "integer_bits": 0,
                    "fractional_bits": 7.0,
                },
            }
    return config


def populate_config_with_all_layers(model, config):
    return create_default_layer_quantization_pruning_config(model, config)


def remove_compression_layers(module, config):
    for name, layer in module.named_children():
        if isinstance(layer, PQDense):
            out_features = layer.out_features
            in_features = layer.in_features
            bias = True if layer.bias is not None else False
            setattr(module, name, nn.Linear(in_features=in_features, out_features=out_features, bias=bias))
            getattr(module, name).weight.data.copy_(layer.weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(layer.bias)
        elif isinstance(layer, (PQConv1d, PQConv2d)):
            bias_values = layer.bias if layer.bias is not None else None
            bias = True if bias_values is not None else False
            conv = nn.Conv2d if isinstance(layer, PQConv2d) else nn.Conv1d
            setattr(
                module,
                name,
                conv(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.groups,
                    bias,
                    layer.padding_mode,
                ),
            )
            getattr(module, name).weight.data.copy_(layer.weight)
            if getattr(module, name).bias is not None:
                getattr(module, name).bias.data.copy_(bias_values.data)
        else:
            remove_compression_layers(layer, config)
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
    return remove_compression_layers(model, config)


def get_ebops(model, **kwargs):
    ebops = 0
    for m in model.modules():
        if isinstance(m, (PQWeightBiasBase)):
            ebops += m.ebops(include_mask=m.enable_pruning)
        elif isinstance(m, (PQAvgPoolBase, PQBatchNorm1d, PQBatchNorm2d, PQActivation)):
            ebops += m.ebops()
    return ebops


def load_torch_hgq_model(model, path_to_checkpoint):
    model.load_state_dict(torch.load(path_to_checkpoint), strict=False)
    for m in model.modules():
        if isinstance(m, Quantizer) and m.quantizer.built:
            # Populate HGQ quantizer bit values from PQuantML quantizer
            m.reload_from_local()
    return model
