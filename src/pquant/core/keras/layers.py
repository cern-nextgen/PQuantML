from math import prod
from typing import Tuple, TypeVar

import keras
from keras import constraints, initializers, ops, regularizers
from keras.layers import (
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Layer,
    ReLU,
    SeparableConv2D,
)
from keras.src.layers.input_spec import InputSpec
from keras.src.ops.operation_utils import (
    compute_conv_output_shape,
    compute_pooling_output_shape,
)

from pquant.core.hyperparameter_optimization import PQConfig
from pquant.core.keras.activations import PQActivation, PQSoftmax
from pquant.core.keras.quantizer import Quantizer
from pquant.core.keras.utils import get_pruning_layer

T = TypeVar("T")


def resolve_data_quant_bits(quant_bits, config):
    """Return (k, i, f) from an explicit tuple, or the config's data-lane defaults."""
    if quant_bits is not None:
        return quant_bits
    parameters = config.quantization_parameters
    return (
        parameters.default_data_keep_negatives,
        parameters.default_data_integer_bits,
        parameters.default_data_fractional_bits,
    )


def resolve_weight_quant_bits(quant_bits, config):
    """Return (k, i, f) from an explicit tuple, or the config's weight defaults."""
    if quant_bits is not None:
        return quant_bits
    parameters = config.quantization_parameters
    return (
        parameters.default_weight_keep_negatives,
        parameters.default_weight_integer_bits,
        parameters.default_weight_fractional_bits,
    )


@keras.saving.register_keras_serializable(package="PQuantML")
class PQWeightBiasBase(keras.layers.Layer):
    def __init__(
        self,
        config,
        layer_type,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        enable_pruning=None,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(config, dict):
            config = PQConfig.load_from_config(config)
        self.k_input, self.i_input, self.f_input = resolve_data_quant_bits(in_quant_bits, config)
        self.k_weight, self.i_weight, self.f_weight = resolve_weight_quant_bits(weight_quant_bits, config)
        self.k_bias, self.i_bias, self.f_bias = resolve_weight_quant_bits(bias_quant_bits, config)
        self.k_output, self.i_output, self.f_output = resolve_data_quant_bits(out_quant_bits, config)

        self.layer_type = layer_type
        self.pruning_layer = get_pruning_layer(config=config, layer_type=self.layer_type)
        self.pruning_method = config.pruning_parameters.pruning_method
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

        self.in_quant_bits = in_quant_bits
        self.weight_quant_bits = weight_quant_bits
        self.bias_quant_bits = bias_quant_bits
        self.out_quant_bits = out_quant_bits
        self.weight_quant_granularity = weight_quant_granularity
        self.in_quant_granularity = in_quant_granularity
        self.bias_quant_granularity = bias_quant_granularity
        self.out_quant_granularity = out_quant_granularity
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
        self.dynamic_data = config.quantization_parameters.dynamic_data_quantization
        self.final_compression_done = False
        self.built = False
        self.parallelization_factor = -1
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.input_shape = None
        self._is_pretraining = True
        self._is_finetuning = False
        self.config = config

        weight_granularity = weight_quant_granularity if weight_quant_granularity is not None else self.granularity
        bias_granularity = bias_quant_granularity if bias_quant_granularity is not None else self.granularity
        in_granularity = in_quant_granularity if in_quant_granularity is not None else self.granularity
        out_granularity = out_quant_granularity if out_quant_granularity is not None else self.granularity

        self.weight_quantizer = Quantizer(
            k=self.k_weight,
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=weight_granularity,
            hgq_gamma=self.hgq_gamma,
            place="weight",
        )
        self.bias_quantizer = Quantizer(
            k=self.k_bias,
            i=self.i_bias,
            f=self.f_bias,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=bias_granularity,
            hgq_gamma=self.hgq_gamma,
            place="bias",
        )
        self.input_quantizer = Quantizer(
            k=self.k_input,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=in_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.dynamic_data,
        )
        self.output_quantizer = Quantizer(
            k=self.k_output,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=out_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.dynamic_data,
        )

    def set_enable_pruning(self, enable_pruning):
        self.enable_pruning = enable_pruning

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def build(self, input_shape):
        self.input_shape = (1,) + tuple(input_shape[1:])
        self.n_parallel = int(prod(input_shape[1:-1]))
        self.parallelization_factor = self.parallelization_factor if self.parallelization_factor > 0 else self.n_parallel
        self.is_pretraining = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.ones(shape) if self._is_pretraining else ops.zeros(shape), dtype),
            name="is_pretraining",
            trainable=False,
            dtype="float32",
        )
        self.is_finetuning = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.ones(shape) if self._is_finetuning else ops.zeros(shape), dtype),
            name="is_finetuning",
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape=input_shape)

    def _build_quantizers(self, input_shape):
        """Build the quantizer lanes in use that are not built yet. The output quantizer is
        only created when the output is actually quantized (nothing reads it otherwise)."""
        output_shape = self.compute_output_shape(input_shape)
        if not self.input_quantizer.built:
            self.input_quantizer.build(input_shape)
        if not self.weight_quantizer.built:
            self.weight_quantizer.build(self._kernel.shape)
        if self.use_bias and not self.bias_quantizer.built:
            self.bias_quantizer.build(self._bias.shape)
        if self.quantize_output and not self.output_quantizer.built:
            self.output_quantizer.build(output_shape)

    def _build_pruning_layer(self):
        if self.enable_pruning and self.pruning_layer is not None and not self.pruning_layer.built:
            pruning_shape = tuple(self._kernel.shape[i] for i in self.weight_transpose)
            self.pruning_layer.build(pruning_shape)

    @property
    def kernel(self):
        if self.final_compression_done:
            return self._kernel
        if self.pruning_first:
            weight = self._prune(self._kernel)
            if self.enable_quantization:
                weight = self.weight_quantizer(weight)
            return weight
        weight = self._kernel
        if self.enable_quantization:
            weight = self.weight_quantizer(weight)
        return self._prune(weight)

    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    @property
    def bias(self):
        if self.final_compression_done or self._bias is None:
            return self._bias
        bias = self._bias
        if self.enable_quantization:
            bias = self.bias_quantizer(self._bias)
        return bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    def apply_final_compression(self):
        self._kernel.assign(self.kernel)
        if self._bias is not None:
            self._bias.assign(self.bias)
        self.final_compression_done = True

    def save_own_variables(self, store):
        if not self.built:
            return
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            store[str(i)] = v

    def load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                f"but received {len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
        for i, v in enumerate(all_vars):
            v.assign(store[str(i)])

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(0.0)
        if self.pruning_layer is not None:
            self.pruning_layer.post_pre_train_function()
        self.input_quantizer.post_pre_train_function()
        self.weight_quantizer.post_pre_train_function()
        self.bias_quantizer.post_pre_train_function()
        self.output_quantizer.post_pre_train_function()

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(1.0)
        if self.pruning_layer is not None:
            self.pruning_layer.pre_finetune_function()

    def pre_epoch_function(self, epoch, total_epochs):
        if self.enable_pruning:
            self.pruning_layer.pre_epoch_function(epoch, total_epochs)

    def post_epoch_function(self, epoch, total_epochs, **kwargs):
        if self.enable_pruning:
            self.pruning_layer.post_epoch_function(epoch, total_epochs, **kwargs)
            self._update_pruning_mask()

    def post_round_function(self):
        self.pruning_layer.post_round_function()

    def _update_pruning_mask(self):
        if self.enable_pruning and hasattr(self.pruning_layer, "update_mask"):
            kernel = self._handle_transpose(self._kernel, self.weight_transpose, True)
            self.pruning_layer.update_mask(kernel)

    def _save_weights(self):
        self.init_weight = ops.copy(self._kernel)

    def _rewind_weights(self):
        self._kernel.assign(self.init_weight)

    def ebops(self):
        return 0.0

    def _masked_weight_bits(self, bw_ker):
        """Zero the bit counts of weights that are pruned away or below the quantization step size."""
        mask = self._handle_transpose(self.pruning_layer.get_hard_mask(), self.weight_transpose_back, do_transpose=True)
        _, _, f = self.get_weight_quantization_bits()
        quantization_step_size = 2 ** (-f - 1)
        step_size_mask = ops.cast(ops.abs(self._kernel) > quantization_step_size, self._kernel.dtype)
        return bw_ker * mask * step_size_mask

    def _bias_ebops(self):
        size = ops.cast(ops.prod(self.input_shape), self.dtype)
        bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
        return ops.mean(bw_bias) * size

    def _conv_ebops(self, conv_bits, rank, include_mask):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._kernel))
        if include_mask:
            bw_ker = self._masked_weight_bits(bw_ker)
        if self.parallelization_factor < 0:
            ebops = ops.sum(conv_bits(bw_inp, bw_ker))
        else:
            if self.do_transpose_data:  # channels_last
                reduce_axis_input = tuple(range(rank + 1))
            else:
                reduce_axis_input = (0,) + tuple(range(2, rank + 2))
            bw_inp = ops.max(bw_inp, axis=reduce_axis_input)
            bw_ker = ops.sum(bw_ker, axis=tuple(range(rank)))
            ebops = ops.sum(bw_inp[:, None] * bw_ker)
        if self.use_bias:
            ebops += self._bias_ebops()
        return ebops

    def hgq_loss(self):
        if not self.use_hgq:
            return ops.convert_to_tensor(0.0)

        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        if self._bias is not None:
            loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return ops.where(ops.cast(self.is_pretraining, "bool"), ops.zeros_like(loss), loss)

    def _handle_transpose(self, x, transpose, do_transpose=False):
        if do_transpose:
            x = ops.transpose(x, transpose)
        return x

    def _prune(self, weight):
        if self.enable_pruning:
            weight = self._handle_transpose(weight, self.weight_transpose, True)
            weight = self.pruning_layer(weight)
            weight = self._handle_transpose(weight, self.weight_transpose_back, True)
        return weight

    def pre_forward(self, x, training):
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x, training=training)
        if self.pruning_method == "wanda" and self.enable_pruning:
            self._collect_input(x, self._kernel, training)
        return x

    def _post_forward(self, x, training):
        if self.quantize_output and self.enable_quantization:
            x = self.output_quantizer(x, training=training)
        if self.pruning_method == "activation_pruning" and self.enable_pruning:
            self._collect_output(x, training)
        return x

    def _collect_input(self, x, weight, training):
        collect_x = self._handle_transpose(x, self.data_transpose, self.do_transpose_data)
        weight_channels_first = self._handle_transpose(weight, self.weight_transpose, True)
        self.pruning_layer.collect_input(collect_x, weight_channels_first, training)

    def _collect_output(self, x, training):
        collect_x = self._handle_transpose(x, self.data_transpose, self.do_transpose_data)
        self.pruning_layer.collect_output(collect_x, training)

    @classmethod
    def from_config(cls, config):
        # Quantizer objects are recreated by __init__ from the parent config;
        # their variable values are restored from the h5 weights file by attribute name.
        config.pop("input_quantizer", None)
        config.pop("weight_quantizer", None)
        config.pop("bias_quantizer", None)
        config.pop("output_quantizer", None)
        final_compression_done = config.pop("final_compression_done", False)
        instance = cls(**config)
        instance.final_compression_done = final_compression_done
        return instance

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "config": self.config.get_dict(),
                "input_quantizer": keras.saving.serialize_keras_object(self.input_quantizer),
                "weight_quantizer": keras.saving.serialize_keras_object(self.weight_quantizer),
                "bias_quantizer": keras.saving.serialize_keras_object(self.bias_quantizer),
                "output_quantizer": keras.saving.serialize_keras_object(self.output_quantizer),
                "quantize_input": self.quantize_input,
                "quantize_output": self.quantize_output,
                "in_quant_bits": self.in_quant_bits,
                "weight_quant_bits": self.weight_quant_bits,
                "bias_quant_bits": self.bias_quant_bits,
                "out_quant_bits": self.out_quant_bits,
                "weight_quant_granularity": self.weight_quant_granularity,
                "in_quant_granularity": self.in_quant_granularity,
                "bias_quant_granularity": self.bias_quant_granularity,
                "out_quant_granularity": self.out_quant_granularity,
                "enable_pruning": self.enable_pruning,
                "final_compression_done": self.final_compression_done,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="PQuantML")
class PQDepthwiseConv2d(PQWeightBiasBase, keras.layers.DepthwiseConv2D):
    def __init__(
        self,
        config,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        quantize_input=True,
        quantize_output=False,
        bias: bool = True,
        device=None,
        dtype=None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        enable_pruning=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            config=config,
            layer_type="depthwise_conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            enable_pruning=enable_pruning,
            **kwargs,
        )
        self.depthwise_regularizer = depthwise_regularizer
        self.use_bias = use_bias
        self.weight_transpose = (2, 3, 0, 1)
        self.weight_transpose_back = (2, 3, 0, 1)
        self.data_transpose = (0, 3, 1, 2)
        self.do_transpose_data = self.data_format == "channels_last"
        self._bias = None

    def build(self, input_shape):
        super().build(input_shape)
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(min_ndim=self.rank + 2, axes={channel_axis: input_channel})
        depthwise_shape = self.kernel_size + (
            input_channel,
            self.depth_multiplier,
        )
        self._kernel = self.add_weight(
            name="kernel",
            shape=depthwise_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.depth_multiplier * input_channel,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self._bias = None
        self._build_quantizers(input_shape)
        self._build_pruning_layer()

    def ebops(self, include_mask=False):
        def conv_bits(bw_inp, bw_ker):
            return ops.depthwise_conv(
                bw_inp,
                bw_ker,
                strides=self.strides,
                padding=self.padding,
                data_format=None,
                dilation_rate=self.dilation_rate,
            )

        return self._conv_ebops(conv_bits, rank=2, include_mask=include_mask)

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = super().call(x)
        x = self._post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x


def _normalize_tuple(value, n):
    if isinstance(value, int):
        return (value,) * n
    return tuple(value)


@keras.saving.register_keras_serializable(package="PQuant")
class PQConv2d(PQWeightBiasBase):
    def __init__(
        self,
        config,
        filters,
        kernel_size,
        quantize_input=True,
        quantize_output=False,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        enable_pruning=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            enable_pruning=enable_pruning,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.filters = filters
        self.kernel_size = _normalize_tuple(kernel_size, 2)
        self.strides = _normalize_tuple(strides, 2)
        self.padding = padding.lower()
        self.data_format = keras.backend.image_data_format() if data_format is None else data_format
        self.dilation_rate = _normalize_tuple(dilation_rate, 2)
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        self.do_transpose_data = self.data_format == "channels_last"

    def build(self, input_shape):
        in_channels = input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        kernel_shape = self.kernel_size + (in_channels // self.groups, self.filters)
        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self._bias = None
        super().build(input_shape)
        self._build_quantizers(input_shape)
        self._build_pruning_layer()

    def ebops(self, include_mask=False):
        def conv_bits(bw_inp, bw_ker):
            return ops.conv(
                bw_inp,
                bw_ker,
                strides=self.strides,
                padding=self.padding,
                data_format=None,
                dilation_rate=self.dilation_rate,
            )

        return self._conv_ebops(conv_bits, rank=2, include_mask=include_mask)

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = ops.conv(
            x,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            bias_shape = (1, 1, 1, self.filters) if self.data_format == "channels_last" else (1, self.filters, 1, 1)
            x = x + ops.reshape(self.bias, bias_shape)
        x = self._post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config


@keras.saving.register_keras_serializable(package="PQuantML")
class PQSeparableConv2d(Layer):
    def __init__(
        self,
        config,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        depth_multiplier=1,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        quantize_input=True,
        quantize_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_transpose = (3, 2, 0, 1)
        self.weight_transpose_back = (2, 3, 1, 0)
        self.data_transpose = (0, 3, 1, 2)
        self.depthwise_conv = PQDepthwiseConv2d(
            config,
            kernel_size,
            strides,
            padding,
            depth_multiplier,
            data_format,
            dilation_rate,
            None,
            use_bias=False,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
            depthwise_constraint=depthwise_constraint,
            quantize_input=quantize_input,
            quantize_output=False,
        )

        self.pointwise_conv = PQConv2d(
            config,
            filters=filters,
            kernel_size=1,
            quantize_input=False,
            quantize_output=quantize_output,
            padding="same",
            data_format=data_format,
            groups=1,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=pointwise_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=pointwise_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=pointwise_constraint,
            bias_constraint=bias_constraint,
        )
        self.do_transpose_data = data_format == "channels_last"

    def apply_final_compression(self):
        self.depthwise_conv.apply_final_compression()
        self.pointwise_conv.apply_final_compression()

    def post_pre_train_function(self):
        self.depthwise_conv.post_pre_train_function()
        self.pointwise_conv.post_pre_train_function()

    def pre_finetune_function(self):
        self.depthwise_conv.pre_finetune_function()
        self.pointwise_conv.pre_finetune_function()

    def pre_epoch_function(self, epoch, total_epochs):
        self.depthwise_conv.pre_epoch_function(epoch, total_epochs)
        self.pointwise_conv.pre_epoch_function(epoch, total_epochs)

    def post_epoch_function(self, epoch, total_epochs, **kwargs):
        self.depthwise_conv.post_epoch_function(epoch, total_epochs, **kwargs)
        self.pointwise_conv.post_epoch_function(epoch, total_epochs, **kwargs)

    def post_round_function(self):
        self.depthwise_conv.post_round_function()
        self.pointwise_conv.post_round_function()

    def _save_weights(self):
        self.depthwise_conv._save_weights()
        self.pointwise_conv._save_weights()

    def _rewind_weights(self):
        self.depthwise_conv._rewind_weights()
        self.pointwise_conv._rewind_weights()

    def call(self, x, training=None):
        x = self.depthwise_conv(x, training=training)
        x = self.pointwise_conv(x, training=training)
        return x

    @classmethod
    def from_config(cls, config):
        final_compression_done = config.pop("final_compression_done", False)
        instance = cls(**config)
        instance.depthwise_conv.final_compression_done = final_compression_done
        instance.pointwise_conv.final_compression_done = final_compression_done
        return instance

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.depthwise_conv.config.model_dump(),
                "filters": self.pointwise_conv.filters,
                "kernel_size": self.depthwise_conv.kernel_size,
                "strides": self.depthwise_conv.strides,
                "padding": self.depthwise_conv.padding,
                "data_format": self.depthwise_conv.data_format,
                "dilation_rate": self.depthwise_conv.dilation_rate,
                "depth_multiplier": self.depthwise_conv.depth_multiplier,
                "use_bias": self.pointwise_conv.use_bias,
                "quantize_input": self.depthwise_conv.quantize_input,
                "quantize_output": self.pointwise_conv.quantize_output,
                "final_compression_done": self.depthwise_conv.final_compression_done,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="PQuant")
class PQConv1d(PQWeightBiasBase):
    def __init__(
        self,
        config,
        filters,
        kernel_size,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        enable_pruning=None,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            layer_type="conv",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            enable_pruning=enable_pruning,
            activity_regularizer=activity_regularizer,
            **kwargs,
        )
        self.filters = filters
        self.kernel_size = _normalize_tuple(kernel_size, 1)
        self.strides = _normalize_tuple(strides, 1)
        self.padding = padding.lower()
        self.data_format = keras.backend.image_data_format() if data_format is None else data_format
        self.dilation_rate = _normalize_tuple(dilation_rate, 1)
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.weight_transpose = (2, 1, 0)
        self.weight_transpose_back = (2, 1, 0)
        self.data_transpose = (0, 2, 1)
        self.do_transpose_data = self.data_format == "channels_last"

    def build(self, input_shape):
        in_channels = input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        kernel_shape = self.kernel_size + (in_channels // self.groups, self.filters)
        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self._bias = None
        super().build(input_shape)
        self._build_quantizers(input_shape)
        self._build_pruning_layer()

    def ebops(self, include_mask=False):
        def conv_bits(bw_inp, bw_ker):
            return ops.conv(
                bw_inp,
                bw_ker,
                strides=self.strides,
                padding=self.padding,
                data_format=None,
                dilation_rate=self.dilation_rate,
            )

        return self._conv_ebops(conv_bits, rank=1, include_mask=include_mask)

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = ops.conv(
            x,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            bias_shape = (1, 1, self.filters) if self.data_format == "channels_last" else (1, self.filters, 1)
            x = x + ops.reshape(self.bias, bias_shape)
        x = self._post_forward(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config


@keras.saving.register_keras_serializable(package="PQuantML")
class PQDense(PQWeightBiasBase):
    def __init__(
        self,
        config,
        units,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        weight_quant_granularity=None,
        in_quant_granularity=None,
        bias_quant_granularity=None,
        out_quant_granularity=None,
        enable_pruning=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            layer_type="linear",
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            weight_quant_bits=weight_quant_bits,
            bias_quant_bits=bias_quant_bits,
            out_quant_bits=out_quant_bits,
            weight_quant_granularity=weight_quant_granularity,
            in_quant_granularity=in_quant_granularity,
            bias_quant_granularity=bias_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            enable_pruning=enable_pruning,
            **kwargs,
        )
        self.weight_transpose = (1, 0)
        self.weight_transpose_back = (1, 0)
        self.data_transpose = (0, 1)  # Always (BATCH_SIZE, OUT_FEATURES)
        self.do_transpose_data = False
        self.use_bias = use_bias
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self._bias = None
        super().build(input_shape)
        self._build_quantizers(input_shape)
        self._build_pruning_layer()

    def ebops(self, include_mask=False):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = self.weight_quantizer.get_total_bits(ops.shape(self._kernel))
        if include_mask:
            bw_ker = self._masked_weight_bits(bw_ker)
        ebops = ops.sum(ops.matmul(bw_inp, bw_ker))
        if self.use_bias:
            bw_bias = self.bias_quantizer.get_total_bits(ops.shape(self._bias))
            size = ops.cast(ops.prod(self.input_shape[:-1]) * self.units, self.dtype)
            ebops += ops.mean(bw_bias) * size
        return ebops * self.parallelization_factor / self.n_parallel

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, x, training=None):
        x = self.pre_forward(x, training)
        x = ops.matmul(x, self.kernel)
        if self.use_bias:
            x = ops.add(x, self.bias)
        x = self._post_forward(x, training)
        if self.use_hgq:
            self.add_loss(self.hgq_loss())
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "use_bias": self.use_bias})
        return config


@keras.saving.register_keras_serializable(package="PQuant")
class PQBatchNormalization(keras.layers.BatchNormalization):
    def __init__(
        self,
        config,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        synchronized=False,
        quantize_input=True,
        quantize_parameters=True,
        in_quant_granularity=None,
        weight_quant_granularity=None,
        bias_quant_granularity=None,
        **kwargs,
    ):
        if isinstance(config, dict):
            config = PQConfig.load_from_config(config)
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            synchronized=synchronized,
            **kwargs,
        )
        self.overflow_mode_parameters = config.quantization_parameters.overflow_mode_parameters
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.round_mode = config.quantization_parameters.round_mode
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.data_k = config.quantization_parameters.default_data_keep_negatives
        self.weight_k = config.quantization_parameters.default_weight_keep_negatives
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.quantize_input = quantize_input
        self.quantize_parameters = quantize_parameters
        self.granularity = config.quantization_parameters.granularity
        self.in_quant_granularity = in_quant_granularity
        self.weight_quant_granularity = weight_quant_granularity
        self.bias_quant_granularity = bias_quant_granularity
        self.dynamic_data = config.quantization_parameters.dynamic_data_quantization
        self.config = config
        self.f_weight = self.f_bias = ops.convert_to_tensor(config.quantization_parameters.default_weight_fractional_bits)
        self.i_weight = self.i_bias = ops.convert_to_tensor(config.quantization_parameters.default_weight_integer_bits)
        self.i_input = ops.convert_to_tensor(config.quantization_parameters.default_data_integer_bits)
        self.f_input = ops.convert_to_tensor(config.quantization_parameters.default_data_fractional_bits)
        self.final_compression_done = False
        self._is_pretraining = True

    def build(self, input_shape):
        super().build(input_shape)
        self.is_pretraining = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.ones(shape), dtype),
            name="is_pretraining",
            trainable=False,
            dtype="float32",
        )
        in_granularity = self.in_quant_granularity if self.in_quant_granularity is not None else self.granularity
        weight_granularity = self.weight_quant_granularity if self.weight_quant_granularity is not None else self.granularity
        bias_granularity = self.bias_quant_granularity if self.bias_quant_granularity is not None else self.granularity
        self.input_quantizer = Quantizer(
            k=1.0,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=in_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.dynamic_data,
        )
        self.weight_quantizer = Quantizer(
            k=1.0,
            i=self.i_weight,
            f=self.f_weight,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=weight_granularity,
            place="weight",
        )
        self.bias_quantizer = Quantizer(
            k=1.0,
            i=self.i_bias,
            f=self.f_bias,
            overflow=self.overflow_mode_parameters,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=False,
            granularity=bias_granularity,
            place="bias",
        )
        self.input_quantizer.build(input_shape)
        self.weight_quantizer.build(self.moving_variance.shape)
        self.bias_quantizer.build(self.moving_mean.shape)
        shape = [1] * len(input_shape)
        shape[self.axis] = input_shape[self.axis]
        self._shape = tuple(shape)
        self.input_shape = (1,) + tuple(input_shape[1:])

    def apply_final_compression(self):
        if self.enable_quantization and self.quantize_parameters:
            if self.gamma is not None:
                self.gamma.assign(self.weight_quantizer(self.gamma))
            if self.beta is not None:
                self.beta.assign(self.bias_quantizer(self.beta))
        self.final_compression_done = True

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        bw_ker = ops.reshape(self.weight_quantizer.get_total_bits(self.moving_mean.shape), self._shape)
        bw_bias = ops.reshape(self.bias_quantizer.get_total_bits(self.moving_mean.shape), self._shape)
        size = ops.cast(ops.prod(self.input_shape), self.dtype)
        ebops = ops.sum(bw_inp * bw_ker) + ops.mean(bw_bias) * size
        return ebops

    def hgq_loss(self):
        if not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        loss += self.weight_quantizer.hgq_loss()
        loss += self.bias_quantizer.hgq_loss()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        return ops.where(ops.cast(self.is_pretraining, "bool"), ops.zeros_like(loss), loss)

    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        compute_dtype = keras.backend.result_type(inputs.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        inputs = ops.cast(inputs, compute_dtype)
        if self.quantize_input and self.enable_quantization:
            inputs = self.input_quantizer(inputs, training=training)
        moving_mean = ops.cast(self.moving_mean, inputs.dtype)
        moving_variance = ops.cast(self.moving_variance, inputs.dtype)

        if training and self.trainable:
            mean, variance = self._moments(inputs, mask)

            self.moving_mean.assign(moving_mean * self.momentum + mean * (1.0 - self.momentum))
            self.moving_variance.assign(moving_variance * self.momentum + variance * (1.0 - self.momentum))
        else:
            mean = moving_mean
            variance = moving_variance

        if self.scale:
            gamma = self.gamma
            if self.enable_quantization and self.quantize_parameters and not self.final_compression_done:
                gamma = self.weight_quantizer(self.gamma)
            gamma = ops.cast(gamma, inputs.dtype)
        else:
            gamma = None

        if self.center:
            beta = self.beta
            if self.enable_quantization and self.quantize_parameters and not self.final_compression_done:
                beta = self.bias_quantizer(self.beta)
            beta = ops.cast(beta, inputs.dtype)
        else:
            beta = None

        outputs = ops.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return ops.cast(outputs, self.compute_dtype)

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_weight_quantization_bits(self):
        return self.weight_quantizer.get_quantization_bits()

    def get_bias_quantization_bits(self):
        return self.bias_quantizer.get_quantization_bits()

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(0.0)

    @classmethod
    def from_config(cls, config):
        final_compression_done = config.pop("final_compression_done", False)
        instance = cls(**config)
        instance.final_compression_done = final_compression_done
        return instance

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "quantize_input": self.quantize_input,
                "quantize_parameters": self.quantize_parameters,
                "in_quant_granularity": self.in_quant_granularity,
                "weight_quant_granularity": self.weight_quant_granularity,
                "bias_quant_granularity": self.bias_quant_granularity,
                "final_compression_done": self.final_compression_done,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="PQuantML")
class PQAvgPoolBase(keras.layers.Layer):
    def __init__(
        self,
        config,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        **kwargs,
    ):

        if isinstance(config, dict):
            config = PQConfig.load_from_config(config)
        super().__init__(**kwargs)

        self.in_quant_bits = in_quant_bits
        self.out_quant_bits = out_quant_bits
        self.in_quant_granularity = in_quant_granularity
        self.out_quant_granularity = out_quant_granularity
        self.k_input, self.i_input, self.f_input = resolve_data_quant_bits(in_quant_bits, config)
        self.k_output, self.i_output, self.f_output = resolve_data_quant_bits(out_quant_bits, config)
        self.overflow_mode_data = config.quantization_parameters.overflow_mode_data
        self.config = config
        self.round_mode = config.quantization_parameters.round_mode
        self.data_k = config.quantization_parameters.default_data_keep_negatives
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.hgq_gamma = config.quantization_parameters.hgq_gamma
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.hgq_heterogeneous = config.quantization_parameters.hgq_heterogeneous
        self.dynamic_data = config.quantization_parameters.dynamic_data_quantization
        self._is_pretraining = True
        self.quantize_input = quantize_input
        self.quantize_output = quantize_output
        # BasePooling.__init__ sets built=True to skip the standard Keras build
        # call, but we need build() to run so quantizers are created.
        self.built = False

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(0.0)

    def build(self, input_shape):
        self.is_pretraining = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.ones(shape), dtype),
            name="is_pretraining",
            trainable=False,
            dtype="float32",
        )
        config_granularity = self.config.quantization_parameters.granularity
        in_granularity = self.in_quant_granularity if self.in_quant_granularity is not None else config_granularity
        out_granularity = self.out_quant_granularity if self.out_quant_granularity is not None else config_granularity
        self.input_quantizer = Quantizer(
            k=1.0,
            i=self.i_input,
            f=self.f_input,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=in_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.dynamic_data,
        )
        self.output_quantizer = Quantizer(
            k=1.0,
            i=self.i_output,
            f=self.f_output,
            overflow=self.overflow_mode_data,
            round_mode=self.round_mode,
            is_heterogeneous=self.use_hgq,
            is_data=True,
            granularity=out_granularity,
            hgq_gamma=self.hgq_gamma,
            place="datalane",
            dynamic_data=self.dynamic_data,
        )
        self.input_quantizer.build(input_shape)
        self.output_quantizer.build(self.compute_output_shape(input_shape))
        self.input_shape = (1,) + tuple(input_shape[1:])

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def compute_output_shape(self, input_shape):
        return compute_pooling_output_shape(
            input_shape,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def _pre_pooling(self, x, training):
        if self.quantize_input and self.enable_quantization:
            x = self.input_quantizer(x, training=training)
        return x

    def _post_pooling(self, x, training):
        if self.quantize_output and self.enable_quantization:
            x = self.output_quantizer(x, training=training)
        return x

    def ebops(self):
        bw_inp = self.input_quantizer.get_total_bits(self.input_shape)
        return ops.sum(bw_inp)

    def hgq_loss(self):
        if not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        loss = self.hgq_beta * self.ebops()
        if self.quantize_input:
            loss += self.input_quantizer.hgq_loss()
        if self.quantize_output:
            loss += self.output_quantizer.hgq_loss()
        return ops.where(ops.cast(self.is_pretraining, "bool"), ops.zeros_like(loss), loss)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "quantize_input": self.quantize_input,
                "quantize_output": self.quantize_output,
                "in_quant_bits": self.in_quant_bits,
                "out_quant_bits": self.out_quant_bits,
                "in_quant_granularity": self.in_quant_granularity,
                "out_quant_granularity": self.out_quant_granularity,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="PQuant")
class PQAvgPool1d(PQAvgPoolBase, keras.layers.AveragePooling1D):
    def __init__(
        self,
        config,
        pool_size,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            out_quant_bits=out_quant_bits,
            in_quant_granularity=in_quant_granularity,
            out_quant_granularity=out_quant_granularity,
            **kwargs,
        )

    def call(self, x, training=None):
        x = self._pre_pooling(x, training)
        x = super().call(x)
        x = self._post_pooling(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="PQuant")
class PQAvgPool2d(PQAvgPoolBase, keras.layers.AveragePooling2D):
    def __init__(
        self,
        config,
        pool_size,
        quantize_input=True,
        quantize_output=False,
        in_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
            config=config,
            quantize_input=quantize_input,
            quantize_output=quantize_output,
            in_quant_bits=in_quant_bits,
            out_quant_bits=out_quant_bits,
            in_quant_granularity=in_quant_granularity,
            out_quant_granularity=out_quant_granularity,
        )

    def call(self, x, training=None):
        x = self._pre_pooling(x, training)
        x = super().call(x)
        x = self._post_pooling(x, training)
        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())
        return x

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="PQuantML")
class PQMultiheadAttention(keras.layers.Layer):
    """Multi-head attention with quantization support.

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
        quantize_input: Whether to quantize the Q/K/V projection inputs (the MHA inputs).
        quantize_output: Whether to quantize the output projection's output (the MHA output).
            The q/k/v projection outputs and the out_proj input (the context) are always
            quantized, mirroring HGQ's QMultiHeadAttention.
        approximate_softmax: Placeholder for approximate softmax (currently uses standard softmax).
        in_quant_bits: (k, i, f) bits for input quantization.
        weight_quant_bits: (k, i, f) bits for weight quantization.
        bias_quant_bits: (k, i, f) bits for bias quantization.
        out_quant_bits: (k, i, f) bits for output quantization.
        attn_quant_bits: (k, i, f) bits for the softmax output quantizer (the attention
            weights). The scores and context need no dedicated quantizers: the softmax's
            input quantizer and the output projection's input quantizer cover them.

    Call args:
        inputs: A tuple (query, key, value) of tensors with shape (batch, seq, features),
            or a single tensor for self-attention.
        training: Python boolean indicating whether the layer should behave in training mode.
        key_padding_mask: Boolean tensor of shape (batch, key_seq). True means the position
            should be ignored.
        attn_mask: Additive mask of shape (query_seq, key_seq) or
            (batch, num_heads, query_seq, key_seq).
        need_weights: If True, returns (output, attn_weights). If False, returns (output, None).
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
        quantize_input: bool = True,
        quantize_output: bool = False,
        approximate_softmax: bool = False,
        in_quant_bits: Tuple[T, T, T] = None,
        weight_quant_bits: Tuple[T, T, T] = None,
        bias_quant_bits: Tuple[T, T, T] = None,
        out_quant_bits: Tuple[T, T, T] = None,
        attn_quant_bits: Tuple[T, T, T] = None,
        in_quant_granularity=None,
        out_quant_granularity=None,
        param_quant_granularity=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        if isinstance(config, dict):
            config = PQConfig.load_from_config(config)

        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.use_bias = bias
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.approximate_softmax = approximate_softmax
        self.scale = self.head_dim**-0.5
        self.enable_quantization = config.quantization_parameters.enable_quantization
        self.use_hgq = config.quantization_parameters.use_high_granularity_quantization
        self.hgq_beta = config.quantization_parameters.hgq_beta
        self.is_pretraining = True

        self.in_quant_bits = in_quant_bits
        self.weight_quant_bits = weight_quant_bits
        self.bias_quant_bits = bias_quant_bits
        self.out_quant_bits = out_quant_bits
        self.attn_quant_bits = attn_quant_bits

        self.in_quant_granularity = in_quant_granularity
        self.out_quant_granularity = out_quant_granularity
        self.param_quant_granularity = param_quant_granularity

        self.softmax = PQSoftmax(config, -1, quantize_input=True, quantize_output=True, out_quant_bits=attn_quant_bits)
        proj_kwargs = dict(
            use_bias=bias,
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
        self.q_proj = PQDense(config, embed_dim, enable_pruning=False, **qkv_kwargs)
        self.k_proj = PQDense(config, embed_dim, enable_pruning=False, **qkv_kwargs)
        self.v_proj = PQDense(config, embed_dim, enable_pruning=False, **qkv_kwargs)
        self.out_proj = PQDense(
            config,
            embed_dim,
            quantize_input=True,
            quantize_output=quantize_output,
            out_quant_granularity=out_quant_granularity,
            **proj_kwargs,
        )

        self.attn_dropout = keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def post_pre_train_function(self):
        self.is_pretraining = False
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.post_pre_train_function()
        self.softmax.post_pre_train_function()

    def pre_finetune_function(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.pre_finetune_function()

    def pre_epoch_function(self, epoch, total_epochs):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.pre_epoch_function(epoch, total_epochs)

    def post_epoch_function(self, epoch, total_epochs, **kwargs):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.post_epoch_function(epoch, total_epochs, **kwargs)

    def post_round_function(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj.post_round_function()

    def _save_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj._save_weights()

    def _rewind_weights(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            proj._rewind_weights()

    def _head_bits(self, proj, seq_len):
        """Bitwidths of a projection's output, in per-head layout (1, H, seq, head_dim)."""
        bw = proj.output_quantizer.get_total_bits((1, seq_len, self.embed_dim))
        bw = ops.reshape(bw, (1, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(bw, (0, 2, 1, 3))

    def attention_ebops(self):
        """EBOPs of the q @ k^T and attn @ v einsums (mirrors HGQ's QMultiHeadAttention)."""
        attn_shape = self.softmax.input_shape  # (1, H, T, S), stored when the softmax was built
        query_len, key_len = attn_shape[2], attn_shape[3]
        bw_q = self._head_bits(self.q_proj, query_len)
        bw_k = self._head_bits(self.k_proj, key_len)
        bw_v = self._head_bits(self.v_proj, key_len)

        bw_attn = self.softmax.output_quantizer.get_total_bits(attn_shape)
        ebops_qk = ops.einsum("bhtd,bhsd->", bw_q, bw_k)
        ebops_av = ops.einsum("bhts,bhsd->", bw_attn, bw_v)
        return ebops_qk + ebops_av

    def ebops(self):
        ebops = self.attention_ebops() + self.softmax.ebops()
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            ebops += proj.ebops(include_mask=proj.enable_pruning)
        return ebops

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return ops.convert_to_tensor(0.0)
        return ops.convert_to_tensor(self.hgq_beta * self.attention_ebops() + self.softmax.hgq_loss())

    @staticmethod
    def _split_qkv_shapes(input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
            q_shape = input_shape[0]
            k_shape = input_shape[1] if len(input_shape) > 1 else q_shape
            v_shape = input_shape[2] if len(input_shape) > 2 else k_shape
        else:
            q_shape = k_shape = v_shape = input_shape
        return q_shape, k_shape, v_shape

    def compute_output_shape(self, input_shape):
        q_shape, k_shape, _ = self._split_qkv_shapes(input_shape)
        batch, tgt_len = q_shape[0], q_shape[1]
        src_len = k_shape[1]
        return (batch, tgt_len, self.embed_dim), (batch, tgt_len, src_len)

    def build(self, input_shape):
        q_shape, k_shape, v_shape = self._split_qkv_shapes(input_shape)
        self.q_proj.build(q_shape)
        self.k_proj.build(k_shape)
        self.v_proj.build(v_shape)
        self.out_proj.build(tuple(q_shape[:-1]) + (self.embed_dim,))
        # Softmax operates on the per-head attention scores (B, H, Tq, Tk).
        self.softmax.build((q_shape[0], self.num_heads, q_shape[1], k_shape[1]))
        super().build(input_shape)

    def call(
        self,
        inputs,
        training=None,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
    ):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                query, key, value = inputs
            elif len(inputs) == 2:
                query, key = inputs
                value = key
            else:
                query = key = value = inputs[0]
        else:
            query = key = value = inputs

        batch_size = ops.shape(query)[0]
        query_len = ops.shape(query)[1]
        key_len = ops.shape(key)[1]

        q = self.q_proj(query, training=training)  # (B, T, E)
        k = self.k_proj(key, training=training)  # (B, S, E)
        v = self.v_proj(value, training=training)  # (B, S, E)

        q = ops.reshape(q, (batch_size, query_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(k, (batch_size, key_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.reshape(v, (batch_size, key_len, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention scores: (B, H, T, S)
        attn_scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        if attn_mask is not None:
            if ops.ndim(attn_mask) == 2:
                # (T, S) -> (1, 1, T, S)
                attn_mask = ops.reshape(attn_mask, (1, 1, query_len, key_len))
            elif ops.ndim(attn_mask) == 3:
                # (B*H, T, S) -> (B, H, T, S)
                attn_mask = ops.reshape(attn_mask, (batch_size, self.num_heads, query_len, key_len))
            attn_scores = attn_scores + ops.cast(attn_mask, attn_scores.dtype)

        mask = None
        if key_padding_mask is not None:
            mask = ops.logical_not(ops.cast(key_padding_mask, "bool"))
            mask = ops.reshape(mask, (batch_size, 1, 1, key_len))  # (B, 1, 1, S)

        # The softmax's own input/output quantizers handle the scores and the attention weights;
        attn_weights = self.softmax(attn_scores, mask=mask)

        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        # Weighted sum of values: (B, H, T, head_dim)
        out = ops.matmul(attn_weights, v)

        # Merge heads: (B, T, E)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, query_len, self.embed_dim))
        out = self.out_proj(out, training=training)

        if self.use_hgq and self.enable_quantization:
            self.add_loss(self.hgq_loss())

        if need_weights:
            # Average attention weights over heads: (B, T, S)
            return out, ops.mean(attn_weights, axis=1)
        return out, None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "bias": self.use_bias,
                "kdim": self.kdim,
                "vdim": self.vdim,
                "quantize_input": self.q_proj.quantize_input,
                "quantize_output": self.out_proj.quantize_output,
                "approximate_softmax": self.approximate_softmax,
                "in_quant_bits": self.in_quant_bits,
                "weight_quant_bits": self.weight_quant_bits,
                "bias_quant_bits": self.bias_quant_bits,
                "out_quant_bits": self.out_quant_bits,
                "attn_quant_bits": self.attn_quant_bits,
                "in_quant_granularity": self.in_quant_granularity,
                "out_quant_granularity": self.out_quant_granularity,
                "param_quant_granularity": self.param_quant_granularity,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("q_proj", None)
        config.pop("k_proj", None)
        config.pop("v_proj", None)
        config.pop("out_proj", None)
        config.pop("softmax", None)
        return cls(**config)


LAYERS_WITH_PRUNING_LAYER = (PQWeightBiasBase, PQSeparableConv2d, PQMultiheadAttention)


def _iter_weight_layers(model):
    for layer in model.layers:
        if isinstance(layer, PQWeightBiasBase):
            yield layer
        elif isinstance(layer, PQSeparableConv2d):
            yield layer.depthwise_conv
            yield layer.pointwise_conv
        elif isinstance(layer, PQMultiheadAttention):
            yield layer.q_proj
            yield layer.k_proj
            yield layer.v_proj
            yield layer.out_proj


def call_post_round_functions(model, rewind, rounds, r):
    last_round = r == rounds - 1
    if rewind == "every-round":
        _rewind_weights_functions(model)
    elif rewind == "post-training-stage" and last_round:
        _rewind_weights_functions(model)
    elif not last_round:
        _post_round_functions(model)


def apply_final_compression(model):
    for layer in model.layers:
        if isinstance(layer, (PQWeightBiasBase, PQSeparableConv2d, PQBatchNormalization)):
            layer.apply_final_compression()
            if hasattr(layer, "input_quantizer"):
                layer.input_quantizer.apply_final_compression()
            if hasattr(layer, "output_quantizer"):
                layer.output_quantizer.apply_final_compression()
        elif isinstance(layer, PQMultiheadAttention):
            for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
                proj.apply_final_compression()
    return model


def post_epoch_functions(model, epoch, total_epochs, **kwargs):
    for layer in model.layers:
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.post_epoch_function(epoch, total_epochs, **kwargs)


def pre_epoch_functions(model, epoch, total_epochs):
    for layer in model.layers:
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.pre_epoch_function(epoch, total_epochs)


def _post_round_functions(model):
    for layer in model.layers:
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.post_round_function()


def save_weights_functions(model):
    for layer in model.layers:
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer._save_weights()


def _rewind_weights_functions(model):
    for layer in model.layers:
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer._rewind_weights()


def pre_finetune_functions(model):
    for layer in model.layers:
        if isinstance(layer, LAYERS_WITH_PRUNING_LAYER):
            layer.pre_finetune_function()


def post_pretrain_functions(model, config):
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PQWeightBiasBase,
                PQSeparableConv2d,
                PQActivation,
                PQAvgPoolBase,
                PQBatchNormalization,
                PQSoftmax,
                PQMultiheadAttention,
            ),
        ):
            layer.post_pre_train_function()
    if config.pruning_parameters.pruning_method == "pdp" or (
        config.pruning_parameters.pruning_method == "wanda" and config.pruning_parameters.calculate_pruning_budget
    ):
        _pdp_setup(model, config)


def _pdp_setup(model, config):
    """
    Calculates a global sparsity threshold. Initializes target sparsity for each layer, which depends on
    how large percentage of weights in the layer is smaller than the global threshold
    """
    global_weights = ops.concatenate([ops.ravel(layer.kernel) for layer in _iter_weight_layers(model)])
    abs_global_weights = ops.abs(global_weights)
    global_weight_topk, _ = ops.top_k(abs_global_weights, ops.size(abs_global_weights))
    threshold = global_weight_topk[int((1 - config.pruning_parameters.sparsity) * float(ops.size(global_weight_topk)))]
    global_weights_below_threshold = ops.where(abs_global_weights < threshold, 1, 0)
    idx = 0
    for layer in _iter_weight_layers(model):
        weight_size = ops.size(layer.kernel)
        w = ops.sum(global_weights_below_threshold[idx : idx + weight_size])
        sparsity = ops.convert_to_tensor(w / weight_size, dtype=layer.kernel.dtype)
        layer.pruning_layer.init_r = sparsity
        layer.pruning_layer.sparsity = sparsity  # Wanda
        idx += weight_size


def get_layer_keep_ratio(model):
    total_w = 0
    remaining_weights = 0
    for layer in model.layers:
        if isinstance(layer, PQWeightBiasBase):
            weight = layer.kernel
            total_w += ops.size(weight)
            remaining_weights += ops.count_nonzero(weight)
        elif isinstance(layer, (PQSeparableConv2d, PQMultiheadAttention)):
            if isinstance(layer, PQSeparableConv2d):
                sublayers = (layer.depthwise_conv, layer.pointwise_conv)
            else:
                sublayers = (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj)
            for sublayer in sublayers:
                weight = sublayer.kernel
                total_w += ops.size(weight)
                remaining_weights += ops.count_nonzero(weight)
        elif isinstance(layer, (Conv2D, Conv1D, DepthwiseConv2D, Dense)):
            weight = layer.kernel
            total_w += ops.size(weight)
            remaining_weights += ops.count_nonzero(weight)
        elif isinstance(layer, SeparableConv2D):
            total_w += ops.size(layer.depthwise_kernel)
            total_w += ops.size(layer.pointwise_kernel)
            remaining_weights += ops.count_nonzero(layer.depthwise_kernel)
            remaining_weights += ops.count_nonzero(layer.pointwise_kernel)
    if total_w != 0:
        return remaining_weights / total_w
    return 0.0


def _is_training_stage(layer):
    return not (layer.pruning_layer._is_finetuning or layer.pruning_layer._is_pretraining)


def get_model_losses(model, losses):
    for layer in _iter_weight_layers(model):
        if layer.enable_pruning and _is_training_stage(layer):
            losses += layer.pruning_layer.calculate_additional_loss()
        if layer.enable_quantization and layer.use_hgq:
            losses += layer.hgq_loss()
    for layer in model.layers:
        if isinstance(layer, (PQActivation, PQAvgPoolBase, PQBatchNormalization, PQSoftmax)):
            if layer.enable_quantization and layer.use_hgq:
                losses += layer.hgq_loss()
    return losses


def _check_activation(layer, config):
    """
    Replaces activations with quantized activations.
    The activation can be a part of another layer such as Conv2D, or an Activation layer
    """
    quantization_enabled = config.quantization_parameters.enable_quantization
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    act = None
    if hasattr(layer.activation, "__name__"):
        if layer.activation.__name__ == "relu":
            act = (
                PQActivation(config, "relu", quantize_input=quantize_input, quantize_output=quantize_output)
                if quantization_enabled
                else ReLU()
            )
            if quantization_enabled:
                _set_quantization_bits_activations(config, layer, act)
            act.build(layer.input.shape)
        elif layer.activation.__name__ == "tanh":
            type_of_tanh = "tanh" if config.quantization_parameters.use_real_tanh else "hard_tanh"
            act = (
                PQActivation(config, type_of_tanh, quantize_input=quantize_input, quantize_output=quantize_output)
                if quantization_enabled
                else Activation(activation="tanh")
            )
            if quantization_enabled:
                _set_quantization_bits_activations(config, layer, act)
                act.build(layer.input.shape)
    return act


def _build_pruning_layer_from_kernel(new_layer, kernel):
    transposed_kernel = ops.transpose(kernel, new_layer.weight_transpose)
    new_layer.pruning_layer.build(transposed_kernel.shape)


def _copy_kernel_and_bias(new_layer, layer):
    new_layer._kernel.assign(layer._kernel)
    if layer.use_bias:
        new_layer._bias.assign(layer.bias)


def add_compression_layers(model, config, input_shape=None):
    # Pruning algorithms assume channels_first format
    # Creates a new functional model from model, replacing certain layers with compressed / quantized variants
    x = model.layers[0].output
    quantize_input = config.quantization_parameters.quantize_input
    quantize_output = config.quantization_parameters.quantize_output
    for layer in model.layers[1:]:
        act = None
        if isinstance(layer, DepthwiseConv2D):
            new_layer = PQDepthwiseConv2d(
                config,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                depth_multiplier=layer.depth_multiplier,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                use_bias=layer.use_bias,
                bias_initializer=layer.bias_initializer,
                depthwise_initializer=layer.depthwise_initializer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                depthwise_constraint=layer.depthwise_constraint,
                bias_constraint=layer.bias_constraint,
                bias=layer.bias,
                dtype=layer.dtype,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            _set_quantization_bits_weight_layers(config, layer, new_layer)
            new_layer.set_enable_pruning(_get_enable_pruning(layer, config))
            _build_pruning_layer_from_kernel(new_layer, layer.kernel)
            x = new_layer(x)
            act = _check_activation(layer, config)
        elif isinstance(layer, Conv2D):
            new_layer = PQConv2d(
                config=config,
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                groups=layer.groups,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            _set_quantization_bits_weight_layers(config, layer, new_layer)
            new_layer.set_enable_pruning(_get_enable_pruning(layer, config))
            _build_pruning_layer_from_kernel(new_layer, layer.kernel)
            new_layer.build(x.shape)
            x = new_layer(x)
            _copy_kernel_and_bias(new_layer, layer)
            act = _check_activation(layer, config)
        elif isinstance(layer, SeparableConv2D):
            new_layer = PQSeparableConv2d(
                config,
                layer.filters,
                layer.kernel_size,
                layer.strides,
                layer.padding,
                layer.data_format,
                layer.dilation_rate,
                layer.depth_multiplier,
                layer.use_bias,
                layer.depthwise_initializer,
                layer.pointwise_initializer,
                layer.bias_initializer,
                layer.depthwise_regularizer,
                layer.pointwise_regularizer,
                layer.bias_regularizer,
                layer.depthwise_constraint,
                layer.pointwise_constraint,
                layer.bias_constraint,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            _set_quantization_bits_weight_layers(config, layer, new_layer)

            enable_pruning_depthwise, enable_pruning_pointwise = _get_enable_pruning(layer, config)
            new_layer.depthwise_conv.set_enable_pruning(enable_pruning_depthwise)
            new_layer.pointwise_conv.set_enable_pruning(enable_pruning_pointwise)
            _build_pruning_layer_from_kernel(new_layer.depthwise_conv, layer.depthwise_kernel)
            _build_pruning_layer_from_kernel(new_layer.pointwise_conv, layer.pointwise_kernel)
            new_layer.depthwise_conv.build(x.shape)
            y = new_layer.depthwise_conv(x).shape
            new_layer.pointwise_conv.build(y)
            x = new_layer(x)
            act = _check_activation(layer, config)
        elif isinstance(layer, Conv1D):
            new_layer = PQConv1d(
                config=config,
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                groups=layer.groups,
                activation=None,
                use_bias=layer.use_bias,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            _set_quantization_bits_weight_layers(config, layer, new_layer)
            new_layer.set_enable_pruning(_get_enable_pruning(layer, config))
            _build_pruning_layer_from_kernel(new_layer, layer.kernel)
            new_layer.build(x.shape)
            x = new_layer(x)
            _copy_kernel_and_bias(new_layer, layer)
            act = _check_activation(layer, config)
        elif isinstance(layer, Dense):
            new_layer = PQDense(
                config=config,
                units=layer.units,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                quantize_input=quantize_input,
                quantize_output=quantize_output,
            )
            _set_quantization_bits_weight_layers(config, layer, new_layer)
            new_layer.set_enable_pruning(_get_enable_pruning(layer, config))
            _build_pruning_layer_from_kernel(new_layer, layer.kernel)
            x = new_layer(x)
            _copy_kernel_and_bias(new_layer, layer)
            act = _check_activation(layer, config)
        # Activation layers
        elif isinstance(layer, ReLU):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQActivation(config, "relu", quantize_input=quantize_input, quantize_output=quantize_output)
                _set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(layer.input.shape)
                x = new_layer(x)

            else:
                x = layer(x)
        elif isinstance(layer, Activation):
            new_layer = _check_activation(layer, config)

            if new_layer is not None:
                x = new_layer(x)
        elif isinstance(layer, AveragePooling1D):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQAvgPool1d(
                    config=config,
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    data_format=layer.data_format,
                )
                _set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(x.shape)
                x = new_layer(x)
        elif isinstance(layer, AveragePooling2D):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQAvgPool2d(
                    config=config,
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    data_format=layer.data_format,
                )
                _set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(x.shape)
                x = new_layer(x)
        elif isinstance(layer, (BatchNormalization)):
            if config.quantization_parameters.enable_quantization:
                new_layer = PQBatchNormalization(
                    config,
                    layer.axis,
                    layer.momentum,
                    layer.epsilon,
                    layer.center,
                    layer.scale,
                    layer.beta_initializer,
                    layer.gamma_initializer,
                    layer.moving_mean_initializer,
                    layer.moving_variance_initializer,
                    layer.beta_regularizer,
                    layer.gamma_regularizer,
                    layer.beta_constraint,
                    layer.gamma_constraint,
                    layer.synchronized,
                    quantize_input=True,
                )
                _set_quantization_bits_activations(config, layer, new_layer)
                new_layer.build(x.shape)
                x = new_layer(x)
            else:
                x = layer(x)
        else:
            x = layer(x)
        if act is not None:
            x = act(x)
    replaced_model = keras.Model(inputs=model.inputs, outputs=x)
    return replaced_model


def _get_quant_section(section, i_default, f_default, target=None, quantize_attr=None):
    """Read integer/fractional bits from one layer_specific section, optionally applying its quantize flag."""
    if section is None:
        return i_default, f_default
    if quantize_attr is not None and "quantize" in section:
        setattr(target, quantize_attr, section["quantize"])
    return section.get("integer_bits", i_default), section.get("fractional_bits", f_default)


def _set_quantization_bits_activations(config, layer, new_layer):
    quant_params = config.quantization_parameters
    i_input = i_output = i_weight = i_bias = quant_params.default_data_integer_bits
    f_input = f_output = f_weight = f_bias = quant_params.default_data_fractional_bits
    if isinstance(layer, ReLU):
        f_input += 1
        f_output += 1  # Unsigned, add 1 bit to default value only
    layer_config = quant_params.layer_specific.get(layer.name)
    if layer_config is not None:
        if hasattr(layer, "activation") and layer.activation.__name__ in layer_config:
            activation_config = layer_config[layer.activation.__name__]
            i_input, f_input = _get_quant_section(
                activation_config.get("input"), i_input, f_input, new_layer, "quantize_input"
            )
            i_output, f_output = _get_quant_section(
                activation_config.get("output"), i_output, f_output, new_layer, "quantize_output"
            )
        else:
            i_input, f_input = _get_quant_section(layer_config.get("input"), i_input, f_input, new_layer, "quantize_input")
            i_weight, f_weight = _get_quant_section(layer_config.get("weight"), i_weight, f_weight)
            i_bias, f_bias = _get_quant_section(layer_config.get("bias"), i_bias, f_bias)
            i_output, f_output = _get_quant_section(
                layer_config.get("output"), i_output, f_output, new_layer, "quantize_output"
            )
    if isinstance(layer, BatchNormalization):
        new_layer.i_weight = i_weight
        new_layer.f_weight = f_weight
        new_layer.i_bias = i_bias
        new_layer.f_bias = f_bias
    new_layer.i_input = i_input
    new_layer.f_input = f_input
    new_layer.i_output = i_output
    new_layer.f_output = f_output


def _set_quantization_bits_weight_layers(config, layer, new_layer):
    quant_params = config.quantization_parameters
    layer_config = quant_params.layer_specific.get(layer.name)
    if isinstance(layer, SeparableConv2D):
        dw_i_weight = pw_i_weight = pw_i_bias = quant_params.default_weight_integer_bits
        dw_f_weight = pw_f_weight = pw_f_bias = quant_params.default_weight_fractional_bits
        i_input = i_output = quant_params.default_data_integer_bits
        f_input = f_output = quant_params.default_data_fractional_bits
        if layer_config is not None:
            i_input, f_input = _get_quant_section(
                layer_config.get("input"), i_input, f_input, new_layer.depthwise_conv, "quantize_input"
            )
            depthwise_config = layer_config.get("depthwise", {})
            dw_i_weight, dw_f_weight = _get_quant_section(depthwise_config.get("weight"), dw_i_weight, dw_f_weight)
            pointwise_config = layer_config.get("pointwise", {})
            pw_i_weight, pw_f_weight = _get_quant_section(pointwise_config.get("weight"), pw_i_weight, pw_f_weight)
            pw_i_bias, pw_f_bias = _get_quant_section(pointwise_config.get("bias"), pw_i_bias, pw_f_bias)
            i_output, f_output = _get_quant_section(
                layer_config.get("output"), i_output, f_output, new_layer, "quantize_output"
            )
        new_layer.depthwise_conv.i_input = i_input
        new_layer.depthwise_conv.f_input = f_input
        new_layer.depthwise_conv.i_weight = dw_i_weight
        new_layer.depthwise_conv.f_weight = dw_f_weight
        new_layer.pointwise_conv.i_weight = pw_i_weight
        new_layer.pointwise_conv.f_weight = pw_f_weight
        new_layer.pointwise_conv.i_bias = pw_i_bias
        new_layer.pointwise_conv.f_bias = pw_f_bias
        new_layer.pointwise_conv.i_output = i_output
        new_layer.pointwise_conv.f_output = f_output
    else:
        i_weight = i_bias = quant_params.default_weight_integer_bits
        f_weight = f_bias = quant_params.default_weight_fractional_bits
        if layer_config is not None:
            new_layer.i_input, new_layer.f_input = _get_quant_section(
                layer_config.get("input"), new_layer.i_input, new_layer.f_input, new_layer, "quantize_input"
            )
            i_weight, f_weight = _get_quant_section(layer_config.get("weight"), i_weight, f_weight)
            i_bias, f_bias = _get_quant_section(layer_config.get("bias"), i_bias, f_bias)
            new_layer.i_output, new_layer.f_output = _get_quant_section(
                layer_config.get("output"), new_layer.i_output, new_layer.f_output, new_layer, "quantize_output"
            )
        new_layer.i_weight = i_weight
        new_layer.f_weight = f_weight
        new_layer.i_bias = i_bias
        new_layer.f_bias = f_bias
        new_layer.weight_quantizer.i_init = float(i_weight)
        new_layer.weight_quantizer.f_init = float(f_weight)
        new_layer.bias_quantizer.i_init = float(i_bias)
        new_layer.bias_quantizer.f_init = float(f_bias)


def _get_enable_pruning(layer, config):
    enable_pruning = config.pruning_parameters.enable_pruning
    if isinstance(layer, (SeparableConv2D, PQSeparableConv2d)):
        enable_pruning_depthwise = enable_pruning_pointwise = True
        if layer.name + "_depthwise" in config.pruning_parameters.disable_pruning_for_layers:
            enable_pruning_depthwise = False
        if layer.name + "_pointwise" in config.pruning_parameters.disable_pruning_for_layers:
            enable_pruning_pointwise = False
        return enable_pruning_depthwise, enable_pruning_pointwise
    else:
        if layer.name in config.pruning_parameters.disable_pruning_for_layers:
            enable_pruning = False
        return enable_pruning


def _populate_config_with_all_layers(model, config):
    """Create a default config, where all the layers are added to the disable_pruning list, and have their
    own default quantization bits in layer_specific. By default input/output quantization is disabled.
    """
    custom_scheme = {"layer_specific": {}, "disable_pruning_for_layers": []}
    for layer in model.layers:
        if isinstance(layer, (Dense, Conv2D, Conv1D, DepthwiseConv2D, PQWeightBiasBase, PQDepthwiseConv2d)):
            if layer.use_bias:
                custom_scheme["layer_specific"][layer.name] = {
                    "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    "input": {"quantize_input": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "output": {"quantize_input": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            else:
                custom_scheme["layer_specific"][layer.name] = {
                    "input": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                    "weight": {"integer_bits": 0, "fractional_bits": 7},
                    "bias": {"integer_bits": 0, "fractional_bits": 7},
                    "output": {"integer_bits": 0, "fractional_bits": 7, "quantize": True},
                }
            if hasattr(layer.activation, "__name__") and layer.activation.__name__ in ["relu", "tanh"]:
                custom_scheme["layer_specific"][layer.name][layer.activation.__name__] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            custom_scheme["disable_pruning_for_layers"].append(layer.name)
        if isinstance(layer, (SeparableConv2D, PQSeparableConv2d)):
            if layer.use_bias:
                custom_scheme["layer_specific"][layer.name] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "depthwise": {
                        "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    },
                    "pointwise": {
                        "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                        "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
                    },
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            else:
                custom_scheme["layer_specific"][layer.name] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "depthwise": {
                        "weight": {
                            "integer_bits": 0.0,
                            "fractional_bits": 7.0,
                        }
                    },
                    "pointwise": {"weight": {"integer_bits": 0.0, "fractional_bits": 7.0}},
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            if hasattr(layer.activation, "__name__") and layer.activation.__name__ in ["relu", "tanh"]:
                custom_scheme["layer_specific"][layer.name][layer.activation.__name__] = {
                    "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                    "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                }
            custom_scheme["disable_pruning_for_layers"].append(layer.name + "_depthwise")
            custom_scheme["disable_pruning_for_layers"].append(layer.name + "_pointwise")
        elif isinstance(
            layer, (Activation, ReLU, AveragePooling1D, AveragePooling2D, AveragePooling3D, PQActivation, PQAvgPoolBase)
        ):
            custom_scheme["layer_specific"][layer.name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "output": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
            }
        elif isinstance(layer, (BatchNormalization, PQBatchNormalization)):
            custom_scheme["layer_specific"][layer.name] = {
                "input": {"quantize": True, "integer_bits": 0.0, "fractional_bits": 7.0},
                "weight": {"integer_bits": 0.0, "fractional_bits": 7.0},
                "bias": {"integer_bits": 0.0, "fractional_bits": 7.0},
            }
    config.quantization_parameters.layer_specific = custom_scheme["layer_specific"]
    config.pruning_parameters.disable_pruning_for_layers = custom_scheme["disable_pruning_for_layers"]
    return config


def post_training_prune(model, config, calibration_data):
    t_delta = config.pruning_parameters.t_delta
    config.pruning_parameters.t_start_collecting_batch = 0

    for i in range(t_delta):
        inputs = calibration_data[i]
        if i == 0:
            model = add_compression_layers(model, config, inputs.shape)
            post_pretrain_functions(model, config)
        model(inputs, training=True)  # True so pruning works
    return apply_final_compression(model)


def get_ebops(model, **kwargs):
    ebops = 0
    for m in model.layers:
        if isinstance(m, PQWeightBiasBase):
            ebops += m.ebops(include_mask=m.enable_pruning)
        elif isinstance(m, (PQAvgPoolBase, PQBatchNormalization, PQActivation, PQSoftmax, PQMultiheadAttention)):
            ebops += m.ebops()
    return ebops
