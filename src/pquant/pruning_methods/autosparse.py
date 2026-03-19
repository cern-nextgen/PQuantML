import keras
import numpy as np
from keras import ops
from keras.initializers import Constant

pi = ops.convert_to_tensor(np.pi)
L0 = ops.convert_to_tensor(-6.0)
L1 = ops.convert_to_tensor(6.0)


def cosine_decay(i, T):
    return (1 + ops.cos(pi * i / T)) / 2


def sigmoid_decay(i, T):
    return 1 - ops.sigmoid(L0 + (L1 - L0) * i / T)


def cosine_sigmoid_decay(i, T):
    return ops.maximum(cosine_decay(i, T), sigmoid_decay(i, T))


def get_threshold_size(config, weight_shape):
    if config.pruning_parameters.threshold_type == "layerwise":
        return (1, 1)
    elif config.pruning_parameters.threshold_type == "channelwise":
        return (weight_shape[0], 1)
    elif config.pruning_parameters.threshold_type == "weightwise":
        return (weight_shape[0], np.prod(weight_shape[1:]))


BACKWARD_SPARSITY = False


@ops.custom_gradient
def autosparse_prune(x, alpha):
    mask = ops.relu(x)
    backward_sparsity = 0.5
    x_flat = ops.ravel(x)
    k = ops.cast(ops.cast(ops.size(x_flat), x.dtype) * backward_sparsity, "int32")
    topks, _ = ops.top_k(x_flat, k)
    kth_value = topks[-1]

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        grads = ops.where(x <= 0, alpha, 1.0)
        if BACKWARD_SPARSITY:
            grads = ops.where(x < kth_value, 0.0, grads)
        return grads * upstream, None

    return mask, grad


@keras.saving.register_keras_serializable(package="Layers")
class AutoSparse(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.g = ops.sigmoid
        self.config = config
        self.layer_type = layer_type
        self._alpha_init = float(config.pruning_parameters.alpha)
        global BACKWARD_SPARSITY
        BACKWARD_SPARSITY = config.pruning_parameters.backward_sparsity
        self._is_pretraining = True
        self._is_finetuning = False

    def build(self, input_shape):
        self.threshold_size = get_threshold_size(self.config, input_shape)
        self.threshold = self.add_weight(
            name="threshold",
            shape=self.threshold_size,
            initializer=Constant(self.config.pruning_parameters.threshold_init),
            trainable=True,
        )
        self.mask = self.add_weight(
            name="mask",
            shape=input_shape,
            initializer="ones",
            trainable=False,
        )
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=Constant(self._alpha_init),
            trainable=False,
        )
        self.is_pretraining = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.ones(shape) if self._is_pretraining else ops.zeros(shape), dtype),
            name="is_pretraining",
            trainable=False,
            dtype="bool",
        )
        self.is_finetuning = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.ones(shape) if self._is_finetuning else ops.zeros(shape), dtype),
            name="is_finetuning",
            trainable=False,
            dtype="bool",
        )
        super().build(input_shape)

    def call(self, weight):
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        w_t = ops.abs(weight_reshaped) - self.g(self.threshold)

        new_binary_mask = ops.cast(ops.reshape(w_t > 0, weight.shape), weight.dtype)
        is_training = ops.logical_not(ops.logical_or(self.is_pretraining, self.is_finetuning))
        self.mask.assign(ops.where(is_training, new_binary_mask, ops.convert_to_tensor(self.mask)))

        sparse_weight = ops.sign(weight) * ops.reshape(autosparse_prune(w_t, self.alpha), weight.shape)

        return ops.where(
            self.is_pretraining,
            weight,
            ops.where(self.is_finetuning, ops.convert_to_tensor(self.mask) * weight, sparse_weight),
        )

    def get_hard_mask(self, weight=None):  # noqa: ARG002
        return ops.convert_to_tensor(self.mask)

    def get_mask(self, weight):
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        w_t = ops.abs(weight_reshaped) - self.g(self.threshold)
        return ops.cast(ops.reshape(w_t > 0, weight.shape), weight.dtype)

    def get_layer_sparsity(self, weight):
        masked_weight = self.get_mask(weight)
        masked_count = ops.count_nonzero(masked_weight)
        return masked_count / ops.size(weight)

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def calculate_additional_loss(self):
        return 0

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(True)

    def post_round_function(self):
        pass

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(False)

    def post_epoch_function(self, epoch, total_epochs):
        decay = cosine_sigmoid_decay(epoch, total_epochs)
        self.alpha.assign(self._alpha_init * decay)
        if epoch >= self.config.pruning_parameters.alpha_reset_epoch:
            self.alpha.assign(ops.zeros_like(self.alpha))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "layer_type": self.layer_type,
            }
        )
        return config
