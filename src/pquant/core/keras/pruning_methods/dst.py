import keras
import numpy as np
from keras import ops


def get_threshold_size(config, weight_shape):
    if config.pruning_parameters.threshold_type == "layerwise":
        return (1, 1)
    elif config.pruning_parameters.threshold_type == "channelwise":
        return (weight_shape[0], 1)
    elif config.pruning_parameters.threshold_type == "weightwise":
        return (weight_shape[0], np.prod(weight_shape[1:]))


@ops.custom_gradient
def binary_step(weight):
    output = ops.cast(weight > 0, dtype=weight.dtype)

    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        abs_weight = ops.abs(weight)
        idx_lt04 = ops.where(abs_weight <= 0.4, 2 - 4 * abs_weight, 0.0)
        idx_04to1 = ops.where(ops.logical_and(abs_weight > 0.4, abs_weight <= 1.0), 0.4, 0.0)
        idx_gt1 = ops.where(abs_weight > 1.0, 0.0, 0.0)
        grads = idx_lt04 + idx_04to1 + idx_gt1
        return grads * upstream

    return output, grad


@keras.saving.register_keras_serializable(package="PQuant")
class DST(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False

    def build(self, input_shape):
        self.threshold_size = get_threshold_size(self.config, input_shape)
        self.threshold = self.add_weight(shape=self.threshold_size, initializer="zeros", trainable=True)
        self.mask = self.add_weight(shape=input_shape, initializer="ones", trainable=False)
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

    def call(self, weight):
        """
        ReLu(|W| - T), with gradient:
            2 - 4*|W| if |W| <= 0.4
            0.4           if 0.4 < |W| <= 1
            0             if |W| > 1
        """
        use_current_mask = ops.logical_or(self.is_pretraining, self.is_finetuning)

        def use_existing():
            return weight * ops.convert_to_tensor(self.mask)

        def compute_new():
            mask = self.get_mask(weight)
            ratio = 1.0 - ops.sum(mask) / ops.cast(ops.size(mask), mask.dtype)
            flag = ratio >= self.config.pruning_parameters.max_pruning_pct

            def reset_and_recalculate():
                self.threshold.assign(ops.zeros(self.threshold.shape))
                return self.get_mask(weight)

            mask = ops.cond(flag, reset_and_recalculate, lambda: mask)
            self.mask.assign(mask)
            return weight * mask

        result = ops.cond(use_current_mask, use_existing, compute_new)
        self.add_loss(self.calculate_additional_loss())
        return result

    def get_hard_mask(self, weight=None):
        return self.mask

    def get_mask(self, weight):
        weight_orig_shape = weight.shape
        weights_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        pre_binarystep_weights = ops.abs(weights_reshaped) - self.threshold
        mask = binary_step(pre_binarystep_weights)
        mask = ops.reshape(mask, weight_orig_shape)
        return mask

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_mask(weight)) / ops.size(weight)

    def calculate_additional_loss(self):
        if self._is_pretraining or self._is_finetuning:
            return ops.cast(0.0, self.threshold.dtype)
        return self.config.pruning_parameters.alpha * ops.sum(ops.exp(-self.threshold))

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(True)

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(False)

    def post_round_function(self):
        pass

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "config": self.config.get_dict(),
                "layer_type": self.layer_type,
            }
        )
        return config
