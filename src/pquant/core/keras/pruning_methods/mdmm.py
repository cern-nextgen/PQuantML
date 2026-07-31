# @Author: Arghya Ranjan Das
# file: src/pquant/pruning_methods/mdmm.py
# modified by:


import inspect

import keras
from keras import ops

from pquant.core.keras.pruning_methods.constraint_functions import (
    EqualityConstraint,
    GreaterThanOrEqualConstraint,
    LessThanOrEqualConstraint,
)
from pquant.core.keras.pruning_methods.metric_functions import (
    StructuredSparsityMetric,
    UnstructuredSparsityMetric,
)

METRIC_REGISTRY = {
    "UnstructuredSparsity": UnstructuredSparsityMetric,
    "StructuredSparsity": StructuredSparsityMetric,
}

CONSTRAINT_REGISTRY = {
    "Equality": EqualityConstraint,
    "LessThanOrEqual": LessThanOrEqualConstraint,
    "GreaterThanOrEqual": GreaterThanOrEqualConstraint,
}

# -------------------------------------------------------------------
#                   MDMM Layer
# -------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="PQuant")
class MDMM(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.layer_type = layer_type
        self.constraint_layer = None
        self._is_finetuning = False
        self._is_pretraining = True
        # TEMP: cache last penalty so calculate_additional_loss() works in
        # custom training loops via get_model_losses(). Remove once the
        # add_loss()/model.fit path is the only supported path.
        self._last_penalty = None

    def build(self, input_shape):
        pruning_parameters = self.config.pruning_parameters
        metric_type = pruning_parameters.metric_type
        constraint_type = pruning_parameters.constraint_type
        target_value = pruning_parameters.target_value
        target_sparsity = pruning_parameters.target_sparsity
        l0_mode = pruning_parameters.l0_mode
        scale_mode = pruning_parameters.scale_mode

        candidate_kwargs = {
            "epsilon": pruning_parameters.epsilon,
            "target_sparsity": target_sparsity,
            "l0_mode": l0_mode,
            "scale_mode": scale_mode,
            "rf": pruning_parameters.rf,
        }

        metric_cls = METRIC_REGISTRY.get(metric_type)
        sig = inspect.signature(getattr(metric_cls, "__init__", metric_cls))
        metric_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None and k in sig.parameters}
        if metric_cls:
            metric_fn = metric_cls(**metric_kwargs)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        common_args = {
            "metric_fn": metric_fn,
            "target_value": target_value,
            "scale": self.config.pruning_parameters.scale,
            "damping": self.config.pruning_parameters.damping,
            "use_grad": self.config.pruning_parameters.use_grad,
            "lr": self.config.pruning_parameters.constraint_lr,
        }

        constraint_type_cls = CONSTRAINT_REGISTRY.get(constraint_type)
        if constraint_type_cls:
            self.constraint_layer = constraint_type_cls(**common_args)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        self.mask = self.add_weight(name="mask", shape=input_shape, initializer="ones", trainable=False)
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
        self.constraint_layer.build(input_shape)
        super().build(input_shape)

    def call(self, weight):
        epsilon = self.config.pruning_parameters.epsilon
        hard_mask = ops.cast(ops.abs(weight) > epsilon, weight.dtype)
        not_active = ops.logical_or(self.is_pretraining, self.is_finetuning)
        self.mask.assign(ops.where(not_active, ops.convert_to_tensor(self.mask), hard_mask))

        penalty = ops.sum(self.constraint_layer(weight))
        gated_penalty = ops.where(not_active, ops.zeros_like(penalty), penalty)
        self.add_loss(gated_penalty)
        # TEMP: cache for calculate_additional_loss() — remove with the
        # _last_penalty attribute once custom-loop callers move to model.losses.
        self._last_penalty = gated_penalty
        return ops.where(self.is_finetuning, weight * hard_mask, weight)

    def get_hard_mask(self, weight=None):
        if weight is None:
            return ops.convert_to_tensor(self.mask)
        epsilon = self.config.pruning_parameters.epsilon
        return ops.cast(ops.abs(weight) > epsilon, weight.dtype)

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask(weight)) / ops.size(weight)

    def calculate_additional_loss(self):
        # Loss is added via self.add_loss() in call() for model.fit.
        # TEMP: also return the cached penalty so custom training loops using
        # get_model_losses() see the constraint term. Remove this branch (and
        # the _last_penalty cache) once those callers switch to model.losses;
        # then this can revert to `return 0.0`.
        if self._last_penalty is not None:
            return self._last_penalty
        return 0.0

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(True)
        if hasattr(self.constraint_layer, "module"):
            self.constraint_layer.module.turn_off()
        else:
            self.constraint_layer.turn_off()

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
