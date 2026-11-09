# @Author: Arghya Ranjan Das
# file: src/pquant/pruning_methods/mdmm.py
# modified by:


import keras
from keras import ops

from pquant.core.keras.pruning_methods import constraint_functions, metric_functions
from pquant.data_models.pruning_model import CONSTRAINT_REGISTRY, METRIC_REGISTRY

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
        metric_model = pruning_parameters.metric
        metric_class = getattr(metric_functions, METRIC_REGISTRY[metric_model.metric_type])
        metric_function = metric_class(**metric_model.runtime_kwargs(pruning_parameters))

        constraint_kwargs = {
            "metric_fn": metric_function,
            "target_value": pruning_parameters.target_value,
            "scale": pruning_parameters.scale,
            "damping": pruning_parameters.damping,
            "use_grad": pruning_parameters.use_grad,
            "lr": pruning_parameters.constraint_lr,
        }
        constraint_class = getattr(constraint_functions, CONSTRAINT_REGISTRY[pruning_parameters.constraint_type])
        self.constraint_layer = constraint_class(**constraint_kwargs)

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

    def _compute_hard_mask(self, weight, epsilon):
        # During fine-tuning, a metric that defines its own projection (e.g. PACA pattern
        # pruning) supplies the mask; otherwise use the magnitude threshold. The layer only
        # checks for the capability, so it stays metric-agnostic (no metric-type branching).
        # The phase switch must be the backend variable, selected with ops.where: a Python
        # bool is frozen into the graph at trace time under tf.function, and flipping it
        # mid-fit (pre_finetune_functions) never triggers a retrace.
        metric_fn = getattr(self.constraint_layer, "metric_fn", None)
        magnitude_mask = ops.cast(ops.abs(weight) > epsilon, weight.dtype)
        if not hasattr(metric_fn, "get_projection_mask"):
            return magnitude_mask
        projection_mask = ops.cast(metric_fn.get_projection_mask(weight), weight.dtype)
        return ops.where(ops.convert_to_tensor(self.is_finetuning), projection_mask, magnitude_mask)

    def call(self, weight):
        epsilon = self.config.pruning_parameters.epsilon
        hard_mask = self._compute_hard_mask(weight, epsilon)
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
        # Route through _compute_hard_mask so sparsity/EBOPs reporting agrees with what the
        # forward pass actually applies (during PACA fine-tuning that is the projection
        # mask, not the magnitude threshold).
        return self._compute_hard_mask(weight, self.config.pruning_parameters.epsilon)

    def get_layer_sparsity(self, weight):
        # Cast size to the mask dtype: ops.sum(mask) is float but ops.size is int, and the
        # TensorFlow backend rejects float/int division (the original float32/int32 bug).
        mask = self.get_hard_mask(weight)
        return ops.sum(mask) / ops.cast(ops.size(weight), mask.dtype)

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
