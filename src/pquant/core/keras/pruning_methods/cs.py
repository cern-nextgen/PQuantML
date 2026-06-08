import keras
from keras import ops
from keras.initializers import Constant


@keras.saving.register_keras_serializable(package="PQuant")
class ContinuousSparsification(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.final_temp = config.pruning_parameters.final_temp
        self._is_finetuning = False
        self.layer_type = layer_type
        self._is_pretraining = True

    def build(self, input_shape):
        self.s_init = ops.convert_to_tensor(self.config.pruning_parameters.threshold_init * ops.ones(input_shape))
        self.s = self.add_weight(name="threshold", shape=input_shape, initializer=Constant(self.s_init), trainable=True)
        self.scaling = 1.0 / ops.sigmoid(self.s_init)
        self.beta = self.add_weight(name="beta", shape=(), initializer=Constant(1.0), trainable=False)
        self.mask = self.add_weight(name="mask", shape=input_shape, initializer=Constant(1.0), trainable=False)
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
        stored_mask = ops.convert_to_tensor(self.mask)
        new_mask = self.get_mask()
        use_current_mask = ops.logical_or(self.is_pretraining, self.is_finetuning)
        updated_mask = ops.where(use_current_mask, stored_mask, new_mask)
        self.mask.assign(updated_mask)
        return updated_mask * weight

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(True)
        if hasattr(self, "mask"):
            self.mask.assign(self.get_hard_mask())

    def get_mask(self):
        return ops.sigmoid(self.beta * self.s) * self.scaling

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(False)

    def pre_epoch_function(self, epoch, total_epochs):  # noqa: ARG002
        pass

    def post_epoch_function(self, epoch, total_epochs):  # noqa: ARG002
        if total_epochs <= 1:
            self.beta.assign(self.beta * self.final_temp)
        else:
            self.beta.assign(self.beta * self.final_temp ** (1 / (total_epochs - 1)))

    def get_hard_mask(self, weight=None):  # noqa: ARG002
        if self.config.pruning_parameters.enable_pruning:
            return ops.cast((self.s > 0), self.s.dtype)
        return ops.convert_to_tensor(1.0)

    def post_round_function(self):
        min_beta_s_s0 = ops.minimum(self.beta * self.s, self.s_init)
        self.s.assign(min_beta_s_s0)
        self.beta.assign(1.0)

    def calculate_additional_loss(self):
        return ops.convert_to_tensor(
            self.config.pruning_parameters.threshold_decay * ops.norm(ops.ravel(self.get_mask()), ord=1)
        )

    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_hard_mask()) / ops.size(weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "layer_type": self.layer_type,
            }
        )
        return config
