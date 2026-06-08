import keras
from keras import ops


@keras.saving.register_keras_serializable(package="PQuant")
class PDP(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self._init_r = float(config.pruning_parameters.sparsity)
        self._epsilon = float(config.pruning_parameters.epsilon)
        self.temp = config.pruning_parameters.temperature
        self.config = config
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False

    def build(self, input_shape):
        self.softmax_shape = list(input_shape) + [1]

        structured = self.config.pruning_parameters.structured_pruning
        if structured:
            if self.layer_type == "linear":
                mask_shape = (input_shape[0], 1)
            elif len(input_shape) == 3:
                mask_shape = (input_shape[0], 1, 1)
            else:
                mask_shape = (input_shape[0], 1, 1, 1)
        else:
            mask_shape = tuple(input_shape)

        self.mask = self.add_weight(shape=mask_shape, initializer="ones", name="mask", trainable=False)
        import math

        self._mask_numel = math.prod(mask_shape)
        self.flat_weight_size = float(self._mask_numel)

        # Dynamic state as Keras variables so they survive tf.function tracing
        # and can be updated between epochs without retracing.
        self.r = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(self._init_r),
            name="r",
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

        # Resolve static config branches at build time — no runtime branching needed.
        if structured:
            self._compute_mask = self._mask_structured_channel if self.layer_type == "conv" else self._mask_structured_linear
        else:
            self._compute_mask = self._mask_unstructured

        super().build(input_shape)

    # --- Lifecycle (called outside the training graph) ---

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(False)

    def pre_epoch_function(self, epoch, _):
        if hasattr(self, "r"):
            self.r.assign(ops.minimum(1.0, self._epsilon * (epoch + 1)) * self._init_r)

    def post_round_function(self):
        pass

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(True)
        if hasattr(self, "mask"):
            self.mask.assign(ops.cast(self.mask >= 0.5, self.mask.dtype))

    def post_epoch_function(self, epoch, total_epochs):
        pass

    # --- Mask computation (graph-compatible, no Python conditionals on dynamic state) ---

    def _mask_unstructured(self, weight):
        weight_reshaped = ops.reshape(weight, self.softmax_shape)
        abs_flat = ops.ravel(ops.abs(weight))
        all_vals, _ = ops.top_k(abs_flat, self._mask_numel)
        ind = ops.cast((1 - self.r) * self.flat_weight_size, "int32") - 1
        lim = ops.clip(ind, 0, int(self.flat_weight_size) - 2)
        Wh, Wt = all_vals[lim], all_vals[lim + 1]
        t = ops.ones_like(weight_reshaped) * (0.5 * (Wh + Wt))
        soft_input = ops.concatenate((t**2, weight_reshaped**2), axis=-1) / self.temp
        _, mw = ops.unstack(ops.softmax(soft_input, axis=-1), axis=-1)
        return ops.reshape(mw, weight.shape)

    def _mask_structured_linear(self, weight):
        norm = ops.norm(weight, axis=1, ord=2, keepdims=True)
        norm_flat = ops.ravel(norm)
        W_all, _ = ops.top_k(norm_flat, self._mask_numel)
        ind = ops.cast((1 - self.r) * self.flat_weight_size, "int32") - 1
        lim = ops.clip(ind, 0, self._mask_numel - 2)
        Wh, Wt = W_all[lim], W_all[lim + 1]
        t = ops.ones(norm.shape) * 0.5 * (Wh + Wt)
        soft_input = ops.concatenate((t**2, norm**2), axis=1) / self.temp
        _, mw = ops.unstack(ops.softmax(soft_input, axis=1), axis=1)
        return ops.expand_dims(mw, -1)

    def _mask_structured_channel(self, weight):
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        norm = ops.norm(weight_reshaped, axis=1, ord=2)
        norm_flat = ops.ravel(norm)
        W_all, _ = ops.top_k(norm_flat, self._mask_numel)
        ind = ops.cast((1 - self.r) * self.flat_weight_size, "int32") - 1
        lim = ops.clip(ind, 0, self._mask_numel - 2)
        Wh, Wt = W_all[lim], W_all[lim + 1]
        norm = ops.expand_dims(norm, -1)
        t = ops.ones(norm.shape) * 0.5 * (Wh + Wt)
        soft_input = ops.concatenate((t**2, norm**2), axis=-1) / self.temp
        zw, mw = ops.unstack(ops.softmax(soft_input, axis=-1), axis=-1)
        for _ in range(len(weight.shape) - len(mw.shape)):
            mw = ops.expand_dims(mw, -1)
        return mw

    def call(self, weight):
        new_mask = self._compute_mask(weight)
        use_current_mask = ops.logical_or(self.is_pretraining, self.is_finetuning)
        mask = ops.where(use_current_mask, ops.convert_to_tensor(self.mask), new_mask)
        return mask * weight

    def update_mask(self, weight):
        """Update stored mask from current weights. Called once per epoch from post_epoch_functions."""
        if not self._is_pretraining and not self._is_finetuning:
            self.mask.assign(self._compute_mask(weight))

    # --- Utilities ---

    def get_hard_mask(self, weight=None):
        # if weight is not None and not bool(self.is_finetuning):
        #    self.mask.assign(self._compute_mask(weight))
        return ops.cast(self.mask >= 0.5, self.mask.dtype)

    def calculate_additional_loss(self):
        return 0

    def get_layer_sparsity(self, weight):
        hard_mask = ops.cast(self.mask >= 0.5, self.mask.dtype)
        masked_weight = hard_mask * weight
        return ops.count_nonzero(masked_weight) / ops.size(masked_weight)

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.get_dict(), "layer_type": self.layer_type})
        return config
