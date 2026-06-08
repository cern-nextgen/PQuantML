import keras
from keras import ops


@keras.saving.register_keras_serializable(package="Layers")
class ActivationPruning(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        super().__init__(*args, **kwargs)
        self.config = config
        self.act_type = "relu"
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False
        self.threshold = ops.convert_to_tensor(config.pruning_parameters.threshold)
        self.t_start_collecting_batch = self.config.pruning_parameters.t_start_collecting_batch

    def build(self, input_shape):
        self.shape = (input_shape[0], 1)
        if self.layer_type in ("conv", "depthwise_conv"):
            if len(input_shape) == 3:
                self.shape = (input_shape[0], 1, 1)
            else:
                self.shape = (input_shape[0], 1, 1, 1)
        n_channels = input_shape[0]
        self.mask = self.add_weight(shape=self.shape, initializer="ones", trainable=False)
        self.mask_placeholder = self.add_weight(shape=self.shape, initializer="ones", trainable=False)
        self.activations = self.add_weight(shape=(n_channels,), initializer="zeros", trainable=False)
        self.batches_collected = self.add_weight(shape=(), initializer="zeros", trainable=False, dtype="int32")
        self.t = self.add_weight(shape=(), initializer="zeros", trainable=False, dtype="int32")
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

    def collect_output(self, output, training):
        """
        Accumulates per-channel activity fractions. Every t_delta batches, updates
        mask_placeholder. The actual mask used in call() is updated from mask_placeholder
        in post_epoch_function (outside the compiled graph, no step-to-step dependency).
        """
        if not training:
            return

        t_delta = self.config.pruning_parameters.t_delta
        # Collect only when training and if above the starting point of collecting
        should_collect = ops.logical_not(
            ops.logical_or(
                ops.logical_or(self.is_pretraining, self.is_finetuning),
                self.t < self.t_start_collecting_batch,
            )
        )

        # Per-channel mean activity fraction
        gt_zero = ops.cast(output > 0, output.dtype)
        if self.layer_type == "linear":
            per_channel = ops.mean(gt_zero, axis=0)
        else:
            # output is channels-first (batch, channels, ...); average over batch + spatial
            axes = (0,) + tuple(range(2, len(output.shape)))
            per_channel = ops.mean(gt_zero, axis=axes)

        # Snapshot current state
        activations_cur = ops.convert_to_tensor(self.activations)
        batches_cur = ops.convert_to_tensor(self.batches_collected)
        mask_ph_cur = ops.convert_to_tensor(self.mask_placeholder)

        # Accumulate (gated by should_collect)
        new_activations = activations_cur + ops.where(should_collect, per_channel, ops.zeros_like(per_channel))
        new_batches = batches_cur + ops.cast(should_collect, "int32")

        # Update mask_placeholder every t_delta batches
        should_update = ops.logical_and(
            should_collect,
            ops.equal(new_batches % t_delta, 0),
        )

        safe_batches = ops.cast(ops.maximum(new_batches, 1), new_activations.dtype)
        pct_active = new_activations / safe_batches
        new_mask_ph = self._compute_mask(pct_active)

        self.mask_placeholder.assign(ops.where(should_update, new_mask_ph, mask_ph_cur))

        # Reset accumulators after mask update, else keep accumulated values
        self.activations.assign(ops.where(should_update, ops.zeros_like(new_activations), new_activations))
        self.batches_collected.assign(ops.where(should_update, ops.zeros_like(new_batches), new_batches))
        self.t.assign(ops.where(should_update, ops.zeros_like(self.t), self.t))

    def _compute_mask(self, pct_active):
        binary = ops.cast(pct_active > self.threshold, pct_active.dtype)
        return ops.reshape(binary, self.shape)

    def call(self, weight):
        stored_mask = ops.convert_to_tensor(self.mask)
        return ops.where(self.is_pretraining, weight, stored_mask * weight)

    def get_hard_mask(self, weight=None):  # noqa: ARG002
        return ops.convert_to_tensor(self.mask)

    def post_pre_train_function(self):
        self._is_pretraining = False
        if hasattr(self, "is_pretraining"):
            self.is_pretraining.assign(False)

    def pre_epoch_function(self, epoch, total_epochs, **kwargs):  # noqa: ARG002
        pass

    def post_round_function(self):
        pass

    def pre_finetune_function(self):
        self._is_finetuning = True
        if hasattr(self, "is_finetuning"):
            self.is_finetuning.assign(True)

    def calculate_additional_loss(self):
        return 0

    def get_layer_sparsity(self, weight):
        pass

    def post_epoch_function(self, epoch, total_epochs, **kwargs):  # noqa: ARG002
        if not self._is_pretraining:
            self.t.assign_add(1)
        self.mask.assign(ops.convert_to_tensor(self.mask_placeholder))

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.get_dict(), "layer_type": self.layer_type})
        return config
