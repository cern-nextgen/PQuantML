import keras
from keras import ops


@keras.saving.register_keras_serializable(package="PQuant")
class Wanda(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.act_type = "relu"
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False
        self.sparsity = self.config.pruning_parameters.sparsity
        self.N = self.config.pruning_parameters.N
        self.M = self.config.pruning_parameters.M
        self.t_start_collecting_batch = self.config.pruning_parameters.t_start_collecting_batch

    def build(self, input_shape):
        # input_shape is the (transposed) weight shape: (out, in) or (out, in, kH, kW)
        # For depthwise_conv, weight shape is (in_ch, depth_mult, kH, kW) so n_in = input_shape[0]
        n_in = input_shape[0] if self.layer_type == "depthwise_conv" else input_shape[1]
        self.mask = self.add_weight(shape=input_shape, initializer="ones", trainable=False)
        # Accumulate per-input-channel sum of squared inputs; shape (n_in,) known at build time.
        # Replaces storing full (batch, n_in, ...) inputs whose spatial/batch dims are unknown.
        self.inputs_sq_sum = self.add_weight(shape=(n_in,), initializer="zeros", trainable=False)
        self.batches_collected = self.add_weight(shape=(), initializer="zeros", trainable=False, dtype="int32")
        self.t = self.add_weight(shape=(), initializer="zeros", trainable=False, dtype="int32")
        self.done = self.add_weight(
            shape=(),
            initializer=lambda shape, dtype: ops.cast(ops.zeros(shape), dtype),
            trainable=False,
            dtype="bool",
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

    def collect_input(self, x, weight, training):
        """
        Accumulates per-input-channel sum-of-squares of layer inputs. After t_delta batches,
        computes mask = Wanda metric (|W| * L2_norm) once and sets done=True. One-shot pruning.
        """
        if not training:
            return

        t_delta = self.config.pruning_parameters.t_delta

        # Only collect in training stage if pruning hasn't been done already, and if above starting epoch
        should_collect = ops.logical_not(
            ops.logical_or(
                ops.logical_or(self.is_pretraining, self.is_finetuning),
                ops.logical_or(self.done, self.t < self.t_start_collecting_batch),
            )
        )

        # Per-batch per-channel sum of squared activations (shape: n_in,)
        if self.layer_type == "linear":
            per_batch_sq = ops.sum(ops.square(x), axis=0)  # (n_in,)
        else:
            # x is channels-first (batch, in_channels, ...); sum over batch + spatial
            axes = (0,) + tuple(range(2, len(x.shape)))
            per_batch_sq = ops.sum(ops.square(x), axis=axes)  # (in_channels,)

        # Snapshot current state
        sq_sum_cur = ops.convert_to_tensor(self.inputs_sq_sum)
        batches_cur = ops.convert_to_tensor(self.batches_collected)

        new_sq_sum = sq_sum_cur + ops.where(should_collect, per_batch_sq, ops.zeros_like(per_batch_sq))
        new_batches = batches_cur + ops.cast(should_collect, "int32")  # Adding 0 if not collecting

        # Prune once when t_delta batches have been collected
        should_prune = ops.equal(new_batches, ops.cast(t_delta, "int32"))

        mask_cur = ops.convert_to_tensor(self.mask)

        def do_prune():
            norm = ops.sqrt(new_sq_sum)
            return self._compute_prune_mask(norm, weight)

        new_mask = ops.cond(should_prune, do_prune, lambda: mask_cur)
        self.mask.assign(new_mask)
        self.done.assign(ops.logical_or(self.done, should_prune))

        # Reset accumulators after pruning
        self.inputs_sq_sum.assign(ops.where(should_prune, ops.zeros_like(new_sq_sum), new_sq_sum))
        self.batches_collected.assign(ops.where(should_prune, ops.zeros_like(new_batches), new_batches))

    def _compute_prune_mask(self, norm, weight):
        if self.layer_type == "linear":
            return self._handle_linear(norm, weight)
        if self.layer_type == "depthwise_conv":
            return self._handle_depthwise_conv(norm, weight)
        return self._handle_conv(norm, weight)

    def _handle_linear(self, norm, weight):
        # norm.shape = (in_features,); weight.shape = (out_features, in_features)
        metric = ops.abs(weight) * norm
        if self.N is not None and self.M is not None:
            metric_reshaped = ops.reshape(metric, (-1, self.M))
            weight_reshaped = ops.reshape(weight, (-1, self.M))
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.N / self.M)
            return ops.reshape(mask, weight.shape)
        metric_reshaped = ops.reshape(metric, (1, -1))
        weight_reshaped = ops.reshape(weight, (1, -1))
        mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.sparsity)
        return ops.reshape(mask, weight.shape)

    def _handle_conv(self, norm, weight):
        # norm.shape = (in_channels,); weight.shape = (out_channels, in_channels, ...)
        if len(weight.shape) == 3:
            norm_reshaped = ops.reshape(norm, [1] + list(norm.shape) + [1])
        else:
            norm_reshaped = ops.reshape(norm, [1] + list(norm.shape) + [1, 1])
        metric = ops.abs(weight) * norm_reshaped
        if self.N is not None and self.M is not None:
            metric_reshaped = ops.reshape(metric, (-1, self.M))
            weight_reshaped = ops.reshape(weight, (-1, self.M))
            mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.N / self.M)
            return ops.reshape(mask, weight.shape)
        metric_reshaped = ops.reshape(metric, (metric.shape[0], -1))
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.sparsity)
        return ops.reshape(mask, weight.shape)

    def _handle_depthwise_conv(self, norm, weight):
        # norm.shape = (in_channels,); weight.shape = (in_channels, depth_mult, kH, kW)
        # Prune per-input-channel: norm[ic] scales all weights for that channel
        norm_reshaped = ops.reshape(norm, list(norm.shape) + [1, 1, 1])
        metric = ops.abs(weight) * norm_reshaped
        metric_reshaped = ops.reshape(metric, (metric.shape[0], -1))
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        mask = self.get_mask(weight_reshaped, metric_reshaped, sparsity=self.sparsity)
        return ops.reshape(mask, weight.shape)

    def get_mask(self, weight, metric, sparsity):
        d0, d1 = metric.shape
        keep_idxs = ops.argsort(metric, axis=1)[:, int(d1 * sparsity) :] + ops.arange(d0)[:, None] * d1
        keep_idxs = ops.ravel(keep_idxs)
        kept_values = ops.reshape(
            ops.scatter(keep_idxs[:, None], ops.take(ops.ravel(weight), keep_idxs), ops.array((ops.size(weight),))),
            weight.shape,
        )
        return ops.cast(kept_values != 0, weight.dtype)

    def call(self, weight):
        return ops.convert_to_tensor(self.mask) * weight

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config.get_dict(),
                "layer_type": self.layer_type,
            }
        )
        return config
