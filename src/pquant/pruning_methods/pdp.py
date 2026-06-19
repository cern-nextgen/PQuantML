import keras
from keras import ops


@keras.saving.register_keras_serializable(package="PQuant")
class PDP(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.init_r = ops.convert_to_tensor(config.pruning_parameters.sparsity)
        self.epsilon = ops.convert_to_tensor(config.pruning_parameters.epsilon)
        self.r = config.pruning_parameters.sparsity
        self.temp = config.pruning_parameters.temperature
        self.is_pretraining = True
        self.config = config
        self.is_finetuning = False
        self.layer_type = layer_type

    def build(self, input_shape):
        input_shape_concatenated = list(input_shape) + [1]
        self.softmax_shape = input_shape_concatenated
        self.t = ops.ones(input_shape_concatenated) * 0.5
        if self.config.pruning_parameters.structured_pruning:
            if self.layer_type == "linear":
                shape = (input_shape[0], 1)
            else:
                if len(input_shape) == 3:
                    shape = (input_shape[0], 1, 1)
                else:
                    shape = (input_shape[0], 1, 1, 1)
        self.mask = self.add_weight(shape=shape, initializer="ones", name="mask", trainable=False)
        self.flat_weight_size = ops.cast(ops.size(self.mask), self.mask.dtype)
        super().build(input_shape)

    def post_pre_train_function(self):
        self.is_pretraining = False  # Enables pruning

    def pre_epoch_function(self, epoch, total_epochs):
        if not self.is_pretraining:
            self.r = ops.minimum(1.0, self.epsilon * (epoch + 1)) * self.init_r

    def post_round_function(self):
        pass

    def get_hard_mask(self, weight=None):
        if self.is_finetuning:
            return self.mask
        if weight is None:
            return ops.cast((self.mask >= 0.5), self.mask.dtype)
        if self.config.pruning_parameters.structured_pruning:
            if self.layer_type == "conv":
                mask = self.get_mask_structured_channel(weight)
            else:
                mask = self.get_mask_structured_linear(weight)
        else:
            mask = self.get_mask(weight)
        self.mask.assign(ops.cast((mask >= 0.5), mask.dtype))
        return self.mask

    def pre_finetune_function(self):
        self.is_finetuning = True
        if hasattr(self, "mask"):
            self.mask.assign(ops.cast((self.mask >= 0.5), self.mask.dtype))

    def get_mask_structured_linear(self, weight):
        """
        Structured pruning. Use the l2 norm of the neurons instead of the absolute weight values
        to calculate threshold point t. Prunes whole neurons.
        """
        if self.is_pretraining:
            return self.mask
        norm = ops.norm(weight, axis=1, ord=2, keepdims=True)
        norm_flat = ops.ravel(norm)
        """ Do top_k for all neuron norms. Returns sorted array, just use the values on both
        sides of the threshold (sparsity * size(norm)) to calculate t directly """
        W_all, _ = ops.top_k(norm_flat, ops.size(norm_flat))
        size = ops.cast(ops.size(W_all), self.mask.dtype)
        ind = ops.cast((1 - self.r) * size, "int32") - 1
        lim = ops.clip(ind, 0, ops.cast(size - 2, "int32"))
        Wh = W_all[lim]
        Wt = W_all[lim + 1]
        # norm = ops.expand_dims(norm, -1)
        t = ops.ones(norm.shape) * 0.5 * (Wh + Wt)
        soft_input = ops.concatenate((t**2, norm**2), axis=1) / self.temp
        softmax_result = ops.softmax(soft_input, axis=1)
        _, mw = ops.unstack(softmax_result, axis=1)
        mw = ops.expand_dims(mw, -1)
        self.mask.assign(mw)
        return mw

    def get_mask_structured_channel(self, weight):
        """
        Structured pruning. Use the l2 norm of the channels instead of the absolute weight values
        to calculate threshold point t. Prunes whole channels.
        """
        if self.is_pretraining:
            return self.mask
        weight_reshaped = ops.reshape(weight, (weight.shape[0], -1))
        norm = ops.norm(weight_reshaped, axis=1, ord=2)

        norm_flat = ops.ravel(norm)
        """ Do top_k for all channel norms. Returns sorted array, just use the values on both
        sides of the threshold (sparsity * size(norm)) to calculate t directly """
        W_all, _ = ops.top_k(norm_flat, ops.size(norm_flat))
        size = ops.cast(ops.size(W_all), self.mask.dtype)
        ind = ops.cast((1 - self.r) * size, "int32") - 1
        lim = ops.clip(ind, 0, ops.cast(size - 2, "int32"))

        Wh = W_all[lim]
        Wt = W_all[lim + 1]
        norm = ops.expand_dims(norm, -1)
        t = ops.ones(norm.shape) * 0.5 * (Wh + Wt)
        soft_input = ops.concatenate((t**2, norm**2), axis=-1) / self.temp
        softmax_result = ops.softmax(soft_input, axis=-1)
        zw, mw = ops.unstack(softmax_result, axis=-1)
        diff = len(weight.shape) - len(mw.shape)
        for _ in range(diff):
            mw = ops.expand_dims(mw, -1)
        self.mask.assign(mw)
        return mw

    def get_mask(self, weight):
        if self.is_pretraining:
            return self.mask
        weight_reshaped = ops.reshape(weight, self.softmax_shape)
        abs_weight_flat = ops.ravel(ops.abs(weight))
        """ Do top_k for all weights. Returns sorted array, just use the values on both
        sides of the threshold (sparsity * size(weight)) to calculate t directly """
        all, _ = ops.top_k(abs_weight_flat, ops.size(abs_weight_flat))
        ind = ops.cast((1 - self.r) * self.flat_weight_size, "int32") - 1  # Index begins from 0
        lim = ops.clip(ind, 0, ops.cast(self.flat_weight_size - 2, "int32"))
        Wh = all[lim]
        Wt = all[lim + 1]
        t = self.t * (Wh + Wt)
        soft_input = ops.concatenate((t**2, weight_reshaped**2), axis=-1) / self.temp
        softmax_result = ops.softmax(soft_input, axis=-1)
        _, mw = ops.unstack(softmax_result, axis=-1)
        mask = ops.reshape(mw, weight.shape)
        self.mask.assign(mask)
        return mask

    def call(self, weight):
        if self.is_finetuning:
            mask = self.mask
        else:
            if self.config.pruning_parameters.structured_pruning:
                if self.layer_type == "conv":
                    mask = self.get_mask_structured_channel(weight)
                else:
                    mask = self.get_mask_structured_linear(weight)
            else:
                mask = self.get_mask(weight)
        return mask * weight

    def calculate_additional_loss(self):
        return 0

    def get_layer_sparsity(self, weight):
        masked_weight_rounded = ops.cast((self.mask >= 0.5), self.mask.dtype)
        masked_weight = masked_weight_rounded * weight
        return ops.count_nonzero(masked_weight) / ops.size(masked_weight)

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.get_dict(), "layer_type": self.layer_type})
        return config
