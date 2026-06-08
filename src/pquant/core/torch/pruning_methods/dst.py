import numpy as np
import torch
import torch.nn as nn


def get_threshold_size(config, weight_shape):
    if config.pruning_parameters.threshold_type == "layerwise":
        return (1, 1)
    elif config.pruning_parameters.threshold_type == "channelwise":
        return (weight_shape[0], 1)
    elif config.pruning_parameters.threshold_type == "weightwise":
        return (weight_shape[0], int(np.prod(weight_shape[1:])))


class _BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return (weight > 0).to(weight.dtype)

    @staticmethod
    def backward(ctx, upstream):
        (weight,) = ctx.saved_tensors
        abs_w = weight.abs()
        idx_lt04 = torch.where(abs_w <= 0.4, 2 - 4 * abs_w, torch.zeros_like(weight))
        idx_04to1 = torch.where((abs_w > 0.4) & (abs_w <= 1.0), torch.full_like(weight, 0.4), torch.zeros_like(weight))
        grads = idx_lt04 + idx_04to1
        return grads * upstream


def binary_step(weight):
    return _BinaryStep.apply(weight)


class DST(nn.Module):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False
        self.is_pretraining = True
        self.is_finetuning = False
        self.built = False

    def build(self, input_shape):
        if self.built:
            return
        threshold_size = get_threshold_size(self.config, input_shape)
        self.threshold = nn.Parameter(torch.zeros(threshold_size))
        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.built = True

    def forward(self, weight):
        if self._is_pretraining or self._is_finetuning:
            return weight * self.mask.to(weight.dtype)

        mask = self.get_mask(weight)
        ratio = 1.0 - mask.sum() / mask.numel()
        if float(ratio.detach()) >= self.config.pruning_parameters.max_pruning_pct:
            with torch.no_grad():
                self.threshold.data.zero_()
            mask = self.get_mask(weight)
        with torch.no_grad():
            self.mask.copy_(mask.detach())
        return weight * mask

    def get_hard_mask(self, weight=None):
        return self.mask

    def get_mask(self, weight):
        weight_orig_shape = weight.shape
        weights_reshaped = weight.reshape(weight.shape[0], -1)
        pre_binarystep = weights_reshaped.abs() - self.threshold
        mask = binary_step(pre_binarystep)
        return mask.reshape(weight_orig_shape)

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def get_layer_sparsity(self, weight):
        return self.get_mask(weight).sum() / weight.numel()

    def calculate_additional_loss(self):
        if self._is_pretraining or self._is_finetuning:
            return torch.zeros((), dtype=self.threshold.dtype, device=self.threshold.device)
        return self.config.pruning_parameters.alpha * torch.sum(torch.exp(-self.threshold))

    def pre_finetune_function(self):
        self._is_finetuning = True
        self.is_finetuning = True

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def post_pre_train_function(self):
        self._is_pretraining = False
        self.is_pretraining = False

    def post_round_function(self):
        pass
