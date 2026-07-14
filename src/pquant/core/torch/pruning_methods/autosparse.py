import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_PI = math.pi
_L0 = -6.0
_L1 = 6.0


def cosine_decay(i, T):
    return (1 + math.cos(_PI * i / T)) / 2


def sigmoid_decay(i, T):
    x = _L0 + (_L1 - _L0) * i / T
    return 1.0 - 1.0 / (1.0 + math.exp(-x))


def cosine_sigmoid_decay(i, T):
    return max(cosine_decay(i, T), sigmoid_decay(i, T))


def get_threshold_size(config, weight_shape):
    if config.pruning_parameters.threshold_type == "layerwise":
        return (1, 1)
    if config.pruning_parameters.threshold_type == "channelwise":
        return (weight_shape[0], 1)
    if config.pruning_parameters.threshold_type == "weightwise":
        return (weight_shape[0], int(np.prod(weight_shape[1:])))
    return None


class _AutoSparsePrune(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, backward_sparsity_flag, backward_sparsity):
        mask = torch.relu(x)
        kth_value = None
        if backward_sparsity_flag:
            flat = x.reshape(-1)
            k = max(int(flat.numel() * backward_sparsity), 1)
            topk_vals, _ = torch.topk(flat, k)
            kth_value = topk_vals[-1]
        ctx.save_for_backward(x, alpha, kth_value if kth_value is not None else torch.zeros((), device=x.device))
        ctx.backward_sparsity_flag = backward_sparsity_flag
        return mask

    @staticmethod
    def backward(ctx, upstream):
        x, alpha, kth_value = ctx.saved_tensors
        grads = torch.where(x <= 0, alpha.to(x.dtype), torch.ones_like(x))
        if ctx.backward_sparsity_flag:
            grads = torch.where(x < kth_value, torch.zeros_like(grads), grads)
        return grads * upstream, None, None, None


def autosparse_prune(x, alpha, backward_sparsity_flag, backward_sparsity):
    return _AutoSparsePrune.apply(x, alpha, backward_sparsity_flag, backward_sparsity)


class AutoSparse(nn.Module):
    mask: Tensor
    alpha: Tensor

    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.layer_type = layer_type
        self._alpha_init = float(config.pruning_parameters.alpha)
        self._backward_sparsity_flag = bool(config.pruning_parameters.backward_sparsity)
        self._backward_sparsity = 0.5
        self._is_pretraining = True
        self._is_finetuning = False
        self.is_pretraining = True
        self.is_finetuning = False
        self.built = False

    def build(self, input_shape):
        if self.built:
            return
        threshold_size = get_threshold_size(self.config, input_shape)
        self.threshold = nn.Parameter(torch.full(threshold_size, float(self.config.pruning_parameters.threshold_init)))
        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.register_buffer("alpha", torch.tensor(self._alpha_init))
        self.built = True

    def _g(self, x):
        return torch.sigmoid(x)

    def forward(self, weight):
        if self._is_pretraining:
            return weight
        if self._is_finetuning:
            return self.mask.to(weight.dtype) * weight

        weight_reshaped = weight.reshape(weight.shape[0], -1)
        w_t = weight_reshaped.abs() - torch.sigmoid(self.threshold)

        new_binary_mask = (w_t > 0).to(weight.dtype).reshape(weight.shape)
        with torch.no_grad():
            self.mask.copy_(new_binary_mask)

        return torch.sign(weight) * autosparse_prune(
            w_t, self.alpha, self._backward_sparsity_flag, self._backward_sparsity
        ).reshape(weight.shape)

    def get_hard_mask(self, weight=None):
        return self.mask

    def get_mask(self, weight):
        weight_reshaped = weight.reshape(weight.shape[0], -1)
        w_t = weight_reshaped.abs() - torch.sigmoid(self.threshold)
        return (w_t > 0).to(weight.dtype).reshape(weight.shape)

    def get_layer_sparsity(self, weight):
        m = self.get_mask(weight)
        return m.count_nonzero() / m.numel()

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def calculate_additional_loss(self):
        return 0.0

    def pre_finetune_function(self):
        self._is_finetuning = True
        self.is_finetuning = True

    def post_round_function(self):
        pass

    def post_pre_train_function(self):
        self._is_pretraining = False
        self.is_pretraining = False

    @torch.no_grad()
    def post_epoch_function(self, epoch, total_epochs):
        decay = cosine_sigmoid_decay(epoch, total_epochs)
        self.alpha.fill_(self._alpha_init * decay)
        if epoch >= self.config.pruning_parameters.alpha_reset_epoch:
            self.alpha.zero_()
