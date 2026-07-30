import math

import torch
import torch.nn as nn


class PDP(nn.Module):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self._init_r = float(config.pruning_parameters.sparsity)
        self._epsilon = float(config.pruning_parameters.epsilon)
        self.temp = float(config.pruning_parameters.temperature)
        self.config = config
        self.layer_type = layer_type
        self._is_pretraining = True
        self._is_finetuning = False
        self.is_pretraining = True
        self.is_finetuning = False
        self.built = False

    # Wanda/PDP setup code externally assigns to .init_r / .sparsity — keep as properties so
    # assignments propagate to the stored Python float used by pre_epoch_function.
    @property
    def init_r(self):
        return self._init_r

    @init_r.setter
    def init_r(self, value):
        self._init_r = float(value.detach().item()) if torch.is_tensor(value) else float(value)

    def build(self, input_shape):
        if self.built:
            return
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

        self.softmax_shape = list(input_shape) + [1]
        self._mask_numel = math.prod(mask_shape)
        self.flat_weight_size = float(self._mask_numel)

        self.register_buffer("mask", torch.ones(mask_shape))
        self.register_buffer("r", torch.tensor(self._init_r))

        if structured:
            self._compute_mask = self._mask_structured_channel if self.layer_type == "conv" else self._mask_structured_linear
        else:
            self._compute_mask = self._mask_unstructured
        self.built = True

    def post_pre_train_function(self):
        self._is_pretraining = False
        self.is_pretraining = False

    @torch.no_grad()
    def pre_epoch_function(self, epoch, _):
        if hasattr(self, "r"):
            val = min(1.0, self._epsilon * (epoch + 1)) * self._init_r
            self.r.fill_(val)

    def post_round_function(self):
        pass

    def pre_finetune_function(self):
        self._is_finetuning = True
        self.is_finetuning = True
        with torch.no_grad():
            self.mask.copy_((self.mask >= 0.5).to(self.mask.dtype))

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def _mask_unstructured(self, weight):
        weight_reshaped = weight.reshape(self.softmax_shape)
        abs_flat = weight.abs().reshape(-1)
        all_vals, _ = torch.topk(abs_flat, self._mask_numel)
        ind = int((1 - float(self.r.item())) * self.flat_weight_size) - 1
        lim = max(0, min(ind, int(self.flat_weight_size) - 2))
        Wh, Wt = all_vals[lim], all_vals[lim + 1]
        t = torch.ones_like(weight_reshaped) * (0.5 * (Wh + Wt))
        soft_input = torch.cat((t**2, weight_reshaped**2), dim=-1) / self.temp
        mw = torch.softmax(soft_input, dim=-1)[..., 1]
        return mw.reshape(weight.shape)

    def _mask_structured_linear(self, weight):
        norm = torch.norm(weight, dim=1, p=2, keepdim=True)
        norm_flat = norm.reshape(-1)
        W_all, _ = torch.topk(norm_flat, self._mask_numel)
        ind = int((1 - float(self.r.item())) * self.flat_weight_size) - 1
        lim = max(0, min(ind, self._mask_numel - 2))
        Wh, Wt = W_all[lim], W_all[lim + 1]
        t = torch.ones_like(norm) * 0.5 * (Wh + Wt)
        soft_input = torch.cat((t**2, norm**2), dim=1) / self.temp
        mw = torch.softmax(soft_input, dim=1)[..., 1]
        return mw.unsqueeze(-1)

    def _mask_structured_channel(self, weight):
        weight_reshaped = weight.reshape(weight.shape[0], -1)
        norm = torch.norm(weight_reshaped, dim=1, p=2)
        norm_flat = norm.reshape(-1)
        W_all, _ = torch.topk(norm_flat, self._mask_numel)
        ind = int((1 - float(self.r.item())) * self.flat_weight_size) - 1
        lim = max(0, min(ind, self._mask_numel - 2))
        Wh, Wt = W_all[lim], W_all[lim + 1]
        norm = norm.unsqueeze(-1)
        t = torch.ones_like(norm) * 0.5 * (Wh + Wt)
        soft_input = torch.cat((t**2, norm**2), dim=-1) / self.temp
        mw = torch.softmax(soft_input, dim=-1)[..., 1]
        while mw.dim() < weight.dim():
            mw = mw.unsqueeze(-1)
        return mw

    def forward(self, weight):
        if self._is_pretraining or self._is_finetuning:
            return self.mask.to(weight.dtype) * weight
        new_mask = self._compute_mask(weight)
        self.mask.data = new_mask
        return self.mask * weight

    def get_hard_mask(self, weight=None):
        return (self.mask >= 0.5).to(self.mask.dtype)

    def calculate_additional_loss(self):
        return 0.0

    def get_layer_sparsity(self, weight):
        hard_mask = (self.mask >= 0.5).to(self.mask.dtype)
        masked_weight = hard_mask * weight
        return masked_weight.count_nonzero() / masked_weight.numel()
