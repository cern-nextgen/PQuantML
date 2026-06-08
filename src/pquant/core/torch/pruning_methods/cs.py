import torch
import torch.nn as nn


class ContinuousSparsification(nn.Module):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.final_temp = float(config.pruning_parameters.final_temp)
        self._is_finetuning = False
        self._is_pretraining = True
        self.is_pretraining = True
        self.is_finetuning = False
        self.layer_type = layer_type
        self.built = False

    def build(self, input_shape):
        if self.built:
            return
        init_val = float(self.config.pruning_parameters.threshold_init)
        s_init = torch.full(tuple(input_shape), init_val)
        self.s = nn.Parameter(s_init.clone())
        self.register_buffer("s_init", s_init.clone())
        self.register_buffer("scaling", 1.0 / torch.sigmoid(s_init))
        self.register_buffer("beta", torch.tensor(1.0))
        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.built = True

    def forward(self, weight):
        if self._is_pretraining or self._is_finetuning:
            return self.mask.to(weight.dtype) * weight
        new_mask = self.get_mask()
        with torch.no_grad():
            self.mask.copy_(new_mask.detach())
        return new_mask * weight

    def pre_finetune_function(self):
        self._is_finetuning = True
        self.is_finetuning = True
        with torch.no_grad():
            self.mask.copy_(self.get_hard_mask().to(self.mask.dtype))

    def get_mask(self):
        return torch.sigmoid(self.beta * self.s) * self.scaling

    def post_pre_train_function(self):
        self._is_pretraining = False
        self.is_pretraining = False

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    @torch.no_grad()
    def post_epoch_function(self, epoch, total_epochs):
        if total_epochs <= 1:
            self.beta.mul_(self.final_temp)
        else:
            self.beta.mul_(self.final_temp ** (1 / (total_epochs - 1)))

    def get_hard_mask(self, weight=None):
        if self.config.pruning_parameters.enable_pruning:
            return (self.s > 0).to(self.s.dtype)
        return torch.tensor(1.0, device=self.s.device, dtype=self.s.dtype)

    @torch.no_grad()
    def post_round_function(self):
        new_s = torch.minimum(self.beta * self.s, self.s_init)
        self.s.data.copy_(new_s)
        self.beta.fill_(1.0)

    def calculate_additional_loss(self):
        return self.config.pruning_parameters.threshold_decay * torch.norm(self.get_mask().reshape(-1), p=1)

    def get_layer_sparsity(self, weight):
        return self.get_hard_mask().sum() / weight.numel()
