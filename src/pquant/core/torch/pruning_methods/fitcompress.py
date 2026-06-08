import torch
import torch.nn as nn


class FITCompress(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.is_pretraining = True
        self.is_finetuning = False
        self.built = False

    def build(self, input_shape):
        if self.built:
            return
        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.built = True

    def forward(self, weight):
        return self.mask.to(weight.dtype) * weight

    def get_hard_mask(self, weight=None):
        return self.mask

    def pre_epoch_function(self, epoch, total_epochs, **kwargs):
        pass

    def calculate_additional_loss(self):
        return 0.0

    def pre_finetune_function(self):
        self.is_finetuning = True

    def post_round_function(self):
        pass

    def post_pre_train_function(self):
        self.is_pretraining = False

    def post_epoch_function(self, epoch, total_epochs, **kwargs):
        pass
