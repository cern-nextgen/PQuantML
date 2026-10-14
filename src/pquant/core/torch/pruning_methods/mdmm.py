import torch
import torch.nn as nn

from pquant.core.torch.pruning_methods import constraint_functions, metric_functions
from pquant.data_models.pruning_model import CONSTRAINT_REGISTRY, METRIC_REGISTRY


class MDMM(nn.Module):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__()
        if isinstance(config, dict):
            from pquant.core.hyperparameter_optimization import PQConfig

            config = PQConfig.load_from_config(config)
        self.config = config
        self.layer_type = layer_type
        self.constraint_layer = None
        self._is_finetuning = False
        self._is_pretraining = True
        self.is_pretraining = True
        self.is_finetuning = False
        self._last_penalty = None
        self.built = False

    def build(self, input_shape):
        if self.built:
            return
        pruning_parameters = self.config.pruning_parameters
        metric_model = pruning_parameters.metric
        metric_class = getattr(metric_functions, METRIC_REGISTRY[metric_model.metric_type])
        metric_function = metric_class(**metric_model.runtime_kwargs(pruning_parameters))

        constraint_kwargs = {
            "metric_fn": metric_function,
            "target_value": pruning_parameters.target_value,
            "scale": pruning_parameters.scale,
            "damping": pruning_parameters.damping,
            "use_grad": pruning_parameters.use_grad,
            "lr": pruning_parameters.constraint_lr,
        }

        constraint_class = getattr(constraint_functions, CONSTRAINT_REGISTRY[pruning_parameters.constraint_type])
        self.constraint_layer = constraint_class(**constraint_kwargs)

        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.built = True

    def _compute_hard_mask(self, weight, epsilon):
        # During fine-tuning, a metric that defines its own projection (e.g. PACA pattern
        # pruning) supplies the mask; otherwise use the magnitude threshold. The layer only
        # checks for the capability, so it stays metric-agnostic (no metric-type branching).
        metric_fn = getattr(self.constraint_layer, "metric_fn", None)
        if self._is_finetuning and hasattr(metric_fn, "get_projection_mask"):
            return metric_fn.get_projection_mask(weight).to(weight.dtype)
        return (weight.abs() > epsilon).to(weight.dtype)

    def forward(self, weight):
        epsilon = self.config.pruning_parameters.epsilon
        hard_mask = self._compute_hard_mask(weight, epsilon)
        not_active = self._is_pretraining or self._is_finetuning

        if not not_active:
            with torch.no_grad():
                self.mask.copy_(hard_mask.detach())

        penalty = self.constraint_layer(weight, training=self.training).sum()

        if not_active:
            self._last_penalty = torch.zeros((), device=weight.device, dtype=weight.dtype)
        else:
            self._last_penalty = penalty

        if self._is_finetuning:
            return weight * hard_mask
        return weight

    def get_hard_mask(self, weight=None):
        if weight is None:
            return self.mask
        # Route through _compute_hard_mask so sparsity/EBOPs reporting agrees with what the
        # forward pass actually applies (during PACA fine-tuning that is the projection
        # mask, not the magnitude threshold).
        return self._compute_hard_mask(weight, self.config.pruning_parameters.epsilon)

    def get_layer_sparsity(self, weight):
        return self.get_hard_mask(weight).sum() / weight.numel()

    def calculate_additional_loss(self):
        if self._last_penalty is None:
            return 0.0
        return self._last_penalty

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def pre_finetune_function(self):
        self._is_finetuning = True
        self.is_finetuning = True
        if hasattr(self.constraint_layer, "module"):
            self.constraint_layer.module.turn_off()
        else:
            self.constraint_layer.turn_off()

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def post_pre_train_function(self):
        self._is_pretraining = False
        self.is_pretraining = False

    def post_round_function(self):
        pass
