import inspect

import torch
import torch.nn as nn

from pquant.core.torch.pruning_methods.constraint_functions import (
    EqualityConstraint,
    GreaterThanOrEqualConstraint,
    LessThanOrEqualConstraint,
)
from pquant.core.torch.pruning_methods.metric_functions import (
    StructuredSparsityMetric,
    UnstructuredSparsityMetric,
)

_METRIC_REGISTRY = {
    "UnstructuredSparsity": UnstructuredSparsityMetric,
    "StructuredSparsity": StructuredSparsityMetric,
}

_CONSTRAINT_REGISTRY = {
    "Equality": EqualityConstraint,
    "LessThanOrEqual": LessThanOrEqualConstraint,
    "GreaterThanOrEqual": GreaterThanOrEqualConstraint,
}


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
        metric_type = pruning_parameters.metric_type
        constraint_type = pruning_parameters.constraint_type
        target_value = pruning_parameters.target_value
        target_sparsity = pruning_parameters.target_sparsity
        l0_mode = pruning_parameters.l0_mode
        scale_mode = pruning_parameters.scale_mode

        candidate_kwargs = {
            "epsilon": pruning_parameters.epsilon,
            "target_sparsity": target_sparsity,
            "l0_mode": l0_mode,
            "scale_mode": scale_mode,
            "rf": pruning_parameters.rf,
        }

        metric_cls = _METRIC_REGISTRY.get(metric_type)
        if metric_cls is None:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        sig = inspect.signature(getattr(metric_cls, "__init__", metric_cls))
        metric_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None and k in sig.parameters}
        metric_fn = metric_cls(**metric_kwargs)

        common_args = {
            "metric_fn": metric_fn,
            "target_value": target_value,
            "scale": pruning_parameters.scale,
            "damping": pruning_parameters.damping,
            "use_grad": pruning_parameters.use_grad,
            "lr": pruning_parameters.constraint_lr,
        }

        constraint_type_cls = _CONSTRAINT_REGISTRY.get(constraint_type)
        if constraint_type_cls is None:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")
        self.constraint_layer = constraint_type_cls(**common_args)

        self.register_buffer("mask", torch.ones(tuple(input_shape)))
        self.built = True

    def forward(self, weight):
        epsilon = self.config.pruning_parameters.epsilon
        hard_mask = (weight.abs() > epsilon).to(weight.dtype)
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
        epsilon = self.config.pruning_parameters.epsilon
        return (weight.abs() > epsilon).to(weight.dtype)

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
