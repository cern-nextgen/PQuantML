import abc

import torch
import torch.nn as nn


class _FlipGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, upstream):
        return upstream * ctx.scale, None


def flip_gradient(x, scale=-1.0):
    return _FlipGradient.apply(x, scale)


class Constraint(nn.Module):
    def __init__(self, lmbda_init=1.0, scale=1.0, damping=1.0, use_grad=True, lr=0.0, **kwargs):
        super().__init__()
        self.use_grad_ = bool(use_grad)
        self.lr_ = float(lr)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("damping", torch.tensor(float(damping)))
        if self.use_grad_:
            self.lmbda = nn.Parameter(torch.tensor(float(lmbda_init)))
        else:
            self.register_buffer("lmbda", torch.tensor(float(lmbda_init)))
            self.register_buffer("prev_infs", torch.tensor(0.0))

    def build(self, input_shape):
        # Exists for API parity with the keras version — no-op in torch.
        pass

    def forward(self, weight, training=None):
        raw_infeasibility = self.get_infeasibility(weight)
        infeasibility = self.pipe_infeasibility(raw_infeasibility)

        if self.use_grad_:
            ascent_lmbda = flip_gradient(self.lmbda)
        else:
            lmbda_step = self.lr_ * self.scale * self.prev_infs
            ascent_lmbda = self.lmbda + lmbda_step
            if training:
                with torch.no_grad():
                    self.lmbda.add_(lmbda_step)
                    self.prev_infs.copy_(infeasibility.detach())

        l_term = ascent_lmbda * infeasibility
        damp_term = self.damping * infeasibility.square() / 2
        return self.scale * (l_term + damp_term)

    @abc.abstractmethod
    def get_infeasibility(self, weight):
        raise NotImplementedError

    def pipe_infeasibility(self, infeasibility):
        return infeasibility

    @torch.no_grad()
    def turn_off(self):
        if not self.use_grad_:
            self.lr_ = 0.0
        self.scale.zero_()
        self.lmbda.data.zero_() if isinstance(self.lmbda, nn.Parameter) else self.lmbda.zero_()


class EqualityConstraint(Constraint):
    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = float(target_value)

    def get_infeasibility(self, weight):
        return (self.metric_fn(weight) - self.target_value).abs()


class LessThanOrEqualConstraint(Constraint):
    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = float(target_value)

    def get_infeasibility(self, weight):
        return torch.clamp(self.metric_fn(weight) - self.target_value, min=0.0)


class GreaterThanOrEqualConstraint(Constraint):
    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = float(target_value)

    def get_infeasibility(self, weight):
        return torch.clamp(self.target_value - self.metric_fn(weight), min=0.0)
