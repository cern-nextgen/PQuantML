import abc

import keras
from keras import ops


@ops.custom_gradient
def flip_gradient(x, scale=-1.0):
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        scale_t = ops.convert_to_tensor(scale, dtype=upstream.dtype)
        return (ops.multiply(upstream, scale_t),)  # ops.abs()

    return x, grad


# Abstract base class for constraints
@keras.utils.register_keras_serializable(name="Constraint")
class Constraint(keras.layers.Layer):
    def __init__(self, lmbda_init=1.0, scale=1.0, damping=1.0, **kwargs):
        self.use_grad_ = bool(kwargs.pop("use_grad", True))
        self.lr_ = float(kwargs.pop("lr", 0.0))
        super().__init__(**kwargs)

        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(scale, dtype=dtype),
            trainable=False,
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(damping, dtype=dtype),
            trainable=False,
        )
        self.lmbda = self.add_weight(
            name=f'{self.name}_lmbda',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(lmbda_init, dtype=dtype),
            trainable=self.use_grad_,
        )

        if not self.use_grad_:
            self.prev_infs = self.add_weight(
                name=f'{self.name}_prev_infs',
                shape=(),
                initializer=lambda shape, dtype: ops.convert_to_tensor(0.0, dtype=dtype),
                trainable=False,
            )

    def call(self, weight, training=None):
        """Calculates the penalty from a given infeasibility measure."""
        raw_infeasibility = self.get_infeasibility(weight)
        infeasibility = self.pipe_infeasibility(raw_infeasibility)

        if self.use_grad_:
            ascent_lmbda = flip_gradient(self.lmbda)
            # ascent_lmbda = ops.maximum(ascent_lmbda, 0.0)
        else:
            lmbda_step = self.lr_ * self.scale * self.prev_infs
            ascent_lmbda = self.lmbda + lmbda_step
            if training:
                self.lmbda.assign_add(lmbda_step)
                self.prev_infs.assign(infeasibility)

        l_term = ascent_lmbda * infeasibility
        damp_term = self.damping * ops.square(infeasibility) / 2
        penalty = self.scale * (l_term + damp_term)

        return penalty

    @abc.abstractmethod
    def get_infeasibility(self, weight):
        """Must be implemented by subclasses to define the violation."""
        raise NotImplementedError

    def pipe_infeasibility(self, infeasibility):
        """Optional transformation of raw infeasibility.
        Default is identity. Subclasses may override."""
        return infeasibility

    def turn_off(self):
        if not self.use_grad_:
            self.lr_ = 0.0
        self.scale.assign(0.0)
        self.lmbda.assign(0.0)


@keras.utils.register_keras_serializable(name="EqualityConstraint")
class EqualityConstraint(Constraint):
    """Constraint for g(w) == target_value."""

    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value

    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.abs(infeasibility)


@keras.utils.register_keras_serializable(name="LessThanOrEqualConstraint")
class LessThanOrEqualConstraint(Constraint):
    """Constraint for g(w) <= target_value."""

    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value

    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.maximum(infeasibility, 0.0)


@keras.utils.register_keras_serializable(name="GreaterThanOrEqualConstraint")
class GreaterThanOrEqualConstraint(Constraint):
    """Constraint for g(w) >= target_value."""

    def __init__(self, metric_fn, target_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value

    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = self.target_value - metric_value
        return ops.maximum(infeasibility, 0.0)
