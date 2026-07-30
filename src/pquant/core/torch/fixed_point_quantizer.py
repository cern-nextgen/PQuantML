# PyTorch version of https://github.com/calad0i/quantizers/blob/master/src/quantizers/fixed_point/_fixed_point_ops.py

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import torch
from numpy.typing import ArrayLike

round_mode_registry: dict[str, Callable[[Any], Any]] = {}
saturation_mode_registry: dict[str, Callable[[Any, Any, Any, Any], Any]] = {}

T = TypeVar('T', bound=ArrayLike)


def _clip(x, min_value, max_value):
    mask_overflow = x > max_value
    mask_underflow = x < min_value
    return torch.where(mask_overflow, max_value, torch.where(mask_underflow, min_value, x))


def rnd_mode(name: str):
    def inner(func):
        assert name not in round_mode_registry, f"Round mode '{name}' already exists."

        @wraps(func)
        def wrapper(x):
            xq = func(x)
            return xq.detach() + (x - x.detach())

        round_mode_registry[name] = wrapper
        return wrapper

    return inner


@rnd_mode('TRN')
def floor(x):
    return torch.floor(x)


@rnd_mode('RND')
def round(x):
    # Round to nearest, ties positive infinity.
    return torch.floor(x + 0.5)


@rnd_mode('RND_CONV')
def round_conv(x):
    # Round to nearest, ties to even.
    return torch.round(x)


@rnd_mode('TRN_ZERO')
def floor_zero(x):
    # Truncate towards zero.
    sign = torch.sign(x)
    return torch.floor(torch.abs(x)) * sign  # type: ignore


@rnd_mode('RND_ZERO')
def round_zero(x):
    # Round to nearest, ties towards zero.
    sign = torch.sign(x)
    return -torch.floor(-torch.abs(x) + 0.5) * sign  # type:ignore


@rnd_mode('RND_MIN_INF')
def round_min_inf(x):
    # Round to nearest, ties towards negative infinity.
    return -torch.floor(-x + 0.5)  # type:ignore


@rnd_mode('RND_INF')
def round_inf(x):
    # Round to nearest, ties away from zero.
    sign = torch.sign(x)
    return torch.floor(torch.abs(x) + 0.5) * sign  # type: ignore


def sat_mode(name: str | list | tuple):
    names = (name,) if isinstance(name, str) else name

    def inner(func):
        for name in names:
            assert name not in saturation_mode_registry, f"Saturation mode '{name}' already exists."
            saturation_mode_registry[name] = func
        saturation_mode_registry[func.__name__.upper()] = func
        return func

    return inner


@sat_mode('WRAP')
def wrap(x, k, i, f):
    xs = x
    bk = i + k
    bias = k * 2.0 ** (bk - 1)
    return (xs + bias) % (2.0**bk) - bias


@sat_mode('SAT')
def sat(x, k, i, f):
    f_eps = 2.0 ** (-f)
    __max = 2.0**i
    _max = __max - f_eps
    _min = -__max * k
    r = _clip(x, _min, _max)
    return r


@sat_mode('SAT_SYM')
def sat_sym(x, k, i, f):
    f_eps = 2.0 ** (-f)
    _max = 2.0**i - f_eps
    _min = -_max * k
    r = _clip(x, _min, _max)
    return r


@sat_mode('WRAP_SM')
def wrap_sm_fn(x, k, i, f, training=None, quant_fn: Callable = lambda x: x):
    # x=ops.round(x*2.**f)
    # High and low bounds are reflective. When overflows, can be less trash than WARP but still more trash than SAT.
    eps = 2.0**-f
    high = 2.0**i - eps
    low = -(high + eps) * k
    interval = 2.0 ** (i + k)
    c1 = ((x) / interval) % 2 >= 1  # type: ignore
    c1 = c1 & (torch.abs(x) > eps / 2)
    c2 = ((x + eps / 2) / (interval / 2)) % 2 >= 1  # type: ignore
    qx = quant_fn(x)
    mapped = ((qx - high - eps) % interval) + low

    mapped = torch.where(c2, -mapped - eps, mapped)  # type: ignore
    mapped = torch.where(c1, -mapped - eps, mapped)  # type: ignore

    return mapped


class FixedPointQuantizer:
    def round(self, x, f: Any = 1.0):
        scale = 2.0**f
        x = x * scale
        xq = self.round_fn(x)
        xq = xq / scale
        return xq

    def saturate(self, x, k, i, f):
        return self.sat_fn(x, k, i, f)

    def __init__(self, round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
        round_mode = round_mode.upper()
        overflow_mode = overflow_mode.upper()
        self.stochastic = False

        if round_mode.startswith('S_'):
            round_mode = round_mode[2:]
            self.stochastic = True

        if overflow_mode == 'WRAP_SM':
            assert round_mode in (
                'RND',
                'RND_CONV',
            ), 'WRAP_SM only supports RND and RND_CONV rounding modes in this implementation.'

        self.round_mode = round_mode
        self.overflow_mode = overflow_mode
        round_fn = round_mode_registry[round_mode]
        sat_fn = saturation_mode_registry[overflow_mode]
        self.sat_fn = sat_fn
        self.round_fn = round_fn

    def forward(self, x, k, i, f, training=False):
        # Workaround for gradient computation around 0.
        # When have values outside boundary rounded to boundary, grad on f presents despite the value
        # will be clipped off anyway. Thus have saturation before rounding, except for
        # wrap mode, which doesn't round during training.

        if self.overflow_mode != 'WRAP':
            x = self.saturate(x, k, i, f)
        x = self.round(x, f)
        if self.overflow_mode == 'WRAP' and not training:
            x = self.saturate(x, k, i, f)
        return x

    def forward_wrap_sm(self, x, k, i, f, training=False):
        def quant_fn(x):
            return self.round(x, f, training and self.stochastic)

        x = wrap_sm_fn(x, k, i, f, training, quant_fn)
        return x

    def __call__(self, x, k, i, f, training=False, seed_gen=None):
        i = torch.maximum(i, -f).detach() + (i - i.detach())  # type: ignore
        if self.stochastic and training:
            assert seed_gen is not None, 'Seed generator must be provided for stochastic rounding.'
        if self.overflow_mode != 'WRAP_SM':
            return self.forward(x, k, i, f, training)
        else:
            return self.forward_wrap_sm(x, k, i, f, training)


def get_fixed_quantizer(round_mode: str = 'TRN', overflow_mode: str = 'WRAP'):
    """Get a stateless fixed-point quantizer given the round and overflow mode.
    The quantizer is differentiable w.r.t. to the input and f, also i if using saturation overflow mode.

    Args:
        round_mode: round mode, one of
    """
    quantizer = FixedPointQuantizer(round_mode, overflow_mode)
    return quantizer  # type: ignore
