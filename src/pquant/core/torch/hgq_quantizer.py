"""
Pure-PyTorch implementation of HGQ FixedPointQuantizerKIF + Quantizer wrapper.

Replaces the Keras hgq2 library dependency for the PyTorch backend.
Combines DefaultBitwidthMapper + FixedPointQuantizerKIF + Quantizer into one
nn.Module with no inheritance chain.

Reference: "HGQ: High Granularity Quantization for Real-time Neural Networks on FPGAs"
           (Sun et al., FPGA '26)
"""

import logging
import math

import torch
import torch.nn as nn

from pquant.core.torch.fixed_point_quantizer import get_fixed_quantizer, round_conv

logger = logging.getLogger(__name__)


def _minimal_i_given_xf(absmax: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Minimum integer bits needed to represent absmax with f fractional bits."""
    eps = 2.0 ** (-f)
    return torch.ceil(torch.log2(absmax + eps + 1e-10))


class HGQQuantizer(nn.Module):
    """
    HGQ fixed-point quantizer parameterized by (k, i, f) — keep_negative, integer
    bits, fractional bits.

    Parameters
    ----------
    k0 : float
        Initial sign bit (0 = unsigned, 1 = signed). Non-trainable.
    i0 : float
        Initial integer bits. Trainable for SAT; tracked buffer for WRAP.
    f0 : float
        Initial fractional bits. Always trainable.
    overflow_mode : str
        One of 'SAT', 'SAT_SYM', 'WRAP', 'WRAP_SM'.
    round_mode : str
        One of 'RND', 'RND_CONV', 'TRN', etc.
    is_data : bool
        True  → data/activation quantizer (batch axis 0 always homogeneous;
                per_channel keys on channel axis 1).
        False → weight/bias quantizer (per_channel keys on output-channel axis 0).
    granularity : str
        One of 'per_tensor' (one shared bit-width) or 'per_weight' (one per element).
        Controls which axes are shared. per_channel is not supported.
    gamma : float
        L1 regularisation coefficient on bit-widths.
    i_decay_speed : float
        WRAP mode only. Rate at which tracked i can decrease per step.
        float('inf') means i is reset each step to the minimum required by data.
    i_min, i_max : float
        Clamp bounds for the i parameter / buffer.
    f_min, f_max : float
        Clamp bounds for the f parameter.
    scaler : float | None
        Optional: inputs are divided by scaler before quantisation and multiplied
        after (equivalent to scaling the fixed-point range).
    qnoise_factor : float | None
        Optional: mix factor for quantisation noise injection during training.
        output = input + qnoise_factor * (quantised - input).
    affine : tuple[float, float] | None
        Optional (scale, shift) applied after quantisation: out = out*scale + shift.
    """

    def __init__(
        self,
        k0: float,
        i0: float,
        f0: float,
        overflow_mode: str,
        round_mode: str,
        is_data: bool,
        granularity: str = "per_weight",
        gamma: float = 1e-8,
        i_decay_speed: float = float("inf"),
        i_min: float = -23.0,
        i_max: float = 23.0,
        f_min: float = -24.0,
        f_max: float = 24.0,
        scaler=None,
        qnoise_factor: float | None = None,
        affine=None,
    ):
        super().__init__()
        assert int(k0) in (0, 1), f"k0 must be 0 or 1, got {k0}"

        self.k0 = float(k0)
        self.i0 = float(i0)
        self.f0 = float(f0)
        self.overflow_mode = overflow_mode.upper()
        self.round_mode = round_mode.upper()
        self.is_data = is_data
        self.granularity = granularity
        self.gamma = gamma
        self.i_decay_speed = i_decay_speed
        self.i_min = i_min
        self.i_max = i_max
        self.f_min = f_min
        self.f_max = f_max
        self.scaler = scaler
        self.qnoise_factor = qnoise_factor
        self.affine = affine

        # Set during build()
        self.homogeneous_axis: tuple[int, ...] = ()
        self._built = False

        self._stateless_quantizer = get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow_mode)

        # Scalar placeholders — replaced with shaped tensors in build().
        # k and i_raw are non-trainable buffers; f (and i for SAT) are Parameters.
        self.register_buffer("_k", torch.tensor(self.k0))
        self._f = nn.Parameter(torch.tensor(self.f0))

        if self.overflow_mode == "WRAP":
            # Integer bits tracked as non-trainable running buffer (not optimised).
            self.register_buffer("_i_raw", torch.tensor(self.i0))
        else:
            # Integer bits are trainable for SAT / SAT_SYM.
            self._i = nn.Parameter(torch.tensor(self.i0))

    def build(self, input_shape: tuple) -> None:
        """
        Initialise shaped parameter / buffer tensors.

        Called automatically on the first forward pass. The optimizer must be
        created *after* build() has been called so that it tracks the shaped
        parameters, not the scalar placeholders from __init__.
        """
        device = self._k.device
        self.homogeneous_axis = self._homogeneous_axis(len(input_shape))
        bw_shape = self._infer_bw_shape(input_shape)

        # k: non-trainable sign-bit buffer
        self.register_buffer("_k", torch.full(bw_shape, self.k0, device=device))

        # f: always trainable
        self._f = nn.Parameter(torch.full(bw_shape, self.f0, device=device))

        if self.overflow_mode == "WRAP":
            self.register_buffer("_i_raw", torch.full(bw_shape, self.i0, device=device))
        else:
            self._i = nn.Parameter(torch.full(bw_shape, self.i0, device=device))

        self._built = True

    def _homogeneous_axis(self, ndim: int) -> tuple[int, ...]:
        """Axes shared (collapsed to 1 in the bit-width tensor), derived from `granularity`.

        HGQ supports only per_tensor and per_weight (data always shares the batch axis 0).
        per_channel is intentionally unsupported (the channel axis is layout-dependent).
        """
        if self.granularity == "per_tensor":
            return tuple(range(ndim))
        if self.granularity == "per_weight":
            return (0,) if self.is_data else ()
        if self.granularity == "per_channel":
            raise ValueError("per_channel granularity is not supported for HGQ. Use 'per_tensor' or 'per_weight'.")
        raise ValueError(f"Unsupported granularity: {self.granularity}")

    def _infer_bw_shape(self, input_shape: tuple) -> tuple:
        """Shape of bit-width parameter tensors: homogeneous axes collapse to 1."""
        return tuple(1 if ax in self.homogeneous_axis else d for ax, d in enumerate(input_shape))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def built(self) -> bool:
        return self._built

    @property
    def k(self) -> torch.Tensor:
        return self._k.float()

    @property
    def i(self) -> torch.Tensor:
        """Integer bits, rounded with STE."""
        if self.overflow_mode == "WRAP":
            # _i_raw is a buffer (no gradient); round_conv is still used for
            # consistency — it degenerates to plain rounding when grad=0.
            return round_conv(self._i_raw.float())
        return round_conv(self._i.float())

    @property
    def f(self) -> torch.Tensor:
        """Fractional bits, rounded with STE."""
        return round_conv(self._f.float())

    @property
    def b(self) -> torch.Tensor:
        """Total non-sign bits = relu(i + f). Zero when quantizer is pruned."""
        return torch.relu(self.i + self.f)

    # ------------------------------------------------------------------
    # Bit-width mapping helpers
    # ------------------------------------------------------------------

    def _bw_to_x(self, bw: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        return bw.expand(x_shape)

    def _x_to_bw_absmax(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.homogeneous_axis) == 0:
            return x.abs()
        return torch.amax(x.abs(), dim=self.homogeneous_axis, keepdim=True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if not self._built:
            self.build(tuple(x.shape))

        if training:
            with torch.no_grad():
                self._f.data.clamp_(self.f_min, self.f_max)
                if self.overflow_mode != "WRAP":
                    self._i.data.clamp_(self.i_min, self.i_max)

        x_in = x  # kept for qnoise

        if self.scaler is not None:
            x = x / self.scaler

        # Round bit-width parameters to integers with STE so gradients flow.
        f_bw = round_conv(self._f)  # bw-shaped
        k = self._k.float()
        f_x = f_bw.expand(x.shape)
        k_x = k.expand(x.shape)

        if self.overflow_mode == "WRAP":
            out = self._stateless_quantizer.round(x, f_x)

            if training:
                # Track minimum integer bits needed for the current data.
                # Done without gradient so it doesn't affect the f gradient path.
                with torch.no_grad():
                    absmax = self._x_to_bw_absmax(out.detach())
                    min_i = _minimal_i_given_xf(absmax, f_bw.detach())
                    if math.isinf(self.i_decay_speed):
                        new_i = min_i
                    else:
                        new_i = torch.maximum(self._i_raw - self.i_decay_speed, min_i)
                    self._i_raw.copy_(new_i.clamp(self.i_min, self.i_max))
            else:
                if self.is_data:
                    # Data quantizer: apply wrap-modulo after rounding (inference).
                    i_x = self.i.expand(x.shape)
                    out = self._stateless_quantizer.saturate(out, k_x, i_x, f_x)
                # Weight quantizer: rounded output is final, no saturation.

                # Zero out pruned (total bits == 0) values.
                i_x = self.i.expand(x.shape)
                out = torch.where(k_x + i_x + f_x > 0, out, torch.zeros_like(out))

        else:  # SAT / SAT_SYM
            i_bw = round_conv(self._i)  # bw-shaped, STE for gradient
            i_x = i_bw.expand(x.shape)
            out = self._stateless_quantizer(x, k_x, i_x, f_x, training)

            if not training:
                out = torch.where(k_x + i_x + f_x > 0, out, torch.zeros_like(out))

        if self.scaler is not None:
            out = out * self.scaler

        if self.qnoise_factor is not None and training:
            out = x_in + self.qnoise_factor * (out - x_in)

        if self.affine is not None:
            out = out * self.affine[0] + self.affine[1]

        return out

    # ------------------------------------------------------------------
    # Regularisation / constraint / utility methods
    # ------------------------------------------------------------------

    def regularization_loss(self) -> torch.Tensor:
        """
        L1 regularisation on bit-widths.

        Replaces Keras `layer.losses` list. Contributes the γ·Σ(bit-widths) term
        from eq. (12) in the paper. For SAT mode both i and f are regularised;
        for WRAP mode only f (i is not a learnable parameter).
        """
        if not self._built or self.gamma == 0.0:
            return torch.tensor(0.0, device=self._k.device)
        loss = self.gamma * self.f.sum()
        if self.overflow_mode != "WRAP":
            loss = loss + self.gamma * self.i.sum()
        return loss

    def post_epoch_constraint_apply(self) -> None:
        """
        Clamp i and f parameters to [*_min, *_max] in-place.

        Replaces Keras `variable.constraint(variable)` + `variable.assign()`.
        Call at the end of each epoch.
        """
        with torch.no_grad():
            self._f.data.clamp_(self.f_min, self.f_max)
            if self.overflow_mode != "WRAP":
                self._i.data.clamp_(self.i_min, self.i_max)
            # WRAP: _i_raw is already clamped inside forward().

    def set_bits(self, i, f) -> None:
        """
        Overwrite i and f with scalar or tensor values.

        Replaces Keras `variable.assign(variable * 0 + value)` pattern.
        Used by `apply_final_compression` and `reload_from_local` in quantizer.py.
        """
        with torch.no_grad():
            i_t = torch.as_tensor(i, dtype=torch.float32)
            f_t = torch.as_tensor(f, dtype=torch.float32)
            self._f.data.copy_(f_t) if f_t.shape == self._f.shape else self._f.data.fill_(f_t.item())
            if self.overflow_mode == "WRAP":
                self._i_raw.copy_(i_t) if i_t.shape == self._i_raw.shape else self._i_raw.fill_(i_t.item())
            else:
                self._i.data.copy_(i_t) if i_t.shape == self._i.shape else self._i.data.fill_(i_t.item())

    def bits_(self, shape: tuple) -> torch.Tensor:
        """
        Return total bits (k + relu(i+f)) broadcast to *shape*.

        Used by `Quantizer.get_total_bits()` for EBOPs calculations.
        """
        if not self._built:
            total = self.k0 + max(self.i0 + self.f0, 0.0)
            return torch.full(shape, total, device=self._k.device)
        return (self.k + self.b).expand(shape)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    logger.debug("=== HGQQuantizer sanity checks ===\n")

    # ---- SAT, signed, weight quantizer (per-element) ----
    x = torch.randn(4, 3) * 0.5
    q_sat = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="SAT", round_mode="RND", is_data=False)
    out = q_sat(x, training=False)
    logger.debug(f"SAT weight forward: input range [{x.min():.3f}, {x.max():.3f}]")
    logger.debug(f"  output range      [{out.min():.3f}, {out.max():.3f}]")
    max_val = 2**2 - 2**-4
    assert out.max() <= max_val + 1e-5, "SAT upper bound violated"
    assert out.min() >= -(2**2) - 1e-5, "SAT lower bound violated"
    logger.debug("  bounds check OK\n")

    # ---- SAT backward (gradient through f) ----
    q_sat2 = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="SAT", round_mode="RND", is_data=False)
    x2 = torch.randn(4, 3, requires_grad=True)
    out2 = q_sat2(x2, training=True)
    out2.sum().backward()
    assert q_sat2._f.grad is not None
    assert q_sat2._f.grad.abs().sum() > 0
    assert x2.grad is not None
    assert x2.grad.abs().sum() > 0
    logger.debug("SAT backward: gradients w.r.t. f and input OK")
    logger.debug(f"  f grad mean: {q_sat2._f.grad.mean():.6f}\n")

    # ---- WRAP, signed, data quantizer (per-batch) ----
    x3 = torch.randn(8, 4) * 3.0
    q_wrap = HGQQuantizer(k0=1, i0=0, f0=4, overflow_mode="WRAP", round_mode="RND", is_data=True)
    out3 = q_wrap(x3, training=True)
    logger.debug(f"WRAP data training: _i_raw after forward: {q_wrap._i_raw.flatten()[:4]}")
    assert q_wrap._i_raw.max() >= 0, "i_raw should track positive range"
    logger.debug("  _i_raw tracking OK\n")

    # ---- WRAP backward (gradient through f, not i_raw) ----
    q_wrap2 = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="WRAP", round_mode="RND", is_data=False)
    x4 = torch.randn(4, 3, requires_grad=True)
    out4 = q_wrap2(x4, training=True)
    out4.sum().backward()
    assert q_wrap2._f.grad is not None
    assert q_wrap2._f.grad.abs().sum() > 0
    assert q_wrap2._i_raw.grad is None, "_i_raw must not accumulate gradient"
    logger.debug("WRAP backward: f has gradient, _i_raw has none OK\n")

    # ---- Zero bits → pruned: k=0,i=0,f=0 → k+i+f=0 → all zero ----
    q_pruned = HGQQuantizer(k0=0, i0=0, f0=0, overflow_mode="SAT", round_mode="RND", is_data=False)
    x5 = torch.randn(4, 3)
    out5 = q_pruned(x5, training=False)
    assert torch.all(out5 == 0), "Zero-bit quantizer should output zeros"
    logger.debug("Pruning (zero bits): output is all zeros OK\n")

    # ---- regularization_loss ----
    q_reg = HGQQuantizer(k0=1, i0=2.0, f0=4.0, overflow_mode="SAT", round_mode="RND", is_data=False, gamma=1e-3)
    q_reg(torch.randn(4, 3))
    loss = q_reg.regularization_loss()
    expected = 1e-3 * (q_reg.f.sum() + q_reg.i.sum())
    assert torch.isclose(loss, expected, rtol=1e-4), f"reg loss mismatch: {loss} vs {expected}"
    logger.debug(f"regularization_loss OK: {loss.item():.6f}\n")

    # ---- bits_ shape ----
    q_bits = HGQQuantizer(k0=1, i0=2.0, f0=4.0, overflow_mode="SAT", round_mode="RND", is_data=False)
    q_bits(torch.randn(4, 3))
    b = q_bits.bits_((8, 4, 3))
    assert b.shape == (8, 4, 3), f"bits_ shape wrong: {b.shape}"
    logger.debug(f"bits_ shape OK: {b.shape}\n")

    # ---- set_bits ----
    q_set = HGQQuantizer(k0=1, i0=2.0, f0=4.0, overflow_mode="SAT", round_mode="RND", is_data=False)
    q_set(torch.randn(4, 3))
    q_set.set_bits(3.0, 5.0)
    assert math.isclose(q_set.i.mean().item(), 3.0, abs_tol=1e-5)
    assert math.isclose(q_set.f.mean().item(), 5.0, abs_tol=1e-5)
    logger.debug("set_bits OK\n")

    logger.debug("=== All checks passed ===")
