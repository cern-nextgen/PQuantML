from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pquant.core.constants import DynamicDataMode
from pquant.core.torch.quantizer import Quantizer

MAX_SCORE_SUM = 32767

#: Output paths of the aie4ml ``softmax_i8`` kernel (see :func:`hccs_normalize_scores`).
Mode = Literal["int16_div", "int16_clb", "uint8_div", "uint8_clb"]
_MODES = ("int16_div", "int16_clb", "uint8_div", "uint8_clb")

# power-of-two reciprocals: the outputs are exact fixed point (int16 x 2**-15, uint8 x 2**-8)
_I16_TO_UNIT = 1.0 / 32768.0
_U8_TO_UNIT = 1.0 / 256.0


def _validate_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {_MODES}.")


def default_inv_shift(mode: Mode) -> int:
    """The kernel's default reciprocal fractional bits: 0 for int16 outputs, 15 for uint8."""
    _validate_mode(mode)
    return 0 if mode.startswith("int16") else 15


def score_budget(n_keys: int) -> int:
    """Largest ``B`` for rows of ``n_keys`` keys: the int16 score sum has to hold ``n_keys`` of them."""
    return max(1, MAX_SCORE_SUM // int(n_keys))


def min_row_sum(mode: Mode = "int16_div", inv_shift: int | torch.Tensor | None = None) -> int | torch.Tensor:
    """Smallest row sum the kernel needs for its reciprocal to fit int16 (``inv_shift``: an int or a scalar tensor)."""
    _validate_mode(mode)
    s = default_inv_shift(mode) if inv_shift is None else inv_shift
    if mode == "int16_div":
        return 1 << s
    if mode == "int16_clb":
        return 1
    if mode == "uint8_div":
        return -(-(255 << s) // 32767)
    return 1 << (torch.clamp(s - 7, min=0) if torch.is_tensor(s) else max(s - 7, 0))


def score_floor(
    n_keys: int,
    B: int | torch.Tensor,
    mode: Mode = "int16_div",
    inv_shift: int | torch.Tensor | None = None,
) -> int | torch.Tensor:
    """Smallest tail score ``B - S * DMAX`` that keeps the row sum at :func:`min_row_sum`, at least 1."""
    n_keys = int(n_keys)
    if n_keys <= 0:
        raise ValueError(f"n_keys must be > 0, got {n_keys}")
    if n_keys == 1:
        return torch.ones_like(B) if torch.is_tensor(B) else 1
    zmin = min_row_sum(mode, inv_shift)
    if torch.is_tensor(B):
        return torch.clamp(torch.ceil((zmin - B) / (n_keys - 1)), min=1.0)
    return max(1, -(-(int(zmin) - int(B)) // (n_keys - 1)))


def _as_head_vector(x: torch.Tensor, n_head: int, name: str, dtype: torch.dtype) -> torch.Tensor:
    v = torch.as_tensor(x, dtype=dtype).view(-1)
    if v.numel() != n_head:
        raise ValueError(f"{name} must have exactly {n_head} values, got {v.numel()}. Head remapping is disabled.")
    return v


def quantize_logits_with_shift(logits: torch.Tensor, shift_eff: int | torch.Tensor) -> torch.Tensor:
    """Saturating int8 quantization after multiplying by ``2**shift_eff``."""
    if torch.is_tensor(shift_eff):
        scale = torch.pow(torch.tensor(2.0, device=logits.device), shift_eff.to(torch.float32))
    else:
        scale = float(1 << int(shift_eff))
    x_scaled = torch.round(logits * scale)
    x_scaled = torch.clamp(x_scaled, -128.0, 127.0)
    return x_scaled.to(torch.int8)


def quantize_logits_scaled(logits: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Saturating int8 quantization ``clamp(round(logits * scale))`` for a real ``scale``."""
    return torch.clamp(torch.round(logits * scale), -128.0, 127.0).to(torch.int8)


def _floor_log2(z: torch.Tensor) -> torch.Tensor:
    """Exact ``floor(log2 z)`` for int32 ``z >= 1``."""
    k = torch.floor(torch.log2(z.to(torch.float32))).to(torch.int32)
    k = torch.where(torch.bitwise_right_shift(z, k + 1) > 0, k + 1, k)
    return torch.where(torch.bitwise_right_shift(z, k) == 0, k - 1, k)


def hccs_normalize_scores(
    scores: torch.Tensor, mode: Mode = "int16_div", inv_shift: int | torch.Tensor | None = None
) -> torch.Tensor:
    """Normalize int16 scores along the last dim as the aie4ml kernel does for ``mode``; floats in [0, 1]."""
    _validate_mode(mode)
    s = default_inv_shift(mode) if inv_shift is None else inv_shift
    Z = torch.clamp_min(scores.sum(dim=-1, keepdim=True).to(torch.int32), 1)
    sc = scores.to(torch.int32)
    if mode == "int16_div":
        inv = torch.clamp((32767 << s) // Z, max=32767)
        out = torch.clamp(torch.bitwise_right_shift(sc * inv, s), -32768, 32767).to(torch.int16)
        return out.to(torch.float32) * _I16_TO_UNIT

    k = _floor_log2(Z)
    if mode == "uint8_div":
        inv = torch.clamp((255 << s) // Z, max=32767)
        out = torch.clamp(torch.bitwise_right_shift(sc * inv, s), 0, 255).to(torch.uint8)
        return out.to(torch.float32) * _U8_TO_UNIT
    if mode == "uint8_clb":
        inv = torch.bitwise_left_shift(torch.tensor(255, device=Z.device, dtype=torch.int32), torch.clamp(s - k, min=0))
        out = torch.clamp(torch.bitwise_right_shift(sc * inv, s), 0, 255).to(torch.uint8)
        return out.to(torch.float32) * _U8_TO_UNIT

    inv = torch.bitwise_right_shift(torch.tensor(32767, device=Z.device, dtype=torch.int32), k)
    out = torch.clamp(sc * inv, -32768, 32767).to(torch.int16)
    return out.to(torch.float32) * _I16_TO_UNIT


def _ramp_scores(d: torch.Tensor, Bm: torch.Tensor, Sm: torch.Tensor, Dm: torch.Tensor) -> torch.Tensor:
    """int16 scores ``B - S * min(d, DMAX)`` of uint8 distances ``d`` to the row max."""
    d = torch.minimum(d, Dm)
    d_i16 = d.to(torch.int16)
    d8 = torch.where(d_i16 < 128, d_i16, d_i16 - 256).to(torch.int8)  # read as int8, as the kernel's lane does
    score32 = Bm + d8.to(torch.int32) * (-Sm)
    return torch.clamp(score32, min=-32768, max=32767).to(torch.int16)


def hccs_forward_per_head(
    attn_weights: torch.Tensor,
    B_per_head: torch.Tensor,
    S_per_head: torch.Tensor,
    DMAX_per_head: torch.Tensor,
    shift_per_head: torch.Tensor | None,
    mode: Mode = "int16_div",
    inv_shift: int | torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    renorm_after_mask: bool = True,
    scale_per_head: torch.Tensor | None = None,
) -> torch.Tensor:
    """HCCS attention weights for ``attn_weights`` ``[B, H, Tq, Tk]`` with per-head params ``[H]``. The logits are
    quantized to int8 with ``scale_per_head`` when given, else with ``2**shift_per_head``. Keys outside ``valid_mask``
    get zero weight (rows are renormalized if ``renorm_after_mask``)."""
    _validate_mode(mode)
    if attn_weights.ndim != 4:
        raise ValueError(f"attn_weights must be rank-4 [B,H,Tq,Tk], got shape {tuple(attn_weights.shape)}")

    _, n_head, _, _ = attn_weights.shape
    Bv = _as_head_vector(B_per_head, n_head, "B_per_head", torch.int32)
    Sv = _as_head_vector(S_per_head, n_head, "S_per_head", torch.int32)
    Dv = _as_head_vector(DMAX_per_head, n_head, "DMAX_per_head", torch.uint8)
    shape = [1, n_head, 1, 1]
    dev = attn_weights.device

    Bm = Bv.view(shape).to(dev, torch.int32)
    Sm = Sv.view(shape).to(dev, torch.int32)
    Dm = Dv.view(shape).to(dev, torch.uint8)

    if scale_per_head is not None:
        Sc = _as_head_vector(scale_per_head, n_head, "scale_per_head", torch.float32)
        x_int8 = quantize_logits_scaled(attn_weights, Sc.view(shape).to(dev))
    elif shift_per_head is not None:
        Sh = _as_head_vector(shift_per_head, n_head, "shift_per_head", torch.int32)
        x_int8 = quantize_logits_with_shift(attn_weights, Sh.view(shape).to(dev, torch.int32))
    else:
        raise ValueError("hccs_forward_per_head: pass shift_per_head or scale_per_head")

    if valid_mask is not None:
        if valid_mask.shape != attn_weights.shape:
            raise ValueError(
                f"valid_mask shape must match attn_weights exactly, got {tuple(valid_mask.shape)} "
                f"vs {tuple(attn_weights.shape)}"
            )
        x_int8 = torch.where(valid_mask, x_int8, torch.full_like(x_int8, -128))

    # as uint8 the difference is the exact distance to the row max
    mu = x_int8.max(dim=-1, keepdim=True).values.to(torch.uint8)
    scores = _ramp_scores(mu - x_int8.to(torch.uint8), Bm, Sm, Dm)
    probs = hccs_normalize_scores(scores, mode=mode, inv_shift=inv_shift)

    if valid_mask is not None:
        probs = probs * valid_mask.to(probs.dtype)
        if renorm_after_mask:
            probs = probs / torch.clamp_min(probs.sum(dim=-1, keepdim=True), 1e-12)

    return probs


def _check_rank4(attn_weights: torch.Tensor) -> None:
    if attn_weights.ndim != 4:
        raise ValueError(f"attn_weights must be rank-4 [B,H,Tq,Tk], got shape {tuple(attn_weights.shape)}")


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through round: forward rounds, backward is identity."""
    return x + (torch.round(x) - x).detach()


def _clamp_ste(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Straight-through clamp: forward clamps, backward is identity."""
    return x + (x.clamp(lo, hi) - x).detach()


def hccs_kernel_ebops(mode: str, num_heads: int, n_keys: int, n_queries: int | None = None) -> float:
    """EBOPs of the kernel for ``num_heads x n_queries x n_keys`` attention weights: a 7 x 7 bit product per key for
    the ramp, plus 16 x 16 per key and per row for the reciprocal in the ``*_div`` modes."""
    n_queries = n_keys if n_queries is None else int(n_queries)
    div = str(mode).endswith("_div")
    per_key = 49.0 + (256.0 if div else 0.0)
    per_row = 256.0 if div else 0.0
    return float(num_heads * n_queries) * (float(n_keys) * per_key + per_row)


def _native_out_bits(mode):
    """(k, i, f) of the kernel's own output: uint8 x 2**-8 or int16 x 2**-15."""
    return (0.0, 0.0, 8.0) if str(mode).startswith("uint8") else (1.0, 0.0, 15.0)


def _pq_data_lane(config, k, i, f, dynamic=None):
    """A PQuant data lane from the config; ``dynamic`` overrides its dynamic_data_mode (False: never, True: training)."""
    qp = config.quantization_parameters
    mode = qp.dynamic_data_mode
    if dynamic is not None:
        mode = DynamicDataMode.TRAINING if dynamic else DynamicDataMode.NEVER
    return Quantizer(
        k=torch.tensor(float(k)),
        i=torch.tensor(float(i)),
        f=torch.tensor(float(f)),
        overflow=qp.overflow_mode_data,
        round_mode=qp.round_mode,
        is_heterogeneous=qp.use_high_granularity_quantization,
        is_data=True,
        hgq_gamma=qp.hgq_gamma,
        place="datalane",
        dynamic_data_mode=mode,
        i_decay=qp.dynamic_data_i_decay,
    )


def _as_config(config):
    if config is None:
        raise ValueError("HCCSLearned: a PQuant config is required, it builds the input and output lanes")
    if isinstance(config, dict):
        from pquant.core.hyperparameter_optimization import PQConfig

        config = PQConfig.load_from_config(config)
    return config


class HCCSLearned(nn.Module):
    """Head-Calibrated Clipped-Linear Softmax for QAT: a clipped linear ramp per head, trained by gradient descent.

    The forward returns the integer kernel's output (:func:`hccs_forward_per_head`) bit for bit and takes its gradient
    from a smooth surrogate. Per head, ``log2_slope`` (ramp slope per logit unit) and ``log2_tail`` (score of the
    clipped keys) train; keep them out of weight decay (:func:`hccs_parameters`). ``log2_scale`` is a buffer that
    mirrors the fractional bits of ``input_quantizer``, a PQuant data lane built from ``config`` (keep it at
    ``k + i + f <= 8``: its codes are the kernel's int8 codes). ``output_quantizer`` is pinned at the kernel's own
    output format unless ``out_quant_bits`` is given.

    The first forward seeds the ramp (``auto_seed``), or use :meth:`cold`; until seeded the module is ``F.softmax``.
    ``max_keys`` is the longest row the parameters fit (taken from the first batch when not given); longer rows are
    refused. :meth:`export_params` gives the integers for deployment. The module takes PQSoftmax's place in an
    attention layer: ``mha.softmax = HCCSLearned(mha.num_heads, config)``.
    """

    slope_ratio0 = 0.35
    tail_ratio0 = 0.05
    _log2_max_S = math.log2(127.0)  # S = sigma / a <= 127

    def __init__(
        self,
        num_heads: int,
        config,
        mode: Mode = "int16_div",
        inv_shift: int | None = None,
        renorm_after_mask: bool = True,
        max_keys: int | None = None,
        auto_seed: bool = True,
        log2_scale_range: tuple[float, float] = (-6.0, 10.0),
        in_quant_bits: tuple | None = None,
        out_quant_bits: tuple | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.inv_shift = default_inv_shift(mode) if inv_shift is None else int(inv_shift)
        self.renorm_after_mask = renorm_after_mask
        self.max_keys = None if max_keys is None else int(max_keys)
        # checkpointed flag, mirrored in a Python bool for forward
        self.register_buffer("_initialized", torch.tensor(False))
        self._ready = False
        # tensor copy for forward (torch.compile)
        self.register_buffer("_inv_shift", torch.tensor(self.inv_shift), persistent=False)
        self.auto_seed = auto_seed
        self.scale_lo, self.scale_hi = float(log2_scale_range[0]), float(log2_scale_range[1])

        self.register_buffer("log2_scale", torch.zeros(num_heads))
        self.log2_slope = nn.Parameter(torch.zeros(num_heads))
        self.log2_tail = nn.Parameter(torch.zeros(num_heads))

        config = _as_config(config)
        qp = config.quantization_parameters
        self.use_hgq = qp.use_high_granularity_quantization
        self.input_shape = None
        if in_quant_bits is None:
            in_quant_bits = (qp.default_data_keep_negatives, qp.default_data_integer_bits, qp.default_data_fractional_bits)
        # the kernel's scale is a constant at inference: the logit lane tracks in training at most
        self.input_quantizer = _pq_data_lane(config, *in_quant_bits, dynamic=qp.dynamic_data_mode != "never")
        # default: pinned at the kernel's own output format (a no-op that carries the width)
        if out_quant_bits is None:
            self.output_quantizer = _pq_data_lane(config, *_native_out_bits(self.mode), dynamic=False)
        else:
            self.output_quantizer = _pq_data_lane(config, *out_quant_bits)

    def _set_ready(self, value: bool) -> None:
        self._initialized.fill_(value)
        self._ready = value

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ready = bool(self._initialized.item())

    def _check_keys(self, n_keys: int) -> None:
        """Refuse rows longer than ``max_keys``: the int16 score sum would overflow."""
        if self.max_keys is not None and n_keys > self.max_keys:
            raise ValueError(
                f"{type(self).__name__}: rows have {n_keys} keys but the params are fitted for at most "
                f"{self.max_keys}; pass max_keys=<longest row> at construction (or seed on it)."
            )

    def _fit_keys(self, n_keys: int) -> None:
        """Take ``n_keys`` as the key length if none is set yet, else refuse longer rows."""
        if self.max_keys is None:
            self.max_keys = int(n_keys)
        else:
            self._check_keys(n_keys)

    @property
    def initialized(self) -> bool:
        return self._ready

    @property
    def budget(self) -> int:
        """``B``: the fixed integer score of the row max, ``32767 // max_keys``."""
        return score_budget(self.max_keys)

    def ebops(self) -> float:
        """Kernel cost per input; 0 until the key length is known."""
        if self.max_keys is None:
            return 0.0
        return hccs_kernel_ebops(self.mode, self.num_heads, self.max_keys)

    def _set(self, slope: torch.Tensor, tail: torch.Tensor | None = None, scale: torch.Tensor | None = None) -> None:
        """Seed the latents from ``sigma``, ``tau`` (default ``tail_ratio0 * budget``) and, when given, ``a``."""
        with torch.no_grad():
            if scale is not None:
                self.log2_scale.copy_(torch.log2(scale.to(self.log2_scale.device, torch.float32)))
            self.log2_slope.copy_(torch.log2(slope.to(self.log2_slope.device, torch.float32)))
            if tail is None:
                tail = torch.full_like(slope, self.tail_ratio0 * self.budget)
            self.log2_tail.copy_(torch.log2(tail.to(self.log2_tail.device, torch.float32).clamp_min(1.0)))
        self._set_ready(True)

    @classmethod
    def cold(cls, num_heads: int, config, seq_len: int, shift: int = 4, **kwargs) -> HCCSLearned:
        """Data-free start for rows of ``seq_len`` keys: ``sigma = slope_ratio0 * B``."""
        m = cls(num_heads, config, max_keys=seq_len, auto_seed=False, **kwargs)
        m._set(torch.full((num_heads,), cls.slope_ratio0 * m.budget), scale=torch.full((num_heads,), float(2**shift)))
        return m

    def seed_from_logits(self, attn_weights: torch.Tensor, valid_mask: torch.Tensor | None = None) -> None:
        """Seed the ramp at ``sigma = slope_ratio0 * B`` and the default tail for this key length."""
        _check_rank4(attn_weights)
        self._fit_keys(int(attn_weights.shape[-1]))
        self._set(torch.full((self.num_heads,), self.slope_ratio0 * self.budget))

    def _integer_params(self, budget: int, n_keys: int) -> tuple[torch.Tensor, ...]:
        """(B, S, DMAX, a) the integer kernel runs. ``B = tau + S * DMAX`` absorbs the remainder, so the tail score is
        exactly ``tau`` for every ``S``."""
        a = torch.exp2(self.log2_scale.clamp(self.scale_lo, self.scale_hi))
        sigma = torch.exp2(self.log2_slope)
        budget = torch.full_like(a, float(budget))
        floor = score_floor(n_keys, budget, self.mode, self._inv_shift)
        S = torch.clamp(torch.round(sigma / a), 1.0, 127.0)
        tau = torch.clamp(torch.round(torch.exp2(self.log2_tail)), floor, budget - S)
        D = torch.clamp(torch.floor((budget - tau) / S), 1.0, 127.0)
        B = torch.minimum(tau + S * D, budget)
        return B, S, D, a

    def export_params(self, seq_len: int | None = None) -> dict[str, torch.Tensor]:
        """Integer params for rows of ``seq_len`` keys (default ``max_keys``), with ``shift = round(log2 a)``."""
        if not self.initialized:
            raise RuntimeError("HCCSLearned: nothing to export -- seed the module first.")
        n_keys = self.max_keys if seq_len is None else int(seq_len)
        with torch.no_grad():
            B, S, D, a = self._integer_params(score_budget(n_keys), n_keys)
            shift = torch.round(torch.log2(a))
            return {
                "B": B.to(torch.int32),
                "S": S.to(torch.int32),
                "DMAX": D.to(torch.uint8),
                "shift_eff": shift.to(torch.int32),
            }

    @property
    def logit_scale(self) -> torch.Tensor:
        """Per-head factor ``a / 2**round(log2 a)`` left over by the integer shift of :meth:`export_params`."""
        with torch.no_grad():
            a = torch.exp2(self.log2_scale.clamp(self.scale_lo, self.scale_hi))
            return a / torch.exp2(torch.round(torch.log2(a)))

    def _clamp_latents_(self, budget: int) -> None:
        """Project the latents back into the deployable box (training only)."""
        with torch.no_grad():
            self.log2_scale.clamp_(self.scale_lo, self.scale_hi)
            self.log2_slope.clamp_(self.log2_scale, self.log2_scale + self._log2_max_S)
            # tensor log2: the budget is symbolic under torch.compile(dynamic=True)
            cap = torch.log2(torch.full((), float(max(2, budget)), device=self.log2_tail.device)) - 0.1
            self.log2_tail.clamp_(min=torch.zeros_like(cap), max=cap)

    def _soft_forward(self, attn_weights: torch.Tensor, valid_mask: torch.Tensor | None, n_keys: int) -> torch.Tensor:
        """The differentiable surrogate: the ramp at the continuous latents, normalised by its exact sum."""
        shape = (1, attn_weights.shape[1], 1, 1)
        budget = self.budget
        if self.training:
            self._clamp_latents_(budget)
        a = torch.exp2(self.log2_scale).view(shape)
        sigma = torch.exp2(self.log2_slope).view(shape)
        B = float(budget)
        floor = score_floor(n_keys, torch.full_like(a, B), self.mode, self._inv_shift)
        S = _clamp_ste(sigma / a, 1.0, 127.0)
        tau = torch.maximum(_clamp_ste(torch.exp2(self.log2_tail).view(shape), 1.0, B - 1.0), floor)
        D = torch.clamp((B - tau) / S, 1.0, 127.0)

        q = _round_ste(attn_weights * a)
        q = q.clamp(-128.0, 127.0)
        if valid_mask is not None:
            q = torch.where(valid_mask, q, torch.full_like(q, -128.0))
        mu = q.max(dim=-1, keepdim=True).values
        d = (mu - q).clamp_min(0.0)
        d = torch.minimum(d, D)
        score = B - S * d
        probs = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if valid_mask is not None:
            probs = probs * valid_mask.to(probs.dtype)
            if self.renorm_after_mask:
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs

    def _hard_forward(self, attn_weights: torch.Tensor, valid_mask: torch.Tensor | None, n_keys: int) -> torch.Tensor:
        """The integer kernel at the deployable params."""
        B, S, D, a = self._integer_params(self.budget, n_keys)
        return hccs_forward_per_head(
            attn_weights,
            B_per_head=B.to(torch.int32),
            S_per_head=S.to(torch.int32),
            DMAX_per_head=D.to(torch.uint8),
            shift_per_head=None,
            mode=self.mode,
            inv_shift=self._inv_shift,
            valid_mask=valid_mask,
            renorm_after_mask=self.renorm_after_mask,
            scale_per_head=a,
        )

    def _sync_scale_from_lane(self) -> None:
        """``log2_scale <- f`` of the input lane, read after the lane's forward."""
        _, _, f = self.input_quantizer.get_quantization_bits()
        f = torch.as_tensor(f, dtype=torch.float32, device=self.log2_scale.device)
        if f.ndim == 4:  # HGQ lane over [B, H, Tq, Tk]: the finest step per head
            f = f.amax(dim=(0, 2, 3))
        elif f.ndim != 0:
            f = f.amax()
        with torch.no_grad():
            self.log2_scale.copy_(f.expand(self.num_heads).clamp(self.scale_lo, self.scale_hi))

    def hgq_loss(self):
        """The lanes' HGQ regularisation."""
        return self.input_quantizer.hgq_loss() + self.output_quantizer.hgq_loss()

    def get_input_quantization_bits(self):
        return self.input_quantizer.get_quantization_bits()

    def get_output_quantization_bits(self):
        return self.output_quantizer.get_quantization_bits()

    def forward(self, attn_weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_weights = self.input_quantizer(attn_weights)
        self._sync_scale_from_lane()
        valid_mask = None if mask is None else mask.bool().expand_as(attn_weights)
        if not self.initialized:
            if not (self.auto_seed and attn_weights.ndim == 4):
                return self.output_quantizer(F.softmax(attn_weights, dim=-1))
            self.seed_from_logits(attn_weights, valid_mask)
        _check_rank4(attn_weights)
        n_keys = int(attn_weights.shape[-1])
        self._fit_keys(n_keys)
        if self.input_shape is None:
            self.input_shape = (1, *map(int, attn_weights.shape[1:]))

        # the surrogate only carries the gradient
        p_soft = self._soft_forward(attn_weights, valid_mask, n_keys) if torch.is_grad_enabled() else None
        with torch.no_grad():
            p_hard = self._hard_forward(attn_weights, valid_mask, n_keys)
        if p_soft is None:
            return self.output_quantizer(p_hard)
        # STE: p_hard's value, the surrogate's gradient
        return self.output_quantizer(p_hard + (p_soft - p_soft.detach()))

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, mode={self.mode!r}, inv_shift={self.inv_shift}, "
            f"initialized={self.initialized}, max_keys={self.max_keys}, "
            f"input_lane={self.input_quantizer.get_quantization_bits()}, "
            f"output_lane={self.output_quantizer.get_quantization_bits()}"
        )


def hccs_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The latents of every :class:`HCCSLearned` in ``model``: give them their own optimizer group with
    ``weight_decay=0`` (decay drags the ramp toward uniform attention)."""
    return [p for m in model.modules() if isinstance(m, HCCSLearned) for p in m.parameters(recurse=False)]
