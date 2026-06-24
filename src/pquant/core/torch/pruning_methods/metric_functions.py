import torch

from pquant.core.constants import (
    DISTANCE_VALUED_HAMMING,
    FPGA_TARGET_RESOURCES,
    TARGET_RESOURCE_BRAM,
    TARGET_RESOURCE_DSP,
)
from pquant.core.torch.pruning_methods import patterns


class UnstructuredSparsityMetric:
    """L0-L1 based metric — torch port of the keras version."""

    def __init__(self, l0_mode='coarse', scale_mode="mean", epsilon=1e-3, target_sparsity=0.8, alpha=100.0):
        assert l0_mode in ['coarse', 'smooth'], "Mode must be 'coarse' or 'smooth'"
        assert scale_mode in ['sum', 'mean'], "Scale mode must be 'sum' or 'mean'"
        assert 0 <= target_sparsity <= 1, "target_sparsity must be between 0 and 1"
        self.l0_mode = l0_mode
        self.scale_mode = scale_mode
        self.target_sparsity = float(target_sparsity)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.l0_fn = self._coarse_l0 if l0_mode == 'coarse' else self._smooth_l0
        self._scaling = self._mean_scaling if scale_mode == 'mean' else self._sum_scaling

    def _sum_scaling(self, fn_value, num):
        return fn_value

    def _mean_scaling(self, fn_value, num):
        return fn_value / num

    def _coarse_l0(self, weight_vector):
        return (weight_vector.abs() <= self.epsilon).to(torch.float32).mean()

    def _smooth_l0(self, weight_vector):
        return torch.exp(-self.alpha * weight_vector.square()).mean()

    def __call__(self, weight):
        num_weights = torch.tensor(float(weight.numel()), dtype=weight.dtype, device=weight.device)
        flat = weight.reshape(-1)
        l0_term = self.l0_fn(flat)
        l1_term = flat.abs().sum()
        factor = (self.target_sparsity**2) - l0_term.square()
        fn_value = factor * l1_term
        return self._scaling(fn_value, num_weights)


class StructuredSparsityMetric:
    def __init__(self, rf=1, epsilon=1e-3):
        self.rf = int(rf)
        self.epsilon = float(epsilon)

    def __call__(self, weight):
        w_reshaped = weight.reshape(weight.shape[0], -1)
        num_weights = w_reshaped.shape[1]
        padding = (self.rf - num_weights % self.rf) % self.rf
        if padding:
            w_padded = torch.nn.functional.pad(w_reshaped, (0, padding))
        else:
            w_padded = w_reshaped
        groups = w_padded.reshape(w_padded.shape[0], -1, self.rf)
        group_norms = torch.sqrt((groups.square()).sum(dim=-1))
        zero_groups = (group_norms <= self.epsilon).to(torch.float32)
        return zero_groups.sum() / float(group_norms.numel())


class FPGAAwareSparsityMetric:
    """Hardware-aware sparsity metric for FPGA targets (torch port).

    Same semantics as the keras version: groups weights into DSP blocks (size ``rf``) and
    BRAM blocks (``c`` DSP blocks, ``c`` derived from ``bram_width``/``precision``), and
    returns the fraction of zero-valued groups at the chosen ``target_resource``. Inputs
    are validated by the Pydantic config model, so no constructor asserts are kept here.
    """

    def __init__(self, rf=1, precision=16, target_resource=TARGET_RESOURCE_DSP, bram_width=36, epsilon=1e-3):
        self.rf = int(rf)
        self.precision = int(precision)
        self.target_resource = target_resource
        self.bram_width = int(bram_width)
        self.epsilon = float(epsilon)
        self.c = self._calculate_c()

    def _calculate_c(self):
        """Number of consecutive DSP groups packed into a single BRAM block."""
        if self.bram_width % self.precision == 0:
            return self.bram_width // self.precision
        return (2 * self.bram_width) // self.precision

    def _prepare_weights(self, weight):
        """Flatten to (rows, -1) and right-pad the row length to a multiple of rf."""
        if weight.dim() == 1:
            w = weight.reshape(1, -1)
        elif weight.dim() > 2:
            w = weight.reshape(weight.shape[0], -1)
        else:
            w = weight
        padding = (self.rf - w.shape[1] % self.rf) % self.rf
        if padding:
            w = torch.nn.functional.pad(w, (0, padding))
        return w

    def __call__(self, weight):
        prepared = self._prepare_weights(weight)
        dsp_groups = prepared.reshape(prepared.shape[0], -1, self.rf)
        if self.target_resource == TARGET_RESOURCE_DSP:
            return self._dsp_sparsity(dsp_groups)
        if self.target_resource == TARGET_RESOURCE_BRAM:
            return self._bram_sparsity(dsp_groups)
        raise ValueError(f"target_resource must be one of {FPGA_TARGET_RESOURCES}, got {self.target_resource!r}")

    def _dsp_sparsity(self, dsp_groups):
        group_norms = torch.sqrt(dsp_groups.square().sum(dim=-1))
        zero_groups = (group_norms <= self.epsilon).to(dsp_groups.dtype)
        return zero_groups.sum() / float(group_norms.numel())

    def _bram_sparsity(self, dsp_groups):
        if self.c < 1:
            raise ValueError(
                f"BRAM packing needs precision <= 2*bram_width (got precision={self.precision}, "
                f"bram_width={self.bram_width} -> c={self.c})."
            )
        num_dsp_groups = dsp_groups.shape[1]
        padding = (self.c - num_dsp_groups % self.c) % self.c
        if padding:
            dsp_groups = torch.nn.functional.pad(dsp_groups, (0, 0, 0, padding))
        bram_groups = dsp_groups.reshape(dsp_groups.shape[0], -1, self.c, self.rf)
        bram_norms = torch.sqrt(bram_groups.square().sum(dim=(-1, -2)))
        zero_bram = (bram_norms <= self.epsilon).to(bram_groups.dtype)
        return zero_bram.sum() / float(bram_norms.numel())


class PACAPatternMetric:
    """Pattern-based pruning metric (PACA, torch port).

    On the first call it selects a small set of dominant binary patterns over the conv
    kernels (cached for the metric instance lifetime), then returns the mean distance of
    every kernel to its closest dominant pattern. Operates on 4D conv weights only (torch
    kernels are OIHW); returns 0 for non-4D inputs. Exposes ``get_projection_mask`` so the
    MDMM layer can snap weights onto their patterns during fine-tuning.
    """

    def __init__(
        self, num_patterns_to_keep=16, beta=0.75, epsilon=1e-5, distance_metric=DISTANCE_VALUED_HAMMING, src="OIHW"
    ):
        self.num_patterns_to_keep = int(num_patterns_to_keep)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.distance_metric = distance_metric
        self.src = src  # torch conv kernels are OIHW
        self.dominant_patterns = None
        self.valid_mask = None

    def _ensure_patterns(self, weight):
        if self.dominant_patterns is None:
            _, pats, _ = patterns.kernels_and_patterns(weight, self.src, self.epsilon)
            self.dominant_patterns, self.valid_mask = patterns.select_dominant_patterns(
                pats, self.num_patterns_to_keep, self.beta
            )

    def __call__(self, weight):
        if weight.dim() != 4:
            return torch.zeros((), dtype=weight.dtype, device=weight.device)
        self._ensure_patterns(weight)
        _, distances = patterns.pattern_distances(
            weight, self.dominant_patterns, self.valid_mask, self.src, self.epsilon, self.distance_metric
        )
        return distances.min(dim=1).values.mean()

    def get_projection_mask(self, weight):
        # Identity mask if patterns were never selected so MDMM's `weight * mask` is a no-op.
        if self.dominant_patterns is None:
            return torch.ones_like(weight)
        return patterns.projection_mask(
            weight, self.dominant_patterns, self.valid_mask, self.src, self.epsilon, self.distance_metric
        )
