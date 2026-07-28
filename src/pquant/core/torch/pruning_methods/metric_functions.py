import abc

import torch

from pquant.core.constants import (
    DISTANCE_VALUED_HAMMING,
    TARGET_RESOURCE_BRAM,
    TARGET_RESOURCE_DSP,
)
from pquant.core.torch.pruning_methods import patterns


class BaseSparsityMetric(abc.ABC):
    """Common interface for MDMM sparsity metrics: callable mapping a weight tensor to a scalar."""

    @abc.abstractmethod
    def __call__(self, weight):
        raise NotImplementedError


class UnstructuredSparsityMetric(BaseSparsityMetric):
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


class StructuredSparsityMetric(BaseSparsityMetric):
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


class FPGAAwareSparsityMetric(BaseSparsityMetric):
    """Hardware-aware sparsity metric for FPGA targets (torch port).

    Same semantics as the keras version: groups weights into DSP blocks (size ``rf``) and
    BRAM blocks (``c`` DSP blocks, ``c`` derived from ``bram_width``/``precision``), and
    returns the fraction of zero-valued groups at the chosen ``target_resource``. With
    ``l0_mode="smooth"`` the group count uses a differentiable surrogate; the "coarse"
    indicator has no weight-gradient. Inputs are validated by the Pydantic config model,
    so no constructor asserts are kept here.
    """

    def __init__(
        self, rf=1, precision=16, target_resource=TARGET_RESOURCE_DSP, bram_width=36, epsilon=1e-3,
        l0_mode='coarse', alpha=100.0,
    ):
        self.rf = int(rf)
        self.precision = int(precision)
        self.target_resource = target_resource
        self.bram_width = int(bram_width)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.c = self._calculate_c()
        self._resource_sparsity = {
            TARGET_RESOURCE_DSP: self._dsp_sparsity,
            TARGET_RESOURCE_BRAM: self._bram_sparsity,
        }
        self._zero_count = self._coarse_zero_count if l0_mode == 'coarse' else self._smooth_zero_count

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

    def _coarse_zero_count(self, norms):
        return (norms <= self.epsilon).to(norms.dtype)

    def _smooth_zero_count(self, norms):
        # Differentiable surrogate of the zero-group indicator (mirrors the smooth l0 of
        # UnstructuredSparsityMetric): ~1 for zero groups, decays to 0 with the group norm.
        return torch.exp(-self.alpha * norms.square())

    def __call__(self, weight):
        prepared = self._prepare_weights(weight)
        dsp_groups = prepared.reshape(prepared.shape[0], -1, self.rf)
        # target_resource is validated by the Pydantic config model; direct misuse fails
        # here as a missing registry key.
        return self._resource_sparsity[self.target_resource](dsp_groups)

    def _dsp_sparsity(self, dsp_groups):
        group_norms = torch.sqrt(dsp_groups.square().sum(dim=-1))
        zero_groups = self._zero_count(group_norms)
        return zero_groups.sum() / float(group_norms.numel())

    def _bram_sparsity(self, dsp_groups):
        # c >= 1 is guaranteed by FPGAAwareSparsityModel at config load.
        num_dsp_groups = dsp_groups.shape[1]
        padding = (self.c - num_dsp_groups % self.c) % self.c
        if padding:
            dsp_groups = torch.nn.functional.pad(dsp_groups, (0, 0, 0, padding))
        bram_groups = dsp_groups.reshape(dsp_groups.shape[0], -1, self.c, self.rf)
        bram_norms = torch.sqrt(bram_groups.square().sum(dim=(-1, -2)))
        zero_bram = self._zero_count(bram_norms)
        return zero_bram.sum() / float(bram_norms.numel())


class PACAPatternMetric(BaseSparsityMetric):
    """Pattern-based pruning metric (PACA, torch port).

    Each call re-derives the dominant binary patterns from the current weights, then
    returns the mean distance of every kernel to its closest dominant pattern (see the
    keras twin for why there is no cross-call cache). Operates on 4D conv weights only
    (torch kernels are natively OIHW); returns 0 for non-4D inputs. Exposes
    ``get_projection_mask`` so the MDMM layer can snap weights onto their patterns
    during fine-tuning.
    """

    def __init__(
        self, num_patterns_to_keep=16, beta=0.75, epsilon=1e-5, distance_metric=DISTANCE_VALUED_HAMMING, src="OIHW"
    ):
        self.num_patterns_to_keep = int(num_patterns_to_keep)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.distance_metric = distance_metric
        self.src = src

    def _dominant_patterns(self, weight):
        _, pats, _ = patterns.kernels_and_patterns(weight, self.src, self.epsilon)
        return patterns.select_dominant_patterns(pats, self.num_patterns_to_keep, self.beta)

    def __call__(self, weight):
        if weight.dim() != 4:
            return torch.zeros((), dtype=weight.dtype, device=weight.device)
        dominant, valid = self._dominant_patterns(weight)
        _, distances = patterns.pattern_distances(
            weight, dominant, valid, self.src, self.epsilon, self.distance_metric
        )
        return distances.min(dim=1).values.mean()

    def get_projection_mask(self, weight):
        if weight.dim() != 4:
            return torch.ones_like(weight)
        dominant, valid = self._dominant_patterns(weight)
        return patterns.projection_mask(weight, dominant, valid, self.src, self.epsilon, self.distance_metric)
