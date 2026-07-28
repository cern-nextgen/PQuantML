# @Author: Arghya Ranjan Das
# PACA pattern utilities (Torch backend) — native-torch port of the keras version.
#
# Torch conv weights are already OIHW (the canonical layout), so layout conversion is
# normally a no-op; `src` is kept for API parity and correctness should a different layout
# ever be passed. Unique-pattern counting uses torch.unique (on-device, no NumPy). The
# dominant set is returned at a fixed size (num_patterns_to_keep) with a validity mask, so
# downstream shapes are static and the weight never leaves the device.

import torch

from pquant.core.constants import (
    CANONICAL_CONV_LAYOUT,
    DISTANCE_COSINE,
    DISTANCE_HAMMING,
    DISTANCE_VALUED_HAMMING,
)
from pquant.core.conv_layout import layout_perm

_INF = 1e30


def convert_conv_layout(w, src, dst=CANONICAL_CONV_LAYOUT):
    """Permute a 4D conv weight from `src` to `dst` layout (no-op if already equal)."""
    perm = layout_perm(src, dst)
    if src != dst or perm != (0, 1, 2, 3):
        return w.permute(*perm).contiguous()
    return w


def kernels_and_patterns(w, src, epsilon):
    """Flatten a 4D conv weight to per-kernel rows and their binary support.

    Returns (kernels, patterns, (C_out, C_in, kH, kW)):
      kernels:  (C_out*C_in, kH*kW) float - flattened kernels, canonical OIHW order.
      patterns: (C_out*C_in, kH*kW) uint8 - binary support, |w| > epsilon.
    """
    w_oihw = convert_conv_layout(w, src=src, dst=CANONICAL_CONV_LAYOUT)
    c_out, c_in, kh, kw = w_oihw.shape
    kernels = w_oihw.reshape(c_out * c_in, kh * kw)
    patterns = (kernels.abs() > epsilon).to(torch.uint8)
    return kernels, patterns, (c_out, c_in, kh, kw)


def select_dominant_patterns(patterns, num_patterns_to_keep, beta):
    """Select the most frequent distinct patterns covering `beta` of the total mass.

    Native torch (torch.unique). Returns (dominant, valid):
      dominant: (num_patterns_to_keep, kH*kW) uint8 - the candidate patterns.
      valid:    (num_patterns_to_keep,)       bool  - which rows are real selections
                (padding rows past the beta cut / distinct-pattern count are ignored).
    """
    alpha = int(num_patterns_to_keep)
    m, k = patterns.shape
    if m == 0:
        return patterns.new_zeros((alpha, k)), patterns.new_zeros((alpha,), dtype=torch.bool)

    uniq, counts = torch.unique(patterns, dim=0, return_counts=True)   # (U, K), (U,)
    pdf = counts.to(torch.float32) / float(m)
    order = torch.argsort(pdf, descending=True)
    uniq, pdf = uniq[order], pdf[order]
    cdf = torch.cumsum(pdf, dim=0)

    reaches = cdf >= beta
    n_beta = int(torch.argmax(reaches.to(torch.int32))) + 1 if bool(reaches.any()) else uniq.shape[0]
    keep = min(n_beta, uniq.shape[0], alpha)

    dom = uniq[:alpha]
    if dom.shape[0] < alpha:
        dom = torch.cat([dom, dom.new_zeros((alpha - dom.shape[0], k))], dim=0)
    valid = torch.arange(alpha, device=patterns.device) < keep
    return dom, valid


def _hamming_distance(kernel_support, kernel_values, dominant_support):
    return (kernel_support - dominant_support).abs().sum(dim=-1)


def _valued_hamming_distance(kernel_support, kernel_values, dominant_support):
    return ((kernel_support - dominant_support).abs() * kernel_values.abs()).sum(dim=-1)


def _cosine_distance(kernel_support, kernel_values, dominant_support):
    projected = kernel_values * dominant_support
    dot = (kernel_values * projected).sum(dim=-1)
    denom = kernel_values.norm(dim=-1) * projected.norm(dim=-1) + 1e-7
    return 1.0 - dot / denom


# Same keys as the keras twin (from core.constants); values are native-torch functions,
# so the registry lives next to the implementations instead of in constants.py.
DISTANCE_FN_REGISTRY = {
    DISTANCE_HAMMING: _hamming_distance,
    DISTANCE_VALUED_HAMMING: _valued_hamming_distance,
    DISTANCE_COSINE: _cosine_distance,
}


def _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric):
    """Distance from every kernel to every dominant pattern. -> (M_kernels, alpha)."""
    kernel_support = kernel_patterns.to(kernels.dtype).unsqueeze(1)        # (M, 1, K)
    kernel_values = kernels.unsqueeze(1)                                   # (M, 1, K)
    dominant_support = dominant_patterns.to(kernels.dtype).unsqueeze(0)    # (1, alpha, K)
    # distance_metric is validated by the Pydantic config model; a bad value on the
    # direct-construction path fails here as a missing registry key.
    return DISTANCE_FN_REGISTRY[distance_metric](kernel_support, kernel_values, dominant_support)


def pattern_distances(w, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Per-kernel distance to each dominant pattern, with invalid patterns masked to +inf."""
    kernels, kernel_patterns, _ = kernels_and_patterns(w, src, epsilon)
    distances = _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric)
    distances = torch.where(valid_mask.unsqueeze(0), distances, distances.new_full((), _INF))
    return kernels, distances


def projection_mask(w, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Binary mask (same layout as `w`) projecting each kernel onto its closest dominant pattern."""
    if w.dim() != 4:
        return torch.ones_like(w)
    _, _, (c_out, c_in, kh, kw) = kernels_and_patterns(w, src, epsilon=0.0)
    _, distances = pattern_distances(w, dominant_patterns, valid_mask, src, epsilon, distance_metric)
    closest = torch.argmin(distances, dim=1)                  # (M,)
    mask_flat = dominant_patterns[closest]                    # (M, K)
    mask_oihw = mask_flat.reshape(c_out, c_in, kh, kw)
    mask_src = convert_conv_layout(mask_oihw, src=CANONICAL_CONV_LAYOUT, dst=src)
    return mask_src.to(w.dtype)
