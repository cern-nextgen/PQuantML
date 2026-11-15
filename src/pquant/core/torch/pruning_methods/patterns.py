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
    PACA_DISTANCE_METRICS,
)
from pquant.core.conv_layout import layout_perm

_INF = 1e30


def convert_conv_layout(weight, src, dst=CANONICAL_CONV_LAYOUT):
    """Permute a 4D conv weight from `src` to `dst` layout (no-op if already equal)."""
    permutation = layout_perm(src, dst)
    if src != dst or permutation != (0, 1, 2, 3):
        return weight.permute(*permutation).contiguous()
    return weight


def kernels_and_patterns(weight, src, epsilon):
    """Flatten a 4D conv weight to per-kernel rows and their binary support.

    Returns (kernels, patterns, (C_out, C_in, kH, kW)):
      kernels:  (C_out*C_in, kH*kW) float - flattened kernels, canonical OIHW order.
      patterns: (C_out*C_in, kH*kW) uint8 - binary support, |weight| > epsilon.
    """
    weight_oihw = convert_conv_layout(weight, src=src, dst=CANONICAL_CONV_LAYOUT)
    out_channels, in_channels, kernel_height, kernel_width = weight_oihw.shape
    kernels = weight_oihw.reshape(out_channels * in_channels, kernel_height * kernel_width)
    patterns = (kernels.abs() > epsilon).to(torch.uint8)
    return kernels, patterns, (out_channels, in_channels, kernel_height, kernel_width)


def select_dominant_patterns(patterns, num_patterns_to_keep, beta):
    """Select the most frequent distinct patterns covering `beta` of the total mass.

    Native torch (torch.unique). Returns (dominant, valid):
      dominant: (num_patterns_to_keep, kH*kW) uint8 - the candidate patterns.
      valid:    (num_patterns_to_keep,)       bool  - which rows are real selections
                (padding rows past the beta cut / distinct-pattern count are ignored).
    """
    pattern_limit = int(num_patterns_to_keep)
    num_kernels, pattern_size = patterns.shape
    if num_kernels == 0:
        return patterns.new_zeros((pattern_limit, pattern_size)), patterns.new_zeros((pattern_limit,), dtype=torch.bool)

    unique_patterns, counts = torch.unique(patterns, dim=0, return_counts=True)  # (U, K), (U,)
    frequencies = counts.to(torch.float32) / float(num_kernels)
    order = torch.argsort(frequencies, descending=True)
    unique_patterns, frequencies = unique_patterns[order], frequencies[order]
    cumulative_frequency = torch.cumsum(frequencies, dim=0)

    coverage_reached = cumulative_frequency >= beta
    patterns_for_coverage = (
        int(torch.argmax(coverage_reached.to(torch.int32))) + 1 if bool(coverage_reached.any()) else unique_patterns.shape[0]
    )
    patterns_to_keep = min(patterns_for_coverage, unique_patterns.shape[0], pattern_limit)

    dominant_patterns = unique_patterns[:pattern_limit]
    if dominant_patterns.shape[0] < pattern_limit:
        dominant_patterns = torch.cat(
            [dominant_patterns, dominant_patterns.new_zeros((pattern_limit - dominant_patterns.shape[0], pattern_size))],
            dim=0,
        )
    valid = torch.arange(pattern_limit, device=patterns.device) < patterns_to_keep
    return dominant_patterns, valid


def _hamming_distance(kernel_support, kernel_values, dominant_support):
    return (kernel_support - dominant_support).abs().sum(dim=-1)


def _valued_hamming_distance(kernel_support, kernel_values, dominant_support):
    return ((kernel_support - dominant_support).abs() * kernel_values.abs()).sum(dim=-1)


def _cosine_distance(kernel_support, kernel_values, dominant_support):
    projected = kernel_values * dominant_support
    dot = (kernel_values * projected).sum(dim=-1)
    denom = kernel_values.norm(dim=-1) * projected.norm(dim=-1) + 1e-7
    return 1.0 - dot / denom


# Shared names resolve to the distance functions implemented by this backend.
DISTANCE_FN_REGISTRY = {distance: globals()[function_name] for distance, function_name in PACA_DISTANCE_METRICS.items()}


def _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric):
    """Distance from every kernel to every dominant pattern. -> (M_kernels, pattern_limit)."""
    kernel_support = kernel_patterns.to(kernels.dtype).unsqueeze(1)  # (M, 1, K)
    kernel_values = kernels.unsqueeze(1)  # (M, 1, K)
    dominant_support = dominant_patterns.to(kernels.dtype).unsqueeze(0)  # (1, pattern_limit, K)
    # distance_metric is validated by the Pydantic config model; a bad value on the
    # direct-construction path fails here as a missing registry key.
    return DISTANCE_FN_REGISTRY[distance_metric](kernel_support, kernel_values, dominant_support)


def pattern_distances(weight, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Per-kernel distance to each dominant pattern, with invalid patterns masked to +inf."""
    kernels, kernel_patterns, _ = kernels_and_patterns(weight, src, epsilon)
    distances = _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric)
    distances = torch.where(valid_mask.unsqueeze(0), distances, distances.new_full((), _INF))
    return kernels, distances


def projection_mask(weight, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Binary mask (same layout as `weight`) projecting each kernel onto its closest dominant pattern."""
    if weight.dim() != 4:
        return torch.ones_like(weight)
    _, _, (out_channels, in_channels, kernel_height, kernel_width) = kernels_and_patterns(weight, src, epsilon=0.0)
    _, distances = pattern_distances(weight, dominant_patterns, valid_mask, src, epsilon, distance_metric)
    closest = torch.argmin(distances, dim=1)  # (M,)
    mask_flat = dominant_patterns[closest]  # (M, K)
    mask_oihw = mask_flat.reshape(out_channels, in_channels, kernel_height, kernel_width)
    mask_src = convert_conv_layout(mask_oihw, src=CANONICAL_CONV_LAYOUT, dst=src)
    return mask_src.to(weight.dtype)
