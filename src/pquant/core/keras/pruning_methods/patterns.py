# @Author: Arghya Ranjan Das
# PACA pattern utilities (Keras backend).
#
# Selects a small set of dominant binary "patterns" (the support, i.e. the non-zero
# positions, of each conv kernel) and measures how far each kernel is from its closest
# dominant pattern. Everything stays in keras.ops (on-device): unique-pattern counting
# is done via bit-pack -> argsort -> bincount over sorted runs, and the dominant set is
# returned at a fixed size (num_patterns_to_keep) together with a validity mask, so no
# host<->device sync or NumPy round-trip of the weight is needed.
#
# Conv weights are canonicalised to OIHW (C_out, C_in, kH, kW) before any pattern logic,
# so both the Keras (HWIO kernel) and Torch (OIHW) layouts are handled by passing `src`.

import keras
from keras import ops

from pquant.core.constants import (
    CANONICAL_CONV_LAYOUT,
    PACA_DISTANCE_METRICS,
)
from pquant.core.conv_layout import layout_perm

_INF = 1e30


def convert_conv_layout(weight, src, dst=CANONICAL_CONV_LAYOUT):
    """Transpose a 4D conv weight from `src` to `dst` layout (no-op if already equal)."""
    permutation = layout_perm(src, dst)
    if src != dst or permutation != (0, 1, 2, 3):
        return ops.transpose(weight, permutation)
    return weight


def kernels_and_patterns(weight, src, epsilon):
    """Flatten a 4D conv weight to per-kernel rows and their binary support.

    Returns (kernels, patterns, (C_out, C_in, kH, kW)):
      kernels:  (C_out*C_in, kH*kW) float - flattened kernels, canonical OIHW order.
      patterns: (C_out*C_in, kH*kW) uint8 - binary support, |weight| > epsilon.
    """
    weight_oihw = convert_conv_layout(weight, src=src, dst=CANONICAL_CONV_LAYOUT)
    out_channels, in_channels, kernel_height, kernel_width = weight_oihw.shape
    kernels = ops.reshape(weight_oihw, (out_channels * in_channels, kernel_height * kernel_width))
    patterns = ops.cast(ops.greater(ops.abs(kernels), epsilon), "uint8")
    return kernels, patterns, (out_channels, in_channels, kernel_height, kernel_width)


def _pattern_codes(patterns):
    """Bit-pack each binary pattern row into a unique int64 code.

    Two rows share a code iff identical. Valid for kH*kW <= 62 (all realistic conv
    kernels); larger kernels are not expected for hardware-aware pattern pruning.
    """
    pattern_size = patterns.shape[1]
    if pattern_size > 62:
        raise ValueError(f"pattern length {pattern_size} exceeds the int64 bit-pack capacity (62 positions)")
    weights = ops.power(ops.full((pattern_size,), 2, dtype="int64"), ops.arange(pattern_size, dtype="int64"))
    return ops.sum(ops.cast(patterns, "int64") * weights, axis=1)  # (M,)


def select_dominant_patterns(patterns, num_patterns_to_keep, beta):
    """Select the most frequent distinct patterns covering `beta` of the total mass.

    Pure keras.ops. Returns (dominant, valid):
      dominant: (num_patterns_to_keep, kH*kW) uint8 - the candidate patterns.
      valid:    (num_patterns_to_keep,)       bool  - which rows are real selections
                (rows past the beta cut / past the number of distinct patterns are
                padding and must be ignored downstream).
    """
    pattern_limit = int(num_patterns_to_keep)
    num_kernels, pattern_size = patterns.shape[0], patterns.shape[1]
    if num_kernels == 0:
        return ops.zeros((pattern_limit, pattern_size), "uint8"), ops.zeros((pattern_limit,), "bool")

    codes = _pattern_codes(patterns)  # (M,) int64
    order = ops.argsort(codes)
    sorted_codes = ops.take(codes, order)
    sorted_patterns = ops.take(patterns, order, axis=0)  # identical patterns now contiguous

    # First position of each distinct code in sorted order.
    not_equal_prev = ops.not_equal(sorted_codes[1:], sorted_codes[:-1])
    is_start = ops.concatenate([ops.ones((1,), "bool"), not_equal_prev], axis=0)  # (M,)

    group = ops.cumsum(ops.cast(is_start, "int32")) - 1  # dense group ids 0..U-1 (M,)
    group_counts = ops.bincount(group, minlength=num_kernels)  # counts per group id
    count_per_position = ops.take(group_counts, group)  # (M,)

    total = ops.cast(num_kernels, "float32")
    frequencies = ops.where(is_start, ops.cast(count_per_position, "float32") / total, ops.zeros((num_kernels,), "float32"))

    # Order representatives by descending frequency; duplicates with zero frequency sink to the end.
    frequency_order = ops.argsort(-frequencies)
    descending_frequencies = ops.take(frequencies, frequency_order)
    patterns_by_frequency = ops.take(sorted_patterns, frequency_order, axis=0)
    cumulative_frequency = ops.cumsum(descending_frequencies)

    # patterns_to_keep = min( #patterns to reach beta coverage, pattern_limit cap, #distinct patterns )
    coverage_reached = ops.cast(cumulative_frequency >= beta, "int32")
    has_hit = ops.sum(coverage_reached) > 0
    patterns_for_coverage = ops.cast(ops.argmax(coverage_reached) + 1, "int32")
    num_distinct_patterns = ops.sum(ops.cast(is_start, "int32"))
    patterns_to_keep = ops.where(has_hit, patterns_for_coverage, num_distinct_patterns)
    patterns_to_keep = ops.minimum(
        ops.minimum(patterns_to_keep, num_distinct_patterns), pattern_limit
    )  # 0-d int tensor, <= pattern_limit

    # Keep a fixed-size slice, padding when there are fewer kernels than pattern_limit.
    dominant_patterns = patterns_by_frequency[:pattern_limit]
    pad_rows = pattern_limit - int(dominant_patterns.shape[0])
    if pad_rows > 0:
        dominant_patterns = ops.concatenate(
            [dominant_patterns, ops.zeros((pad_rows, pattern_size), dominant_patterns.dtype)], axis=0
        )
    valid = ops.arange(pattern_limit) < ops.cast(patterns_to_keep, "int32")  # (pattern_limit,) bool, stays on-device
    return dominant_patterns, valid


def _hamming_distance(kernel_support, kernel_values, dominant_support):
    return ops.sum(ops.abs(kernel_support - dominant_support), axis=-1)


def _valued_hamming_distance(kernel_support, kernel_values, dominant_support):
    return ops.sum(ops.abs(kernel_support - dominant_support) * ops.abs(kernel_values), axis=-1)


def _cosine_distance(kernel_support, kernel_values, dominant_support):
    projected = kernel_values * dominant_support
    dot = ops.sum(kernel_values * projected, axis=-1)
    denom = ops.norm(kernel_values, axis=-1) * ops.norm(projected, axis=-1) + keras.backend.epsilon()
    return 1.0 - dot / denom


# Shared names resolve to the distance functions implemented by this backend.
DISTANCE_FN_REGISTRY = {distance: globals()[function_name] for distance, function_name in PACA_DISTANCE_METRICS.items()}


def _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric):
    """Distance from every kernel to every dominant pattern. -> (M_kernels, pattern_limit)."""
    kernel_support = ops.expand_dims(ops.cast(kernel_patterns, kernels.dtype), 1)  # (M, 1, K)
    kernel_values = ops.expand_dims(kernels, 1)  # (M, 1, K)
    dominant_support = ops.expand_dims(ops.cast(dominant_patterns, kernels.dtype), 0)  # (1, pattern_limit, K)
    # distance_metric is validated by the Pydantic config model; a bad value on the
    # direct-construction path fails here as a missing registry key.
    return DISTANCE_FN_REGISTRY[distance_metric](kernel_support, kernel_values, dominant_support)


def pattern_distances(weight, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Per-kernel distance to each dominant pattern, with invalid patterns masked to +inf."""
    kernels, kernel_patterns, _ = kernels_and_patterns(weight, src, epsilon)
    distances = _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric)
    distances = ops.where(valid_mask[None, :], distances, ops.cast(_INF, distances.dtype))
    return kernels, distances


def projection_mask(weight, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Binary mask (same layout as `weight`) projecting each kernel onto its closest dominant pattern."""
    if len(weight.shape) != 4:
        return ops.ones_like(weight)
    _, _, (out_channels, in_channels, kernel_height, kernel_width) = kernels_and_patterns(weight, src, epsilon=0.0)
    _, distances = pattern_distances(weight, dominant_patterns, valid_mask, src, epsilon, distance_metric)
    closest = ops.argmin(distances, axis=1)  # (M,)
    mask_flat = ops.take(dominant_patterns, closest, axis=0)  # (M, K)
    mask_oihw = ops.reshape(mask_flat, (out_channels, in_channels, kernel_height, kernel_width))
    mask_src = convert_conv_layout(mask_oihw, src=CANONICAL_CONV_LAYOUT, dst=src)  # back to weight layout
    return ops.cast(mask_src, weight.dtype)
