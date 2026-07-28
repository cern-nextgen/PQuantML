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
    DISTANCE_COSINE,
    DISTANCE_HAMMING,
    DISTANCE_VALUED_HAMMING,
)
from pquant.core.conv_layout import layout_perm

_INF = 1e30


def convert_conv_layout(w, src, dst=CANONICAL_CONV_LAYOUT):
    """Transpose a 4D conv weight from `src` to `dst` layout (no-op if already equal)."""
    perm = layout_perm(src, dst)
    if src != dst or perm != (0, 1, 2, 3):
        return ops.transpose(w, perm)
    return w


def kernels_and_patterns(w, src, epsilon):
    """Flatten a 4D conv weight to per-kernel rows and their binary support.

    Returns (kernels, patterns, (C_out, C_in, kH, kW)):
      kernels:  (C_out*C_in, kH*kW) float - flattened kernels, canonical OIHW order.
      patterns: (C_out*C_in, kH*kW) uint8 - binary support, |w| > epsilon.
    """
    w_oihw = convert_conv_layout(w, src=src, dst=CANONICAL_CONV_LAYOUT)
    c_out, c_in, kh, kw = w_oihw.shape
    kernels = ops.reshape(w_oihw, (c_out * c_in, kh * kw))
    patterns = ops.cast(ops.greater(ops.abs(kernels), epsilon), "uint8")
    return kernels, patterns, (c_out, c_in, kh, kw)


def _pattern_codes(patterns):
    """Bit-pack each binary pattern row into a unique int64 code.

    Two rows share a code iff identical. Valid for kH*kW <= 62 (all realistic conv
    kernels); larger kernels are not expected for hardware-aware pattern pruning.
    """
    k = patterns.shape[1]
    if k > 62:
        raise ValueError(f"pattern length {k} exceeds the int64 bit-pack capacity (62 positions)")
    weights = ops.power(ops.full((k,), 2, dtype="int64"), ops.arange(k, dtype="int64"))
    return ops.sum(ops.cast(patterns, "int64") * weights, axis=1)  # (M,)


def select_dominant_patterns(patterns, num_patterns_to_keep, beta):
    """Select the most frequent distinct patterns covering `beta` of the total mass.

    Pure keras.ops. Returns (dominant, valid):
      dominant: (num_patterns_to_keep, kH*kW) uint8 - the candidate patterns.
      valid:    (num_patterns_to_keep,)       bool  - which rows are real selections
                (rows past the beta cut / past the number of distinct patterns are
                padding and must be ignored downstream).
    """
    alpha = int(num_patterns_to_keep)
    m, k = patterns.shape[0], patterns.shape[1]
    if m == 0:
        return ops.zeros((alpha, k), "uint8"), ops.zeros((alpha,), "bool")

    codes = _pattern_codes(patterns)                       # (M,) int64
    order = ops.argsort(codes)
    codes_sorted = ops.take(codes, order)
    pat_sorted = ops.take(patterns, order, axis=0)         # identical patterns now contiguous

    # First position of each distinct code in sorted order.
    not_equal_prev = ops.not_equal(codes_sorted[1:], codes_sorted[:-1])
    is_start = ops.concatenate([ops.ones((1,), "bool"), not_equal_prev], axis=0)  # (M,)

    group = ops.cumsum(ops.cast(is_start, "int32")) - 1    # dense group ids 0..U-1 (M,)
    group_counts = ops.bincount(group, minlength=m)        # counts per group id
    count_per_pos = ops.take(group_counts, group)          # (M,)

    total = ops.cast(m, "float32")
    pdf = ops.where(is_start, ops.cast(count_per_pos, "float32") / total, ops.zeros((m,), "float32"))

    # Order representatives by descending frequency; zero-pdf duplicates sink to the end.
    order2 = ops.argsort(-pdf)
    pdf_desc = ops.take(pdf, order2)
    pat_desc = ops.take(pat_sorted, order2, axis=0)
    cdf = ops.cumsum(pdf_desc)

    # keep = min( #patterns to reach beta coverage, alpha cap, #distinct patterns )
    reaches = ops.cast(cdf >= beta, "int32")
    has_hit = ops.sum(reaches) > 0
    n_beta = ops.cast(ops.argmax(reaches) + 1, "int32")
    n_distinct = ops.sum(ops.cast(is_start, "int32"))
    keep = ops.where(has_hit, n_beta, n_distinct)
    keep = ops.minimum(ops.minimum(keep, n_distinct), alpha)   # 0-d int tensor, <= alpha

    # Static-size top-alpha slice (alpha is a Python int), zero-padded if M < alpha.
    pat_top = pat_desc[:alpha]
    pad_rows = alpha - int(pat_top.shape[0])
    if pad_rows > 0:
        pat_top = ops.concatenate([pat_top, ops.zeros((pad_rows, k), pat_top.dtype)], axis=0)
    valid = ops.arange(alpha) < ops.cast(keep, "int32")        # (alpha,) bool, stays on-device
    return pat_top, valid


def _hamming_distance(kernel_support, kernel_values, dominant_support):
    return ops.sum(ops.abs(kernel_support - dominant_support), axis=-1)


def _valued_hamming_distance(kernel_support, kernel_values, dominant_support):
    return ops.sum(ops.abs(kernel_support - dominant_support) * ops.abs(kernel_values), axis=-1)


def _cosine_distance(kernel_support, kernel_values, dominant_support):
    projected = kernel_values * dominant_support
    dot = ops.sum(kernel_values * projected, axis=-1)
    denom = ops.norm(kernel_values, axis=-1) * ops.norm(projected, axis=-1) + keras.backend.epsilon()
    return 1.0 - dot / denom


# Key strings are defined once in core.constants; the function values are backend-specific
# (keras.ops here, native torch in the torch twin), which is why the registry lives next to
# the implementations instead of in constants.py.
DISTANCE_FN_REGISTRY = {
    DISTANCE_HAMMING: _hamming_distance,
    DISTANCE_VALUED_HAMMING: _valued_hamming_distance,
    DISTANCE_COSINE: _cosine_distance,
}


def _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric):
    """Distance from every kernel to every dominant pattern. -> (M_kernels, alpha)."""
    kernel_support = ops.expand_dims(ops.cast(kernel_patterns, kernels.dtype), 1)      # (M, 1, K)
    kernel_values = ops.expand_dims(kernels, 1)                                        # (M, 1, K)
    dominant_support = ops.expand_dims(ops.cast(dominant_patterns, kernels.dtype), 0)  # (1, alpha, K)
    try:
        distance_fn = DISTANCE_FN_REGISTRY[distance_metric]
    except KeyError:
        raise ValueError(f"Unsupported distance metric: {distance_metric!r}") from None
    return distance_fn(kernel_support, kernel_values, dominant_support)


def pattern_distances(w, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Per-kernel distance to each dominant pattern, with invalid patterns masked to +inf."""
    kernels, kernel_patterns, _ = kernels_and_patterns(w, src, epsilon)
    distances = _kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric)
    distances = ops.where(valid_mask[None, :], distances, ops.cast(_INF, distances.dtype))
    return kernels, distances


def projection_mask(w, dominant_patterns, valid_mask, src, epsilon, distance_metric):
    """Binary mask (same layout as `w`) projecting each kernel onto its closest dominant pattern."""
    if len(w.shape) != 4:
        return ops.ones_like(w)
    _, _, (c_out, c_in, kh, kw) = kernels_and_patterns(w, src, epsilon=0.0)
    _, distances = pattern_distances(w, dominant_patterns, valid_mask, src, epsilon, distance_metric)
    closest = ops.argmin(distances, axis=1)                       # (M,)
    mask_flat = ops.take(dominant_patterns, closest, axis=0)      # (M, K)
    mask_oihw = ops.reshape(mask_flat, (c_out, c_in, kh, kw))
    mask_src = convert_conv_layout(mask_oihw, src=CANONICAL_CONV_LAYOUT, dst=src)  # back to weight layout
    return ops.cast(mask_src, w.dtype)
