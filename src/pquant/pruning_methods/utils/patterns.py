from functools import lru_cache
import numpy as np
import keras
from keras import ops

_AX = {"H": 0,   # kernel height  (kH)
       "W": 1,   # kernel width   (kW)
       "I": 2,   # input  channels (C_in)
       "O": 3}   # output channels (C_out)

def _layout_to_axes(layout: str):
    if len(layout) != 4 or set(layout) != set("HWIO"):
        raise ValueError("layout must be a permutation of 'HWIO'")
    return tuple(_AX[ch] for ch in layout)

@lru_cache(maxsize=None)
def _perm(src: str, dst: str):
    """
    Constant-time (cached) permutation tuple that reorders *src*→*dst*.
    """
    s = _layout_to_axes(src)
    d = _layout_to_axes(dst)
    return tuple(s.index(ax) for ax in d)


def convert_conv_layout(w, src: str, dst: str = "HWIO"):
    if src == dst:
        return w
    perm = _perm(src, dst)                     # Python‑level, cached
    if perm == (0, 1, 2, 3):                   # identity permutation
        return w
    return ops.transpose(w, perm)


def _get_kernels_and_patterns(w, src="OIHW", epsilon=1e-5):
    # src:
    #   PyTorch: (out, in, kH, kW): OIHW
    #   Keras  : (kH, kW, in, out): HWIO
    w_permuted = convert_conv_layout(w, src="OIHW", dst="OIHW")
    C_out, C_in, kH, kW = ops.shape(w_permuted)
    kernels = ops.reshape(w_permuted, (C_out * C_in, -1))
    all_patterns = ops.cast(ops.greater(ops.abs(kernels), epsilon), dtype="uint8")

    return kernels, all_patterns, (C_out, C_in, kH, kW)

def _get_unique_patterns_with_counts(all_patterns):
    """Returns the unique patterns and their counts."""
    np_patterns = ops.convert_to_numpy(all_patterns)
    # This is currently in nummpy implementation
    uniq_np, counts_np = np.unique(np_patterns, axis=0, return_counts=True)

    unique_patterns = ops.convert_to_tensor(uniq_np, dtype=all_patterns.dtype)
    counts          = ops.convert_to_tensor(counts_np.astype("int32"), dtype="int32")
    return unique_patterns, counts

def _select_dominant_patterns(all_patterns, unique_patterns, counts, alpha, beta, dtype=None):
    """Selects the most frequent patterns based on alpha and beta."""
    if not dtype:
        raise ValueError("dtype must be provided")
    if ops.shape(unique_patterns)[0] == 0:
        return unique_patterns

    total = ops.cast(ops.shape(all_patterns)[0], dtype)
    pdf   = ops.cast(counts, dtype) / total

    order   = ops.argsort(-pdf)
    pdf_s   = ops.take(pdf, order)
    pat_s   = ops.take(unique_patterns, order, axis=0)

    cdf     = ops.cumsum(pdf_s)
    mask    = cdf >= beta
    has_hit = ops.any(mask)

    idx      = ops.argmax(mask)
    n_beta   = ops.cast(idx + 1, counts.dtype)
    n_total  = ops.cast(ops.shape(cdf)[0], counts.dtype)

    keep_beta = ops.where(has_hit, n_beta, n_total)
    keep      = ops.minimum(keep_beta, ops.cast(alpha, counts.dtype))

    return pat_s[:keep]


def count_dominant_patterns(dominant_patterns, unique_patterns, counts):
    """
    Shapes:
      dominant_patterns: [M, K]
      unique_patterns:   [U, K]
      counts:            [U]
    """
    dp_exp = ops.expand_dims(dominant_patterns, axis=1)  # [M, 1, K]
    up_exp = ops.expand_dims(unique_patterns, axis=0)    # [1, U, K]
    matches = ops.all(ops.equal(dp_exp, up_exp), axis=-1)  # [M, U]

    idx = ops.argmax(matches, axis=1)  # [M]
    dom_counts = ops.take(counts, idx, axis=0)  # [M]
    return dom_counts


def calc_pattern_distances(Tk, P, k, distance_metric='cosine'):
    """
    Compute distances between a set of target kernels Tk and a set of patterns P,
    using the specified distance metric.
    """
    # Cast uint8 patterns/kernel-binarization to the kernel float dtype so
    # arithmetic with the float kernels works under TF's strict-dtype rules.
    Tk = ops.cast(Tk, k.dtype)
    P = ops.cast(P, k.dtype)
    Tk_exp = ops.expand_dims(Tk, 1)
    k_exp = ops.expand_dims(k, 1)
    P_exp = ops.expand_dims(P, 0)


    if distance_metric == 'hamming':
        distances = ops.sum(ops.abs(Tk_exp - P_exp), axis=-1)
    elif distance_metric == 'valued_hamming':
        abs_diff = ops.abs(Tk_exp - P_exp)
        distances = ops.sum(abs_diff * ops.abs(k_exp), axis=-1)
    elif distance_metric == 'cosine':
        projected_kernels = k_exp * P_exp
        k_dot_projected = ops.sum(k_exp * projected_kernels, axis=-1)
        norm_k = ops.norm(k_exp, axis=-1)
        norm_projected = ops.norm(projected_kernels, axis=-1)
        cosine_similarity = k_dot_projected / (norm_k * norm_projected + keras.backend.epsilon())
        distances = 1.0 - cosine_similarity
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    return distances


def _pattern_distances(w, dominant_patterns, src="OIHW", epsilon=1e-5, distance_metric='cosine'):
    """Calculates the distance of all kernels to the set of dominant patterns."""
    if dominant_patterns is None:
        raise ValueError("Dominant patterns have not been selected yet.")

    w_kernels, w_patterns, _ = _get_kernels_and_patterns(w, src, epsilon)
    distances = calc_pattern_distances(w_patterns, dominant_patterns,
                                       w_kernels, distance_metric)

    return w_kernels, distances

def _get_projection_mask(weight, dominant_patterns, src="OIHW", epsilon=1e-5, distance_metric='cosine'):
    if len(ops.shape(weight)) != 4:
        return ops.ones_like(weight)
    _, _, (C_out, C_in, kH, kW) = _get_kernels_and_patterns(weight, src, epsilon=0.0)
    _, distances = _pattern_distances(weight, dominant_patterns,
                                      src,
                                      epsilon,
                                      distance_metric)  # Shape: (C_out*C_in, num_dominant)
    closest_pattern_indices = ops.argmin(distances, axis=1)  # Shape: (C_out*C_in,)
    projection_mask_flat = ops.take(dominant_patterns, closest_pattern_indices, axis=0)
    projection_mask = ops.reshape(projection_mask_flat, (C_out, C_in, kH, kW))
    # Cast back to the weight dtype so downstream `weight * mask` works under TF strict dtypes.
    return ops.cast(projection_mask, weight.dtype)
