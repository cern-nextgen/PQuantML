from keras import ops

from pquant.core.constants import (
    DISTANCE_VALUED_HAMMING,
    FPGA_TARGET_RESOURCES,
    TARGET_RESOURCE_BRAM,
    TARGET_RESOURCE_DSP,
)
from pquant.core.keras.pruning_methods import patterns


class UnstructuredSparsityMetric:
    """L0-L1 based metric"""

    """Calculates the ratio of non-zero weights in a tensor."""

    def __init__(self, l0_mode='coarse', scale_mode="mean", epsilon=1e-3, target_sparsity=0.8, alpha=100.0):
        # Note: scale_mode:"sum" give very high losses for large model
        assert l0_mode in ['coarse', 'smooth'], "Mode must be 'coarse' or 'smooth'"
        assert scale_mode in ['sum', 'mean'], "Scale mode must be 'sum' or 'mean'"
        assert 0 <= target_sparsity <= 1, "target_sparsity must be between 0 and 1"
        self.l0_mode = l0_mode
        self.scale_mode = scale_mode
        self.target_sparsity = float(target_sparsity)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)

        self.l0_fn = None
        self._scaling = None

        self.build()

    def build(self):
        # l0 term -> number of zero weights/number of weights
        if self.l0_mode == 'coarse':
            self.l0_fn = self._coarse_l0
        elif self.l0_mode == 'smooth':
            self.l0_fn = self._smooth_l0

        if self.scale_mode == 'mean':
            self._scaling = self._mean_scaling
        elif self.scale_mode == 'sum':
            self._scaling = self._sum_scaling

    def _sum_scaling(self, fn_value, num):
        return fn_value

    def _mean_scaling(self, fn_value, num):
        return fn_value / num

    def _coarse_l0(self, weight_vector):
        return ops.mean(ops.cast(ops.abs(weight_vector) <= self.epsilon, "float32"))

    def _smooth_l0(self, weight_vector):
        """Differentiable approximation of L0 norm using Keras ops."""
        return ops.mean(ops.exp(-self.alpha * ops.square(weight_vector)))

    def __call__(self, weight):
        num_weights = ops.cast(ops.size(weight), weight.dtype)
        weights_vector = ops.reshape(weight, [-1])

        l0_term = self.l0_fn(weights_vector)
        l1_term = ops.sum(ops.abs(weights_vector))

        # farctor by constrction goes to zero when l0_term == target_sparsiity
        factor = ops.square(self.target_sparsity) - ops.square(l0_term)
        fn_value = factor * l1_term
        fn_value = self._scaling(fn_value, num_weights)

        return fn_value


class StructuredSparsityMetric:
    """Calculates the ratio of near-zero weight groups (based on Reuse Factor: rf)."""

    def __init__(self, rf=1, epsilon=1e-3):
        self.rf = rf
        self.epsilon = epsilon

    def __call__(self, weight):
        original_shape = weight.shape
        w_reshaped = ops.reshape(weight, (original_shape[0], -1))
        num_weights = ops.shape(w_reshaped)[1]

        padding = (self.rf - num_weights % self.rf) % self.rf
        w_padded = ops.pad(w_reshaped, [[0, 0], [0, padding]])

        groups = ops.reshape(w_padded, (original_shape[0], -1, self.rf))
        group_norms = ops.sqrt(ops.sum(ops.square(groups), axis=-1))
        zero_groups = ops.less_equal(group_norms, self.epsilon)
        num_groups = ops.cast(ops.size(group_norms), "float32")

        return ops.sum(ops.cast(zero_groups, "float32")) / num_groups


class FPGAAwareSparsityMetric:
    """Hardware-aware sparsity metric for FPGA targets.

    Models how weights are packed into DSP blocks (groups of size ``rf``) and further
    into BRAM blocks (groups of ``c`` DSP blocks, where ``c`` derives from ``bram_width``
    and ``precision``). Returns the fraction of zero-valued groups at the chosen
    ``target_resource`` level. Constructor inputs are validated by the Pydantic config
    model (MDMMPruningModel), so no input asserts are kept here.
    """

    def __init__(self, rf=1, precision=16, target_resource=TARGET_RESOURCE_DSP, bram_width=36, epsilon=1e-3):
        self.rf = rf
        self.precision = precision
        self.target_resource = target_resource
        self.bram_width = bram_width
        self.epsilon = epsilon
        self.c = self._calculate_c()

    def _calculate_c(self):
        """Number of consecutive DSP groups packed into a single BRAM block."""
        if self.bram_width % self.precision == 0:
            return self.bram_width // self.precision
        return (2 * self.bram_width) // self.precision

    def _prepare_weights(self, weight):
        """Flatten to (rows, -1) and right-pad the row length to a multiple of rf."""
        original_shape = weight.shape
        # 1D (e.g. bias) -> single row; 2D -> as-is; >2D (e.g. Conv2D) -> flatten trailing dims.
        if len(original_shape) == 1:
            weight_reshaped = ops.reshape(weight, (1, -1))
        elif len(original_shape) > 2:
            weight_reshaped = ops.reshape(weight, (original_shape[0], -1))
        else:
            weight_reshaped = weight
        num_weights = ops.shape(weight_reshaped)[1]
        padding_needed = (self.rf - num_weights % self.rf) % self.rf
        return ops.pad(weight_reshaped, [[0, 0], [0, padding_needed]])

    def __call__(self, weight):
        prepared = self._prepare_weights(weight)
        dsp_groups = ops.reshape(prepared, (prepared.shape[0], -1, self.rf))
        if self.target_resource == TARGET_RESOURCE_DSP:
            return self._dsp_sparsity(dsp_groups)
        if self.target_resource == TARGET_RESOURCE_BRAM:
            return self._bram_sparsity(dsp_groups)
        raise ValueError(f"target_resource must be one of {FPGA_TARGET_RESOURCES}, got {self.target_resource!r}")

    def _dsp_sparsity(self, dsp_groups):
        """A DSP block is pruned when the L2-norm of its weight group is below epsilon."""
        group_norms = ops.sqrt(ops.sum(ops.square(dsp_groups), axis=-1))
        zero_groups = ops.less_equal(group_norms, self.epsilon)
        num_groups = ops.cast(ops.size(group_norms), dsp_groups.dtype)
        return ops.sum(ops.cast(zero_groups, dsp_groups.dtype)) / num_groups

    def _bram_sparsity(self, dsp_groups):
        """A BRAM block is pruned when the L2-norm of all weights stored in it is below epsilon."""
        if self.c < 1:
            raise ValueError(
                f"BRAM packing needs precision <= 2*bram_width (got precision={self.precision}, "
                f"bram_width={self.bram_width} -> c={self.c})."
            )
        num_dsp_groups = ops.shape(dsp_groups)[1]
        bram_padding = (self.c - num_dsp_groups % self.c) % self.c
        dsp_padded = ops.pad(dsp_groups, [[0, 0], [0, bram_padding], [0, 0]])
        bram_groups = ops.reshape(dsp_padded, (dsp_groups.shape[0], -1, self.c, self.rf))
        bram_norms = ops.sqrt(ops.sum(ops.square(bram_groups), axis=(-1, -2)))
        zero_bram = ops.less_equal(bram_norms, self.epsilon)
        num_bram = ops.cast(ops.size(bram_norms), dsp_groups.dtype)
        return ops.sum(ops.cast(zero_bram, dsp_groups.dtype)) / num_bram


class PACAPatternMetric:
    """Pattern-based pruning metric (PACA).

    On the first call it selects a small set of dominant binary patterns over the conv
    kernels (cached for the metric instance lifetime), then returns the mean distance of
    every kernel to its closest dominant pattern. Operates on 4D conv weights only (Keras
    kernels are HWIO); returns 0 for non-4D inputs. Exposes ``get_projection_mask`` so the
    MDMM layer can snap weights onto their patterns during fine-tuning. Constructor inputs
    are validated by the Pydantic config model (MDMMPruningModel).
    """

    def __init__(
        self, num_patterns_to_keep=16, beta=0.75, epsilon=1e-5, distance_metric=DISTANCE_VALUED_HAMMING, src="HWIO"
    ):
        self.num_patterns_to_keep = num_patterns_to_keep
        self.beta = beta
        self.epsilon = epsilon
        self.distance_metric = distance_metric
        self.src = src  # Keras conv kernels are HWIO
        self.dominant_patterns = None
        self.valid_mask = None

    def _ensure_patterns(self, weight):
        if self.dominant_patterns is None:
            _, pats, _ = patterns.kernels_and_patterns(weight, self.src, self.epsilon)
            self.dominant_patterns, self.valid_mask = patterns.select_dominant_patterns(
                pats, self.num_patterns_to_keep, self.beta
            )

    def __call__(self, weight):
        if len(weight.shape) != 4:
            return ops.convert_to_tensor(0.0, dtype=weight.dtype)
        self._ensure_patterns(weight)
        _, distances = patterns.pattern_distances(
            weight, self.dominant_patterns, self.valid_mask, self.src, self.epsilon, self.distance_metric
        )
        return ops.mean(ops.min(distances, axis=1))

    def get_projection_mask(self, weight):
        # Identity mask if patterns were never selected (e.g. metric never invoked) so the
        # MDMM layer's `weight * mask` is a safe no-op.
        if self.dominant_patterns is None:
            return ops.ones_like(weight)
        return patterns.projection_mask(
            weight, self.dominant_patterns, self.valid_mask, self.src, self.epsilon, self.distance_metric
        )
