import keras
from keras import ops

from pquant.pruning_methods.utils import patterns


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

    Models how weights are packed into DSP blocks (groups of size `rf`) and
    further into BRAM blocks (groups of `c` DSP blocks, where `c` derives from
    `bram_width` and `precision`). Returns the fraction of zero-valued groups
    at the chosen target_resource level.
    """

    def __init__(self, rf=1, precision=16, target_resource='DSP',
                 bram_width=36, epsilon=1e-3):
        assert target_resource in ['DSP', 'BRAM'], "target_resource must be 'DSP' or 'BRAM'."
        assert rf >= 1, "rf must be >= 1"
        assert precision >= 1, "precision must be >= 1"
        assert bram_width >= 1, "bram_width must be >= 1"
        self.rf = rf
        self.precision = precision
        self.target_resource = target_resource
        self.bram_width = bram_width
        self.epsilon = epsilon

        self.c = self._calculate_c()
        assert self.c >= 1, (
            f"Computed c={self.c} from precision={precision}, bram_width={bram_width}. "
            "BRAM packing requires precision <= 2*bram_width."
        )

    def _calculate_c(self):
        """Calculates 'C', the number of consecutive DSP groups packed into a single BRAM block."""
        if self.bram_width % self.precision == 0:
            return self.bram_width // self.precision
        else:
            return (2 * self.bram_width) // self.precision

    def _prepare_weights(self, weight):
        """Reshapes and pads the weight tensor to align with the Reuse Factor (RF)."""
        original_shape = weight.shape
        # 1D (e.g. bias) -> single-row matrix; 2D -> as-is; >2D (e.g. Conv2D) -> flatten trailing dims.
        if len(original_shape) == 1:
            weight_reshaped = ops.reshape(weight, (1, -1))
        elif len(original_shape) > 2:
            weight_reshaped = ops.reshape(weight, (original_shape[0], -1))
        else:
            weight_reshaped = weight

        num_weights = ops.shape(weight_reshaped)[1]
        padding_needed = (self.rf - num_weights % self.rf) % self.rf
        weight_padded = ops.pad(weight_reshaped, [[0, 0], [0, padding_needed]])

        return weight_padded

    def __call__(self, weight):
        prepared_weights = self._prepare_weights(weight)
        dsp_groups = ops.reshape(prepared_weights, (prepared_weights.shape[0], -1, self.rf))

        if self.target_resource == 'DSP':
            return self._calculate_dsp_sparsity(dsp_groups)
        elif self.target_resource == 'BRAM':
            return self._calculate_bram_sparsity(dsp_groups)

    def _calculate_dsp_sparsity(self, dsp_groups):
        """A DSP block is "pruned" if the L2-norm of its weight group is below epsilon."""
        group_norms = ops.sqrt(ops.sum(ops.square(dsp_groups), axis=-1))
        zero_groups = ops.less_equal(group_norms, self.epsilon)
        num_groups = ops.cast(ops.size(group_norms), dsp_groups.dtype)
        # TODO Align with some target
        return ops.sum(ops.cast(zero_groups, dsp_groups.dtype)) / num_groups

    def _calculate_bram_sparsity(self, dsp_groups):
        """A BRAM block is "pruned" if the L2-norm of all weights stored in it is below epsilon."""
        num_dsp_groups = ops.shape(dsp_groups)[1]
        bram_padding = (self.c - num_dsp_groups % self.c) % self.c
        dsp_groups_padded = ops.pad(dsp_groups, [[0, 0], [0, bram_padding], [0, 0]])
        bram_groups = ops.reshape(dsp_groups_padded, (dsp_groups.shape[0], -1, self.c, self.rf))

        bram_group_norms = ops.sqrt(ops.sum(ops.square(bram_groups), axis=(-1, -2)))
        zero_bram_groups = ops.less_equal(bram_group_norms, self.epsilon)
        num_bram_groups = ops.cast(ops.size(bram_group_norms), dsp_groups.dtype)
        return ops.sum(ops.cast(zero_bram_groups, dsp_groups.dtype)) / num_bram_groups


class PACAPatternMetric:
    """Pattern-based pruning metric (PACA).

    Selects a small set of dominant binary patterns over kernel layouts on the
    first call (cached for the metric instance lifetime). Returns the mean
    distance of every kernel to its closest dominant pattern. Operates on 4D
    Conv2D weights only; returns 0 for non-4D inputs.
    """

    def __init__(self, num_patterns_to_keep=16, beta=0.75,
                 epsilon=1e-5, distance_metric='valued_hamming'):
        assert num_patterns_to_keep > 0, "num_patterns_to_keep must be > 0"
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        assert distance_metric in ('hamming', 'valued_hamming', 'cosine'), (
            f"distance_metric must be one of hamming/valued_hamming/cosine, got {distance_metric!r}"
        )
        self.alpha = num_patterns_to_keep
        self.beta = beta
        self.distance_metric = distance_metric
        self.epsilon = epsilon
        self.dominant_patterns = None
        self.projection_mask = None
        self.src = "OIHW"

    def __call__(self, weight):
        if len(weight.shape) != 4:
            return ops.convert_to_tensor(0.0, dtype=weight.dtype)

        if self.dominant_patterns is None:
            _, all_patterns, _ = patterns._get_kernels_and_patterns(weight, self.src, self.epsilon)
            unique_patterns, counts = patterns._get_unique_patterns_with_counts(all_patterns)
            self.dominant_patterns = patterns._select_dominant_patterns(
                all_patterns, unique_patterns, counts,
                alpha=self.alpha, beta=self.beta, dtype=weight.dtype,
            )

        if self.dominant_patterns is None or self.dominant_patterns.shape[0] == 0:
            return ops.convert_to_tensor(0.0, dtype=weight.dtype)

        w_kernels, distances = patterns._pattern_distances(
            weight, self.dominant_patterns,
            self.src, self.epsilon, self.distance_metric,
        )
        min_distances = ops.min(distances, axis=1)
        return ops.mean(min_distances)

    def get_projection_mask(self, weight):
        # If patterns weren't selected (e.g. metric was never invoked, or weight is non-4D),
        # return an identity mask so MDMM.call's `weight * mask` is a no-op.
        if self.dominant_patterns is None or self.dominant_patterns.shape[0] == 0:
            return ops.ones_like(weight)
        if self.projection_mask is None:
            self.projection_mask = patterns._get_projection_mask(
                weight, self.dominant_patterns, self.src, self.epsilon, self.distance_metric,
            )
        return self.projection_mask

    def get_dominant_patterns(self):
        return self.dominant_patterns
