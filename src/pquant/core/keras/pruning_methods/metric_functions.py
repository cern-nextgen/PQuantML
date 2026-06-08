from keras import ops


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
