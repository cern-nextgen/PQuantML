import torch


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
