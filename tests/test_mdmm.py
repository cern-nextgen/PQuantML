import math

import numpy as np
import pytest
from keras import ops

from pquant.pruning_methods.constraint_functions import (
    EqualityConstraint,
    GreaterThanOrEqualConstraint,
    LessThanOrEqualConstraint,
)
from pquant.pruning_methods.mdmm import MDMM
from pquant.pruning_methods.metric_functions import (
    # StructuredSparsityMetric,
    UnstructuredSparsityMetric,
    # TODO: add PACA and FPGA metric functions after integrating them
)

IN_FEATURES = 8
OUT_FEATURES = 16


@pytest.fixture
def config():
    return {
        "pruning_parameters": {
            "pruning_method": "mdmm",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "constraint_type": "Equality",
            "target_value": 0.0,
            "metric_type": "UnstructuredSparsity",
            "target_sparsity": 0.5,
            "rf": 1,
            "epsilon": 1e-3,
            "scale": 1.0,
            "damping": 1.0,
            "use_grad": False,
            "l0_mode": "coarse",
            "scale_mode": "mean",
            "constraint_lr": 0.0,
        }
    }


# Metric functions in isolation


def test_unstructured_sparsity_coarse_mean_satisfied():
    """weight=[0,0,1,2], target=0.5 -> L0=0.5, factor=0, metric=0."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 1.0, 2.0], dtype="float32")
    result = metric(weight)
    assert ops.abs(result) < 1e-6


def test_unstructured_sparsity_coarse_mean_unsatisfied():
    """weight=[0,0,0,2], target=0.5 -> L0=0.75, factor=-0.3125, metric=-0.15625."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 0.0, 2.0], dtype="float32")
    result = metric(weight)
    # L0 = mean([T,T,T,F]) = 0.75
    # L1 = 2.0
    # factor = 0.25 - 0.5625 = -0.3125
    # fn = -0.3125 * 2.0 / 4 = -0.15625
    expected = -0.15625
    assert ops.abs(result - expected) < 1e-5


def test_unstructured_sparsity_coarse_sum():
    """Same as unsatisfied but scale_mode='sum' -> no division by num_weights."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="sum", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 0.0, 2.0], dtype="float32")
    result = metric(weight)
    # fn = -0.3125 * 2.0 = -0.625
    expected = -0.625
    assert ops.abs(result - expected) < 1e-5


def test_unstructured_sparsity_smooth_at_zeros():
    """weight=[0,0,1,2], target=0.5 -> smooth L0 ~ 0.5, metric ~ 0."""
    metric = UnstructuredSparsityMetric(l0_mode="smooth", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 1.0, 2.0], dtype="float32")
    result = metric(weight)
    # exp(-100*0)=1, exp(-100*1)~0, exp(-100*4)~0 -> smooth_L0 ~ 0.5
    assert ops.abs(result) < 1e-4


def test_unstructured_sparsity_smooth_nonzero():
    """weight=[0.1,0.1,1,2], target=0.5 -> smooth L0 ~ 0.184, metric ~ 0.173."""
    metric = UnstructuredSparsityMetric(l0_mode="smooth", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.1, 0.1, 1.0, 2.0], dtype="float32")
    result = metric(weight)
    # exp(-100*0.01)=exp(-1)=0.3679, exp(-100*1)~0, exp(-100*4)~0
    # smooth_L0 = mean([0.3679, 0.3679, ~0, ~0]) = 0.18395
    # factor = 0.25 - 0.18395^2 = 0.25 - 0.03384 = 0.21616
    # L1 = 0.1+0.1+1.0+2.0 = 3.2
    # fn = 0.21616 * 3.2 / 4 = 0.17293
    smooth_l0 = (2 * math.exp(-1.0)) / 4.0
    factor = 0.5**2 - smooth_l0**2
    l1 = 3.2
    expected = factor * l1 / 4.0
    assert ops.abs(result - expected) < 1e-4


# Constraint functions in isolation


class ConstantMetric:
    """A simple metric that always returns a fixed value, for isolating constraint logic."""

    def __init__(self, value):
        self.value = value

    def __call__(self, weight):
        return ops.convert_to_tensor(self.value, dtype="float32")


def test_equality_constraint():
    """metric=0.3, target=0.0 -> inf=0.3, penalty=1*(1*0.3 + 1*0.09/2) = 0.345."""
    metric_fn = ConstantMetric(0.3)
    constraint = EqualityConstraint(metric_fn=metric_fn, target_value=0.0, scale=1.0, damping=1.0, use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    result = constraint(dummy)
    # inf = |0.3 - 0.0| = 0.3
    # lmbda_step = 0 (lr=0), ascent_lmbda = 1.0
    # l_term = 1.0 * 0.3, damp_term = 1.0 * 0.09 / 2 = 0.045
    # penalty = 1.0 * (0.3 + 0.045) = 0.345
    expected = 0.345
    assert ops.abs(result - expected) < 1e-5


def test_leq_constraint_satisfied():
    """metric=0.3, target=0.5 -> inf=max(-0.2, 0)=0, penalty=0."""
    metric_fn = ConstantMetric(0.3)
    constraint = LessThanOrEqualConstraint(metric_fn=metric_fn, target_value=0.5, scale=1.0, damping=1.0, use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    result = constraint(dummy)
    assert ops.abs(result) < 1e-6


def test_leq_constraint_violated():
    """metric=0.7, target=0.5 -> inf=0.2, penalty=0.22."""
    metric_fn = ConstantMetric(0.7)
    constraint = LessThanOrEqualConstraint(metric_fn=metric_fn, target_value=0.5, scale=1.0, damping=1.0, use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    result = constraint(dummy)
    # inf = max(0.7 - 0.5, 0) = 0.2
    # l_term = 1.0 * 0.2, damp_term = 1.0 * 0.04 / 2 = 0.02
    # penalty = 1.0 * (0.2 + 0.02) = 0.22
    expected = 0.22
    assert ops.abs(result - expected) < 1e-5


def test_geq_constraint_satisfied():
    """metric=0.7, target=0.5 -> inf=max(-0.2, 0)=0, penalty=0."""
    metric_fn = ConstantMetric(0.7)
    constraint = GreaterThanOrEqualConstraint(metric_fn=metric_fn, target_value=0.5, scale=1.0, damping=1.0, use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    result = constraint(dummy)
    assert ops.abs(result) < 1e-6


def test_geq_constraint_violated():
    """metric=0.3, target=0.5 -> inf=0.2, penalty=0.22."""
    metric_fn = ConstantMetric(0.3)
    constraint = GreaterThanOrEqualConstraint(metric_fn=metric_fn, target_value=0.5, scale=1.0, damping=1.0, use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    result = constraint(dummy)
    expected = 0.22
    assert ops.abs(result - expected) < 1e-5


def test_constraint_turn_off():
    """After turn_off(), penalty should be 0."""
    metric_fn = ConstantMetric(0.5)
    constraint = EqualityConstraint(metric_fn=metric_fn, target_value=0.0, scale=10.0, damping=5.0, use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    constraint.turn_off()
    result = constraint(dummy)
    assert ops.abs(result) < 1e-6


def test_constraint_lambda_update_no_grad():
    """With lr>0, lambda should update between calls, changing the penalty."""
    metric_fn = ConstantMetric(0.3)
    constraint = EqualityConstraint(metric_fn=metric_fn, target_value=0.0, scale=2.0, damping=1.0, use_grad=False, lr=0.1)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)

    # First call (training=True): prev_infs=0, so lmbda_step=0, ascent_lmbda=1.0
    result1 = constraint(dummy, training=True)
    # After first call: prev_infs = 0.3, lmbda += 0 (still 1.0)

    # Second call (training=True): lmbda_step = 0.1 * 2.0 * 0.3 = 0.06
    # ascent_lmbda = 1.0 + 0.06 = 1.06
    result2 = constraint(dummy, training=True)

    # Penalty should differ between calls due to lambda update
    assert not ops.isclose(result1, result2)


# Mask correctness


def test_hard_mask_threshold(config):
    """Verify mask at epsilon boundary: |w| > epsilon -> 1, else 0."""
    weight = ops.convert_to_tensor(
        [[1e-4, 1e-3, 1e-2, 0.1],
         [-1e-4, -1e-3, -1e-2, -0.1]],
        dtype="float32",
    )
    expected_mask = ops.convert_to_tensor(
        [[0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]],
        dtype="float32",
    )
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()  # transition to active phase
    mdmm(weight)
    assert ops.all(ops.equal(ops.convert_to_tensor(mdmm.mask), expected_mask))



def test_get_layer_sparsity(config):
    """Half zero, half nonzero -> sparsity = 0.5."""
    weight = ops.convert_to_tensor(
        [[1e-4, 1e-3, 1e-2, 0.1],
         [-1e-4, -1e-3, -1e-2, -0.1]],
        dtype="float32",
    )
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    # 4 out of 8 above epsilon -> keep ratio = 0.5
    sparsity = mdmm.get_layer_sparsity(weight)
    assert ops.abs(sparsity - 0.5) < 1e-5


# MDMM training phases


def _make_mixed_weight_linear():
    """Weight with ~75% zeros (below epsilon) and ~25% nonzero.
    With target_sparsity=0.5, this ensures L0 != target -> nonzero penalty."""
    vals = np.zeros(OUT_FEATURES * IN_FEATURES, dtype=np.float32)
    quarter = len(vals) // 4
    vals[-quarter:] = np.linspace(0.01, 1.0, quarter)
    return ops.convert_to_tensor(vals.reshape(OUT_FEATURES, IN_FEATURES))



def test_pretraining_phase_linear(config):
    """Pretraining: mask stays ones, no penalty, output unchanged."""
    weight = _make_mixed_weight_linear()
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    # Still in pretraining phase (default)
    result = mdmm(weight)

    assert ops.all(ops.equal(result, weight))
    assert ops.all(ops.equal(ops.convert_to_tensor(mdmm.mask), ops.ones(weight.shape)))
    assert mdmm.losses[-1] < 1e-8


def test_active_phase_linear(config):
    """Active: mask updated, penalty nonzero, output unchanged."""
    weight = _make_mixed_weight_linear()
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()  # -> active

    result = mdmm(weight)
    expected_mask = ops.cast(ops.abs(weight) > config["pruning_parameters"]["epsilon"], weight.dtype)

    assert ops.all(ops.equal(result, weight))
    assert ops.all(ops.equal(ops.convert_to_tensor(mdmm.mask), expected_mask))
    assert mdmm.losses[-1] > 1e-8


def test_finetuning_phase_linear(config):
    """Finetuning: mask frozen, no penalty, output = weight * hard_mask."""
    weight = _make_mixed_weight_linear()
    epsilon = config["pruning_parameters"]["epsilon"]
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()  # -> active
    mdmm(weight)  # update mask
    mdmm.pre_finetune_function()  # -> finetuning

    result = mdmm(weight)
    hard_mask = ops.cast(ops.abs(weight) > epsilon, weight.dtype)
    expected_output = weight * hard_mask

    assert ops.all(ops.equal(result, expected_output))
    assert mdmm.losses[-1] < 1e-8



# Penalty numerical verification (full chain)


def test_penalty_satisfied(config):
    """weight=[0,0,1,2] at target=0.5 -> metric=0 -> penalty=0."""
    config["pruning_parameters"]["target_sparsity"] = 0.5
    weight = ops.convert_to_tensor([[0.0, 0.0], [1.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] < 1e-6


def test_penalty_unsatisfied(config):
    """weight=[0,0,0,2] at target=0.5 -> known penalty value."""
    config["pruning_parameters"]["target_sparsity"] = 0.5
    weight = ops.convert_to_tensor([[0.0, 0.0], [0.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)

    # metric = -0.15625 (from Group A)
    # EqualityConstraint: inf = |-0.15625| = 0.15625
    # penalty_per_element = 1.0 * (1.0 * 0.15625 + 1.0 * 0.15625^2 / 2) = 0.16846
    # MDMM does ops.sum(constraint(weight)) -> single scalar, so penalty = 0.16846
    expected = 0.15625 + 0.15625**2 / 2.0
    assert ops.abs(mdmm.losses[-1] - expected) < 1e-4


def test_penalty_with_scale_and_damping(config):
    """Same weight but scale=10, damping=2 -> larger penalty."""
    config["pruning_parameters"]["target_sparsity"] = 0.5
    config["pruning_parameters"]["scale"] = 10.0
    config["pruning_parameters"]["damping"] = 2.0
    weight = ops.convert_to_tensor([[0.0, 0.0], [0.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)

    # inf = 0.15625
    # l_term = 1.0 * 0.15625 (lmbda=1.0)
    # damp_term = 2.0 * 0.15625^2 / 2 = 0.02441
    # penalty = 10.0 * (0.15625 + 0.02441) = 1.80664
    inf_val = 0.15625
    expected = 10.0 * (1.0 * inf_val + 2.0 * inf_val**2 / 2.0)
    assert ops.abs(mdmm.losses[-1] - expected) < 1e-3


# Constraint type variations (full chain)


def test_leq_full_chain_satisfied(config):
    """LEQ constraint with metric < 0 (satisfied) -> penalty ~ 0."""
    config["pruning_parameters"]["constraint_type"] = "LessThanOrEqual"
    config["pruning_parameters"]["target_sparsity"] = 0.5
    # weight=[0,0,0,2]: metric = -0.15625. LEQ target_value=0 -> inf = max(-0.15625, 0) = 0
    weight = ops.convert_to_tensor([[0.0, 0.0], [0.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] < 1e-6


def test_geq_full_chain_violated(config):
    """GEQ constraint with metric < 0 (violated) -> nonzero penalty."""
    config["pruning_parameters"]["constraint_type"] = "GreaterThanOrEqual"
    config["pruning_parameters"]["target_sparsity"] = 0.5
    # metric = -0.15625, GEQ target=0 -> inf = max(0 - (-0.15625), 0) = 0.15625
    weight = ops.convert_to_tensor([[0.0, 0.0], [0.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] > 1e-4


# L0 mode variations



def test_smooth_differs_from_coarse_in_soft_region(config):
    """For w=0.05, smooth L0 differs from coarse -> different penalties."""
    weight = ops.convert_to_tensor([[0.05, 0.05], [1.0, 2.0]], dtype="float32")

    config["pruning_parameters"]["l0_mode"] = "coarse"
    mdmm_coarse = MDMM(config, "linear")
    mdmm_coarse.build(weight.shape)
    mdmm_coarse.post_pre_train_function()
    mdmm_coarse(weight)
    loss_coarse = mdmm_coarse.losses[-1]

    config["pruning_parameters"]["l0_mode"] = "smooth"
    mdmm_smooth = MDMM(config, "linear")
    mdmm_smooth.build(weight.shape)
    mdmm_smooth.post_pre_train_function()
    mdmm_smooth(weight)
    loss_smooth = mdmm_smooth.losses[-1]

    # Coarse: |0.05| > 1e-3 so L0=0 (no zeros). Smooth: exp(-100*0.0025)=exp(-0.25)~0.78
    # These produce different L0 values -> different penalties
    assert not ops.isclose(loss_coarse, loss_smooth)



# Edge cases


def test_all_zero_weights(config):
    """All-zero weights -> L1=0, so metric=factor*0=0 -> penalty=0."""
    weight = ops.zeros((4, 4))
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] < 1e-6


def test_all_large_weights(config):
    """No weights below epsilon -> L0=0 -> factor=target^2 > 0 -> nonzero penalty."""
    weight = ops.ones((4, 4))
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] > 1e-4


