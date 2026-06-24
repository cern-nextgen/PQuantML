"""MDMM core behaviour (Keras layer): metrics, constraints, masking and training phases.

Imports the Keras MDMM implementation but uses backend-agnostic keras.ops, so the same
file runs under both KERAS_BACKEND=tensorflow and KERAS_BACKEND=torch (see run_tests.sh).
Native-torch MDMM is covered separately in test_torch_mdmm.py.
"""

import math

import numpy as np
import pytest
from keras import ops

from pquant.core.keras.pruning_methods.constraint_functions import (
    EqualityConstraint,
    GreaterThanOrEqualConstraint,
    LessThanOrEqualConstraint,
)
from pquant.core.keras.pruning_methods.mdmm import MDMM
from pquant.core.keras.pruning_methods.metric_functions import UnstructuredSparsityMetric

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


# --- UnstructuredSparsityMetric in isolation ---


def test_unstructured_sparsity_coarse_mean_satisfied():
    """weight=[0,0,1,2], target=0.5 -> L0=0.5, factor=0, metric=0."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 1.0, 2.0], dtype="float32")
    assert ops.abs(metric(weight)) < 1e-6


def test_unstructured_sparsity_coarse_mean_unsatisfied():
    """weight=[0,0,0,2], target=0.5 -> L0=0.75, factor=-0.3125, metric=-0.15625."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 0.0, 2.0], dtype="float32")
    assert ops.abs(metric(weight) - (-0.15625)) < 1e-5


def test_unstructured_sparsity_coarse_sum():
    """Same as unsatisfied but scale_mode='sum' -> no division by num_weights."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="sum", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 0.0, 2.0], dtype="float32")
    assert ops.abs(metric(weight) - (-0.625)) < 1e-5


def test_unstructured_sparsity_smooth_at_zeros():
    """weight=[0,0,1,2], target=0.5 -> smooth L0 ~ 0.5, metric ~ 0."""
    metric = UnstructuredSparsityMetric(l0_mode="smooth", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.0, 0.0, 1.0, 2.0], dtype="float32")
    assert ops.abs(metric(weight)) < 1e-4


def test_unstructured_sparsity_smooth_nonzero():
    """weight=[0.1,0.1,1,2], target=0.5 -> smooth L0 ~ 0.184, metric ~ 0.173."""
    metric = UnstructuredSparsityMetric(l0_mode="smooth", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    weight = ops.convert_to_tensor([0.1, 0.1, 1.0, 2.0], dtype="float32")
    smooth_l0 = (2 * math.exp(-1.0)) / 4.0
    expected = (0.5**2 - smooth_l0**2) * 3.2 / 4.0
    assert ops.abs(metric(weight) - expected) < 1e-4


# --- Constraint functions in isolation ---


class ConstantMetric:
    """A metric that always returns a fixed value, to isolate constraint logic."""

    def __init__(self, value):
        self.value = value

    def __call__(self, weight):
        return ops.convert_to_tensor(self.value, dtype="float32")


def test_equality_constraint():
    """metric=0.3, target=0.0 -> inf=0.3, penalty=1*(0.3 + 0.09/2)=0.345."""
    constraint = EqualityConstraint(metric_fn=ConstantMetric(0.3), target_value=0.0, scale=1.0, damping=1.0,
                                    use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    assert ops.abs(constraint(dummy) - 0.345) < 1e-5


def test_leq_constraint_satisfied():
    constraint = LessThanOrEqualConstraint(metric_fn=ConstantMetric(0.3), target_value=0.5, scale=1.0, damping=1.0,
                                           use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    assert ops.abs(constraint(dummy)) < 1e-6


def test_leq_constraint_violated():
    """metric=0.7, target=0.5 -> inf=0.2, penalty=0.22."""
    constraint = LessThanOrEqualConstraint(metric_fn=ConstantMetric(0.7), target_value=0.5, scale=1.0, damping=1.0,
                                           use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    assert ops.abs(constraint(dummy) - 0.22) < 1e-5


def test_geq_constraint_satisfied():
    constraint = GreaterThanOrEqualConstraint(metric_fn=ConstantMetric(0.7), target_value=0.5, scale=1.0, damping=1.0,
                                              use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    assert ops.abs(constraint(dummy)) < 1e-6


def test_geq_constraint_violated():
    """metric=0.3, target=0.5 -> inf=0.2, penalty=0.22."""
    constraint = GreaterThanOrEqualConstraint(metric_fn=ConstantMetric(0.3), target_value=0.5, scale=1.0, damping=1.0,
                                              use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    assert ops.abs(constraint(dummy) - 0.22) < 1e-5


def test_constraint_turn_off():
    """After turn_off(), penalty should be 0."""
    constraint = EqualityConstraint(metric_fn=ConstantMetric(0.5), target_value=0.0, scale=10.0, damping=5.0,
                                    use_grad=False, lr=0.0)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    constraint.turn_off()
    assert ops.abs(constraint(dummy)) < 1e-6


def test_constraint_lambda_update_no_grad():
    """With lr>0, lambda updates between calls, changing the penalty."""
    constraint = EqualityConstraint(metric_fn=ConstantMetric(0.3), target_value=0.0, scale=2.0, damping=1.0,
                                    use_grad=False, lr=0.1)
    dummy = ops.zeros((2, 2))
    constraint.build(dummy.shape)
    result1 = constraint(dummy, training=True)
    result2 = constraint(dummy, training=True)
    assert not ops.isclose(result1, result2)


# --- Mask correctness ---


def test_hard_mask_threshold(config):
    """|w| > epsilon -> 1, else 0 (mask updated during active phase)."""
    weight = ops.convert_to_tensor([[1e-4, 1e-3, 1e-2, 0.1], [-1e-4, -1e-3, -1e-2, -0.1]], dtype="float32")
    expected_mask = ops.convert_to_tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert ops.all(ops.equal(ops.convert_to_tensor(mdmm.mask), expected_mask))


def test_get_layer_sparsity(config):
    """Half above epsilon -> keep ratio 0.5 (also guards the float/int dtype fix)."""
    weight = ops.convert_to_tensor([[1e-4, 1e-3, 1e-2, 0.1], [-1e-4, -1e-3, -1e-2, -0.1]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    assert ops.abs(mdmm.get_layer_sparsity(weight) - 0.5) < 1e-5


# --- MDMM training phases ---


def _make_mixed_weight_linear():
    """~75% zeros (below epsilon), ~25% nonzero, so L0 != target_sparsity=0.5."""
    vals = np.zeros(OUT_FEATURES * IN_FEATURES, dtype=np.float32)
    quarter = len(vals) // 4
    vals[-quarter:] = np.linspace(0.01, 1.0, quarter)
    return ops.convert_to_tensor(vals.reshape(OUT_FEATURES, IN_FEATURES))


def test_pretraining_phase_linear(config):
    """Pretraining: mask stays ones, no penalty, output unchanged."""
    weight = _make_mixed_weight_linear()
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    result = mdmm(weight)
    assert ops.all(ops.equal(result, weight))
    assert ops.all(ops.equal(ops.convert_to_tensor(mdmm.mask), ops.ones(weight.shape)))
    assert mdmm.losses[-1] < 1e-8


def test_active_phase_linear(config):
    """Active: mask updated, penalty nonzero, output unchanged."""
    weight = _make_mixed_weight_linear()
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
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
    mdmm.post_pre_train_function()
    mdmm(weight)
    mdmm.pre_finetune_function()
    result = mdmm(weight)
    expected_output = weight * ops.cast(ops.abs(weight) > epsilon, weight.dtype)
    assert ops.all(ops.equal(result, expected_output))
    assert mdmm.losses[-1] < 1e-8


# --- Penalty numerical verification (full chain) ---


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
    inf_val = 0.15625
    expected = 10.0 * (1.0 * inf_val + 2.0 * inf_val**2 / 2.0)
    assert ops.abs(mdmm.losses[-1] - expected) < 1e-3


# --- Constraint-type variations (full chain) ---


def test_leq_full_chain_satisfied(config):
    """LEQ with metric < 0 (satisfied) -> penalty ~ 0."""
    config["pruning_parameters"]["constraint_type"] = "LessThanOrEqual"
    config["pruning_parameters"]["target_sparsity"] = 0.5
    weight = ops.convert_to_tensor([[0.0, 0.0], [0.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] < 1e-6


def test_geq_full_chain_violated(config):
    """GEQ with metric < 0 (violated) -> nonzero penalty."""
    config["pruning_parameters"]["constraint_type"] = "GreaterThanOrEqual"
    config["pruning_parameters"]["target_sparsity"] = 0.5
    weight = ops.convert_to_tensor([[0.0, 0.0], [0.0, 2.0]], dtype="float32")
    mdmm = MDMM(config, "linear")
    mdmm.build(weight.shape)
    mdmm.post_pre_train_function()
    mdmm(weight)
    assert mdmm.losses[-1] > 1e-4


def test_smooth_differs_from_coarse_in_soft_region(config):
    """For w=0.05, smooth L0 differs from coarse -> different penalties."""
    weight = ops.convert_to_tensor([[0.05, 0.05], [1.0, 2.0]], dtype="float32")

    config["pruning_parameters"]["l0_mode"] = "coarse"
    mdmm_coarse = MDMM(config, "linear")
    mdmm_coarse.build(weight.shape)
    mdmm_coarse.post_pre_train_function()
    mdmm_coarse(weight)

    config["pruning_parameters"]["l0_mode"] = "smooth"
    mdmm_smooth = MDMM(config, "linear")
    mdmm_smooth.build(weight.shape)
    mdmm_smooth.post_pre_train_function()
    mdmm_smooth(weight)

    assert not ops.isclose(mdmm_coarse.losses[-1], mdmm_smooth.losses[-1])


# --- Edge cases ---


def test_all_zero_weights(config):
    """All-zero weights -> L1=0 -> metric=0 -> penalty=0."""
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
