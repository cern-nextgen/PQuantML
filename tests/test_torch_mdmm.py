"""Native-torch MDMM implementation: FPGA/PACA metrics, pattern helpers and phase cycle.

Exercises pquant.core.torch.pruning_methods directly (raw torch.nn.Module, native torch
ops). Run with KERAS_BACKEND=torch (see run_tests.sh). Torch conv kernels are OIHW.
"""

import math

import numpy as np
import pytest
import torch

from pquant.core.torch.pruning_methods import patterns
from pquant.core.torch.pruning_methods.constraint_functions import EqualityConstraint
from pquant.core.torch.pruning_methods.mdmm import MDMM
from pquant.core.torch.pruning_methods.metric_functions import (
    FPGAAwareSparsityMetric,
    PACAPatternMetric,
    UnstructuredSparsityMetric,
)


def base_config():
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
            "constraint_lr": 0.0,
        }
    }


def _conv_weight(c_out=4, c_in=2, kh=3, kw=3, seed=42):
    """Torch conv kernel: OIHW (C_out, C_in, kH, kW)."""
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.standard_normal((c_out, c_in, kh, kw)).astype(np.float32))


# --- Metrics ---


def test_unstructured_sparsity_value():
    """weight=[0,0,0,2], target=0.5 -> L0=0.75, factor=-0.3125, metric=-0.15625."""
    metric = UnstructuredSparsityMetric(l0_mode="coarse", scale_mode="mean", epsilon=1e-3, target_sparsity=0.5)
    assert abs(float(metric(torch.tensor([0.0, 0.0, 0.0, 2.0]))) - (-0.15625)) < 1e-5


@pytest.mark.parametrize(
    "target_resource,fill,expected",
    [("DSP", 0.0, 1.0), ("DSP", 1.0, 0.0), ("BRAM", 0.0, 1.0), ("BRAM", 1.0, 0.0)],
)
def test_fpga_extreme_weights(target_resource, fill, expected):
    metric = FPGAAwareSparsityMetric(rf=2, precision=16, bram_width=36, target_resource=target_resource, epsilon=1e-3)
    assert abs(float(metric(torch.full((4, 16), fill))) - expected) < 1e-5


def test_fpga_dsp_l2_grouping_math():
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    assert abs(float(metric(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))) - 0.5) < 1e-5


@pytest.mark.parametrize("precision,bram_width,expected_c", [(16, 36, 4), (18, 36, 2)])
def test_fpga_calculate_c(precision, bram_width, expected_c):
    assert FPGAAwareSparsityMetric(precision=precision, bram_width=bram_width, target_resource="DSP").c == expected_c


def test_fpga_handles_1d_weight():
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    assert abs(float(metric(torch.ones(8)))) < 1e-5


def test_fpga_invalid_target_resource_raises_on_call():
    with pytest.raises(ValueError):
        FPGAAwareSparsityMetric(target_resource="LUT")(torch.ones(4, 8))


def test_fpga_bram_packing_impossible_raises_on_call():
    with pytest.raises(ValueError):
        FPGAAwareSparsityMetric(precision=128, bram_width=16, target_resource="BRAM")(torch.ones(4, 16))


def test_paca_returns_zero_for_non_4d():
    assert abs(float(PACAPatternMetric()(torch.rand(8, 4)))) < 1e-6


def test_paca_patterns_follow_current_weights():
    """No first-call latch: a used metric agrees with a fresh one on new weights."""
    used = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    _ = used(_conv_weight())
    w2 = _conv_weight(c_out=4, c_in=4)
    fresh = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    assert torch.allclose(used(w2), fresh(w2))


def test_fpga_smooth_mode_produces_weight_gradient():
    """The smooth zero-count surrogate must be trainable; the coarse indicator is not."""
    w = torch.randn(4, 16, requires_grad=True)
    metric = FPGAAwareSparsityMetric(rf=4, l0_mode='smooth')
    metric(w).backward()
    assert w.grad is not None and float(w.grad.abs().sum()) > 0.0


def test_paca_projection_mask_shape_and_binary():
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    w = _conv_weight()
    _ = metric(w)
    mask = metric.get_projection_mask(w)
    assert tuple(mask.shape) == tuple(w.shape)
    assert set(np.unique(mask.detach().cpu().numpy()).tolist()).issubset({0, 1})


@pytest.mark.parametrize("distance_metric", ["hamming", "valued_hamming", "cosine"])
def test_paca_all_distance_metrics(distance_metric):
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, distance_metric=distance_metric)
    assert math.isfinite(float(metric(_conv_weight())))


def test_paca_projection_mask_identity_when_no_patterns():
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    w = _conv_weight()
    np.testing.assert_array_equal(
        metric.get_projection_mask(w).detach().cpu().numpy(), np.ones(tuple(w.shape), dtype=np.float32)
    )


def test_paca_invalid_distance_metric_raises_on_call():
    with pytest.raises(ValueError):
        PACAPatternMetric(distance_metric="manhattan")(_conv_weight())


# --- Pattern helpers ---


def test_patterns_kernels_and_patterns():
    weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1,1,2,2) OIHW
    kernels, pats, _ = patterns.kernels_and_patterns(weight, src="OIHW", epsilon=0.5)
    assert tuple(kernels.shape) == (1, 4)
    np.testing.assert_array_equal(pats.detach().cpu().numpy(), np.array([[1, 0, 0, 1]], dtype=np.uint8))


def test_patterns_select_dominant_fixed_size_with_validity():
    pats = torch.tensor(np.array([[1, 1, 0, 0]] * 6 + [[0, 0, 1, 1]] * 2, dtype=np.uint8))
    dom, valid = patterns.select_dominant_patterns(pats, num_patterns_to_keep=4, beta=0.99)
    assert tuple(dom.shape) == (4, 4)
    valid_np = valid.detach().cpu().numpy()
    assert valid_np[:2].all() and not valid_np[2:].any()


# --- MDMM phase cycle ---


def test_mdmm_get_layer_sparsity():
    weight = torch.tensor([[1e-4, 1e-3, 1e-2, 0.1], [-1e-4, -1e-3, -1e-2, -0.1]])
    mdmm = MDMM(base_config(), "linear")
    mdmm.build(weight.shape)
    assert abs(float(mdmm.get_layer_sparsity(weight)) - 0.5) < 1e-5


def test_mdmm_fpga_phase_cycle():
    cfg = base_config()
    cfg["pruning_parameters"].update(
        {"metric_type": "FPGAAwareSparsity", "precision": 16, "target_resource": "DSP", "rf": 2}
    )
    weight = _conv_weight()
    mdmm = MDMM(cfg, "conv")
    mdmm.build(weight.shape)
    assert torch.equal(mdmm(weight), weight)            # pretraining: unchanged
    assert abs(float(mdmm.calculate_additional_loss())) < 1e-8
    mdmm.post_pre_train_function()
    _ = mdmm(weight)                                    # active
    mdmm.pre_finetune_function()
    _ = mdmm(weight)                                    # finetuning
    assert abs(float(mdmm.calculate_additional_loss())) < 1e-8


def test_mdmm_paca_forces_equality():
    cfg = base_config()
    cfg["pruning_parameters"].update(
        {"metric_type": "PACAPatternSparsity", "constraint_type": "GreaterThanOrEqual", "target_value": 99.0,
         "num_patterns_to_keep": 4, "beta": 0.85, "distance_metric": "cosine"}
    )
    mdmm = MDMM(cfg, "conv")
    mdmm.build(_conv_weight().shape)
    assert isinstance(mdmm.constraint_layer, EqualityConstraint)
    assert mdmm.constraint_layer.target_value == 0.0


def test_mdmm_paca_phase_cycle_projection_mask():
    cfg = base_config()
    cfg["pruning_parameters"].update(
        {"metric_type": "PACAPatternSparsity", "num_patterns_to_keep": 4, "beta": 0.85, "distance_metric": "cosine"}
    )
    weight = _conv_weight(c_out=4, c_in=4)
    mdmm = MDMM(cfg, "conv")
    mdmm.build(weight.shape)
    assert torch.equal(mdmm(weight), weight)            # pretraining
    assert torch.isfinite(mdmm.constraint_layer.metric_fn(weight))
    mdmm.post_pre_train_function()
    _ = mdmm(weight)                                    # active
    mdmm.pre_finetune_function()
    out = mdmm(weight)                                  # finetuning -> weight * binary mask
    out_np, weight_np = out.detach().cpu().numpy(), weight.detach().cpu().numpy()
    assert np.logical_or(np.isclose(out_np, 0.0), np.isclose(out_np, weight_np)).all()
