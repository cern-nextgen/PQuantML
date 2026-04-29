import math

import numpy as np
import pytest
from keras import ops

from pquant.pruning_methods.mdmm import MDMM
from pquant.pruning_methods.metric_functions import (
    FPGAAwareSparsityMetric,
    PACAPatternMetric,
)
from pquant.pruning_methods.utils import patterns


@pytest.fixture
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
            "l0_mode": "coarse",
            "scale_mode": "mean",
            "constraint_lr": 0.0,
        }
    }


def _make_conv_weight(c_out=4, c_in=2, kh=3, kw=3, seed=42):
    rng = np.random.default_rng(seed)
    return ops.convert_to_tensor(rng.standard_normal((c_out, c_in, kh, kw)).astype(np.float32))


# FPGAAwareSparsityMetric


@pytest.mark.parametrize(
    "target_resource,fill,expected",
    [
        ("DSP", 0.0, 1.0),   # all zero -> 100% pruned
        ("DSP", 1.0, 0.0),   # all large -> 0% pruned
        ("BRAM", 0.0, 1.0),
        ("BRAM", 1.0, 0.0),
    ],
)
def test_fpga_extreme_weights(target_resource, fill, expected):
    metric = FPGAAwareSparsityMetric(
        rf=2, precision=16, bram_width=36, target_resource=target_resource, epsilon=1e-3,
    )
    weight = ops.cast(ops.full((4, 16), fill), "float32")
    result = metric(weight)
    assert ops.abs(result - expected) < 1e-5


def test_fpga_dsp_l2_grouping_math():
    """[[1,1,0,0]] with rf=2 -> groups [1,1] (norm sqrt(2)) and [0,0] (norm 0). Result = 1/2."""
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    weight = ops.convert_to_tensor([[1.0, 1.0, 0.0, 0.0]], dtype="float32")
    result = metric(weight)
    assert ops.abs(result - 0.5) < 1e-5


@pytest.mark.parametrize(
    "precision,bram_width,expected_c",
    [
        (16, 36, 4),  # 36 % 16 != 0 -> (2*36)//16 = 4
        (18, 36, 2),  # 36 % 18 == 0 -> 36//18 = 2
    ],
)
def test_fpga_calculate_c(precision, bram_width, expected_c):
    metric = FPGAAwareSparsityMetric(precision=precision, bram_width=bram_width, target_resource="DSP")
    assert metric.c == expected_c


def test_fpga_handles_1d_weight():
    """1D bias vector reshapes to (1, N) and works. Was crashing before fix."""
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    weight = ops.ones((8,), dtype="float32")
    result = metric(weight)
    assert ops.abs(result) < 1e-5


def test_fpga_dtype_propagates_for_float64():
    """float64 weight returns float64 scalar (not hardcoded float32)."""
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    weight = ops.cast(ops.zeros((4, 8)), "float64")
    result = metric(weight)
    assert "float64" in str(result.dtype)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_resource": "LUT"},          # invalid resource
        {"precision": 128, "bram_width": 16},  # c=0 (precision > 2*bram_width)
        {"rf": 0},                            # rf must be >= 1
    ],
)
def test_fpga_invalid_kwargs_raise(kwargs):
    with pytest.raises(AssertionError):
        FPGAAwareSparsityMetric(**kwargs)


# PACAPatternMetric


def test_paca_returns_zero_for_non_4d():
    """Dense (2D) weight -> PACA returns 0.0 without crashing."""
    metric = PACAPatternMetric()
    weight = ops.convert_to_tensor(np.random.rand(8, 4).astype(np.float32))
    result = metric(weight)
    assert ops.abs(result) < 1e-6


def test_paca_dominant_patterns_lazy_caching():
    """dominant_patterns is None until first __call__, then cached across calls."""
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    weight = _make_conv_weight()
    assert metric.dominant_patterns is None

    _ = metric(weight)
    first = metric.dominant_patterns
    assert first is not None

    _ = metric(weight)
    assert metric.dominant_patterns is first  # not recomputed


def test_paca_get_projection_mask_shape_and_binary():
    """Projection mask matches weight shape and contains only 0s and 1s."""
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    weight = _make_conv_weight()
    _ = metric(weight)  # populate dominant_patterns
    mask = metric.get_projection_mask(weight)
    assert tuple(ops.shape(mask)) == tuple(ops.shape(weight))
    mask_np = ops.convert_to_numpy(mask)
    assert set(np.unique(mask_np).tolist()).issubset({0, 1})


@pytest.mark.parametrize("distance_metric", ["hamming", "valued_hamming", "cosine"])
def test_paca_all_distance_metrics(distance_metric):
    """Each supported distance metric produces a finite scalar."""
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, distance_metric=distance_metric)
    weight = _make_conv_weight()
    val = float(ops.convert_to_numpy(metric(weight)))
    assert math.isfinite(val)


def test_paca_get_projection_mask_returns_identity_when_no_patterns():
    """If __call__ was never invoked, get_projection_mask returns ones_like (no-op)."""
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75)
    weight = _make_conv_weight()
    mask = metric.get_projection_mask(weight)
    assert tuple(ops.shape(mask)) == tuple(ops.shape(weight))
    mask_np = ops.convert_to_numpy(mask)
    np.testing.assert_array_equal(mask_np, np.ones_like(mask_np))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_patterns_to_keep": 0},
        {"beta": 1.5},
        {"distance_metric": "manhattan"},
    ],
)
def test_paca_invalid_kwargs_raise(kwargs):
    with pytest.raises(AssertionError):
        PACAPatternMetric(**kwargs)


# Pattern helpers


def test_patterns_helper_kernels_and_patterns():
    """_get_kernels_and_patterns flattens correctly and binarizes by epsilon."""
    weight = ops.convert_to_tensor(
        np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    )  # shape (1, 1, 2, 2)
    kernels, all_patterns, _ = patterns._get_kernels_and_patterns(weight, src="OIHW", epsilon=0.5)
    assert tuple(ops.shape(kernels)) == (1, 4)
    pattern_np = ops.convert_to_numpy(all_patterns)
    np.testing.assert_array_equal(pattern_np, np.array([[1, 0, 0, 1]], dtype=np.uint8))


# MDMM integration through the registry


def _conv_weight_for_mdmm():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((4, 4, 3, 3)).astype(np.float32)
    arr[:2] = 0.0  # zero half the kernels for a non-trivial mask
    return ops.convert_to_tensor(arr)


def test_mdmm_fpga_full_phase_cycle(base_config):
    """MDMM with FPGAAwareSparsity runs through pretraining/active/finetuning."""
    base_config["pruning_parameters"]["metric_type"] = "FPGAAwareSparsity"
    base_config["pruning_parameters"]["precision"] = 16
    base_config["pruning_parameters"]["target_resource"] = "DSP"
    base_config["pruning_parameters"]["bram_width"] = 36
    base_config["pruning_parameters"]["rf"] = 2

    weight = _conv_weight_for_mdmm()
    mdmm = MDMM(base_config, "conv")
    mdmm.build(weight.shape)

    out = mdmm(weight)  # pretraining
    assert ops.all(ops.equal(out, weight))
    assert mdmm.losses[-1] < 1e-8

    mdmm.post_pre_train_function()
    _ = mdmm(weight)  # active

    mdmm.pre_finetune_function()
    _ = mdmm(weight)  # finetuning
    assert mdmm.losses[-1] < 1e-8


def test_mdmm_paca_forces_equality_constraint(base_config):
    """PACA always pairs with EqualityConstraint(target=0), even if config says GEQ."""
    base_config["pruning_parameters"]["metric_type"] = "PACAPatternSparsity"
    base_config["pruning_parameters"]["constraint_type"] = "GreaterThanOrEqual"  # ignored
    base_config["pruning_parameters"]["target_value"] = 99.0  # ignored
    base_config["pruning_parameters"]["num_patterns_to_keep"] = 4
    base_config["pruning_parameters"]["beta"] = 0.85
    base_config["pruning_parameters"]["distance_metric"] = "cosine"

    weight = _conv_weight_for_mdmm()
    mdmm = MDMM(base_config, "conv")
    mdmm.build(weight.shape)

    from pquant.pruning_methods.constraint_functions import EqualityConstraint
    assert isinstance(mdmm.constraint_layer, EqualityConstraint)
    assert mdmm.constraint_layer.target_value == 0.0


def test_mdmm_paca_full_phase_cycle(base_config):
    """MDMM with PACA: dominant_patterns set in pretraining, finetuning uses projection mask."""
    base_config["pruning_parameters"]["metric_type"] = "PACAPatternSparsity"
    base_config["pruning_parameters"]["num_patterns_to_keep"] = 4
    base_config["pruning_parameters"]["beta"] = 0.85
    base_config["pruning_parameters"]["distance_metric"] = "cosine"

    weight = _conv_weight_for_mdmm()
    mdmm = MDMM(base_config, "conv")
    mdmm.build(weight.shape)

    out = mdmm(weight)  # pretraining
    assert ops.all(ops.equal(out, weight))
    assert mdmm.constraint_layer.metric_fn.dominant_patterns is not None

    mdmm.post_pre_train_function()
    _ = mdmm(weight)  # active

    mdmm.pre_finetune_function()
    out = mdmm(weight)  # finetuning -> output is weight * binary_mask
    out_np = ops.convert_to_numpy(out)
    weight_np = ops.convert_to_numpy(weight)
    is_zero_or_equal = np.logical_or(np.isclose(out_np, 0.0), np.isclose(out_np, weight_np))
    assert is_zero_or_equal.all()


# Pydantic schema validation


def test_pydantic_accepts_new_metric_types():
    """Both new metric_types validate with their associated parameters."""
    from pquant.data_models.pruning_model import MDMMPruningModel
    fpga = MDMMPruningModel(
        metric_type="FPGAAwareSparsity",
        precision=16,
        target_resource="DSP",
        bram_width=36,
    )
    assert fpga.metric_type == "FPGAAwareSparsity"
    assert fpga.precision == 16

    paca = MDMMPruningModel(
        metric_type="PACAPatternSparsity",
        num_patterns_to_keep=8,
        beta=0.9,
        distance_metric="hamming",
    )
    assert paca.metric_type == "PACAPatternSparsity"
    assert paca.beta == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric_type": "BogusMetric"},
        {"target_resource": "LUT"},
        {"distance_metric": "manhattan"},
    ],
)
def test_pydantic_rejects_invalid_values(kwargs):
    from pydantic import ValidationError
    from pquant.data_models.pruning_model import MDMMPruningModel
    with pytest.raises(ValidationError):
        MDMMPruningModel(**kwargs)
