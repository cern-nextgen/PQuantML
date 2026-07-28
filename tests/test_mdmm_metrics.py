"""Hardware-aware MDMM metrics (Keras): FPGAAwareSparsity, PACAPattern + pattern helpers.

Runs under both KERAS_BACKEND=tensorflow and torch (see run_tests.sh). The native-torch
metric implementations are exercised in test_torch_mdmm.py. Conv weights here use the Keras
HWIO kernel layout.
"""

import math

import numpy as np
import pytest
from keras import ops
from pydantic import ValidationError

from pquant.core.keras.pruning_methods import patterns
from pquant.core.keras.pruning_methods.constraint_functions import EqualityConstraint
from pquant.core.keras.pruning_methods.mdmm import MDMM
from pquant.core.keras.pruning_methods.metric_functions import FPGAAwareSparsityMetric, PACAPatternMetric
from pquant.data_models.pruning_model import MDMMPruningModel


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
            "constraint_lr": 0.0,
        }
    }


def _make_conv_weight(kh=3, kw=3, c_in=2, c_out=4, seed=42):
    """Keras conv kernel: HWIO layout (kH, kW, C_in, C_out)."""
    rng = np.random.default_rng(seed)
    return ops.convert_to_tensor(rng.standard_normal((kh, kw, c_in, c_out)).astype(np.float32))


def _conv_weight_for_mdmm():
    """HWIO weight with half the (out-channel) kernels zeroed for a non-trivial mask."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 3, 4, 4)).astype(np.float32)
    arr[..., :2] = 0.0  # zero half the output channels
    return ops.convert_to_tensor(arr)


# --- FPGAAwareSparsityMetric ---


@pytest.mark.parametrize(
    "target_resource,fill,expected",
    [("DSP", 0.0, 1.0), ("DSP", 1.0, 0.0), ("BRAM", 0.0, 1.0), ("BRAM", 1.0, 0.0)],
)
def test_fpga_extreme_weights(target_resource, fill, expected):
    metric = FPGAAwareSparsityMetric(rf=2, precision=16, bram_width=36, target_resource=target_resource, epsilon=1e-3)
    weight = ops.cast(ops.full((4, 16), fill), "float32")
    assert ops.abs(metric(weight) - expected) < 1e-5


def test_fpga_dsp_l2_grouping_math():
    """[[1,1,0,0]] with rf=2 -> groups [1,1] (norm sqrt 2) and [0,0] (norm 0). Result = 0.5."""
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    weight = ops.convert_to_tensor([[1.0, 1.0, 0.0, 0.0]], dtype="float32")
    assert ops.abs(metric(weight) - 0.5) < 1e-5


@pytest.mark.parametrize("precision,bram_width,expected_c", [(16, 36, 4), (18, 36, 2)])
def test_fpga_calculate_c(precision, bram_width, expected_c):
    metric = FPGAAwareSparsityMetric(precision=precision, bram_width=bram_width, target_resource="DSP")
    assert metric.c == expected_c


def test_fpga_handles_1d_weight():
    """1D bias vector reshapes to (1, N) and works without crashing."""
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    assert ops.abs(metric(ops.ones((8,), dtype="float32"))) < 1e-5


def test_fpga_dtype_propagates_for_float64():
    """float64 weight returns a float64 scalar (not hardcoded float32)."""
    metric = FPGAAwareSparsityMetric(rf=2, target_resource="DSP", epsilon=1e-3)
    result = metric(ops.cast(ops.zeros((4, 8)), "float64"))
    assert "float64" in str(result.dtype)


def test_fpga_invalid_target_resource_raises_on_call():
    """Validation now lives in Pydantic, so a bad target_resource surfaces at call time."""
    metric = FPGAAwareSparsityMetric(target_resource="LUT")
    with pytest.raises(ValueError):
        metric(ops.ones((4, 8), dtype="float32"))


def test_fpga_bram_packing_impossible_raises_on_call():
    """precision > 2*bram_width -> c=0 -> clear error when BRAM sparsity is computed."""
    metric = FPGAAwareSparsityMetric(precision=128, bram_width=16, target_resource="BRAM")
    with pytest.raises(ValueError):
        metric(ops.ones((4, 16), dtype="float32"))


# --- PACAPatternMetric ---


def test_paca_returns_zero_for_non_4d():
    metric = PACAPatternMetric()
    weight = ops.convert_to_tensor(np.random.rand(8, 4).astype(np.float32))
    assert ops.abs(metric(weight)) < 1e-6


def test_paca_patterns_follow_current_weights():
    """No first-call latch: a used metric agrees with a fresh one on new weights."""
    used = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, src="HWIO")
    _ = used(_make_conv_weight())
    sparse = _conv_weight_for_mdmm()
    fresh = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, src="HWIO")
    np.testing.assert_allclose(
        ops.convert_to_numpy(used(sparse)), ops.convert_to_numpy(fresh(sparse)), rtol=1e-6
    )


def test_paca_get_projection_mask_shape_and_binary():
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, src="HWIO")
    weight = _make_conv_weight()
    _ = metric(weight)
    mask = metric.get_projection_mask(weight)
    assert tuple(ops.shape(mask)) == tuple(ops.shape(weight))
    assert set(np.unique(ops.convert_to_numpy(mask)).tolist()).issubset({0, 1})


@pytest.mark.parametrize("distance_metric", ["hamming", "valued_hamming", "cosine"])
def test_paca_all_distance_metrics(distance_metric):
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, distance_metric=distance_metric, src="HWIO")
    assert math.isfinite(float(ops.convert_to_numpy(metric(_make_conv_weight()))))


def test_paca_projection_mask_identity_on_dense_weight():
    """A fully dense weight has the all-ones support as its only pattern -> identity mask."""
    metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, src="HWIO")
    weight = _make_conv_weight()
    mask = metric.get_projection_mask(weight)
    np.testing.assert_array_equal(ops.convert_to_numpy(mask), np.ones_like(ops.convert_to_numpy(mask)))


def test_paca_invalid_distance_metric_raises_on_call():
    metric = PACAPatternMetric(distance_metric="manhattan")
    with pytest.raises(ValueError):
        metric(_make_conv_weight())


# --- Pattern helpers ---


def test_patterns_kernels_and_patterns():
    """Flatten per-kernel and binarize by epsilon (explicit OIHW input)."""
    weight = ops.convert_to_tensor(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))  # (1,1,2,2)
    kernels, all_patterns, _ = patterns.kernels_and_patterns(weight, src="OIHW", epsilon=0.5)
    assert tuple(ops.shape(kernels)) == (1, 4)
    np.testing.assert_array_equal(ops.convert_to_numpy(all_patterns), np.array([[1, 0, 0, 1]], dtype=np.uint8))


def test_patterns_select_dominant_returns_fixed_size_with_validity():
    """select_dominant_patterns returns (num_patterns_to_keep, K) + a validity mask."""
    # 6 identical-support rows + 2 of another support -> 2 distinct patterns.
    pats = ops.convert_to_tensor(
        np.array([[1, 1, 0, 0]] * 6 + [[0, 0, 1, 1]] * 2, dtype=np.uint8)
    )
    dom, valid = patterns.select_dominant_patterns(pats, num_patterns_to_keep=4, beta=0.99)
    assert tuple(ops.shape(dom)) == (4, 4)
    valid_np = ops.convert_to_numpy(valid)
    assert valid_np[:2].all() and not valid_np[2:].any()  # exactly 2 distinct patterns are valid


# --- MDMM integration through the registry ---


def test_mdmm_fpga_full_phase_cycle(base_config):
    base_config["pruning_parameters"].update(
        {"metric_type": "FPGAAwareSparsity", "precision": 16, "target_resource": "DSP", "bram_width": 36, "rf": 2}
    )
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
    """PACA always pairs with EqualityConstraint(target=0), even if config asks for GEQ/99."""
    base_config["pruning_parameters"].update(
        {"metric_type": "PACAPatternSparsity", "constraint_type": "GreaterThanOrEqual", "target_value": 99.0,
         "num_patterns_to_keep": 4, "beta": 0.85, "distance_metric": "cosine"}
    )
    mdmm = MDMM(base_config, "conv")
    mdmm.build(_conv_weight_for_mdmm().shape)
    assert isinstance(mdmm.constraint_layer, EqualityConstraint)
    assert mdmm.constraint_layer.target_value == 0.0


def test_mdmm_paca_full_phase_cycle(base_config):
    """PACA: metric evaluates on every call; finetuning output = weight * binary mask."""
    base_config["pruning_parameters"].update(
        {"metric_type": "PACAPatternSparsity", "num_patterns_to_keep": 4, "beta": 0.85, "distance_metric": "cosine"}
    )
    weight = _conv_weight_for_mdmm()
    mdmm = MDMM(base_config, "conv")
    mdmm.build(weight.shape)

    out = mdmm(weight)  # pretraining (constraint still evaluates the metric)
    assert ops.all(ops.equal(out, weight))
    metric_value = mdmm.constraint_layer.metric_fn(weight)
    assert math.isfinite(float(ops.convert_to_numpy(metric_value)))

    mdmm.post_pre_train_function()
    _ = mdmm(weight)  # active
    mdmm.pre_finetune_function()
    out = mdmm(weight)  # finetuning
    out_np, weight_np = ops.convert_to_numpy(out), ops.convert_to_numpy(weight)
    assert np.logical_or(np.isclose(out_np, 0.0), np.isclose(out_np, weight_np)).all()


# --- Pydantic schema validation (backend-agnostic) ---


def test_pydantic_accepts_new_metric_types():
    """The legacy flat layout is lifted into the nested metric block."""
    fpga = MDMMPruningModel(metric_type="FPGAAwareSparsity", precision=16, target_resource="DSP", bram_width=36)
    assert fpga.metric_type.value == "FPGAAwareSparsity" and fpga.metric.precision == 16
    paca = MDMMPruningModel(metric_type="PACAPatternSparsity", num_patterns_to_keep=8, beta=0.9, distance_metric="hamming")
    assert paca.metric_type.value == "PACAPatternSparsity" and paca.metric.beta == 0.9


def test_pydantic_nested_metric_layout():
    """The nested layout is the canonical one; per-metric params live in the sub-model."""
    m = MDMMPruningModel(metric={"metric_type": "FPGAAwareSparsity", "precision": 8, "target_resource": "BRAM"})
    assert m.metric.precision == 8 and m.metric.bram_width == 36
    assert m.metric_type.value == "FPGAAwareSparsity"
    default = MDMMPruningModel()
    assert default.metric_type.value == "UnstructuredSparsity"


def test_pydantic_bram_packing_validated_at_load():
    """precision > 2*bram_width makes BRAM packing impossible; the sub-model rejects it."""
    with pytest.raises(ValidationError):
        MDMMPruningModel(metric={"metric_type": "FPGAAwareSparsity", "precision": 128, "bram_width": 16, "target_resource": "BRAM"})
    # Fine when the resource is DSP: the BRAM fields are unused.
    MDMMPruningModel(metric={"metric_type": "FPGAAwareSparsity", "precision": 128, "bram_width": 16, "target_resource": "DSP"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric_type": "BogusMetric"},
        {"target_resource": "LUT"},
        {"distance_metric": "manhattan"},
        {"precision": 0},
        {"bram_width": 0},
        {"num_patterns_to_keep": 0},
        {"beta": 1.5},
        {"beta": -0.1},
    ],
)
def test_pydantic_rejects_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        MDMMPruningModel(**kwargs)


def test_pydantic_paca_forces_equality_at_model_level():
    """The model_validator coerces PACA to Equality/0 regardless of requested constraint."""
    m = MDMMPruningModel(metric_type="PACAPatternSparsity", constraint_type="GreaterThanOrEqual", target_value=42.0)
    assert m.constraint_type.value == "Equality" and m.target_value == 0.0


def test_mdmm_config_serializes_without_warnings():
    """Regression for the reported bug: MDMM configs must round-trip through model_dump
    with no PydanticSerializationUnexpectedValue (constraint_type enum / discriminated union)."""
    import warnings

    from pquant.core.hyperparameter_optimization import PQConfig

    cases = [
        {},
        {"metric_type": "FPGAAwareSparsity", "precision": 8, "target_resource": "DSP"},
        {"metric_type": "PACAPatternSparsity", "num_patterns_to_keep": 8},
    ]
    for pp in cases:
        cfg = PQConfig.load_from_config({"pruning_parameters": {"pruning_method": "mdmm", **pp}})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg.model_dump(mode="json")
            cfg.model_dump_json()
        serial = [w for w in caught if "Serialization" in str(w.message) or "serialized value" in str(w.message)]
        assert not serial, f"serialization warnings for {pp}: {[str(w.message) for w in serial]}"
