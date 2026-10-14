"""Hardware-aware MDMM metrics (Keras): FPGAAwareSparsity, PACAPattern + pattern helpers.

Runs under both KERAS_BACKEND=tensorflow and torch (see run_tests.sh). The native-torch
metric implementations are exercised in test_torch_mdmm.py. Conv weights here use the Keras
HWIO kernel layout.
"""

import math
import warnings
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
from keras import ops
from pydantic import ValidationError

from pquant.core.keras.pruning_methods import patterns
from pquant.core.keras.pruning_methods.constraint_functions import EqualityConstraint
from pquant.core.keras.pruning_methods.mdmm import MDMM
from pquant.core.keras.pruning_methods.metric_functions import FPGAAwareSparsityMetric, PACAPatternMetric
from pquant.data_models.pruning_model import ConstraintType, MDMMPruningModel, MetricType


@pytest.fixture
def base_config():
    return {
        "pruning_parameters": {
            "pruning_method": "mdmm",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "constraint_type": "Equality",
            "target_value": 0.0,
            "metric": {"metric_type": "UnstructuredSparsity"},
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
    random_generator = np.random.default_rng(seed)
    return ops.convert_to_tensor(random_generator.standard_normal((kh, kw, c_in, c_out)).astype(np.float32))


def _conv_weight_for_mdmm():
    """HWIO weight with half the (out-channel) kernels zeroed for a non-trivial mask."""
    random_generator = np.random.default_rng(0)
    weight = random_generator.standard_normal((3, 3, 4, 4)).astype(np.float32)
    weight[..., :2] = 0.0  # zero half the output channels
    return ops.convert_to_tensor(weight)


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
    """Config values are validated by Pydantic; direct misuse fails as a registry miss."""
    metric = FPGAAwareSparsityMetric(target_resource="LUT")
    with pytest.raises(KeyError):
        metric(ops.ones((4, 8), dtype="float32"))


# --- PACAPatternMetric ---


def test_paca_returns_zero_for_non_4d():
    metric = PACAPatternMetric()
    weight = ops.convert_to_tensor(np.random.rand(8, 4).astype(np.float32))
    assert ops.abs(metric(weight)) < 1e-6


def test_paca_patterns_follow_current_weights():
    """A previously evaluated metric must recompute patterns for the current weights."""
    evaluated_metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, src="HWIO")
    evaluated_metric(_make_conv_weight())
    sparse_weight = _conv_weight_for_mdmm()
    reference_metric = PACAPatternMetric(num_patterns_to_keep=4, beta=0.75, src="HWIO")
    np.testing.assert_allclose(
        ops.convert_to_numpy(evaluated_metric(sparse_weight)),
        ops.convert_to_numpy(reference_metric(sparse_weight)),
        rtol=1e-6,
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
    """Config values are validated by Pydantic; direct misuse fails as a registry miss."""
    metric = PACAPatternMetric(distance_metric="manhattan")
    with pytest.raises(KeyError):
        metric(_make_conv_weight())


# --- Pattern helpers ---


@pytest.mark.parametrize("backend", ["keras", "torch"])
@pytest.mark.parametrize("distance_metric,expected", [("hamming", 1.0), ("valued_hamming", 4.0), ("cosine", 0.4)])
def test_pattern_distance_dispatch(backend, distance_metric, expected):
    """The shared distance names must resolve to the intended formula in each backend."""
    to_tensor = ops.convert_to_tensor if backend == "keras" else pytest.importorskip("torch").as_tensor
    pattern_module = import_module(f"pquant.core.{backend}.pruning_methods.patterns")
    kernels = to_tensor(np.array([[3.0, 4.0]], dtype=np.float32))
    kernel_patterns = to_tensor(np.array([[1, 1]], dtype=np.uint8))
    dominant_patterns = to_tensor(np.array([[1, 0]], dtype=np.uint8))
    distances = pattern_module._kernel_pattern_distances(kernel_patterns, kernels, dominant_patterns, distance_metric)
    actual = ops.convert_to_numpy(distances) if backend == "keras" else distances.detach().cpu().numpy()
    np.testing.assert_allclose(actual, [[expected]], atol=1e-6)


def test_patterns_kernels_and_patterns():
    """Flatten per-kernel and binarize by epsilon (explicit OIHW input)."""
    weight = ops.convert_to_tensor(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))  # (1,1,2,2)
    kernels, all_patterns, _ = patterns.kernels_and_patterns(weight, src="OIHW", epsilon=0.5)
    assert tuple(ops.shape(kernels)) == (1, 4)
    np.testing.assert_array_equal(ops.convert_to_numpy(all_patterns), np.array([[1, 0, 0, 1]], dtype=np.uint8))


def test_patterns_select_dominant_returns_fixed_size_with_validity():
    """select_dominant_patterns returns (num_patterns_to_keep, K) + a validity mask."""
    # 6 identical-support rows + 2 of another support -> 2 distinct patterns.
    kernel_patterns = ops.convert_to_tensor(np.array([[1, 1, 0, 0]] * 6 + [[0, 0, 1, 1]] * 2, dtype=np.uint8))
    dominant_patterns, valid = patterns.select_dominant_patterns(kernel_patterns, num_patterns_to_keep=4, beta=0.99)
    assert tuple(ops.shape(dominant_patterns)) == (4, 4)
    valid_np = ops.convert_to_numpy(valid)
    assert valid_np[:2].all() and not valid_np[2:].any()  # exactly 2 distinct patterns are valid


# --- MDMM integration through the registry ---


def test_mdmm_fpga_full_phase_cycle(base_config):
    base_config["pruning_parameters"].update(
        {
            "metric": {"metric_type": "FPGAAwareSparsity", "precision": 16, "target_resource": "DSP", "bram_width": 36},
            "rf": 2,
        }
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
        {
            "metric": {
                "metric_type": "PACAPatternSparsity",
                "num_patterns_to_keep": 4,
                "beta": 0.85,
                "distance_metric": "cosine",
            },
            "constraint_type": "GreaterThanOrEqual",
            "target_value": 99.0,
        }
    )
    mdmm = MDMM(base_config, "conv")
    mdmm.build(_conv_weight_for_mdmm().shape)
    assert isinstance(mdmm.constraint_layer, EqualityConstraint)
    assert mdmm.constraint_layer.target_value == 0.0


def test_mdmm_paca_full_phase_cycle(base_config):
    """PACA: metric evaluates on every call; finetuning output = weight * binary mask."""
    base_config["pruning_parameters"].update(
        {
            "metric": {
                "metric_type": "PACAPatternSparsity",
                "num_patterns_to_keep": 4,
                "beta": 0.85,
                "distance_metric": "cosine",
            }
        }
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


@pytest.mark.parametrize(
    "metric_type,expected_parameters",
    [
        (MetricType.UNSTRUCTURED, {}),
        (MetricType.STRUCTURED, {}),
        (MetricType.FPGA_AWARE, {"precision": 16, "target_resource": "DSP", "bram_width": 36}),
        (MetricType.PACA_PATTERN, {"num_patterns_to_keep": 16, "beta": 0.75, "distance_metric": "valued_hamming"}),
    ],
)
def test_pydantic_metric_defaults(metric_type, expected_parameters):
    pruning_parameters = MDMMPruningModel(metric={"metric_type": metric_type.value})
    assert pruning_parameters.metric.metric_type is metric_type
    assert pruning_parameters.metric.model_dump(exclude={"metric_type"}) == expected_parameters


def test_pydantic_default_mdmm_parameters():
    pruning_parameters = MDMMPruningModel()
    assert pruning_parameters.metric.metric_type is MetricType.UNSTRUCTURED
    assert pruning_parameters.constraint_type is ConstraintType.EQUALITY
    assert pruning_parameters.target_value == 0.0
    assert pruning_parameters.target_sparsity == 0.9
    assert pruning_parameters.rf == 1
    assert pruning_parameters.epsilon == 1e-3
    assert pruning_parameters.scale == 10.0
    assert pruning_parameters.damping == 1.0
    assert pruning_parameters.use_grad is False
    assert pruning_parameters.l0_mode == "coarse"
    assert pruning_parameters.scale_mode == "mean"
    assert pruning_parameters.constraint_lr == 1e-3


@pytest.mark.parametrize("backend", ["keras", "torch"])
@pytest.mark.parametrize(
    "metric_config,class_name,expected_kwargs",
    [
        (
            {"metric_type": "UnstructuredSparsity"},
            "UnstructuredSparsityMetric",
            {"epsilon": 0.02, "target_sparsity": 0.65, "l0_mode": "smooth", "scale_mode": "sum"},
        ),
        (
            {"metric_type": "StructuredSparsity"},
            "StructuredSparsityMetric",
            {"epsilon": 0.02, "rf": 3},
        ),
        (
            {"metric_type": "FPGAAwareSparsity", "precision": 8, "target_resource": "BRAM", "bram_width": 18},
            "FPGAAwareSparsityMetric",
            {"epsilon": 0.02, "rf": 3, "l0_mode": "smooth", "precision": 8, "target_resource": "BRAM", "bram_width": 18},
        ),
        (
            {"metric_type": "PACAPatternSparsity", "num_patterns_to_keep": 7, "beta": 0.9, "distance_metric": "hamming"},
            "PACAPatternMetric",
            {"epsilon": 0.02, "num_patterns_to_keep": 7, "beta": 0.9, "distance_metric": "hamming"},
        ),
    ],
)
def test_mdmm_routes_metric_parameters_to_backend(backend, metric_config, class_name, expected_kwargs):
    """Each backend must construct the selected metric with its complete supported parameters."""
    if backend == "torch":
        pytest.importorskip("torch")
    mdmm_module = import_module(f"pquant.core.{backend}.pruning_methods.mdmm")
    metric_module = import_module(f"pquant.core.{backend}.pruning_methods.metric_functions")
    metric_class = getattr(metric_module, class_name)
    config = {
        "pruning_parameters": {
            "pruning_method": "mdmm",
            "metric": metric_config,
            "epsilon": 0.02,
            "target_sparsity": 0.65,
            "rf": 3,
            "l0_mode": "smooth",
            "scale_mode": "sum",
        }
    }
    with patch.object(metric_module, class_name, wraps=metric_class) as constructor:
        mdmm = mdmm_module.MDMM(config, "conv")
        mdmm.build((4, 2, 3, 3))
    constructor.assert_called_once_with(**expected_kwargs)
    assert isinstance(mdmm.constraint_layer.metric_fn, metric_class)


def test_pydantic_bram_packing_validated_at_load():
    """precision > 2*bram_width makes BRAM packing impossible; the sub-model rejects it."""
    with pytest.raises(ValidationError):
        MDMMPruningModel(
            metric={"metric_type": "FPGAAwareSparsity", "precision": 128, "bram_width": 16, "target_resource": "BRAM"}
        )
    # Fine when the resource is DSP: the BRAM fields are unused.
    MDMMPruningModel(
        metric={"metric_type": "FPGAAwareSparsity", "precision": 128, "bram_width": 16, "target_resource": "DSP"}
    )


@pytest.mark.parametrize(
    "metric_config,invalid_field",
    [
        ({"metric_type": "FPGAAwareSparsity", "target_resource": "LUT"}, "target_resource"),
        ({"metric_type": "PACAPatternSparsity", "distance_metric": "manhattan"}, "distance_metric"),
        ({"metric_type": "FPGAAwareSparsity", "precision": 0}, "precision"),
        ({"metric_type": "FPGAAwareSparsity", "bram_width": 0}, "bram_width"),
        ({"metric_type": "PACAPatternSparsity", "num_patterns_to_keep": 0}, "num_patterns_to_keep"),
        ({"metric_type": "PACAPatternSparsity", "beta": 1.5}, "beta"),
        ({"metric_type": "PACAPatternSparsity", "beta": -0.1}, "beta"),
    ],
)
def test_pydantic_rejects_invalid_values(metric_config, invalid_field):
    with pytest.raises(ValidationError) as caught:
        MDMMPruningModel(metric=metric_config)
    assert any(error["loc"][-1] == invalid_field for error in caught.value.errors())


def test_pydantic_rejects_unknown_metric_type():
    with pytest.raises(ValidationError) as caught:
        MDMMPruningModel(metric={"metric_type": "BogusMetric"})
    assert caught.value.errors()[0]["type"] == "union_tag_invalid"


@pytest.mark.parametrize("metric_type", list(MetricType))
@pytest.mark.parametrize("constraint_type", list(ConstraintType))
def test_pydantic_constraint_override_precedence(metric_type, constraint_type):
    """Only PACA overrides the requested constraint, after parsing the nested metric."""
    pruning_parameters = MDMMPruningModel(
        metric={"metric_type": metric_type}, constraint_type=constraint_type, target_value=42.0
    )
    if metric_type is MetricType.PACA_PATTERN:
        assert pruning_parameters.constraint_type is ConstraintType.EQUALITY
        assert pruning_parameters.target_value == 0.0
    else:
        assert pruning_parameters.constraint_type is constraint_type
        assert pruning_parameters.target_value == 42.0


@pytest.mark.parametrize(
    "metric_config",
    [
        None,
        {"metric_type": "StructuredSparsity"},
        {"metric_type": "FPGAAwareSparsity", "precision": 8, "target_resource": "BRAM"},
        {"metric_type": "PACAPatternSparsity", "num_patterns_to_keep": 8},
    ],
)
def test_mdmm_config_serializes_without_warnings(metric_config):
    """Nested configs round-trip without enum or discriminated-union serialization warnings."""

    from pquant.core.hyperparameter_optimization import PQConfig

    pruning_parameters = {"pruning_method": "mdmm"}
    if metric_config is not None:
        pruning_parameters["metric"] = metric_config
    config = PQConfig.load_from_config({"pruning_parameters": pruning_parameters})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        serialized_config = config.model_dump(mode="json")
        serialized_json = config.model_dump_json()
    serialization_warnings = [
        warning
        for warning in caught
        if "Serialization" in str(warning.message) or "serialized value" in str(warning.message)
    ]
    assert not serialization_warnings, [str(warning.message) for warning in serialization_warnings]
    assert "metric_type" not in serialized_config["pruning_parameters"]
    assert PQConfig.model_validate_json(serialized_json).model_dump(mode="json") == serialized_config
