import os

import pytest
import torch
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant.activations import PQActivation  # noqa: E402
from pquant.core.hyperparameter_optimization import PQConfig  # noqa: E402
from pquant.layers import (  # noqa: E402
    PQDense,
    check_quantization,
)

BATCH_SIZE = 4
OUT_FEATURES = 32
IN_FEATURES = 16


@pytest.fixture
def config_pdp():
    cfg = {
        "pruning_parameters": {
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "epsilon": 1.0,
            "pruning_method": "pdp",
            "sparsity": 0.75,
            "temperature": 1e-5,
            "threshold_decay": 0.0,
            "structured_pruning": False,
        },
        "quantization_parameters": {
            "default_weight_integer_bits": 0.0,
            "default_weight_fractional_bits": 7.0,
            "default_data_integer_bits": 0.0,
            "default_data_fractional_bits": 7.0,
            "default_data_keep_negatives": 0.0,
            "default_weight_keep_negatives": 1.0,
            "quantize_input": True,
            "quantize_output": False,
            "enable_quantization": False,
            "hgq_gamma": 0.0003,
            "hgq_beta": 1e-5,
            "hgq_heterogeneous": True,
            "layer_specific": {},
            "use_high_granularity_quantization": False,
            "use_real_tanh": False,
            "use_relu_multiplier": True,
            "use_symmetric_quantization": False,
            "round_mode": "RND",
            "overflow_mode_parameters": "SAT",
            "overflow_mode_data": "SAT",
            "granularity": "per_tensor",
        },
        "training_parameters": {"pruning_first": False},
        "fitcompress_parameters": {"enable_fitcompress": False},
    }
    return PQConfig.load_from_config(cfg)


def test_check_quantization_passes(config_pdp):
    class GoodModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=True)
            self.act = PQActivation(config_pdp, "relu", quantize_input=False, quantize_output=True)
            self.d2 = PQDense(config_pdp, OUT_FEATURES, OUT_FEATURES, quantize_input=False, quantize_output=True)

        def forward(self, x):
            return self.d2(self.act(self.d1(x)))

    assert check_quantization(GoodModel()) is True


def test_check_quantization_fails(config_pdp):
    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=False)
            self.d2 = PQDense(config_pdp, OUT_FEATURES, OUT_FEATURES, quantize_input=False, quantize_output=True)

        def forward(self, x):
            a = self.d1(x)
            b = a + a
            return self.d2(b)

    result = check_quantization(BadModel())
    assert isinstance(result, list)
    assert len(result) >= 1
    joined = "\n".join(result)
    assert "add" in joined
    assert "d2" in joined
    assert "not quantized" in joined


def _build_chain_model(config, a_qin, a_qout, b_qin, b_qout, op="add"):
    class ChainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = PQDense(config, IN_FEATURES, OUT_FEATURES, quantize_input=a_qin, quantize_output=a_qout)
            self.b = PQDense(config, OUT_FEATURES, OUT_FEATURES, quantize_input=b_qin, quantize_output=b_qout)

        def forward(self, x):
            y = self.a(x)
            if op == "add":
                y = y + y
            elif op == "sigmoid":
                y = torch.sigmoid(y)
            else:
                raise ValueError(op)
            return self.b(y)

    return ChainModel()


def test_check_quantization_chain_all_true_passes(config_pdp):
    model = _build_chain_model(config_pdp, True, True, True, True)
    assert check_quantization(model) is True


def test_check_quantization_chain_a_qin_false(config_pdp):
    model = _build_chain_model(config_pdp, False, True, True, True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'a'" in joined
    assert "quantize_input=False" in joined
    assert "'b'" not in joined
    assert "add" not in joined


def test_check_quantization_chain_b_qin_false(config_pdp):
    model = _build_chain_model(config_pdp, True, True, True, True)
    model.b.quantize_input = False
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'b'" in joined
    assert "quantize_input=False" in joined
    assert "'a'" not in joined


def test_check_quantization_chain_a_qout_false(config_pdp):
    model = _build_chain_model(config_pdp, True, False, True, True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'add'" in joined
    assert "'a'" in joined
    assert "not quantized" in joined
    assert "'b'" not in joined


def test_check_quantization_chain_b_qout_false(config_pdp):
    model = _build_chain_model(config_pdp, True, True, True, False)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'output'" in joined
    assert "'b'" in joined
    assert "not quantized" in joined


def test_check_quantization_chain_both_qin_false(config_pdp):
    model = _build_chain_model(config_pdp, False, True, False, True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'a'" in joined
    assert "'b'" in joined
    assert joined.count("quantize_input=False") == 2


def test_check_quantization_chain_both_qout_false(config_pdp):
    model = _build_chain_model(config_pdp, True, False, True, False)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'add'" in joined
    assert "'output'" in joined
    assert joined.count("not quantized") == 2


def test_check_quantization_chain_all_false(config_pdp):
    model = _build_chain_model(config_pdp, False, False, False, False)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'a'" in joined
    assert "'add'" in joined
    assert "'b'" in joined
    assert "'output'" in joined
    assert joined.count("quantize_input=False") == 2


def test_check_quantization_unary_op_between(config_pdp):
    model = _build_chain_model(config_pdp, True, False, True, True, op="sigmoid")
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "sigmoid" in joined
    assert "'a'" in joined


def test_check_quantization_fix_pqml_producer_flips_flag_unbuilt(config_pdp):
    model = _build_chain_model(config_pdp, True, False, True, True)
    assert model.a.quantize_output is False
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.a.quantize_output is True
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added == []
    assert check_quantization(traced) is True


def test_check_quantization_fix_pqml_producer_flips_flag_built(config_pdp):
    model = _build_chain_model(config_pdp, True, False, True, True)
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    model(x)
    assert model.a.built is True
    assert not hasattr(model.a, "output_quantizer")
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.a.quantize_output is True
    assert hasattr(model.a, "output_quantizer")
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added == []
    assert check_quantization(traced) is True


def test_check_quantization_fix_output_flips_pqml_flag(config_pdp):
    model = _build_chain_model(config_pdp, True, True, True, False)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.b.quantize_output is True
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added == []
    assert check_quantization(traced) is True


def test_check_quantization_fix_placeholder_producer(config_pdp):
    class PHModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=True)

        def forward(self, x):
            return self.b(x + x)

    model = PHModel()
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert len(added) == 1
    assert check_quantization(traced) is True


def test_check_quantization_fix_functional_producer_inserts_quantizer(config_pdp):
    class ChainedOps(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=True)
            self.b = PQDense(config_pdp, OUT_FEATURES, OUT_FEATURES, quantize_input=False, quantize_output=True)

        def forward(self, x):
            y = self.a(x)
            y = torch.sigmoid(y + y)
            return self.b(y)

    model = ChainedOps()
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert len(added) == 2
    assert check_quantization(traced) is True


def test_check_quantization_fix_roundtrip_all_false(config_pdp):
    model = _build_chain_model(config_pdp, False, False, False, False)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.a.quantize_output is True
    assert model.b.quantize_output is True
    assert check_quantization(traced) is True


def test_check_quantization_fix_requires_config(config_pdp):
    model = _build_chain_model(config_pdp, True, False, True, True)
    with pytest.raises(ValueError):
        check_quantization(model, add_missing_quantizers=True)


def test_check_quantization_pqactivation_producer_fix(config_pdp):
    class ActChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = PQActivation(config_pdp, "relu", quantize_input=True, quantize_output=False)
            self.b = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=True)

        def forward(self, x):
            y = self.act(x)
            y = y + y
            return self.b(y)

    model = ActChain()
    result = check_quantization(model)
    assert isinstance(result, list)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.act.quantize_output is True
    assert check_quantization(traced) is True


def test_check_quantization_multiple_consumers_of_pqml_output(config_pdp):
    class MultiConsumer(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = PQDense(config_pdp, IN_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=False)
            self.b = PQDense(config_pdp, OUT_FEATURES, OUT_FEATURES, quantize_input=True, quantize_output=True)

        def forward(self, x):
            y = self.a(x)
            z = y + y
            return self.b(z), y

    model = MultiConsumer()
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'add'" in joined
    assert "'output'" in joined
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.a.quantize_output is True
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added == []
    assert check_quantization(traced) is True
