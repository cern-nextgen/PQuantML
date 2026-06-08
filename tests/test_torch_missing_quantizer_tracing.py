import os

import pytest
import torch
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant.activations import PQActivation  # noqa: E402
from pquant.core.hyperparameter_optimization import PQConfig  # noqa: E402
from pquant.core.torch.quantizer import Quantizer  # noqa: E402
from pquant.core.torch.tracing import check_quantization  # noqa: E402
from pquant.layers import PQDense  # noqa: E402

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
            "enable_quantization": True,
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
    # d1.quantize_output=False, so the `a + a` add receives unquantized inputs.
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
    # 'a' has quantize_input=False but receives the raw (unquantized) model input.
    model = _build_chain_model(config_pdp, False, True, True, True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'a'" in joined
    assert "quantize_input=False" in joined
    assert "'b'" not in joined
    assert "add" not in joined


def test_check_quantization_chain_b_qin_false(config_pdp):
    # 'b' has quantize_input=False but its input comes from the unquantized `add` op output.
    model = _build_chain_model(config_pdp, True, True, True, True)
    model.b.quantize_input = False
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'b'" in joined
    assert "quantize_input=False" in joined
    assert "'a'" not in joined


def test_check_quantization_chain_a_qout_false(config_pdp):
    # 'a'.quantize_output=False, so the `add` op consuming 'a' receives unquantized input.
    model = _build_chain_model(config_pdp, True, False, True, True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'add'" in joined
    assert "'a'" in joined
    assert "not quantized" in joined
    assert "'b'" not in joined


def test_check_quantization_chain_b_qout_false(config_pdp):
    # 'b'.quantize_output=False, so the model output is left unquantized.
    model = _build_chain_model(config_pdp, True, True, True, False)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'output'" in joined
    assert "'b'" in joined
    assert "not quantized" in joined


def test_check_quantization_chain_both_qin_false(config_pdp):
    # Both 'a' and 'b' have quantize_input=False while their inputs are unquantized.
    model = _build_chain_model(config_pdp, False, True, False, True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'a'" in joined
    assert "'b'" in joined
    assert joined.count("quantize_input=False") == 2


def test_check_quantization_chain_both_qout_false(config_pdp):
    # First layer output and model output unquantized
    model = _build_chain_model(config_pdp, True, False, True, False)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'add'" in joined
    assert "'output'" in joined
    assert joined.count("not quantized") == 2


def test_check_quantization_chain_all_false(config_pdp):
    # All quantize_input/output flags are False.
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
    # 'a'.quantize_output=False.
    model = _build_chain_model(config_pdp, True, False, True, True, op="sigmoid")
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "sigmoid" in joined
    assert "'a'" in joined


def test_check_quantization_fix_pqml_producer_flips_flag_unbuilt(config_pdp):
    # 'a'.quantize_output=False feeds the `add` op, change quantize_output to True.
    model = _build_chain_model(config_pdp, True, False, True, True)
    assert model.a.quantize_output is False
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.a.quantize_output is True
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added == []
    assert check_quantization(traced) is True


def test_check_quantization_fix_pqml_producer_flips_flag_built(config_pdp):
    # Same as above but 'a' is already built: the fix flips the flag and constructs its output_quantizer in-place.
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
    # 'b'.quantize_output=False leaves the model output unquantized, so the fix flips 'b'.quantize_output.
    model = _build_chain_model(config_pdp, True, True, True, False)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.b.quantize_output is True
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added == []
    assert check_quantization(traced) is True


def test_check_quantization_fix_placeholder_producer(config_pdp):
    # `x + x` consumes the raw placeholder input (unquantized), so the fix inserts a standalone quantizer.
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
    # `add` feeds `sigmoid` and `sigmoid` feeds 'b' (both function outputs are unquantized).
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
    # All flags False, so the fix must flip both layers' quantize_output to fully quantize the graph.
    model = _build_chain_model(config_pdp, False, False, False, False)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.a.quantize_output is True
    assert model.b.quantize_output is True
    assert check_quantization(traced) is True


def test_check_quantization_fix_requires_config(config_pdp):
    # Fixing missing quantizer requires a config (here omitted, so it raises an error).
    model = _build_chain_model(config_pdp, True, False, True, True)
    with pytest.raises(ValueError):
        check_quantization(model, add_missing_quantizers=True)


def test_check_quantization_pqactivation_producer_fix(config_pdp):
    # The PQActivation's quantize_output=False feeds the `add` op, so the fix flips the activation's quantize_output.
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
    # 'a'.quantize_output=False feeds both the `add` op and the model output, quantize a output.
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


def _make_quantizer(config):
    qp = config.quantization_parameters
    return Quantizer(
        k=qp.default_data_keep_negatives,
        i=qp.default_data_integer_bits,
        f=qp.default_data_fractional_bits,
        overflow=qp.overflow_mode_data,
        round_mode=qp.round_mode,
        is_heterogeneous=False,
        is_data=True,
        granularity="per_tensor",
        hgq_gamma=qp.hgq_gamma,
    )


def _build_dense_matmul_skip_model(config, dense_qout):
    class DenseMatmulSkip(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = PQDense(config, IN_FEATURES, IN_FEATURES, quantize_input=True, quantize_output=dense_qout)
            self.register_buffer("w", torch.randn(IN_FEATURES, IN_FEATURES))
            # Route the constant matrix through a Quantizer so it counts as
            # "assumed quantized" (the tracer treats a raw get_attr as unquantized).
            self.wq = _make_quantizer(config)

        def forward(self, x):
            y = self.d(x)
            y = torch.matmul(y, self.wq(self.w))  # matmul with constant matrix (assumed quantized)
            y = y + x  # skip connection from the (unquantized) model input
            return y

    return DenseMatmulSkip()


def test_check_quantization_dense_matmul_skip_from_input(config_pdp):
    # matmul output is unquantized and the skip connection feeds the raw model input 'x' into the
    # `add`, so the add inputs and the model output are flagged; the assumed-quantized constant is not.
    model = _build_dense_matmul_skip_model(config_pdp, dense_qout=True)
    result = check_quantization(model)
    assert isinstance(result, list)
    joined = "\n".join(result)
    assert "'add'" in joined
    assert "input 'x' is not quantized" in joined
    assert "input 'matmul' is not quantized" in joined
    assert "'output'" in joined
    # The constant matmul operand is quantized, so matmul itself reports no missing input.
    assert not any(r.startswith("'matmul'") for r in result)


def test_check_quantization_dense_matmul_skip_from_input_fix(config_pdp):
    # Fixing inserts standalone quantizers on the unquantized skip input, the matmul output, and the
    # model output (the assumed-quantized constant needs none).
    model = _build_dense_matmul_skip_model(config_pdp, dense_qout=True)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    added = [name for name, _ in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert len(added) == 3
    assert check_quantization(traced) is True


def test_check_quantization_dense_matmul_skip_unquantized_dense(config_pdp):
    # With dense.quantize_output=False the matmul also sees an unquantized data input from 'd'; the fix
    # flips the PQDense flag (a PQuantML producer) and inserts quantizers for the rest.
    model = _build_dense_matmul_skip_model(config_pdp, dense_qout=False)
    result = check_quantization(model)
    assert isinstance(result, list)
    assert any(r.startswith("'matmul'") and "input 'd' is not quantized" in r for r in result)
    traced = check_quantization(model, add_missing_quantizers=True, config=config_pdp)
    assert model.d.quantize_output is True
    assert check_quantization(traced) is True
