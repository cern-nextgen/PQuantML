"""Tests for convert_to_onnx / convert_to_onnx_fx.

Each test builds a small model (one PQ layer + ReLU where applicable), runs a
forward pass to initialise any running statistics, calls apply_final_compression
on every PQ module, exports to ONNX with convert_to_onnx(), and then verifies
that onnxruntime produces the same output as the PyTorch model.

The same check is repeated with bias=True and bias=False via parametrize.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

os.environ["KERAS_BACKEND"] = "torch"

import pquant  # noqa: E402
from pquant.core.torch.convert_to_onnx import (  # noqa: E402
    convert_to_onnx,
    convert_to_onnx_fx,
    export_qdq_layernorm,
)
from pquant.core.torch.layers import Quantizer  # noqa: E402
from pquant.layers import (  # noqa: E402
    PQActivation,
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQLayerNorm,
    PQMultiheadAttention,
)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

ATOL = 1e-4  # float32 Gemm/Conv can differ by ~1 ULP; keep some slack
# When quantization is enabled, torch fake-quant and ONNX QuantizeLinear can round a
# few values to opposite sides of a 0.5 boundary (the rounding inputs differ by float
# ULPs from differing op/accumulation order), so allow ~1 quantization level of slack
# for graphs that re-quantize intermediate activations.
QUANT_ATOL = 5e-3


def _atol(cfg):
    return QUANT_ATOL if cfg.quantization_parameters.enable_quantization else ATOL


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["float", "quant"])
def cfg(request):
    # Run every cfg-based test twice: once with the plain float path and once with
    # quantization enabled so the emitted Quantize/DequantizeLinear nodes are
    # actually exercised against onnxruntime (they are skipped entirely when
    # enable_quantization is False).
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = request.param
    return c


@pytest.fixture
def cfg_quant():
    # Quantization-enabled config for tests that specifically target the QDQ path.
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = True
    return c


def _apply_compression(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "apply_final_compression"):
            m.apply_final_compression()


def _onnx_run(model: nn.Module, x: torch.Tensor, input_shape: tuple, tmp_path) -> np.ndarray:
    """Export model → ONNX file in tmp_path, run with onnxruntime, return output."""
    path = str(tmp_path / "model.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.cpu().numpy()})[0]


def _onnx_run_fx(model: nn.Module, x: torch.Tensor, input_shape: tuple, tmp_path) -> np.ndarray:
    """FX-based export → ONNX, run with onnxruntime."""
    path = str(tmp_path / "model_fx.onnx")
    convert_to_onnx_fx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.cpu().numpy()})[0]


def _torch_out(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


@pytest.mark.parametrize("bias", [True, False])
def test_dense_onnx(cfg, bias, tmp_path):
    IN, OUT = 16, 8
    model = nn.Sequential(
        PQDense(cfg, in_features=IN, out_features=OUT, bias=bias),
        nn.ReLU(),
    )
    x = torch.randn(4, IN)
    with torch.no_grad():
        model(x)  # warm-up (needed for any running stats)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(IN,), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQDense bias={bias}: torch vs ONNX mismatch")


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_onnx(cfg, bias, tmp_path):
    IN_C, OUT_C, H, W = 3, 8, 8, 8
    model = nn.Sequential(
        PQConv2d(cfg, in_channels=IN_C, out_channels=OUT_C, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
    )
    x = torch.randn(2, IN_C, H, W)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(IN_C, H, W), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQConv2d bias={bias}: torch vs ONNX mismatch")


@pytest.mark.parametrize("bias", [True, False])
def test_conv1d_onnx(cfg, bias, tmp_path):
    IN_C, OUT_C, L = 4, 8, 16
    model = nn.Sequential(
        PQConv1d(cfg, in_channels=IN_C, out_channels=OUT_C, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(),
    )
    x = torch.randn(2, IN_C, L)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(IN_C, L), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQConv1d bias={bias}: torch vs ONNX mismatch")


def test_batchnorm2d_onnx(cfg, tmp_path):
    C, H, W = 8, 4, 4
    model = nn.Sequential(
        PQBatchNorm2d(cfg, num_features=C),
        nn.ReLU(),
    )
    x = torch.randn(4, C, H, W)
    with torch.no_grad():
        model(x)
    _apply_compression(model)
    model.eval()  # switch BN to use running stats

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, H, W), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQBatchNorm2d: torch vs ONNX mismatch")


def test_batchnorm1d_onnx(cfg, tmp_path):
    C, L = 8, 16
    model = nn.Sequential(
        PQBatchNorm1d(cfg, num_features=C),
        nn.ReLU(),
    )
    x = torch.randn(4, C, L)
    with torch.no_grad():
        model(x)
    _apply_compression(model)
    model.eval()

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, L), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQBatchNorm1d: torch vs ONNX mismatch")


def test_avgpool2d_onnx(cfg, tmp_path):
    C, H, W = 8, 8, 8
    model = nn.Sequential(
        PQAvgPool2d(cfg, kernel_size=2, stride=2),
    )
    x = torch.randn(2, C, H, W)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, H, W), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQAvgPool2d: torch vs ONNX mismatch")


def test_avgpool1d_onnx(cfg, tmp_path):
    C, L = 8, 16
    model = nn.Sequential(
        PQAvgPool1d(cfg, kernel_size=2, stride=2),
    )
    x = torch.randn(2, C, L)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(C, L), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQAvgPool1d: torch vs ONNX mismatch")


class _SelfAttnModel(nn.Module):
    """Thin wrapper so FX tracing sees a single-input model."""

    def __init__(self, mha: PQMultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


@pytest.mark.parametrize("bias", [True, False])
def test_mha_onnx(cfg, bias, tmp_path):
    E, H, T = 16, 4, 8
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H, bias=bias, batch_first=True)
    model = _SelfAttnModel(mha)

    x = torch.randn(2, T, E)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    # The quantized softmax (exp/inv LUTs) re-quantizes intermediates, so allow ~1 LSB
    # of rounding-boundary slack when quantization is enabled (see _atol).
    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run_fx(model, x, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_out, onnx_out, atol=_atol(cfg), err_msg=f"PQMultiheadAttention bias={bias}: torch vs ONNX mismatch"
    )


class _CausalSelfAttnModel(nn.Module):
    """Self-attention with a constant additive causal mask (the decoder-inference case)."""

    def __init__(self, mha: PQMultiheadAttention, seq_len: int):
        super().__init__()
        self.mha = mha
        # (T, S) additive mask: 0 on/below the diagonal, large-negative above it.
        self.register_buffer("attn_mask", torch.triu(torch.full((seq_len, seq_len), -1e4), diagonal=1))

    def forward(self, x):
        out, _ = self.mha(x, x, x, attn_mask=self.attn_mask)
        return out


@pytest.mark.parametrize("bias", [True, False])
def test_mha_causal_attn_mask_onnx(cfg, bias, tmp_path):
    E, H, T = 16, 4, 8
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H, bias=bias, batch_first=True)
    model = _CausalSelfAttnModel(mha, T)

    x = torch.randn(2, T, E)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run_fx(model, x, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_out, onnx_out, atol=_atol(cfg), err_msg=f"MHA causal attn_mask bias={bias}: torch vs ONNX mismatch"
    )


class _PaddedSelfAttnModel(nn.Module):
    """Self-attention with a runtime bool key_padding_mask input (True == padding)."""

    def __init__(self, mha: PQMultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(self, x, key_padding_mask):
        out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return out


@pytest.mark.parametrize("bias", [True, False])
def test_mha_key_padding_mask_onnx(cfg, bias, tmp_path):
    import onnx

    E, H, T = 16, 4, 8
    mha = PQMultiheadAttention(cfg, embed_dim=E, num_heads=H, bias=bias, batch_first=True)
    model = _PaddedSelfAttnModel(mha)

    x = torch.randn(2, T, E)
    key_padding_mask = torch.zeros(2, T, dtype=torch.bool)
    key_padding_mask[:, -2:] = True  # last two key positions are padding
    with torch.no_grad():
        model(x, key_padding_mask)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(x, key_padding_mask).cpu().numpy()

    path = str(tmp_path / "mha_kpm.onnx")
    proto = convert_to_onnx_fx(model, input_shape=[(T, E), (T,)], output_path=path, input_dtypes=["float32", "bool"])

    # The padding mask must be a genuine bool graph input, not baked away.
    kpm_vi = next(i for i in proto.graph.input if i.name == "key_padding_mask")
    assert kpm_vi.type.tensor_type.elem_type == onnx.TensorProto.BOOL

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {"x": x.cpu().numpy(), "key_padding_mask": key_padding_mask.cpu().numpy()})[0]

    np.testing.assert_allclose(
        torch_out, onnx_out, atol=_atol(cfg), err_msg=f"MHA key_padding_mask bias={bias}: torch vs ONNX mismatch"
    )


@pytest.mark.parametrize("input_shape", [(4, 64), (1, 4, 64)])
def test_qdq_layernorm_export(input_shape, tmp_path):
    import onnx

    D = input_shape[-1]
    rng = np.random.default_rng(0)
    # Q7 representable: gamma = k / 128, k integer, |k| < 32768
    gamma_q = rng.integers(low=64, high=192, size=(D,), dtype=np.int32)  # ~0.5 .. 1.5
    gamma = (gamma_q.astype(np.float32)) / (1 << 7)
    # Q15 representable: beta = k / 32768, k integer, |k| < 32768 (so |beta| < 1)
    beta_q = rng.integers(low=-1024, high=1024, size=(D,), dtype=np.int32)
    beta = (beta_q.astype(np.float32)) / (1 << 15)

    input_scale_log2 = -7  # input_scale = 2**-7
    output_scale_log2 = -6  # output_scale = 2**-6
    eps_q0 = 1

    path = str(tmp_path / "qdq_layernorm.onnx")
    model_proto = export_qdq_layernorm(
        output_path=path,
        input_shape=input_shape,
        gamma=gamma,
        beta=beta,
        input_scale_log2=input_scale_log2,
        output_scale_log2=output_scale_log2,
        eps_q0=eps_q0,
    )

    # ----- structural checks -----
    op_types = [n.op_type for n in model_proto.graph.node]
    assert op_types == ["DequantizeLinear", "LayerNormalization", "QuantizeLinear", "DequantizeLinear"]

    ln_node = model_proto.graph.node[1]
    axis = next(a.i for a in ln_node.attribute if a.name == "axis")
    eps_attr = next(a.f for a in ln_node.attribute if a.name == "epsilon")
    assert axis == -1
    expected_eps = eps_q0 * (2.0**input_scale_log2) ** 2
    assert abs(eps_attr - expected_eps) < 1e-12

    # input must be int8, output float
    assert len(model_proto.graph.input) == 1
    assert model_proto.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT8
    assert model_proto.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    in_dims = [d.dim_value for d in model_proto.graph.input[0].type.tensor_type.shape.dim]
    assert tuple(in_dims) == input_shape

    # zero-points must be int8 zero
    inits = {t.name: t for t in model_proto.graph.initializer}
    for zp_name in ("input_zero_point", "output_zero_point"):
        zp = onnx.numpy_helper.to_array(inits[zp_name])
        assert zp.dtype == np.int8
        assert int(zp) == 0

    # scales must be exact powers of two
    in_scale = float(onnx.numpy_helper.to_array(inits["input_scale"]))
    out_scale = float(onnx.numpy_helper.to_array(inits["output_scale"]))
    assert in_scale == 2.0**input_scale_log2
    assert out_scale == 2.0**output_scale_log2

    # ----- numerical check via onnxruntime -----
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    x_q = rng.integers(low=-64, high=64, size=input_shape, dtype=np.int8)
    onnx_out = sess.run(None, {in_name: x_q})[0]

    # Reference: dequantize -> layernorm(axis=-1) -> quantize -> dequantize
    x_f = x_q.astype(np.float32) * in_scale
    mean = x_f.mean(axis=-1, keepdims=True)
    var = x_f.var(axis=-1, keepdims=True)
    x_norm = (x_f - mean) / np.sqrt(var + expected_eps)
    y_f = x_norm * gamma + beta
    y_q = np.clip(np.round(y_f / out_scale), -128, 127).astype(np.int8)
    y_ref = y_q.astype(np.float32) * out_scale

    np.testing.assert_allclose(onnx_out, y_ref, atol=out_scale * 0.5)


class _TwoInputModel(nn.Module):
    """Two tensor inputs merged by addition after independent Dense layers."""

    def __init__(self, cfg, in_a: int, in_b: int, out: int, bias: bool):
        super().__init__()
        self.dense_a = PQDense(cfg, in_features=in_a, out_features=out, bias=bias)
        self.dense_b = PQDense(cfg, in_features=in_b, out_features=out, bias=bias)

    def forward(self, a, b):
        return torch.relu(self.dense_a(a) + self.dense_b(b))


@pytest.mark.parametrize("bias", [True, False])
def test_two_input_onnx(cfg, bias, tmp_path):
    IN_A, IN_B, OUT = 16, 4, 8
    model = _TwoInputModel(cfg, IN_A, IN_B, OUT, bias)

    a = torch.randn(3, IN_A)
    b = torch.randn(3, IN_B)
    with torch.no_grad():
        model(a, b)  # warm-up
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(a, b).cpu().numpy()

    path = str(tmp_path / "two_input.onnx")
    model_proto = convert_to_onnx_fx(model, input_shape=[(IN_A,), (IN_B,)], output_path=path)

    # Graph must declare exactly two inputs, named after the forward parameters,
    # each with a dynamic batch dim and its own feature shape.
    assert [i.name for i in model_proto.graph.input] == ["a", "b"]
    in_shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model_proto.graph.input}
    assert in_shapes["a"] == [0, IN_A]  # dim_value 0 == dynamic (no batch fixed)
    assert in_shapes["b"] == [0, IN_B]

    sess = ort.InferenceSession(path)
    names = [i.name for i in sess.get_inputs()]
    assert set(names) == {"a", "b"}
    onnx_out = sess.run(None, {"a": a.cpu().numpy(), "b": b.cpu().numpy()})[0]

    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"two-input bias={bias}: torch vs ONNX mismatch")


def test_two_input_shape_count_mismatch(cfg, tmp_path):
    """Wrong number of shapes for the model's tensor inputs is a clear error."""
    model = _TwoInputModel(cfg, 16, 4, 8, bias=True)
    with torch.no_grad():
        model(torch.randn(2, 16), torch.randn(2, 4))
    _apply_compression(model)

    path = str(tmp_path / "bad_count.onnx")
    with pytest.raises(ValueError, match="tensor input"):
        convert_to_onnx_fx(model, input_shape=(16,), output_path=path)  # only one shape


class _FlaggedModel(nn.Module):
    """One tensor input plus a bool flag selecting an optional scaling branch."""

    def __init__(self, cfg, in_features: int, out: int):
        super().__init__()
        self.dense = PQDense(cfg, in_features=in_features, out_features=out, bias=True)

    def forward(self, x, scale_up: bool = False):
        out = self.dense(x)
        if scale_up:
            out = out * 2.0
        return torch.relu(out)


@pytest.mark.parametrize("scale_up", [False, True])
def test_concrete_args_specialization(cfg, scale_up, tmp_path):
    IN, OUT = 16, 8
    model = _FlaggedModel(cfg, IN, OUT)

    x = torch.randn(3, IN)
    with torch.no_grad():
        model(x, scale_up)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(x, scale_up).cpu().numpy()

    path = str(tmp_path / f"flag_{scale_up}.onnx")
    model_proto = convert_to_onnx_fx(model, input_shape=(IN,), output_path=path, concrete_args={"scale_up": scale_up})

    # The bool flag is baked in as a constant, so it must NOT appear as a graph
    # input — only the single tensor input "input" remains.
    assert [i.name for i in model_proto.graph.input] == ["input"]

    sess = ort.InferenceSession(path)
    assert [i.name for i in sess.get_inputs()] == ["input"]
    onnx_out = sess.run(None, {"input": x.cpu().numpy()})[0]

    np.testing.assert_allclose(
        torch_out, onnx_out, atol=ATOL, err_msg=f"concrete_args scale_up={scale_up}: torch vs ONNX mismatch"
    )


class _ResidualConcatModel(nn.Module):
    """Exercises the FX converter's branch handling: a skip-add and a concat."""

    def __init__(self, cfg, dim: int, out: int):
        super().__init__()
        self.d1 = PQDense(cfg, in_features=dim, out_features=dim)
        self.d2 = PQDense(cfg, in_features=dim, out_features=dim)
        self.d3 = PQDense(cfg, in_features=2 * dim, out_features=out)

    def forward(self, x):
        h = self.d1(x)
        h = h + self.d2(h)  # residual / skip add
        h = torch.cat([h, x], dim=1)  # branch merge by concatenation
        return self.d3(h)


def test_residual_concat_onnx(cfg_quant, tmp_path):
    DIM, OUT = 16, 8
    model = _ResidualConcatModel(cfg_quant, DIM, OUT)

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(x).cpu().numpy()

    path = str(tmp_path / "residual_concat.onnx")
    model_proto = convert_to_onnx_fx(model, input_shape=(DIM,), output_path=path)
    op_types = [n.op_type for n in model_proto.graph.node]
    assert "Add" in op_types  # the skip connection
    assert "Concat" in op_types  # the branch merge

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_out, onnx_out, atol=QUANT_ATOL, err_msg="residual+concat: torch vs ONNX mismatch")


@pytest.mark.parametrize("activation", ["relu", "tanh", "hard_tanh", "leaky_relu", "gelu"])
def test_pqactivation_onnx(cfg_quant, activation, tmp_path):
    DIM = 16
    model = nn.Sequential(PQActivation(cfg_quant, activation=activation))

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=(DIM,), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"PQActivation {activation}: torch vs ONNX mismatch")


def test_standalone_quantizer_onnx(cfg_quant, tmp_path):
    qp = cfg_quant.quantization_parameters
    quant = Quantizer(
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
    model = nn.Sequential(quant)

    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    # A standalone quantizer must emit a Quantize/Dequantize pair.
    path = str(tmp_path / "quantizer.onnx")
    model_proto = convert_to_onnx(model, input_shape=(16,), output_path=path)
    op_types = [n.op_type for n in model_proto.graph.node]
    assert "QuantizeLinear" in op_types and "DequantizeLinear" in op_types

    torch_out = _torch_out(model, x)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="standalone Quantizer: torch vs ONNX mismatch")


class _CNNFlattenModel(nn.Module):
    def __init__(self, cfg, in_c: int, hw: int, out: int, use_reshape: bool):
        super().__init__()
        self.conv = PQConv2d(cfg, in_channels=in_c, out_channels=4, kernel_size=3, padding=1)
        self.dense = PQDense(cfg, in_features=4 * hw * hw, out_features=out)
        self.use_reshape = use_reshape
        self._flat = 4 * hw * hw

    def forward(self, x):
        h = torch.relu(self.conv(x))
        # Both are common CNN→Dense transitions; reshape uses a constant (-1, N)
        # shape because the FX exporter requires static reshape targets.
        h = h.reshape(-1, self._flat) if self.use_reshape else torch.flatten(h, 1)
        return self.dense(h)


@pytest.mark.parametrize("use_reshape", [False, True])
def test_cnn_flatten_to_dense_onnx(cfg_quant, use_reshape, tmp_path):
    IN_C, HW, OUT = 3, 8, 8
    model = _CNNFlattenModel(cfg_quant, IN_C, HW, OUT, use_reshape)

    x = torch.randn(2, IN_C, HW, HW)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(x).cpu().numpy()

    path = str(tmp_path / f"cnn_flatten_{use_reshape}.onnx")
    convert_to_onnx_fx(model, input_shape=(IN_C, HW, HW), output_path=path)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(
        torch_out, onnx_out, atol=QUANT_ATOL, err_msg=f"CNN→Dense reshape={use_reshape}: torch vs ONNX mismatch"
    )


class _ScalarOpsModel(nn.Module):
    def __init__(self, cfg, dim: int):
        super().__init__()
        self.d = PQDense(cfg, in_features=dim, out_features=dim)

    def forward(self, x):
        h = self.d(x)
        h = h * 2.0  # Mul with scalar literal
        h = h - 1.0  # Sub with scalar literal
        h = h / 3.0  # Div with scalar literal
        return torch.sigmoid(h)


def test_scalar_ops_onnx(cfg_quant, tmp_path):
    DIM = 16
    model = _ScalarOpsModel(cfg_quant, DIM)

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(x).cpu().numpy()

    path = str(tmp_path / "scalar_ops.onnx")
    model_proto = convert_to_onnx_fx(model, input_shape=(DIM,), output_path=path)
    op_types = [n.op_type for n in model_proto.graph.node]
    for expected in ("Mul", "Sub", "Div", "Sigmoid"):
        assert expected in op_types, f"missing {expected} node"

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="scalar ops+sigmoid: torch vs ONNX mismatch")


class _MultiOutputModel(nn.Module):
    def __init__(self, cfg, dim: int):
        super().__init__()
        self.a = PQDense(cfg, in_features=dim, out_features=8)
        self.b = PQDense(cfg, in_features=dim, out_features=4)

    def forward(self, x):
        return self.a(x), self.b(x)


def test_multi_output_onnx(cfg_quant, tmp_path):
    DIM = 16
    model = _MultiOutputModel(cfg_quant, DIM)

    x = torch.randn(3, DIM)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        t0, t1 = (t.cpu().numpy() for t in model(x))

    path = str(tmp_path / "multi_output.onnx")
    model_proto = convert_to_onnx_fx(model, input_shape=(DIM,), output_path=path)
    assert len(model_proto.graph.output) == 2

    sess = ort.InferenceSession(path)
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})
    by_name = dict(zip(out_names, outs))
    # Match outputs by shape (order is preserved, but be explicit about which is which).
    o0 = next(v for v in by_name.values() if v.shape[1] == 8)
    o1 = next(v for v in by_name.values() if v.shape[1] == 4)
    np.testing.assert_allclose(t0, o0, atol=ATOL, err_msg="multi-output[0]: torch vs ONNX mismatch")
    np.testing.assert_allclose(t1, o1, atol=ATOL, err_msg="multi-output[1]: torch vs ONNX mismatch")


def test_pqlayernorm_onnx(cfg_quant, tmp_path):
    DIM = 16
    model = nn.Sequential(PQLayerNorm(cfg_quant, normalized_shape=DIM))

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    _apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(x).cpu().numpy()

    path = str(tmp_path / "pqlayernorm.onnx")
    # LayerNormalization is an opset-17 op; the converter default (13) cannot host it.
    convert_to_onnx(model, input_shape=(DIM,), output_path=path, opset=17)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="PQLayerNorm: torch vs ONNX mismatch")


@pytest.mark.parametrize(
    "make_model,input_shape,batch",
    [
        (lambda: nn.Sequential(nn.LeakyReLU(0.1)), (8, 16), 4),
        (lambda: nn.Sequential(nn.MaxPool2d(2, 2)), (3, 8, 8), 2),
        (lambda: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest")), (3, 4, 4), 2),
        (lambda: nn.Sequential(nn.Dropout(0.5)), (16,), 4),
        (lambda: nn.Sequential(nn.BatchNorm2d(3)), (3, 8, 8), 2),
    ],
    ids=["leaky_relu", "maxpool2d", "upsample", "dropout", "batchnorm2d"],
)
def test_plain_passthrough_layers_onnx(make_model, input_shape, batch, tmp_path):
    model = make_model()
    model.eval()  # Dropout/BatchNorm must be in eval mode for a deterministic compare
    x = torch.randn(batch, *input_shape)

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run(model, x, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg="plain passthrough layer: torch vs ONNX mismatch")


def _quantized_dense_model(cfg_quant):
    model = nn.Sequential(PQDense(cfg_quant, in_features=16, out_features=8), nn.ReLU())
    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)
    _apply_compression(model)
    return model, x


@pytest.mark.parametrize("integer_ops", [False, True])
def test_integer_weight_storage_onnx(cfg_quant, integer_ops, tmp_path):
    """store_integer_weights and integer_ops (MatMulInteger) must stay numerically exact."""
    model, x = _quantized_dense_model(cfg_quant)
    torch_out = _torch_out(model, x)

    path = str(tmp_path / f"int_{integer_ops}.onnx")
    kwargs = {"integer_ops": True} if integer_ops else {"store_integer_weights": True}
    convert_to_onnx(model, input_shape=(16,), output_path=path, **kwargs)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_out, onnx_out, atol=ATOL, err_msg=f"integer_ops={integer_ops}: mismatch")


def test_include_clip_toggle_structure(cfg_quant, tmp_path):
    """include_clip controls whether a Clip node precedes each input QuantizeLinear."""
    model, _ = _quantized_dense_model(cfg_quant)

    proto_clip = convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "clip.onnx"), include_clip=True)
    proto_noclip = convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "noclip.onnx"), include_clip=False)
    assert "Clip" in [n.op_type for n in proto_clip.graph.node]
    assert "Clip" not in [n.op_type for n in proto_noclip.graph.node]


def test_batch_size_fixes_input_dim(cfg_quant, tmp_path):
    """batch_size pins the graph's batch dimension instead of leaving it dynamic."""
    model, _ = _quantized_dense_model(cfg_quant)

    proto = convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "bs.onnx"), batch_size=4)
    in_dims = [d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim]
    assert in_dims[0] == 4  # batch fixed
    assert in_dims[1] == 16


def test_qonnx_export_builds(cfg_quant, tmp_path):
    """use_qonnx emits QONNX Quant nodes and produces a structurally valid model."""
    import onnx

    model, _ = _quantized_dense_model(cfg_quant)
    path = str(tmp_path / "qonnx.onnx")
    proto = convert_to_onnx(model, input_shape=(16,), output_path=path, use_qonnx=True)
    onnx.checker.check_model(onnx.load(path))
    assert any(n.op_type == "Quant" for n in proto.graph.node)


def test_qdq_layernorm_validation(tmp_path):
    path = str(tmp_path / "bad.onnx")
    D = 64
    gamma = np.ones(D, dtype=np.float32)
    beta = np.zeros(D, dtype=np.float32)

    # rank-1 input: rejected
    with pytest.raises(ValueError, match="rank"):
        export_qdq_layernorm(path, (D,), gamma, beta, -7, -6)

    # last dim not multiple of 32
    with pytest.raises(ValueError, match="multiple of 32"):
        export_qdq_layernorm(path, (4, 16), np.ones(16, np.float32), np.zeros(16, np.float32), -7, -6)

    # last dim not power of two (96 = 32*3)
    with pytest.raises(ValueError, match="power of two"):
        export_qdq_layernorm(path, (4, 96), np.ones(96, np.float32), np.zeros(96, np.float32), -7, -6)

    # gamma not Q7-representable (1/3 is not k/128 exactly)
    with pytest.raises(ValueError, match="gamma"):
        export_qdq_layernorm(path, (4, D), np.full(D, 1.0 / 3.0, np.float32), beta, -7, -6)

    # beta not Q15-representable (1/3 is not k/32768 exactly)
    with pytest.raises(ValueError, match="beta"):
        export_qdq_layernorm(path, (4, D), gamma, np.full(D, 1.0 / 3.0, np.float32), -7, -6)

    # eps_q0 < 1
    with pytest.raises(ValueError, match="eps_q0"):
        export_qdq_layernorm(path, (4, D), gamma, beta, -7, -6, eps_q0=0)
