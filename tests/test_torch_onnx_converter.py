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
from pquant.layers import (  # noqa: E402
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQMultiheadAttention,
)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

ATOL = 1e-4  # float32 Gemm/Conv can differ by ~1 ULP; keep some slack


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = False
    return c


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQDense
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQConv2d
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQConv1d
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQBatchNorm2d
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQBatchNorm1d
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQAvgPool2d
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQAvgPool1d
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PQMultiheadAttention  (uses FX converter; self-attention, batch_first=True)
# ---------------------------------------------------------------------------


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

    torch_out = _torch_out(model, x)
    onnx_out = _onnx_run_fx(model, x, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_out, onnx_out, atol=ATOL, err_msg=f"PQMultiheadAttention bias={bias}: torch vs ONNX mismatch"
    )


# ---------------------------------------------------------------------------
# Static-QDQ LayerNormalization graph
# ---------------------------------------------------------------------------


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
