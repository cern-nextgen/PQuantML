"""Tests for convert_to_onnx.

Each test builds a small model (one PQ layer + ReLU where applicable), runs a
forward pass to initialise any running statistics, calls apply_final_compression
on every PQ module, exports to ONNX with convert_to_onnx(), and then verifies
that onnxruntime produces the same output as the PyTorch model.

The same check is repeated with bias=True and bias=False via parametrize.
"""

import os

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

os.environ["KERAS_BACKEND"] = "torch"

import pquant  # noqa: E402
from pquant.core.torch.layers import Quantizer  # noqa: E402
from pquant.core.torch.onnx import convert_to_onnx  # noqa: E402
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

ATOL = 1e-4
QUANT_ATOL = 5e-3


def atol(cfg):
    return QUANT_ATOL if cfg.quantization_parameters.enable_quantization else ATOL


@pytest.fixture(params=[False, True], ids=["float", "quant"])
def cfg(request):
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = request.param
    return c


@pytest.fixture
def cfg_quant():
    c = pquant.cs_config()
    c.quantization_parameters.enable_quantization = True
    return c


def apply_compression(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "apply_final_compression"):
            m.apply_final_compression()


def onnx_run(model: nn.Module, x: torch.Tensor, input_shape: tuple, tmp_path) -> np.ndarray:
    """Export model → ONNX file in tmp_path, run with onnxruntime, return output."""
    path = str(tmp_path / "model.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.cpu().numpy()})[0]


def onnx_run_fx(model: nn.Module, x: torch.Tensor, input_shape: tuple, tmp_path) -> np.ndarray:
    """FX-based export → ONNX, run with onnxruntime."""
    path = str(tmp_path / "model_fx.onnx")
    convert_to_onnx(model, input_shape=input_shape, output_path=path)
    sess = ort.InferenceSession(path)
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: x.cpu().numpy()})[0]


def torch_out(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


# (model factory, input shape without batch dim, batch size)
SINGLE_LAYER_CASES = [
    pytest.param(
        lambda cfg: nn.Sequential(PQDense(cfg, in_features=16, out_features=8, bias=True), nn.ReLU()),
        (16,),
        4,
        id="dense-bias",
    ),
    pytest.param(
        lambda cfg: nn.Sequential(PQDense(cfg, in_features=16, out_features=8, bias=False), nn.ReLU()),
        (16,),
        4,
        id="dense-nobias",
    ),
    pytest.param(
        lambda cfg: nn.Sequential(
            PQConv2d(cfg, in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=True), nn.ReLU()
        ),
        (3, 8, 8),
        2,
        id="conv2d-bias",
    ),
    pytest.param(
        lambda cfg: nn.Sequential(
            PQConv2d(cfg, in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=False), nn.ReLU()
        ),
        (3, 8, 8),
        2,
        id="conv2d-nobias",
    ),
    pytest.param(
        lambda cfg: nn.Sequential(
            PQConv1d(cfg, in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=True), nn.ReLU()
        ),
        (4, 16),
        2,
        id="conv1d-bias",
    ),
    pytest.param(
        lambda cfg: nn.Sequential(
            PQConv1d(cfg, in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False), nn.ReLU()
        ),
        (4, 16),
        2,
        id="conv1d-nobias",
    ),
    pytest.param(lambda cfg: nn.Sequential(PQBatchNorm2d(cfg, num_features=8), nn.ReLU()), (8, 4, 4), 4, id="batchnorm2d"),
    pytest.param(lambda cfg: nn.Sequential(PQBatchNorm1d(cfg, num_features=8), nn.ReLU()), (8, 16), 4, id="batchnorm1d"),
    pytest.param(lambda cfg: nn.Sequential(PQAvgPool2d(cfg, kernel_size=2, stride=2)), (8, 8, 8), 2, id="avgpool2d"),
    pytest.param(lambda cfg: nn.Sequential(PQAvgPool1d(cfg, kernel_size=2, stride=2)), (8, 16), 2, id="avgpool1d"),
]


@pytest.mark.parametrize("make_model,input_shape,batch", SINGLE_LAYER_CASES)
def test_single_layer_onnx(cfg, make_model, input_shape, batch, tmp_path):
    model = make_model(cfg)
    x = torch.randn(batch, *input_shape)
    with torch.no_grad():
        model(x)  # warm-up in train mode (initialises any running stats)
    apply_compression(model)

    torch_output = torch_out(model, x)  # eval mode: BN uses running stats
    onnx_out = onnx_run(model, x, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg="torch vs ONNX mismatch")


class SelfAttnModel(nn.Module):
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
    model = SelfAttnModel(mha)

    x = torch.randn(2, T, E)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    # The quantized softmax (exp/inv LUTs) re-quantizes intermediates, so allow ~1 LSB
    # of rounding-boundary slack when quantization is enabled (see atol).
    torch_output = torch_out(model, x)
    onnx_out = onnx_run_fx(model, x, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_output, onnx_out, atol=atol(cfg), err_msg=f"PQMultiheadAttention bias={bias}: torch vs ONNX mismatch"
    )


class CausalSelfAttnModel(nn.Module):
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
    model = CausalSelfAttnModel(mha, T)

    x = torch.randn(2, T, E)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    torch_output = torch_out(model, x)
    onnx_out = onnx_run_fx(model, x, input_shape=(T, E), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_output, onnx_out, atol=atol(cfg), err_msg=f"MHA causal attn_mask bias={bias}: torch vs ONNX mismatch"
    )


class PaddedSelfAttnModel(nn.Module):
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
    model = PaddedSelfAttnModel(mha)

    x = torch.randn(2, T, E)
    key_padding_mask = torch.zeros(2, T, dtype=torch.bool)
    key_padding_mask[:, -2:] = True  # last two key positions are padding
    with torch.no_grad():
        model(x, key_padding_mask)
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(x, key_padding_mask).cpu().numpy()

    path = str(tmp_path / "mha_kpm.onnx")
    proto = convert_to_onnx(model, input_shape=[(T, E), (T,)], output_path=path, input_dtypes=["float32", "bool"])

    # The padding mask must be a genuine bool graph input, not baked away.
    kpm_vi = next(i for i in proto.graph.input if i.name == "key_padding_mask")
    assert kpm_vi.type.tensor_type.elem_type == onnx.TensorProto.BOOL

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {"x": x.cpu().numpy(), "key_padding_mask": key_padding_mask.cpu().numpy()})[0]

    np.testing.assert_allclose(
        torch_output, onnx_out, atol=atol(cfg), err_msg=f"MHA key_padding_mask bias={bias}: torch vs ONNX mismatch"
    )


class TwoInputModel(nn.Module):
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
    model = TwoInputModel(cfg, IN_A, IN_B, OUT, bias)

    a = torch.randn(3, IN_A)
    b = torch.randn(3, IN_B)
    with torch.no_grad():
        model(a, b)  # warm-up
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(a, b).cpu().numpy()

    path = str(tmp_path / "two_input.onnx")
    model_proto = convert_to_onnx(model, input_shape=[(IN_A,), (IN_B,)], output_path=path)

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

    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg=f"two-input bias={bias}: torch vs ONNX mismatch")


def test_two_input_shape_count_mismatch(cfg, tmp_path):
    """Wrong number of shapes for the model's tensor inputs is a clear error."""
    model = TwoInputModel(cfg, 16, 4, 8, bias=True)
    with torch.no_grad():
        model(torch.randn(2, 16), torch.randn(2, 4))
    apply_compression(model)

    path = str(tmp_path / "bad_count.onnx")
    with pytest.raises(ValueError, match="tensor input"):
        convert_to_onnx(model, input_shape=(16,), output_path=path)  # only one shape


class FlaggedModel(nn.Module):
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
    model = FlaggedModel(cfg, IN, OUT)

    x = torch.randn(3, IN)
    with torch.no_grad():
        model(x, scale_up)
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(x, scale_up).cpu().numpy()

    path = str(tmp_path / f"flag_{scale_up}.onnx")
    model_proto = convert_to_onnx(model, input_shape=(IN,), output_path=path, concrete_args={"scale_up": scale_up})

    # The bool flag is baked in as a constant, so it must NOT appear as a graph
    # input — only the single tensor input "input" remains.
    assert [i.name for i in model_proto.graph.input] == ["input"]

    sess = ort.InferenceSession(path)
    assert [i.name for i in sess.get_inputs()] == ["input"]
    onnx_out = sess.run(None, {"input": x.cpu().numpy()})[0]

    np.testing.assert_allclose(
        torch_output, onnx_out, atol=ATOL, err_msg=f"concrete_args scale_up={scale_up}: torch vs ONNX mismatch"
    )


class ResidualConcatModel(nn.Module):
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
    model = ResidualConcatModel(cfg_quant, DIM, OUT)

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(x).cpu().numpy()

    path = str(tmp_path / "residual_concat.onnx")
    model_proto = convert_to_onnx(model, input_shape=(DIM,), output_path=path)
    op_types = [n.op_type for n in model_proto.graph.node]
    assert "Add" in op_types  # the skip connection
    assert "Concat" in op_types  # the branch merge

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_out, atol=QUANT_ATOL, err_msg="residual+concat: torch vs ONNX mismatch")


@pytest.mark.parametrize("activation", ["relu", "tanh", "hard_tanh", "leaky_relu", "gelu"])
def test_pqactivation_onnx(cfg_quant, activation, tmp_path):
    DIM = 16
    model = nn.Sequential(PQActivation(cfg_quant, activation=activation))

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    torch_output = torch_out(model, x)
    onnx_out = onnx_run(model, x, input_shape=(DIM,), tmp_path=tmp_path)
    np.testing.assert_allclose(
        torch_output, onnx_out, atol=ATOL, err_msg=f"PQActivation {activation}: torch vs ONNX mismatch"
    )


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
    apply_compression(model)

    # A standalone quantizer must emit a Quantize/Dequantize pair.
    path = str(tmp_path / "quantizer.onnx")
    model_proto = convert_to_onnx(model, input_shape=(16,), output_path=path)
    op_types = [n.op_type for n in model_proto.graph.node]
    assert "QuantizeLinear" in op_types and "DequantizeLinear" in op_types

    torch_output = torch_out(model, x)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg="standalone Quantizer: torch vs ONNX mismatch")


class CNNFlattenModel(nn.Module):
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
    model = CNNFlattenModel(cfg_quant, IN_C, HW, OUT, use_reshape)

    x = torch.randn(2, IN_C, HW, HW)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(x).cpu().numpy()

    path = str(tmp_path / f"cnn_flatten_{use_reshape}.onnx")
    convert_to_onnx(model, input_shape=(IN_C, HW, HW), output_path=path)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(
        torch_output, onnx_out, atol=QUANT_ATOL, err_msg=f"CNN→Dense reshape={use_reshape}: torch vs ONNX mismatch"
    )


class ScalarOpsModel(nn.Module):
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
    model = ScalarOpsModel(cfg_quant, DIM)

    x = torch.randn(4, DIM)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(x).cpu().numpy()

    path = str(tmp_path / "scalar_ops.onnx")
    model_proto = convert_to_onnx(model, input_shape=(DIM,), output_path=path)
    op_types = [n.op_type for n in model_proto.graph.node]
    for expected in ("Mul", "Sub", "Div", "Sigmoid"):
        assert expected in op_types, f"missing {expected} node"

    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg="scalar ops+sigmoid: torch vs ONNX mismatch")


class MultiOutputModel(nn.Module):
    def __init__(self, cfg, dim: int):
        super().__init__()
        self.a = PQDense(cfg, in_features=dim, out_features=8)
        self.b = PQDense(cfg, in_features=dim, out_features=4)

    def forward(self, x):
        return self.a(x), self.b(x)


def test_multi_output_onnx(cfg_quant, tmp_path):
    DIM = 16
    model = MultiOutputModel(cfg_quant, DIM)

    x = torch.randn(3, DIM)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        t0, t1 = (t.cpu().numpy() for t in model(x))

    path = str(tmp_path / "multi_output.onnx")
    model_proto = convert_to_onnx(model, input_shape=(DIM,), output_path=path)
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
    apply_compression(model)

    model.eval()
    with torch.no_grad():
        torch_output = model(x).cpu().numpy()

    path = str(tmp_path / "pqlayernorm.onnx")
    # LayerNormalization is an opset-17 op; the converter default (13) cannot host it.
    convert_to_onnx(model, input_shape=(DIM,), output_path=path, opset=17)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg="PQLayerNorm: torch vs ONNX mismatch")


@pytest.mark.parametrize(
    "make_model,input_shape,batch",
    [
        (lambda: nn.Sequential(nn.LeakyReLU(0.1)), (8, 16), 4),
        (lambda: nn.Sequential(nn.MaxPool2d(2, 2)), (3, 8, 8), 2),
        (lambda: nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest")), (3, 4, 4), 2),
        (lambda: nn.Sequential(nn.Dropout(0.5)), (16,), 4),
    ],
    ids=["leaky_relu", "maxpool2d", "upsample", "dropout"],
)
def test_plain_passthrough_layers_onnx(make_model, input_shape, batch, tmp_path):
    model = make_model()
    model.eval()  # Dropout/BatchNorm must be in eval mode for a deterministic compare
    x = torch.randn(batch, *input_shape)

    torch_output = torch_out(model, x)
    onnx_out = onnx_run(model, x, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg="plain passthrough layer: torch vs ONNX mismatch")


def quantized_dense_model(cfg_quant):
    model = nn.Sequential(PQDense(cfg_quant, in_features=16, out_features=8), nn.ReLU())
    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)
    apply_compression(model)
    return model, x


@pytest.mark.parametrize("integer_ops", [False, True])
def test_integer_weight_storage_onnx(cfg_quant, integer_ops, tmp_path):
    """store_integer_weights and integer_ops (MatMulInteger) must stay numerically exact."""
    model, x = quantized_dense_model(cfg_quant)
    torch_output = torch_out(model, x)

    path = str(tmp_path / f"int_{integer_ops}.onnx")
    kwargs = {"integer_ops": True} if integer_ops else {"store_integer_weights": True}
    convert_to_onnx(model, input_shape=(16,), output_path=path, **kwargs)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg=f"integer_ops={integer_ops}: mismatch")


def rank3_dense_model(cfg):
    """PQDense applied to a [batch, seq, dim] input (F.linear maps over the last axis)."""
    SEQ, DIM, OUT = 5, 16, 8
    model = nn.Sequential(PQDense(cfg, in_features=DIM, out_features=OUT), nn.ReLU())
    x = torch.randn(4, SEQ, DIM)
    with torch.no_grad():
        model(x)
    apply_compression(model)
    return model, x, (SEQ, DIM)


def test_dense_rank3_input_onnx(cfg, tmp_path):
    """A standalone PQDense on a rank-3 input must export as MatMul + Add (Gemm is rank-2 only)."""
    model, x, input_shape = rank3_dense_model(cfg)
    torch_output = torch_out(model, x)
    onnx_out = onnx_run(model, x, input_shape=input_shape, tmp_path=tmp_path)
    np.testing.assert_allclose(torch_output, onnx_out, atol=atol(cfg), err_msg="rank-3 dense: torch vs ONNX mismatch")


@pytest.mark.parametrize("integer_ops", [False, True])
def test_dense_rank3_integer_onnx(cfg_quant, integer_ops, tmp_path):
    """Integer weight storage and MatMulInteger must also handle rank-3 dense inputs."""
    model, x, input_shape = rank3_dense_model(cfg_quant)
    torch_output = torch_out(model, x)

    path = str(tmp_path / f"rank3_int_{integer_ops}.onnx")
    kwargs = {"integer_ops": True} if integer_ops else {"store_integer_weights": True}
    convert_to_onnx(model, input_shape=input_shape, output_path=path, **kwargs)
    sess = ort.InferenceSession(path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    np.testing.assert_allclose(torch_output, onnx_out, atol=ATOL, err_msg=f"rank-3 integer_ops={integer_ops}: mismatch")


def test_include_clip_toggle_structure(cfg_quant, tmp_path):
    """include_clip controls whether a Clip node precedes each input QuantizeLinear."""
    model, _ = quantized_dense_model(cfg_quant)

    proto_clip = convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "clip.onnx"), include_clip=True)
    proto_noclip = convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "noclip.onnx"), include_clip=False)
    assert "Clip" in [n.op_type for n in proto_clip.graph.node]
    assert "Clip" not in [n.op_type for n in proto_noclip.graph.node]


def test_batch_size_fixes_input_dim(cfg_quant, tmp_path):
    """batch_size pins the graph's batch dimension instead of leaving it dynamic."""
    model, _ = quantized_dense_model(cfg_quant)

    proto = convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "bs.onnx"), batch_size=4)
    in_dims = [d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim]
    assert in_dims[0] == 4  # batch fixed
    assert in_dims[1] == 16


def test_qonnx_export_builds(cfg_quant, tmp_path):
    """use_qonnx emits QONNX Quant nodes and produces a structurally valid model."""
    import onnx

    model, _ = quantized_dense_model(cfg_quant)
    path = str(tmp_path / "qonnx.onnx")
    proto = convert_to_onnx(model, input_shape=(16,), output_path=path, use_qonnx=True)
    onnx.checker.check_model(onnx.load(path))
    assert any(n.op_type == "Quant" for n in proto.graph.node)


@pytest.mark.parametrize(
    "slicer",
    [
        lambda y: y[:, 2:6],
        lambda y: y[:, 0],
        lambda y: y[..., 1:8:2],
        lambda y: y[:, -1],
    ],
    ids=["range", "int_squeeze", "ellipsis_step", "neg_int"],
)
def test_tensor_slicing_onnx(cfg, slicer, tmp_path):
    """Constant tensor slicing in forward must export as ONNX Slice (+ Squeeze)."""

    class SliceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = PQDense(cfg, in_features=16, out_features=8)

        def forward(self, x):
            return slicer(self.dense(x))

    model = SliceModel()
    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    torch_output = torch_out(model, x)
    onnx_out = onnx_run(model, x, input_shape=(16,), tmp_path=tmp_path)
    np.testing.assert_allclose(torch_output, onnx_out, atol=atol(cfg), err_msg="tensor slicing: torch vs ONNX mismatch")


@pytest.mark.parametrize(
    "reshaper",
    [
        lambda y: y.unsqueeze(1),
        lambda y: torch.unsqueeze(y, -1),
        lambda y: y.unsqueeze(2).squeeze(2),
        lambda y: torch.squeeze(y.unsqueeze(1)),
    ],
    ids=["method_unsqueeze", "fn_unsqueeze_neg", "roundtrip", "squeeze_all"],
)
def test_squeeze_unsqueeze_onnx(cfg, reshaper, tmp_path):
    """squeeze/unsqueeze in forward must export as ONNX Squeeze/Unsqueeze."""

    class ReshapeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = PQDense(cfg, in_features=16, out_features=8)

        def forward(self, x):
            return reshaper(self.dense(x))

    model = ReshapeModel()
    x = torch.randn(4, 16)
    with torch.no_grad():
        model(x)
    apply_compression(model)

    torch_output = torch_out(model, x)
    onnx_out = onnx_run(model, x, input_shape=(16,), tmp_path=tmp_path)
    assert torch_output.shape == onnx_out.shape
    np.testing.assert_allclose(torch_output, onnx_out, atol=atol(cfg), err_msg="squeeze/unsqueeze: torch vs ONNX mismatch")
