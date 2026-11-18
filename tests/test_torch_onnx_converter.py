"""Tests for the torch → ONNX converter: onnxruntime must reproduce the PyTorch outputs of every exported model."""

import inspect
import os
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

os.environ["KERAS_BACKEND"] = "torch"

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

import pquant  # noqa: E402
from pquant.core.torch.layers import Quantizer  # noqa: E402
from pquant.core.torch.onnx import (  # noqa: E402
    TARGETS,
    ExportOptions,
    OnnxExportWarning,
    convert_to_onnx,
    list_targets,
    resolve_export_options,
)

ATOL = 1e-4
MHA_ATOL = 5e-3  # exp and reciprocal differ in the last bit between runtimes, which can flip a quantizer after them


def make_cfg(quantize=True, **quantization_parameters):
    """cs_config that rounds half to even like QuantizeLinear, so that quantized exports are exact."""
    cfg = pquant.cs_config()
    cfg.quantization_parameters.enable_quantization = quantize
    cfg.quantization_parameters.round_mode = "RND_CONV"
    for name, value in quantization_parameters.items():
        setattr(cfg.quantization_parameters, name, value)
    return cfg


@pytest.fixture(params=[False, True], ids=["float", "quant"])
def cfg(request):
    return make_cfg(request.param)


@pytest.fixture
def cfg_quant():
    return make_cfg()


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)  # the session-wide seed makes tests depend on their order


class Model(nn.Module):
    """A model whose forward is ``fn(self, x)``; ``parts`` become its submodules and buffers."""

    def __init__(self, fn, **parts):
        super().__init__()
        self.fn = fn
        for name, part in parts.items():
            if isinstance(part, torch.Tensor):
                self.register_buffer(name, part)
            else:
                setattr(self, name, part)

    def forward(self, x):
        return self.fn(self, x)


class TwoInputModel(Model):
    def forward(self, a, b):
        return self.fn(self, a, b)


def compressed(model, *inputs, **kwargs):
    """Warm up ``model`` in train mode, apply the final compression and switch to eval mode."""
    with torch.no_grad():
        model(*inputs, **kwargs)
    for module in model.modules():
        if hasattr(module, "apply_final_compression"):
            module.apply_final_compression()
    return model.eval()


def run_onnx(path, *inputs):
    """Outputs of the graph as exported; onnxruntime's optimizers would re-quantize float weights."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path, options)
    return session.run(None, {i.name: x.cpu().numpy() for i, x in zip(session.get_inputs(), inputs)})


def check_export(model, *inputs, tmp_path, atol=ATOL, compare=True, **export_options):
    """Export ``model`` and compare all of its outputs with onnxruntime.  Returns the ModelProto."""
    kwargs = export_options.get("concrete_args") or {}
    compressed(model, *inputs, **kwargs)
    shapes = [tuple(x.shape[1:]) for x in inputs]
    path = str(tmp_path / "model.onnx")
    proto = convert_to_onnx(model, shapes if len(shapes) > 1 else shapes[0], path, **export_options)
    if compare:
        with torch.no_grad():
            expected = model(*inputs, **kwargs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        outputs = run_onnx(path, *inputs)
        assert [tuple(t.shape) for t in expected] == [o.shape for o in outputs]
        for want, got in zip(expected, outputs):
            np.testing.assert_allclose(want.cpu().numpy(), got, atol=atol)
    return proto


def dense_relu(cfg, **kwargs):
    return nn.Sequential(PQDense(cfg, 16, 8, **kwargs), nn.ReLU())


# name: (layers, input shape without the batch dim, export options)
LAYERS = {
    "dense": (lambda c: [PQDense(c, 16, 8), nn.ReLU()], (16,), {}),
    "dense-nobias": (lambda c: [PQDense(c, 16, 8, bias=False), nn.ReLU()], (16,), {}),
    "dense-rank3": (lambda c: [PQDense(c, 16, 8), nn.ReLU()], (5, 16), {}),
    "conv2d": (lambda c: [PQConv2d(c, 3, 8, 3, padding=1), nn.ReLU()], (3, 8, 8), {}),
    "conv2d-nobias": (lambda c: [PQConv2d(c, 3, 8, 3, padding=1, bias=False), nn.ReLU()], (3, 8, 8), {}),
    "conv1d": (lambda c: [PQConv1d(c, 4, 8, 3, padding=1), nn.ReLU()], (4, 16), {}),
    "conv1d-nobias": (lambda c: [PQConv1d(c, 4, 8, 3, padding=1, bias=False), nn.ReLU()], (4, 16), {}),
    "batchnorm2d": (lambda c: [PQBatchNorm2d(c, 8), nn.ReLU()], (8, 4, 4), {}),
    "batchnorm1d": (lambda c: [PQBatchNorm1d(c, 8), nn.ReLU()], (8, 16), {}),
    "avgpool2d": (lambda c: [PQAvgPool2d(c, 2, 2)], (8, 8, 8), {}),
    "avgpool1d": (lambda c: [PQAvgPool1d(c, 2, 2)], (8, 16), {}),
    "layernorm": (lambda c: [PQLayerNorm(c, 16)], (16,), {"opset": 17}),  # LayerNormalization exists from opset 17 on
}


@pytest.mark.parametrize("name", LAYERS)
def test_single_layer_onnx(cfg, name, tmp_path):
    layers, input_shape, export_options = LAYERS[name]
    check_export(nn.Sequential(*layers(cfg)), torch.randn(4, *input_shape), tmp_path=tmp_path, **export_options)


def data_quantizer():
    qp = make_cfg().quantization_parameters
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


PLAIN_LAYERS = {
    "quantizer": (data_quantizer, (16,)),
    "leaky_relu": (lambda: nn.LeakyReLU(0.1), (8, 16)),
    "maxpool2d": (lambda: nn.MaxPool2d(2, 2), (3, 8, 8)),
    "upsample": (lambda: nn.Upsample(scale_factor=2, mode="nearest"), (3, 4, 4)),
    "dropout": (lambda: nn.Dropout(0.5), (16,)),
}


@pytest.mark.parametrize("name", PLAIN_LAYERS)
def test_plain_layer_onnx(name, tmp_path):
    """A standalone Quantizer and the torch.nn layers, which do not depend on the config."""
    make_layer, input_shape = PLAIN_LAYERS[name]
    check_export(nn.Sequential(make_layer()), torch.randn(4, *input_shape), tmp_path=tmp_path)


@pytest.mark.parametrize(
    "activation, multiplier",
    [("relu", False), ("tanh", False), ("hard_tanh", False), ("leaky_relu", False), ("gelu", False), ("relu", True)],
)
def test_pqactivation_onnx(activation, multiplier, tmp_path):
    cfg = make_cfg(use_relu_multiplier=multiplier)
    check_export(nn.Sequential(PQActivation(cfg, activation=activation)), torch.randn(4, 16), tmp_path=tmp_path)


# name: (forward, PQMultiheadAttention options, export options)
ATTENTION = {
    "plain": (lambda m, x: m.mha(x, x, x)[0], {}, {}),
    "no-bias": (lambda m, x: m.mha(x, x, x)[0], {"bias": False}, {}),
    "causal-mask": (lambda m, x: m.mha(x, x, x, attn_mask=m.mask)[0], {}, {}),
    # Both outputs; ReduceMax/ReduceMean take their axes as an input from opset 18 on.
    "with-weights": (lambda m, x: m.mha(x, x, x), {}, {"opset": 18}),
    "sequence-first": (lambda m, x: m.mha(x, x, x), {"batch_first": False}, {"opset": 18}),
}


def mha_atol(cfg):
    return MHA_ATOL if cfg.quantization_parameters.enable_quantization else ATOL


@pytest.mark.parametrize("name", ATTENTION)
def test_mha_onnx(cfg, name, tmp_path):
    forward, mha_options, export_options = ATTENTION[name]
    mha = PQMultiheadAttention(cfg, embed_dim=16, num_heads=4, **{"batch_first": True, **mha_options})
    causal_mask = torch.triu(torch.full((8, 8), -1e4), diagonal=1)
    model = Model(forward, mha=mha, mask=causal_mask)
    check_export(model, torch.randn(2, 8, 16), tmp_path=tmp_path, atol=mha_atol(cfg), **export_options)


def test_mha_key_padding_mask_onnx(cfg, tmp_path):
    """A runtime key_padding_mask (True = padding) stays a bool graph input."""
    mha = PQMultiheadAttention(cfg, embed_dim=16, num_heads=4, batch_first=True)
    model = TwoInputModel(lambda m, x, mask: m.mha(x, x, x, key_padding_mask=mask)[0], mha=mha)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, -2:] = True
    proto = check_export(
        model, torch.randn(2, 8, 16), mask, tmp_path=tmp_path, atol=mha_atol(cfg), input_dtypes=["float32", "bool"]
    )
    assert proto.graph.input[1].type.tensor_type.elem_type == onnx.TensorProto.BOOL


def test_two_input_onnx(cfg_quant, tmp_path):
    """Graph inputs are named after the forward parameters, and input_shape must describe all of them."""
    dense_a, dense_b = PQDense(cfg_quant, 16, 8), PQDense(cfg_quant, 4, 8)
    model = TwoInputModel(lambda m, a, b: torch.relu(m.dense_a(a) + m.dense_b(b)), dense_a=dense_a, dense_b=dense_b)
    proto = check_export(model, torch.randn(3, 16), torch.randn(3, 4), tmp_path=tmp_path)
    assert [i.name for i in proto.graph.input] == ["a", "b"]
    shapes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in proto.graph.input]
    assert shapes == [[0, 16], [0, 4]]  # 0 means dynamic
    with pytest.raises(ValueError, match="tensor input"):
        convert_to_onnx(model, input_shape=(16,), output_path=str(tmp_path / "one_shape.onnx"))


class OptionalArgsModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = PQDense(cfg, 16, 8)

    def forward(self, x, scale_up: bool = False, mask=None):
        out = self.dense(x)
        if scale_up:
            out = out * 2.0
        if mask is not None:
            out = out * mask
        return torch.relu(out)


@pytest.mark.parametrize("scale_up", [False, True])
def test_concrete_args_onnx(cfg_quant, scale_up, tmp_path):
    """Arguments baked in with concrete_args (a bool flag, an optional mask left at None) are not graph inputs."""
    concrete_args = {"scale_up": scale_up, "mask": None}
    proto = check_export(OptionalArgsModel(cfg_quant), torch.randn(3, 16), tmp_path=tmp_path, concrete_args=concrete_args)
    assert [i.name for i in proto.graph.input] == ["input"]


def test_residual_concat_onnx(cfg_quant, tmp_path):
    def forward(m, x):
        h = m.d1(x)
        return m.d3(torch.cat([h + m.d2(h), x], dim=1))

    model = Model(forward, d1=PQDense(cfg_quant, 16, 16), d2=PQDense(cfg_quant, 16, 16), d3=PQDense(cfg_quant, 32, 8))
    check_export(model, torch.randn(4, 16), tmp_path=tmp_path)


def test_multi_output_onnx(cfg_quant, tmp_path):
    model = Model(lambda m, x: (m.a(x), m.b(x)), a=PQDense(cfg_quant, 16, 8), b=PQDense(cfg_quant, 16, 4))
    check_export(model, torch.randn(3, 16), tmp_path=tmp_path)


# Operations on the (N, 8) output of a dense layer ...
DENSE_OUTPUT_OPS = {
    "slice-range": lambda y: y[:, 2:6],
    "slice-int": lambda y: y[:, 0],
    "slice-ellipsis-step": lambda y: y[..., 1:8:2],
    "slice-negative-int": lambda y: y[:, -1],
    "unsqueeze": lambda y: y.unsqueeze(1),
    "unsqueeze-negative": lambda y: torch.unsqueeze(y, -1),
    "unsqueeze-squeeze": lambda y: y.unsqueeze(2).squeeze(2),
    "squeeze-all": lambda y: torch.squeeze(y.unsqueeze(1)),
    "scalar-arithmetic": lambda y: torch.sigmoid((y * 2.0 - 1.0) / 3.0),
}
# ... and on the (N, 4, 4, 4) output of a convolution; only flatten(1) maps to ONNX Flatten.
CONV_OUTPUT_OPS = {
    "flatten": lambda y: torch.flatten(y, 1),
    "flatten-all": lambda y: y.flatten(),
    "flatten-from-2": lambda y: torch.flatten(y, 2),
    "flatten-1-to-2": lambda y: y.flatten(1, 2),
    "flatten-module": nn.Flatten(start_dim=2),
    "reshape": lambda y: y.reshape(-1, 64),
    "reshape-tuple": lambda y: y.reshape((-1, 4, 16)),
    "reshape-function": lambda y: torch.reshape(y, (-1, 64)),
}


@pytest.mark.parametrize("name", [*DENSE_OUTPUT_OPS, *CONV_OUTPUT_OPS])
def test_tensor_op_onnx(cfg_quant, name, tmp_path):
    if name in DENSE_OUTPUT_OPS:
        layer, op, x = PQDense(cfg_quant, 16, 8), DENSE_OUTPUT_OPS[name], torch.randn(4, 16)
    else:
        layer, op, x = PQConv2d(cfg_quant, 3, 4, 3, padding=1), CONV_OUTPUT_OPS[name], torch.randn(2, 3, 4, 4)
    check_export(Model(lambda m, x: m.op(m.layer(x)), layer=layer, op=op), x, tmp_path=tmp_path)


@pytest.mark.parametrize("target", ["qdq-int8", "integer-ops"])
@pytest.mark.parametrize("input_shape", [(16,), (5, 16)], ids=["rank2", "rank3"])
def test_integer_dense_onnx(cfg_quant, target, input_shape, tmp_path):
    """Integer weight storage and MatMulInteger stay exact, for rank-3 inputs too."""
    check_export(dense_relu(cfg_quant), torch.randn(4, *input_shape), tmp_path=tmp_path, target=target, include_clip=True)


@pytest.mark.parametrize("target", ["qdq-int8", "qonnx"])
def test_unquantized_model_keeps_float_weights(target, tmp_path):
    """Without quantization the weights are not on a fixed-point grid, so no target may quantize them."""
    model = dense_relu(make_cfg(quantize=False))
    proto = check_export(model, torch.randn(4, 16), tmp_path=tmp_path, target=target, batch_size="dynamic")
    assert [node.op_type for node in proto.graph.node] == ["Gemm", "Relu"]


def test_integer_ops_needs_quantized_layer(tmp_path):
    model = compressed(dense_relu(make_cfg(quantize=False)), torch.randn(4, 16))
    with pytest.raises(ValueError, match="integer_ops requires"):
        convert_to_onnx(model, (16,), str(tmp_path / "model.onnx"), target="integer-ops")


# name: (quant_format, include_clip, batch_size)
TARGET_OPTIONS = {
    "reference": ("qdq", True, None),
    "qonnx": ("qonnx", True, 1),
    "aie4ml": ("qdq_int_weights", False, 1),
    "qdq-int8": ("qdq_int_weights", False, None),
    "onnxruntime": ("qdq_int_weights", False, None),
    "tensorrt": ("qdq_int_weights", False, None),
    "integer-ops": ("integer_ops", True, None),
}


def test_target_registry():
    assert {name: (o.quant_format.value, o.include_clip, o.batch_size) for name, o in TARGETS.items()} == TARGET_OPTIONS
    assert list_targets().keys() == TARGETS.keys() and all(list_targets().values())
    assert inspect.signature(convert_to_onnx).parameters["target"].default == "reference"


def test_target_overrides():
    resolve = resolve_export_options
    assert resolve("aie4ml", include_clip=True, batch_size="dynamic") == ExportOptions("qdq_int_weights", include_clip=True)
    assert resolve("reference", opset=17, strict=True) == ExportOptions(opset=17, strict=True)
    assert resolve("qdq-int8", batch_size=2).batch_size == 2

    custom = ExportOptions(quant_format="qonnx", opset=17)
    assert resolve(custom) == custom
    assert resolve(custom, batch_size=2) == ExportOptions(quant_format="qonnx", opset=17, batch_size=2)

    with pytest.raises(ValueError, match="unknown export target"):
        resolve("fpga")
    with pytest.raises(ValueError, match="opset 11 is not supported"):
        resolve("reference", opset=11)


# One export per distinct set of options (the aliases of qdq-int8 share its), plus integer ops without Clip.
INTEGER_OPS_NO_CLIP = ExportOptions("integer_ops", include_clip=False)
EXPORTED_TARGETS = ["reference", "qonnx", "aie4ml", "qdq-int8", "integer-ops", INTEGER_OPS_NO_CLIP]


@pytest.mark.parametrize("target", EXPORTED_TARGETS, ids=lambda t: t if isinstance(t, str) else "integer-ops-noclip")
def test_target_export_onnx(cfg_quant, target, tmp_path):
    """Each target yields a valid graph with the structure its options promise, which onnxruntime reproduces."""
    options = resolve_export_options(target)
    x = torch.rand(options.batch_size or 4, 16) * 0.9  # inside the unsigned 7-bit input range: exact without Clip too
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OnnxExportWarning)  # 7-bit activations without Clip, see EXPORT_CHECKS
        # onnxruntime cannot execute the Quant custom op
        proto = check_export(dense_relu(cfg_quant), x, tmp_path=tmp_path, target=target, compare=not options.use_qonnx)
    onnx.checker.check_model(proto)

    op_types = {node.op_type for node in proto.graph.node}
    int_types = (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)
    int_weights = any(t.name.endswith("_weight_int") and t.data_type in int_types for t in proto.graph.initializer)
    assert ("Quant" in op_types) == options.use_qonnx
    assert ("MatMulInteger" in op_types) == options.integer_ops
    assert ("Clip" in op_types) == (options.include_clip and not options.use_qonnx)
    assert int_weights == options.store_integer_weights
    assert proto.graph.input[0].type.tensor_type.shape.dim[0].dim_value == (options.batch_size or 0)  # 0 means dynamic


# (config change, export options, expected outcome); the default config has unsigned 7-bit activations
EXPORT_CHECKS = [
    ({"default_data_fractional_bits": 9.0}, {}, ("raises", "container")),  # 9-bit activations
    ({"default_data_fractional_bits": 9.0}, {"target": "qonnx"}, None),  # Quant carries the bit width
    ({"default_weight_fractional_bits": 11.0}, {}, None),  # 12-bit weights, stored as floats
    ({"default_weight_fractional_bits": 11.0}, {"target": "aie4ml"}, ("raises", "container")),
    ({}, {"target": "qdq-int8"}, ("warns", "saturates")),  # narrower than uint8, and no Clip
    ({}, {"target": "qdq-int8", "strict": True}, ("raises", "saturates")),
    ({}, {"target": "qdq-int8", "include_clip": True}, None),
    ({"round_mode": "RND"}, {}, None),  # another tie rule than QuantizeLinear's is only logged
    ({"round_mode": "TRN"}, {}, ("warns", "truncation")),
    ({"round_mode": "TRN"}, {"target": "qonnx"}, None),  # Quant carries the rounding mode
    ({"overflow_mode_data": "WRAP"}, {}, ("warns", "WRAP")),
]


@pytest.mark.parametrize("config, export_options, outcome", EXPORT_CHECKS)
def test_export_checks(config, export_options, outcome, tmp_path):
    model = compressed(dense_relu(make_cfg(**config)), torch.randn(4, 16))
    path = str(tmp_path / "model.onnx")

    def export():
        convert_to_onnx(model, (16,), path, **export_options)

    if outcome is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OnnxExportWarning)
            export()
    elif outcome[0] == "warns":
        with pytest.warns(OnnxExportWarning, match=outcome[1]) as caught:
            export()
        assert caught[0].filename == __file__  # reported at the convert_to_onnx call, not inside the converter
    else:
        with pytest.raises(ValueError, match=outcome[1]):
            export()
        assert not os.path.exists(path)
