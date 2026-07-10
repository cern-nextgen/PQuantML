"""Convert a pruned + quantized PQuant Torch model with Alkaid"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from alkaid.codegen import RTLModel  # noqa: E402
from alkaid.converter import trace_model
from alkaid.trace import trace  # noqa: E402
from alkaid.trace import FVArray, HWConfig  # noqa: E402

from pquant import pdp_config
from pquant._alkaid_plugin import _alkaid_torch_plugin  # noqa: E402
from pquant.core.torch.activations import PQActivation
from pquant.core.torch.layers import (
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQMultiheadAttention,
    PQSoftmax,
    apply_final_compression,
)
from pquant.core.torch.quantizer import Quantizer

_alkaid_torch_plugin.register()

IN_FEATURES = 3
OUT_FEATURES = 4
KERNEL_SIZE = 3
H = W = 6
SEQ_LEN = H * W

PRUNE_FRACTION = 0.9
HWCONF = HWConfig(1, 1, -1)
INPUT_KIF = (1, 4, 4)


class TwoBranchNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv2d = PQConv2d(config, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, padding="same")
        self.act2d = PQActivation(config, "relu", quantize_input=True, quantize_output=True)
        self.flat2d = nn.Flatten()

        self.conv1d = PQConv1d(config, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, padding="same")
        self.act1d = PQActivation(config, "relu", quantize_input=True, quantize_output=True)
        self.flat1d = nn.Flatten()

        self.dense = PQDense(config, OUT_FEATURES * SEQ_LEN, OUT_FEATURES)
        self.act = PQActivation(config, "relu", quantize_input=True, quantize_output=True)

    def forward(self, img, seq):
        a = self.flat2d(self.act2d(self.conv2d(img)))
        b = self.flat1d(self.act1d(self.conv1d(seq)))
        x = a + b
        return self.act(self.dense(x))


def random_prune(layer, fraction, rng):
    mask = layer.pruning_layer.mask
    numel = int(np.prod(tuple(mask.shape)))
    n_zero = int(round(fraction * numel))
    flat = np.ones(numel, dtype="float32")
    flat[rng.permutation(numel)[:n_zero]] = 0.0
    mask.copy_(torch.tensor(flat.reshape(tuple(mask.shape)), dtype=mask.dtype, device=mask.device))
    return n_zero / numel


def fixed_point_input(shape, kif=INPUT_KIF):
    """Bounded fixed-point symbolic input so the SAT input quantizer can be replayed."""
    k, i, f = (np.full(shape, v, dtype=np.int8) for v in kif)
    return FVArray.from_kif(k, i, f, HWCONF, 0, None)


def rtl_predict(comb, path, data):
    rtl_model = RTLModel(comb, str(path), "model", flavor="verilog", latency_cutoff=5, clock_period=5.0, print_latency=False)
    rtl_model.write()
    rtl_model.compile()
    if isinstance(data, list):
        data = [a.astype(np.float64) for a in data]
    else:
        data = data.astype(np.float64)
    return rtl_model.predict(data)


def build_pruned_compressed_model(config):
    model = TwoBranchNet(config)
    device = next(model.parameters()).device
    img = torch.zeros(1, IN_FEATURES, H, W, device=device)
    seq = torch.zeros(1, IN_FEATURES, SEQ_LEN, device=device)
    with torch.no_grad():
        model(img, seq)
    apply_final_compression(model)
    model.eval()
    return model, device


def test_alkaid_conversion_pruned_quantized_model():
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    model, _ = build_pruned_compressed_model(config)

    inputs = (fixed_point_input((1, IN_FEATURES, H, W)), fixed_point_input((1, IN_FEATURES, SEQ_LEN)))
    inp, out = trace_model(model, hwconf=HWCONF, inputs=inputs, framework="torch")

    assert out.shape == (OUT_FEATURES,)
    expected_inputs = IN_FEATURES * H * W + IN_FEATURES * SEQ_LEN
    assert inp.shape == (expected_inputs,)


def test_alkaid_rtl_matches_model(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model, device = build_pruned_compressed_model(config)

    inputs = (fixed_point_input((1, IN_FEATURES, H, W)), fixed_point_input((1, IN_FEATURES, SEQ_LEN)))
    inp_fv, out_fv = trace_model(model, hwconf=HWCONF, inputs=inputs, framework="torch")
    comb = trace(inp_fv, out_fv, optimize=True)
    n_samples = 16
    img = rng.integers(0, 16, size=(n_samples, IN_FEATURES, H, W)).astype("float32") / 16.0
    seq = rng.integers(0, 16, size=(n_samples, IN_FEATURES, SEQ_LEN)).astype("float32") / 16.0

    with torch.no_grad():
        reference = (
            model(torch.tensor(img, device=device), torch.tensor(seq, device=device)).cpu().numpy().astype(np.float64)
        )  # (n_samples, OUT_FEATURES)

    emulated = rtl_predict(comb, tmp_path, [img, seq])
    assert (tmp_path / "src" / "model.v").exists()

    assert np.any(reference != 0)  # the comparison is non-trivial
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)


ALL_C = 4
ALL_H = ALL_W = 6
ALL_LIN = (ALL_H // 2) * (ALL_W // 2) * 2

ALL_TORCH_LAYER_TYPES = {
    "PQConv2d",
    "PQBatchNorm2d",
    "PQAvgPool2d",
    "PQConv1d",
    "PQBatchNorm1d",
    "PQAvgPool1d",
    "PQDense",
    "PQActivation",
}


class AllLayersNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv2d = PQConv2d(config, IN_FEATURES, ALL_C, KERNEL_SIZE, padding="same")
        self.bn2d = PQBatchNorm2d(config, ALL_C)
        self.act2d = PQActivation(config, "relu", quantize_input=True, quantize_output=True)
        self.pool2d = PQAvgPool2d(config, kernel_size=2, stride=2)
        self.flat2d = nn.Flatten()

        self.conv1d = PQConv1d(config, IN_FEATURES, ALL_C, KERNEL_SIZE, padding="same")
        self.bn1d = PQBatchNorm1d(config, ALL_C)
        self.act1d = PQActivation(config, "relu", quantize_input=True, quantize_output=True)
        self.pool1d = PQAvgPool1d(config, kernel_size=2, stride=2)
        self.flat1d = nn.Flatten()

        self.dense = PQDense(config, ALL_C * (ALL_H // 2) * (ALL_W // 2), OUT_FEATURES)
        self.act = PQActivation(config, "relu", quantize_input=True, quantize_output=True)

    def forward(self, img, seq):
        a = self.flat2d(self.pool2d(self.act2d(self.bn2d(self.conv2d(img)))))
        b = self.flat1d(self.pool1d(self.act1d(self.bn1d(self.conv1d(seq)))))
        return self.act(self.dense(a + b))


def test_alkaid_conversion_all_layer_types(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model = AllLayersNet(config)
    device = next(model.parameters()).device

    model.train()
    with torch.no_grad():
        model(
            torch.tensor(rng.standard_normal((4, IN_FEATURES, ALL_H, ALL_W)), dtype=torch.float32, device=device),
            torch.tensor(rng.standard_normal((4, IN_FEATURES, ALL_LIN)), dtype=torch.float32, device=device),
        )

    with torch.no_grad():
        for module in model.modules():
            if not hasattr(module, "pruning_layer"):
                continue
            random_prune(module, PRUNE_FRACTION, rng)
    apply_final_compression(model)
    model.eval()

    inputs = (fixed_point_input((1, IN_FEATURES, ALL_H, ALL_W)), fixed_point_input((1, IN_FEATURES, ALL_LIN)))
    inp_fv, out_fv = trace_model(model, hwconf=HWCONF, inputs=inputs, framework="torch")
    comb = trace(inp_fv, out_fv, optimize=True)
    assert out_fv.shape == (OUT_FEATURES,)

    n_samples = 16
    img = rng.integers(0, 16, size=(n_samples, IN_FEATURES, ALL_H, ALL_W)).astype("float32") / 16.0
    seq = rng.integers(0, 16, size=(n_samples, IN_FEATURES, ALL_LIN)).astype("float32") / 16.0
    with torch.no_grad():
        reference = (
            model(torch.tensor(img, device=device), torch.tensor(seq, device=device)).cpu().numpy().astype(np.float64)
        )
    emulated = rtl_predict(comb, tmp_path, [img, seq])
    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)


def make_data_quantizer(config):
    qp = config.quantization_parameters
    return Quantizer(
        k=qp.default_data_keep_negatives,
        i=qp.default_data_integer_bits,
        f=qp.default_data_fractional_bits,
        overflow=qp.overflow_mode_data,
        round_mode=qp.round_mode,
        is_heterogeneous=qp.use_high_granularity_quantization,
        is_data=True,
        hgq_gamma=qp.hgq_gamma,
        place="datalane",
        dynamic_data=qp.dynamic_data_quantization,
    )


class SingleLayer(nn.Module):
    def __init__(self, layer, tail=None):
        super().__init__()
        self.layer = layer
        self.tail = tail

    def forward(self, x):
        x = self.layer(x)
        return x if self.tail is None else self.tail(x)


SINGLE_LAYER_CASES = {
    "conv2d": lambda c: ((1, 2, 4, 4), SingleLayer(PQConv2d(c, 2, 3, KERNEL_SIZE, padding="same", quantize_output=True))),
    "conv1d": lambda c: ((1, 2, 8), SingleLayer(PQConv1d(c, 2, 3, KERNEL_SIZE, padding="same", quantize_output=True))),
    "dense": lambda c: ((1, 6), SingleLayer(PQDense(c, 6, OUT_FEATURES, quantize_output=True))),
    "batchnorm2d": lambda c: ((1, 3, 4, 4), SingleLayer(PQBatchNorm2d(c, 3), make_data_quantizer(c))),
    "batchnorm1d": lambda c: ((1, 3, 8), SingleLayer(PQBatchNorm1d(c, 3), make_data_quantizer(c))),
    "avgpool2d": lambda c: ((1, 3, 4, 4), SingleLayer(PQAvgPool2d(c, kernel_size=2, stride=2, quantize_output=True))),
    "avgpool1d": lambda c: ((1, 3, 8), SingleLayer(PQAvgPool1d(c, kernel_size=2, stride=2, quantize_output=True))),
    "activation": lambda c: ((1, 6), SingleLayer(PQActivation(c, "relu", quantize_input=True, quantize_output=True))),
    "quantizer": lambda c: ((1, 6), SingleLayer(make_data_quantizer(c))),
    "softmax": lambda c: ((1, 6), SingleLayer(PQSoftmax(c, axis=-1))),
}


@pytest.mark.parametrize("case_id", list(SINGLE_LAYER_CASES))
def test_alkaid_single_layer(case_id, tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True
    shape, model = SINGLE_LAYER_CASES[case_id](config)
    rng = np.random.default_rng(0)

    model.train()
    with torch.no_grad():
        model(torch.tensor(rng.standard_normal((4,) + shape[1:]), dtype=torch.float32))
    apply_final_compression(model)
    model.eval()

    inp_fv, out_fv = trace_model(model, hwconf=HWCONF, inputs=(fixed_point_input(shape),), framework="torch")
    comb = trace(inp_fv, out_fv, optimize=True)

    n_samples = 16
    x = rng.integers(0, 16, size=(n_samples,) + shape[1:]).astype("float32") / 16.0
    with torch.no_grad():
        reference = model(torch.tensor(x)).cpu().numpy().reshape(n_samples, -1).astype(np.float64)
    emulated = rtl_predict(comb, tmp_path, x)

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)


MHA_SEQ_LEN = 4
MHA_EMBED_DIM = 4
MHA_NUM_HEADS = 2


class _MHANet(nn.Module):
    """Self-attention PQMultiheadAttention with every data quantizer enabled.

    The MHA lives inside a wrapper module so torch.fx inlines its forward with
    concrete (None) mask arguments; tracing the MHA as the fx root would turn the
    masks into proxies and hit data-dependent control flow.
    """

    def __init__(self, config):
        super().__init__()
        self.mha = PQMultiheadAttention(
            config,
            embed_dim=MHA_EMBED_DIM,
            num_heads=MHA_NUM_HEADS,
            batch_first=True,
            quantize_output=True,
        )

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


def test_alkaid_multihead_attention(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model = _MHANet(config)
    with torch.no_grad():
        model(torch.tensor(rng.standard_normal((4, MHA_SEQ_LEN, MHA_EMBED_DIM)), dtype=torch.float32))  # build
    apply_final_compression(model)
    model.eval()

    shape = (1, MHA_SEQ_LEN, MHA_EMBED_DIM)
    inp_fv, out_fv = trace_model(model, hwconf=HWCONF, inputs=(fixed_point_input(shape),), framework="torch")
    comb = trace(inp_fv, out_fv, optimize=True)
    assert out_fv.shape == (MHA_SEQ_LEN * MHA_EMBED_DIM,)

    n_samples = 16
    x = rng.integers(0, 16, size=(n_samples, MHA_SEQ_LEN, MHA_EMBED_DIM)).astype("float32") / 16.0
    with torch.no_grad():
        reference = model(torch.tensor(x)).cpu().numpy().reshape(n_samples, -1).astype(np.float64)
    emulated = rtl_predict(comb, tmp_path, x)

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)
