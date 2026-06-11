"""Convert a pruned + quantized PQuant Torch model with Alkaid.

Torch counterpart of ``test_keras_alkaid_conversion``: exercises the
``alkaid_torch`` second-level plugin (``pquant._alkaid_plugin``). Build a model
from PQConv2d / PQConv1d / PQDense with relu PQActivation, randomly prune 90% of
every weight tensor, bake the pruning and quantization in with
``apply_final_compression``, then trace the model into Alkaid's IR. A second test
additionally lowers the trace to an RTL project and checks that the RTL's
software model reproduces the Torch model's outputs bit-for-bit.

Alkaid traces Torch models through ``torch.fx``, so the model must be in eval
mode and already built (one real forward pass) before conversion, and the
symbolic inputs must be supplied explicitly (shapes cannot be inferred).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from alkaid.codegen import RTLModel  # noqa: E402

# Alkaid is required for this test; skip cleanly if it (or its deps) isn't installed.
from alkaid.converter import trace_model
from alkaid.trace import trace  # noqa: E402
from alkaid.trace import FVArray, HWConfig  # noqa: E402

from pquant import pdp_config

# Load the pquant Alkaid plugin explicitly instead of relying on the `alkaid_torch`
# entry point. register() patches the PQ layers for fx tracing and marks the plugin
# loaded so Alkaid's lazy loader skips re-discovery. Without it (e.g. when the
# installed package metadata doesn't expose the entry point, as with some editable
# installs) tracing hits the un-patched `weight` property and torch.fx raises a
# control-flow TraceError.
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
    apply_final_compression,
)
from pquant.core.torch.quantizer import Quantizer

_alkaid_torch_plugin.register()

IN_FEATURES = 3
OUT_FEATURES = 4
KERNEL_SIZE = 3
H = W = 6
# Match the conv1d flatten length to the conv2d branch's so the two can be merged.
SEQ_LEN = H * W

PRUNE_FRACTION = 0.9
HWCONF = HWConfig(1, 1, -1)
# Input fixed-point format (signed, 4 integer, 4 fractional bits) for the hardware
# input ports; bounded so the SAT input quantizer can be replayed.
INPUT_KIF = (1, 4, 4)


class TwoBranchNet(nn.Module):
    """conv2d branch + conv1d branch merged (matched flatten lengths) -> dense head.

    A conv after a reshape/flatten cannot be traced by Alkaid (its reshape folds
    the batch axis), and its ``Concatenate`` merge is unreliable, so the two convs
    live on separate branches that are summed before the dense head.
    """

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


def _random_prune(layer, fraction, rng):
    """Zero exactly ``fraction`` of the layer's weights via its pruning mask."""
    mask = layer.pruning_layer.mask
    numel = int(np.prod(tuple(mask.shape)))
    n_zero = int(round(fraction * numel))
    flat = np.ones(numel, dtype="float32")
    flat[rng.permutation(numel)[:n_zero]] = 0.0
    mask.copy_(torch.tensor(flat.reshape(tuple(mask.shape)), dtype=mask.dtype, device=mask.device))
    return n_zero / numel


def _fixed_point_input(shape, kif=INPUT_KIF):
    """Bounded fixed-point symbolic input so the SAT input quantizer can be replayed."""
    k, i, f = (np.full(shape, v, dtype=np.int8) for v in kif)
    return FVArray.from_kif(k, i, f, HWCONF, 0, None)


def _build_pruned_compressed_model(config, rng):
    """Build the model, build it (one forward), prune 90%, apply final compression, eval."""
    model = TwoBranchNet(config)
    device = next(model.parameters()).device
    img = torch.zeros(1, IN_FEATURES, H, W, device=device)
    seq = torch.zeros(1, IN_FEATURES, SEQ_LEN, device=device)

    with torch.no_grad():
        model(img, seq)  # build quantizers + pruning masks

        pq_layers = [m for m in model.modules() if isinstance(m, (PQConv2d, PQConv1d, PQDense))]
        # Give the kernels a healthy scale so the un-pruned 10% survive quantization.
        for layer in pq_layers:
            layer._weight.copy_(
                torch.tensor(rng.standard_normal(tuple(layer._weight.shape)), dtype=layer._weight.dtype, device=device)
            )
        expected_sparsity = {id(layer): _random_prune(layer, PRUNE_FRACTION, rng) for layer in pq_layers}

    apply_final_compression(model)
    model.eval()
    return model, pq_layers, expected_sparsity, device


def test_alkaid_conversion_pruned_quantized_model():
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model, pq_layers, expected_sparsity, _ = _build_pruned_compressed_model(config, rng)
    assert {type(layer).__name__ for layer in pq_layers} == {"PQConv2d", "PQConv1d", "PQDense"}

    for layer in pq_layers:
        weight = layer._weight.detach().cpu().numpy()
        sparsity = float((weight == 0).mean())
        assert bool(layer.final_compression_done)
        # Quantization can only add zeros, so realized sparsity >= the masked fraction.
        assert sparsity >= expected_sparsity[id(layer)] - 1e-9
        assert sparsity >= 0.88  # ~90% pruned
        assert sparsity < 1.0  # some weights survive

    # Convert with Alkaid. fx tracing requires eval mode + explicit inputs.
    inputs = (_fixed_point_input((1, IN_FEATURES, H, W)), _fixed_point_input((1, IN_FEATURES, SEQ_LEN)))
    inp, out = trace_model(model, hwconf=HWCONF, inputs=inputs, framework="torch")

    assert out.shape == (OUT_FEATURES,)
    expected_inputs = IN_FEATURES * H * W + IN_FEATURES * SEQ_LEN
    assert inp.shape == (expected_inputs,)


def test_alkaid_rtl_matches_model(tmp_path):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True

    rng = np.random.default_rng(0)
    model, _, _, device = _build_pruned_compressed_model(config, rng)

    inputs = (_fixed_point_input((1, IN_FEATURES, H, W)), _fixed_point_input((1, IN_FEATURES, SEQ_LEN)))
    inp_fv, out_fv = trace_model(model, hwconf=HWCONF, inputs=inputs, framework="torch")
    # Lower the trace to combinational logic: the pure-Python interpreter `comb(...)`
    # is the exact software model the RTL is generated from.
    comb = trace(inp_fv, out_fv, optimize=True)

    # Sample inputs exactly representable in INPUT_KIF (non-negative multiples of
    # 2**-4) so the input-port quantization is a no-op and model == hardware input.
    n_samples = 16
    img = rng.integers(0, 16, size=(n_samples, IN_FEATURES, H, W)).astype("float32") / 16.0
    seq = rng.integers(0, 16, size=(n_samples, IN_FEATURES, SEQ_LEN)).astype("float32") / 16.0

    with torch.no_grad():
        reference = (
            model(torch.tensor(img, device=device), torch.tensor(seq, device=device)).cpu().numpy().astype(np.float64)
        )  # (n_samples, OUT_FEATURES)

    emulated = np.stack(
        [
            np.asarray(comb(np.concatenate([img[n].ravel(), seq[n].ravel()]), quantize=True), dtype=np.float64)
            for n in range(n_samples)
        ]
    )

    assert np.any(reference != 0)  # the comparison is non-trivial
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)

    # Generate the actual RTL project from the same combinational logic.
    RTLModel(comb, str(tmp_path), "model", flavor="verilog", print_latency=False).write()
    assert (tmp_path / "src" / "model.v").exists()


# --- Coverage of every PQ layer the torch Alkaid plugin handles ---------------

ALL_C = 4
ALL_H = ALL_W = 6
# conv1d branch is pooled to ALL_LIN/2; matches the conv2d branch flatten length
# (ALL_C * (ALL_H//2) * (ALL_W//2)) when ALL_LIN//2 == (ALL_H//2) * (ALL_W//2).
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
    """Exercises every PQ layer type the torch Alkaid plugin handles.

    conv2d -> batchnorm2d -> relu -> avgpool2d branch, and a
    conv1d -> batchnorm1d -> relu -> avgpool1d branch, merged (matched flatten
    lengths) into a dense head. Each layer also drives an inner Quantizer.
    """

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

    # Build with random input (in train mode) so batchnorm running stats are sane.
    model.train()
    with torch.no_grad():
        model(
            torch.tensor(rng.standard_normal((4, IN_FEATURES, ALL_H, ALL_W)), dtype=torch.float32, device=device),
            torch.tensor(rng.standard_normal((4, IN_FEATURES, ALL_LIN)), dtype=torch.float32, device=device),
        )

    assert ALL_TORCH_LAYER_TYPES <= {type(m).__name__ for m in model.modules()}

    # Randomize + prune 90% of every prunable layer (conv/dense; batchnorm/pool aren't pruned).
    with torch.no_grad():
        for layer in [m for m in model.modules() if getattr(m, "pruning_layer", None) is not None]:
            layer._weight.copy_(
                torch.tensor(rng.standard_normal(tuple(layer._weight.shape)), dtype=layer._weight.dtype, device=device)
            )
            _random_prune(layer, PRUNE_FRACTION, rng)

    apply_final_compression(model)
    model.eval()

    inputs = (_fixed_point_input((1, IN_FEATURES, ALL_H, ALL_W)), _fixed_point_input((1, IN_FEATURES, ALL_LIN)))
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
    emulated = np.stack(
        [
            np.asarray(comb(np.concatenate([img[n].ravel(), seq[n].ravel()]), quantize=True), dtype=np.float64)
            for n in range(n_samples)
        ]
    )

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)

    RTLModel(comb, str(tmp_path), "model", flavor="verilog", print_latency=False).write()
    assert (tmp_path / "src" / "model.v").exists()


# --- Per-layer conversion: a model that is a single layer ---------------------


def _data_quantizer(config):
    """A data Quantizer built from the config's default data settings."""
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


class _SingleLayer(nn.Module):
    """One PQ layer, optionally followed by a Quantizer.

    Layers with a ``quantize_output`` option set it directly; layers without one
    (batchnorm) get an explicit trailing Quantizer so the output is fixed-point.
    """

    def __init__(self, layer, tail=None):
        super().__init__()
        self.layer = layer
        self.tail = tail

    def forward(self, x):
        x = self.layer(x)
        return x if self.tail is None else self.tail(x)


# id -> lambda(config) -> (input shape incl. batch=1, single-layer module)
_SINGLE_LAYER_CASES = {
    "conv2d": lambda c: ((1, 2, 4, 4), _SingleLayer(PQConv2d(c, 2, 3, KERNEL_SIZE, padding="same", quantize_output=True))),
    "conv1d": lambda c: ((1, 2, 8), _SingleLayer(PQConv1d(c, 2, 3, KERNEL_SIZE, padding="same", quantize_output=True))),
    "dense": lambda c: ((1, 6), _SingleLayer(PQDense(c, 6, OUT_FEATURES, quantize_output=True))),
    "batchnorm2d": lambda c: ((1, 3, 4, 4), _SingleLayer(PQBatchNorm2d(c, 3), _data_quantizer(c))),
    "batchnorm1d": lambda c: ((1, 3, 8), _SingleLayer(PQBatchNorm1d(c, 3), _data_quantizer(c))),
    "avgpool2d": lambda c: ((1, 3, 4, 4), _SingleLayer(PQAvgPool2d(c, kernel_size=2, stride=2, quantize_output=True))),
    "avgpool1d": lambda c: ((1, 3, 8), _SingleLayer(PQAvgPool1d(c, kernel_size=2, stride=2, quantize_output=True))),
    "activation": lambda c: ((1, 6), _SingleLayer(PQActivation(c, "relu", quantize_input=True, quantize_output=True))),
    "quantizer": lambda c: ((1, 6), _SingleLayer(_data_quantizer(c))),
}


@pytest.mark.parametrize("case_id", list(_SINGLE_LAYER_CASES))
def test_alkaid_single_layer(case_id):
    config = pdp_config()
    config.quantization_parameters.enable_quantization = True
    shape, model = _SINGLE_LAYER_CASES[case_id](config)
    rng = np.random.default_rng(0)

    # Build (random input, train mode so any batchnorm running stats are sane).
    # Tensors use torch's default device (conftest sets it), matching the model.
    model.train()
    with torch.no_grad():
        model(torch.tensor(rng.standard_normal((4,) + shape[1:]), dtype=torch.float32))
    apply_final_compression(model)
    model.eval()

    inp_fv, out_fv = trace_model(model, hwconf=HWCONF, inputs=(_fixed_point_input(shape),), framework="torch")
    comb = trace(inp_fv, out_fv, optimize=True)

    n_samples = 16
    x = rng.integers(0, 16, size=(n_samples,) + shape[1:]).astype("float32") / 16.0
    with torch.no_grad():
        reference = model(torch.tensor(x)).cpu().numpy().reshape(n_samples, -1).astype(np.float64)
    emulated = np.stack([np.asarray(comb(x[i].ravel(), quantize=True), dtype=np.float64) for i in range(n_samples)])

    assert np.any(reference != 0)
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-9)
