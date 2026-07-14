"""
Parity tests: PyTorch `HGQQuantizer` vs Keras `hgq.Quantizer` (hgq2 library).

Both implementations should produce matching forward outputs, matching
gradients on the fractional-bit parameter `f`, and should follow a similar
training trajectory when fed the same data with the same initial state.
"""

import pytest
import torch

hgq = pytest.importorskip("hgq")
from hgq.quantizer import Quantizer as KerasHGQQuantizer  # noqa: E402
from hgq.quantizer import QuantizerConfig  # noqa: E402

from pquant.core.torch.hgq_quantizer import HGQQuantizer  # noqa: E402

RTOL = 1e-4
ATOL = 1e-5

# HGQ supports only per_tensor and per_weight; per_channel is rejected.
GRANULARITIES = ["per_tensor", "per_weight"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keras_q(k, i, f, overflow, round_mode, is_data, place="datalane"):
    """Construct the Keras hgq2 quantizer mirroring create_hgq_*_quantizer()."""
    homogeneous_axis = (0,) if is_data else ()
    cfg = QuantizerConfig(
        q_type="kif",
        place="datalane" if is_data else place,
        k0=k,
        i0=i,
        f0=f,
        overflow_mode=overflow,
        round_mode=round_mode,
        homogeneous_axis=homogeneous_axis,
    )
    return KerasHGQQuantizer(config=cfg)


def _as_torch(x):
    """Unwrap a keras tensor produced with the torch backend to a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x
    import keras.ops as ops

    return ops.convert_to_tensor(x)


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overflow", "round_mode", "is_data"),
    [
        ("SAT", "RND", False),
        ("SAT", "RND_CONV", False),
        ("SAT_SYM", "RND", False),
        ("WRAP", "RND", False),
        ("WRAP", "RND", True),
        ("SAT", "RND", True),
    ],
)
def test_forward_matches_keras(overflow, round_mode, is_data):
    torch.manual_seed(0)
    k, i, f = 1, 2, 4
    x = torch.randn(8, 5) * 0.8

    torch_q = HGQQuantizer(
        k0=k,
        i0=i,
        f0=f,
        overflow_mode=overflow,
        round_mode=round_mode,
        is_data=is_data,
    )
    keras_q = _make_keras_q(k, i, f, overflow, round_mode, is_data)

    out_torch = torch_q(x, training=False).detach()
    # Keras quantizer: call with training=False
    out_keras = _as_torch(keras_q(x, training=False)).detach()

    assert out_torch.shape == out_keras.shape, f"shape mismatch: {out_torch.shape} vs {out_keras.shape}"
    assert torch.allclose(out_torch, out_keras, rtol=RTOL, atol=ATOL), (
        f"[{overflow}/{round_mode}/is_data={is_data}] max diff = {(out_torch - out_keras).abs().max().item():.6g}"
    )


def test_forward_matches_keras_training_sat():
    torch.manual_seed(1)
    x = torch.randn(4, 6) * 1.2
    torch_q = HGQQuantizer(k0=1, i0=1, f0=3, overflow_mode="SAT", round_mode="RND", is_data=False)
    keras_q = _make_keras_q(1, 1, 3, "SAT", "RND", is_data=False)

    out_torch = torch_q(x, training=True).detach()
    out_keras = _as_torch(keras_q(x, training=True)).detach()

    assert torch.allclose(out_torch, out_keras, rtol=RTOL, atol=ATOL), (
        f"training forward diverged, max diff = {(out_torch - out_keras).abs().max().item():.6g}"
    )


# ---------------------------------------------------------------------------
# Backward parity — gradient through f
# ---------------------------------------------------------------------------


def test_backward_f_gradient_sat():
    torch.manual_seed(2)
    x_data = torch.randn(4, 5) * 0.6

    torch_q = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="SAT", round_mode="RND", is_data=False)
    keras_q = _make_keras_q(1, 2, 4, "SAT", "RND", is_data=False)

    # Forward to trigger build for both
    x_t = x_data.clone().requires_grad_(True)
    out_t = torch_q(x_t, training=True)
    loss_t = out_t.pow(2).sum()
    loss_t.backward()
    grad_f_torch = torch_q._f.grad.detach().clone()

    x_k = x_data.clone().requires_grad_(True)
    out_k = _as_torch(keras_q(x_k, training=True))
    loss_k = out_k.pow(2).sum()
    loss_k.backward()

    # hgq2 stores the f parameter as keras_q.quantizer._f — access via state
    keras_f = _find_f_param(keras_q)
    grad_f_keras = keras_f.grad.detach().clone() if keras_f.grad is not None else None

    assert grad_f_keras is not None, "Keras hgq quantizer produced no gradient on _f"
    assert grad_f_torch.shape == grad_f_keras.shape, f"grad f shape mismatch: {grad_f_torch.shape} vs {grad_f_keras.shape}"
    # Gradient direction/magnitude should match up to STE discretisation noise.
    assert torch.allclose(grad_f_torch, grad_f_keras, rtol=1e-3, atol=1e-5), (
        f"grad f mismatch, max diff = {(grad_f_torch - grad_f_keras).abs().max().item():.6g}"
    )


def test_backward_input_gradient_ste():
    """STE: gradient w.r.t. input should be approximately identity within sat range."""
    torch.manual_seed(3)
    x = (torch.randn(3, 4) * 0.3).requires_grad_(True)  # small → within sat bounds

    torch_q = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="SAT", round_mode="RND", is_data=False)
    out = torch_q(x, training=True)
    out.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-5), (
        f"STE grad should be ~1 inside sat range, got max deviation {(x.grad - 1).abs().max().item():.6g}"
    )


# ---------------------------------------------------------------------------
# Training trajectory parity
# ---------------------------------------------------------------------------


def test_training_trajectory_matches():
    """Run several SGD steps on both implementations with same data/init. The
    learned f parameter should stay close (exact equality not guaranteed due to
    STE non-determinism in rounding ties, but trajectories should track)."""
    torch.manual_seed(4)
    k, i, f = 1, 2, 4
    lr = 0.05
    steps = 20

    torch_q = HGQQuantizer(k0=k, i0=i, f0=f, overflow_mode="SAT", round_mode="RND", is_data=False)
    keras_q = _make_keras_q(k, i, f, "SAT", "RND", is_data=False)

    # Prime build with a dummy pass
    dummy = torch.zeros(4, 3)
    torch_q(dummy, training=False)
    keras_q(dummy, training=False)

    # Optimizer: plain SGD on f (and i for SAT) for both
    torch_opt = torch.optim.SGD([p for p in torch_q.parameters() if p.requires_grad], lr=lr)
    keras_f = _find_f_param(keras_q)
    keras_i = _find_i_param(keras_q)
    keras_params = [p for p in (keras_f, keras_i) if p is not None and p.requires_grad]
    keras_opt = torch.optim.SGD(keras_params, lr=lr)

    for step in range(steps):
        torch.manual_seed(100 + step)
        x = torch.randn(4, 3) * 0.5

        torch_opt.zero_grad()
        loss_t = torch_q(x, training=True).pow(2).sum()
        loss_t.backward()
        torch_opt.step()

        keras_opt.zero_grad()
        loss_k = _as_torch(keras_q(x, training=True)).pow(2).sum()
        loss_k.backward()
        keras_opt.step()

    # After training, the learned f should match within reasonable tolerance.
    diff_f = (torch_q._f.detach() - keras_f.detach()).abs().max().item()
    assert diff_f < 0.25, f"f parameters diverged after {steps} steps: max diff = {diff_f:.4f}"

    # Forward outputs on a held-out batch should also be close.
    torch.manual_seed(999)
    x_test = torch.randn(4, 3) * 0.5
    out_t = torch_q(x_test, training=False).detach()
    out_k = _as_torch(keras_q(x_test, training=False)).detach()
    diff_out = (out_t - out_k).abs().max().item()
    assert diff_out < 0.1, f"trained outputs diverged: max diff = {diff_out:.4f}"


# ---------------------------------------------------------------------------
# Granularity → bit-width tensor shape
#
# per_tensor : one shared bit-width        -> shape collapses to all-ones
# per_weight : one per element             -> shape matches the tensor
#              (data always shares the batch axis, so axis 0 collapses to 1)
# per_channel is not supported via granularity (see axis-override tests).
# ---------------------------------------------------------------------------


def _is_single_value(shape):
    return int(torch.tensor(shape).prod().item()) == 1


def _build_hgq(shape, is_data, granularity):
    q = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="SAT", round_mode="RND", is_data=is_data, granularity=granularity)
    q.build(shape)
    return tuple(q.i.shape), tuple(q.f.shape)


# weight layout (output-channel first): Linear (out, in), Conv1d (out, in, k), Conv2d (out, in, kH, kW)
WEIGHT_SHAPES = [(32, 16), (32, 16, 3), (32, 16, 3, 3)]
# data layout (channels-first): (batch, channels, *spatial)
DATA_SHAPES = [(4, 32), (4, 32, 6), (4, 32, 6, 6)]


@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
def test_weight_granularity_shape(granularity, shape):
    for bw_shape in _build_hgq(shape, is_data=False, granularity=granularity):
        if granularity == "per_tensor":
            assert _is_single_value(bw_shape), f"expected single value, got {bw_shape}"
        else:  # per_weight
            assert bw_shape == shape, f"expected {shape}, got {bw_shape}"


@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize("shape", DATA_SHAPES)
def test_data_granularity_shape(granularity, shape):
    for bw_shape in _build_hgq(shape, is_data=True, granularity=granularity):
        if granularity == "per_tensor":
            assert _is_single_value(bw_shape), f"expected single value, got {bw_shape}"
        else:  # per_weight: batch axis shared, rest per-element
            assert bw_shape == (1, *shape[1:]), f"expected batch-collapsed, got {bw_shape}"


@pytest.mark.parametrize("is_data", [False, True])
def test_per_channel_rejected_for_hgq(is_data):
    # per_channel is not a valid HGQ granularity; building must raise.
    q = HGQQuantizer(k0=1, i0=2, f0=4, overflow_mode="SAT", round_mode="RND", is_data=is_data, granularity="per_channel")
    with pytest.raises(ValueError, match="per_channel"):
        q.build((32, 16, 3, 3))


# ---------------------------------------------------------------------------
# Per-quantizer granularity override
# ---------------------------------------------------------------------------


def test_per_quantizer_granularity_override():
    from pquant.core.torch.layers import PQDense

    # input per_tensor, everything else per_weight
    layer = PQDense(
        _hgq_config("per_weight"),
        in_features=16,
        out_features=32,
        quantize_output=True,
        in_quant_granularity="per_tensor",
    )
    layer(torch.randn(4, 16))
    _, wi, _ = layer.get_weight_quantization_bits()
    assert tuple(wi.shape) == (32, 16)  # weight follows config per_weight
    _, ii, _ = layer.get_input_quantization_bits()
    assert _is_single_value(tuple(ii.shape))  # input overridden to per_tensor
    _, oi, _ = layer.get_output_quantization_bits()
    assert tuple(oi.shape) == (1, 32)  # output keeps config per_weight (batch-collapsed)


def test_per_quantizer_granularity_defaults_to_config():
    from pquant.core.torch.layers import PQDense

    layer = PQDense(_hgq_config("per_tensor"), in_features=16, out_features=32, weight_quant_granularity="per_weight")
    layer(torch.randn(4, 16))
    _, wi, _ = layer.get_weight_quantization_bits()
    assert tuple(wi.shape) == (32, 16)  # weight overridden to per_weight
    _, ii, _ = layer.get_input_quantization_bits()
    assert _is_single_value(tuple(ii.shape))  # input uses config per_tensor


# ---------------------------------------------------------------------------
# Per-quantizer granularity override on auxiliary (boundary) layers.
# A model can start/end with a BatchNorm / LayerNorm / AvgPool whose input/output
# quantizer is effectively the model's I/O quantizer, so that must be overridable.
# ---------------------------------------------------------------------------


def test_batchnorm_input_granularity_override():
    from pquant.core.torch.layers import PQBatchNorm1d

    # config per_weight, but this BN sits on the model boundary -> input per_tensor
    layer = PQBatchNorm1d(_hgq_config("per_weight"), num_features=16, in_quant_granularity="per_tensor")
    layer(torch.randn(4, 16))
    _, ii, _ = layer.get_input_quantization_bits()
    assert _is_single_value(tuple(ii.shape))  # input overridden to per_tensor
    _, wi, _ = layer.get_weight_quantization_bits()
    assert tuple(wi.shape) == (16,)  # weight follows config per_weight


def test_layernorm_io_granularity_override():
    from pquant.core.torch.layers import PQLayerNorm

    layer = PQLayerNorm(
        _hgq_config("per_weight"),
        normalized_shape=16,
        quantize_output=True,
        in_quant_granularity="per_tensor",
        out_quant_granularity="per_tensor",
    )
    layer(torch.randn(4, 16))
    _, ii, _ = layer.get_input_quantization_bits()
    assert _is_single_value(tuple(ii.shape))  # input overridden to per_tensor
    _, oi, _ = layer.get_output_quantization_bits()
    assert _is_single_value(tuple(oi.shape))  # output overridden to per_tensor
    _, wi, _ = layer.get_weight_quantization_bits()
    assert tuple(wi.shape) == (16,)  # weight follows config per_weight


def test_avgpool_io_granularity_override():
    from pquant.core.torch.layers import PQAvgPool1d

    layer = PQAvgPool1d(
        _hgq_config("per_weight"),
        kernel_size=2,
        quantize_output=True,
        in_quant_granularity="per_tensor",
        out_quant_granularity="per_tensor",
    )
    layer(torch.randn(4, 16, 8))
    _, ii, _ = layer.get_input_quantization_bits()
    assert _is_single_value(tuple(ii.shape))  # input overridden to per_tensor
    _, oi, _ = layer.get_output_quantization_bits()
    assert _is_single_value(tuple(oi.shape))  # output overridden to per_tensor


def test_mha_io_param_granularity_override():
    from pquant.core.torch.layers import PQMultiheadAttention

    # in/out are the model-boundary granularities; param (weight+bias) is uniform across projections.
    layer = PQMultiheadAttention(
        _hgq_config("per_weight"),
        embed_dim=16,
        num_heads=4,
        batch_first=True,
        quantize_output=True,
        in_quant_granularity="per_tensor",
        out_quant_granularity="per_tensor",
    )
    x = torch.randn(2, 5, 16)
    layer(x, x, x)
    # Q/K/V projection inputs (boundary) overridden to per_tensor
    for proj in (layer.q_proj, layer.k_proj, layer.v_proj):
        _, ii, _ = proj.get_input_quantization_bits()
        assert _is_single_value(tuple(ii.shape))
    # out_proj output (boundary) overridden to per_tensor
    _, oi, _ = layer.out_proj.get_output_quantization_bits()
    assert _is_single_value(tuple(oi.shape))
    # weights follow param granularity = config per_weight
    _, wi, _ = layer.q_proj.get_weight_quantization_bits()
    assert tuple(wi.shape) == (16, 16)
    # out_proj input is internal -> stays config per_weight (batch-collapsed, not single)
    _, oii, _ = layer.out_proj.get_input_quantization_bits()
    assert not _is_single_value(tuple(oii.shape))


def test_mha_qkv_always_output_quantized():
    from pquant.core.torch.layers import PQMultiheadAttention

    # MHA-level quantize_output=False (default): Q/K/V outputs are matmul operands and stay
    # output-quantized; only out_proj follows the MHA-level flag.
    layer = PQMultiheadAttention(_hgq_config("per_tensor"), embed_dim=16, num_heads=4, batch_first=True)
    assert layer.q_proj.quantize_output is True
    assert layer.k_proj.quantize_output is True
    assert layer.v_proj.quantize_output is True
    assert layer.out_proj.quantize_output is False


def test_mha_param_granularity_override():
    from pquant.core.torch.layers import PQMultiheadAttention

    # param overrides weight+bias across all projections; inputs/outputs follow config per_tensor.
    layer = PQMultiheadAttention(
        _hgq_config("per_tensor"),
        embed_dim=16,
        num_heads=4,
        batch_first=True,
        param_quant_granularity="per_weight",
    )
    x = torch.randn(2, 5, 16)
    layer(x, x, x)
    _, wi, _ = layer.out_proj.get_weight_quantization_bits()
    assert tuple(wi.shape) == (16, 16)  # weight overridden to per_weight
    _, ii, _ = layer.q_proj.get_input_quantization_bits()
    assert _is_single_value(tuple(ii.shape))  # input uses config per_tensor


# ---------------------------------------------------------------------------
# Granularity end-to-end through the PQ layers
# ---------------------------------------------------------------------------


def _hgq_config(granularity):
    from pquant import pdp_config

    config = pdp_config()
    config.quantization_parameters.use_high_granularity_quantization = True
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.granularity = granularity
    config.pruning_parameters.enable_pruning = False
    return config


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_layer_weight_granularity(granularity):
    from pquant.core.torch.layers import PQConv1d, PQConv2d, PQDense

    cases = [
        (PQDense(_hgq_config(granularity), in_features=16, out_features=32), torch.randn(4, 16), (32, 16)),
        (PQConv1d(_hgq_config(granularity), 16, 32, kernel_size=3), torch.randn(4, 16, 8), (32, 16, 3)),
        (PQConv2d(_hgq_config(granularity), 16, 32, kernel_size=3), torch.randn(4, 16, 8, 8), (32, 16, 3, 3)),
    ]
    for layer, x, weight_shape in cases:
        layer(x)  # triggers lazy build of the weight quantizer
        _, i, f = layer.get_weight_quantization_bits()
        for bw_shape in (tuple(i.shape), tuple(f.shape)):
            if granularity == "per_tensor":
                assert _is_single_value(bw_shape), f"{type(layer).__name__}: expected single value, got {bw_shape}"
            else:  # per_weight
                assert bw_shape == weight_shape, f"{type(layer).__name__}: expected {weight_shape}, got {bw_shape}"


# ---------------------------------------------------------------------------
# Utility: locate f / i parameters inside the Keras hgq2 quantizer
# ---------------------------------------------------------------------------


def _find_f_param(keras_q):
    """Locate the fractional-bit parameter on the Keras hgq2 quantizer.

    hgq2 stores it as `keras_q.quantizer._f` (Keras Variable, which the torch
    backend exposes as a torch.nn.Parameter).
    """
    inner = getattr(keras_q, "quantizer", keras_q)
    for name in ("_f", "f"):
        p = getattr(inner, name, None)
        if p is None:
            continue
        # Keras Variables expose a torch `.value` parameter under the torch backend
        val = getattr(p, "value", p)
        if isinstance(val, (torch.nn.Parameter, torch.Tensor)):
            return val
    raise RuntimeError("Could not locate f parameter on Keras hgq quantizer")


def _find_i_param(keras_q):
    inner = getattr(keras_q, "quantizer", keras_q)
    for name in ("_i", "i"):
        p = getattr(inner, name, None)
        if p is None:
            continue
        val = getattr(p, "value", p)
        if isinstance(val, (torch.nn.Parameter, torch.Tensor)):
            return val
    return None
