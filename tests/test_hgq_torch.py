"""
Parity tests: PyTorch `HGQQuantizer` vs Keras `hgq.Quantizer` (hgq2 library).

Both implementations should produce matching forward outputs, matching
gradients on the fractional-bit parameter `f`, and should follow a similar
training trajectory when fed the same data with the same initial state.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import pytest  # noqa: E402
import torch  # noqa: E402

hgq = pytest.importorskip("hgq")
from hgq.quantizer import Quantizer as KerasHGQQuantizer  # noqa: E402
from hgq.quantizer import QuantizerConfig  # noqa: E402

from pquant.core.torch.hgq_quantizer import HGQQuantizer  # noqa: E402

RTOL = 1e-4
ATOL = 1e-5


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
    "overflow,round_mode,is_data",
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
        f"[{overflow}/{round_mode}/is_data={is_data}] " f"max diff = {(out_torch - out_keras).abs().max().item():.6g}"
    )


def test_forward_matches_keras_training_sat():
    torch.manual_seed(1)
    x = torch.randn(4, 6) * 1.2
    torch_q = HGQQuantizer(k0=1, i0=1, f0=3, overflow_mode="SAT", round_mode="RND", is_data=False)
    keras_q = _make_keras_q(1, 1, 3, "SAT", "RND", is_data=False)

    out_torch = torch_q(x, training=True).detach()
    out_keras = _as_torch(keras_q(x, training=True)).detach()

    assert torch.allclose(out_torch, out_keras, rtol=RTOL, atol=ATOL), (
        f"training forward diverged, max diff = " f"{(out_torch - out_keras).abs().max().item():.6g}"
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
        f"grad f mismatch, max diff = " f"{(grad_f_torch - grad_f_keras).abs().max().item():.6g}"
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
        f"STE grad should be ~1 inside sat range, got max deviation " f"{(x.grad - 1).abs().max().item():.6g}"
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
        if isinstance(val, torch.nn.Parameter) or isinstance(val, torch.Tensor):
            return val
    raise RuntimeError("Could not locate f parameter on Keras hgq quantizer")


def _find_i_param(keras_q):
    inner = getattr(keras_q, "quantizer", keras_q)
    for name in ("_i", "i"):
        p = getattr(inner, name, None)
        if p is None:
            continue
        val = getattr(p, "value", p)
        if isinstance(val, torch.nn.Parameter) or isinstance(val, torch.Tensor):
            return val
    return None
