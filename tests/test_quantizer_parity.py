"""
Parity tests: PyTorch `Quantizer` vs Keras `Quantizer` in non-HGQ (fixed-point) mode.

Both implementations must produce matching forward outputs, matching gradients
w.r.t. the quantized tensor, and must freeze their dynamically-computed
bitwidths identically once training ends (``apply_final_compression``).

``per_channel`` granularity needs special handling: keras stores weights
output-channel-last while torch stores them output-channel-first, so the raw
``Quantizer`` reduces over different axes in each backend. A real layer
reconciles this with a transpose before quantizing (keras ``_handle_transpose``);
here we do the same transpose manually (see ``to_channel_last``/``to_channel_first``)
so both backends reduce over the same channel groups before comparing.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from keras import ops  # noqa: E402

from pquant.core.keras.quantizer import Quantizer as KQuantizer  # noqa: E402
from pquant.core.torch.quantizer import Quantizer as TQuantizer  # noqa: E402

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-4


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(ops.convert_to_numpy(x))


def to_channel_last(x):
    """Move torch's channel-first axis (0) to keras's channel-last axis (-1)."""
    return np.moveaxis(x, 0, -1)


def to_channel_first(x):
    """Move keras's channel-last axis (-1) back to torch's channel-first axis (0)."""
    return np.moveaxis(to_numpy(x), -1, 0)


def assert_close(a, b, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE, msg=""):
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    assert a_np.shape == b_np.shape, f"{msg}: shape mismatch: {a_np.shape} vs {b_np.shape}"
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol, err_msg=msg)


def keras_tensor(arr):
    return ops.convert_to_tensor(np.asarray(arr).astype(np.float32))


def torch_tensor(arr, requires_grad=False):
    return torch.as_tensor(np.asarray(arr).astype(np.float32)).requires_grad_(requires_grad)


def reset_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def keras_grad(fn, x, training=True):
    """Gradient of scalar-reducing ``fn(x).sum()`` w.r.t. keras tensor ``x``."""
    xt = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        loss = ops.sum(fn(xt, training=training))
    return tape.gradient(loss, xt)


def make_quantizers(shape, k=1.0, i=2.0, f=2.0, overflow="SAT", round_mode="RND", is_data=False, granularity="per_tensor"):
    k_layer = KQuantizer(
        k=k,
        i=i,
        f=f,
        overflow=overflow,
        round_mode=round_mode,
        is_heterogeneous=False,
        is_data=is_data,
        granularity=granularity,
    )
    k_layer.build(shape)

    t_layer = TQuantizer(
        k=k,
        i=i,
        f=f,
        overflow=overflow,
        round_mode=round_mode,
        is_heterogeneous=False,
        is_data=is_data,
        granularity=granularity,
    )
    return k_layer, t_layer


@pytest.mark.parametrize(
    "shape,granularity,is_data",
    [
        ((8, 4), "per_tensor", False),
        ((8, 4), "per_weight", False),
        ((4, 8, 3, 3), "per_weight", False),
        ((16,), "per_tensor", True),
    ],
)
def test_quantizer_matches_keras(shape, granularity, is_data):
    reset_seed()
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)

    k_layer, t_layer = make_quantizers(shape, is_data=is_data, granularity=granularity)

    k_x = keras_tensor(x_np)
    t_x = torch_tensor(x_np, requires_grad=True)
    k_out = k_layer(k_x, training=True)
    t_out = t_layer(t_x)
    assert_close(k_out, t_out, msg=f"Quantizer forward ({granularity}, is_data={is_data})")
    assert_close(k_layer.i, t_layer.i, msg=f"Quantizer i ({granularity}, is_data={is_data})")
    assert_close(k_layer.f, t_layer.f, msg=f"Quantizer f ({granularity}, is_data={is_data})")

    k_grad = keras_grad(k_layer, x_np, training=True)
    t_out.sum().backward()
    assert_close(k_grad, t_x.grad, msg=f"Quantizer backward ({granularity}, is_data={is_data})")


@pytest.mark.parametrize("shape", [(8, 4), (4, 8, 3, 3)])
def test_quantizer_per_channel_matches_keras(shape):
    reset_seed()
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)
    x_np_channel_last = to_channel_last(x_np)

    k_layer, t_layer = make_quantizers(x_np_channel_last.shape, is_data=False, granularity="per_channel")

    k_x = keras_tensor(x_np_channel_last)
    t_x = torch_tensor(x_np, requires_grad=True)
    k_out = k_layer(k_x, training=True)
    t_out = t_layer(t_x)
    assert_close(to_channel_first(k_out), t_out, msg=f"Quantizer per_channel forward {shape}")
    assert_close(to_channel_first(k_layer.i), t_layer.i, msg=f"Quantizer per_channel i {shape}")
    assert_close(to_channel_first(k_layer.f), t_layer.f, msg=f"Quantizer per_channel f {shape}")

    k_grad = keras_grad(k_layer, x_np_channel_last, training=True)
    t_out.sum().backward()
    assert_close(to_channel_first(k_grad), t_x.grad, msg=f"Quantizer per_channel backward {shape}")


@pytest.mark.parametrize("shape", [(8, 4), (4, 8, 3, 3)])
def test_quantizer_per_channel_freezes_bits_after_final_compression(shape):
    reset_seed()
    train_np = (np.random.randn(*shape) * 2.0).astype(np.float32)
    eval_np = (np.random.randn(*shape) * 500.0).astype(np.float32)
    train_np_channel_last = to_channel_last(train_np)
    eval_np_channel_last = to_channel_last(eval_np)

    k_layer, t_layer = make_quantizers(train_np_channel_last.shape, is_data=False, granularity="per_channel")

    k_layer(keras_tensor(train_np_channel_last), training=True)
    t_layer.train()
    t_layer(torch_tensor(train_np))

    k_layer.apply_final_compression()
    t_layer.apply_final_compression()
    assert_close(to_channel_first(k_layer.i), t_layer.i, msg=f"per_channel i after apply_final_compression {shape}")
    assert_close(to_channel_first(k_layer.f), t_layer.f, msg=f"per_channel f after apply_final_compression {shape}")

    frozen_k_i, frozen_k_f = to_numpy(k_layer.i).copy(), to_numpy(k_layer.f).copy()

    k_layer(keras_tensor(eval_np_channel_last), training=False)
    t_layer.eval()
    t_layer(torch_tensor(eval_np))

    assert_close(k_layer.i, frozen_k_i, msg=f"keras per_channel i drifted after eval forward {shape}")
    assert_close(k_layer.f, frozen_k_f, msg=f"keras per_channel f drifted after eval forward {shape}")
    assert_close(t_layer.i, to_channel_first(frozen_k_i), msg=f"torch per_channel i drifted after eval forward {shape}")
    assert_close(t_layer.f, to_channel_first(frozen_k_f), msg=f"torch per_channel f drifted after eval forward {shape}")


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel", "per_weight"])
def test_is_data_ignores_granularity(granularity):
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)

    k_layer, t_layer = make_quantizers(shape, is_data=True, granularity=granularity)

    k_out = k_layer(keras_tensor(x_np), training=True)
    t_out = t_layer(torch_tensor(x_np))

    assert to_numpy(k_layer.i).size == 1, f"keras is_data i should stay scalar for granularity={granularity}"
    assert to_numpy(t_layer.i).size == 1, f"torch is_data i should stay scalar for granularity={granularity}"
    assert_close(k_out, t_out, msg=f"Quantizer forward (is_data=True, granularity={granularity})")


@pytest.mark.parametrize("overflow", ["SAT", "SAT_SYM", "WRAP", "WRAP_SM"])
def test_quantizer_matches_keras_overflow_modes(overflow):
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)

    k_layer, t_layer = make_quantizers(shape, overflow=overflow, is_data=False, granularity="per_tensor")

    k_x = keras_tensor(x_np)
    t_x = torch_tensor(x_np, requires_grad=True)
    k_out = k_layer(k_x, training=True)
    t_out = t_layer(t_x)
    assert_close(k_out, t_out, msg=f"Quantizer forward (overflow={overflow})")

    k_grad = keras_grad(k_layer, x_np, training=True)
    t_out.sum().backward()
    assert_close(k_grad, t_x.grad, msg=f"Quantizer backward (overflow={overflow})")


def test_quantizer_matches_keras_wrap_eval_saturates():
    """WRAP overflow only saturates when training=False -- exercise that branch specifically."""
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 50.0).astype(np.float32)  # large enough to trigger saturation

    k_layer, t_layer = make_quantizers(shape, overflow="WRAP", is_data=False, granularity="per_tensor")

    k_out = k_layer(keras_tensor(x_np), training=False)
    t_layer.eval()
    t_out = t_layer(torch_tensor(x_np))
    assert_close(k_out, t_out, msg="Quantizer WRAP eval-mode saturate branch")


@pytest.mark.parametrize(
    "round_mode",
    ["RND", "TRN", "RND_CONV", "TRN_ZERO", "RND_ZERO", "RND_MIN_INF", "RND_INF"],
)
def test_quantizer_matches_keras_round_modes(round_mode):
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)

    k_layer, t_layer = make_quantizers(shape, round_mode=round_mode, is_data=False, granularity="per_tensor")

    k_x = keras_tensor(x_np)
    t_x = torch_tensor(x_np, requires_grad=True)
    k_out = k_layer(k_x, training=True)
    t_out = t_layer(t_x)
    assert_close(k_out, t_out, msg=f"Quantizer forward (round_mode={round_mode})")

    k_grad = keras_grad(k_layer, x_np, training=True)
    t_out.sum().backward()
    assert_close(k_grad, t_x.grad, msg=f"Quantizer backward (round_mode={round_mode})")


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_quantizer_matches_keras_k_values(k):
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)
    if k == 0.0:
        x_np = np.abs(x_np)

    k_layer, t_layer = make_quantizers(shape, k=k, is_data=False, granularity="per_tensor")

    k_x = keras_tensor(x_np)
    t_x = torch_tensor(x_np, requires_grad=True)
    k_out = k_layer(k_x, training=True)
    t_out = t_layer(t_x)
    assert_close(k_out, t_out, msg=f"Quantizer forward (k={k})")

    k_grad = keras_grad(k_layer, x_np, training=True)
    t_out.sum().backward()
    assert_close(k_grad, t_x.grad, msg=f"Quantizer backward (k={k})")


def test_get_total_bits_matches_keras_non_hgq():
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)

    k_layer, t_layer = make_quantizers(shape, is_data=False, granularity="per_tensor")
    k_layer(keras_tensor(x_np), training=True)
    t_layer(torch_tensor(x_np))

    assert_close(k_layer.get_total_bits(shape), t_layer.get_total_bits(shape), msg="get_total_bits (non-HGQ)")


def test_get_total_bits_matches_keras_hgq():
    reset_seed()
    shape = (8, 4)
    x_np = (np.random.randn(*shape) * 2.0).astype(np.float32)

    k_layer = KQuantizer(
        k=1.0,
        i=2.0,
        f=2.0,
        overflow="SAT",
        round_mode="RND",
        is_heterogeneous=True,
        is_data=False,
        granularity="per_tensor",
        place="weight",
    )
    k_layer.build(shape)
    t_layer = TQuantizer(
        k=1.0,
        i=2.0,
        f=2.0,
        overflow="SAT",
        round_mode="RND",
        is_heterogeneous=True,
        is_data=False,
        granularity="per_tensor",
        place="weight",
    )

    k_layer(keras_tensor(x_np), training=True)
    t_layer(torch_tensor(x_np))

    assert_close(k_layer.get_total_bits(shape), t_layer.get_total_bits(shape), msg="get_total_bits (HGQ)")


@pytest.mark.parametrize(
    "shape,granularity",
    [
        ((8, 4), "per_weight"),
        ((4, 8, 3, 3), "per_weight"),
    ],
)
def test_quantizer_freezes_bits_after_final_compression(shape, granularity):
    reset_seed()
    train_np = (np.random.randn(*shape) * 2.0).astype(np.float32)
    eval_np = (np.random.randn(*shape) * 500.0).astype(np.float32)

    k_layer, t_layer = make_quantizers(shape, is_data=False, granularity=granularity)

    k_layer(keras_tensor(train_np), training=True)
    t_layer.train()
    t_layer(torch_tensor(train_np))

    k_layer.apply_final_compression()
    t_layer.apply_final_compression()
    assert_close(k_layer.i, t_layer.i, msg=f"i after apply_final_compression ({granularity})")
    assert_close(k_layer.f, t_layer.f, msg=f"f after apply_final_compression ({granularity})")

    frozen_k_i, frozen_k_f = to_numpy(k_layer.i).copy(), to_numpy(k_layer.f).copy()

    k_layer(keras_tensor(eval_np), training=False)
    t_layer.eval()
    t_layer(torch_tensor(eval_np))

    assert_close(k_layer.i, frozen_k_i, msg=f"keras i drifted after eval forward ({granularity})")
    assert_close(k_layer.f, frozen_k_f, msg=f"keras f drifted after eval forward ({granularity})")
    assert_close(t_layer.i, frozen_k_i, msg=f"torch i drifted after eval forward ({granularity})")
    assert_close(t_layer.f, frozen_k_f, msg=f"torch f drifted after eval forward ({granularity})")
