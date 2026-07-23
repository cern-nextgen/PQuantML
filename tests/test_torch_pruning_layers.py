"""
Parity tests: PyTorch pruning methods vs Keras reference.

For each pruning layer (Activation, PDP, CS, DST, Wanda, AutoSparse, MDMM)
the torch implementation must produce numerically matching outputs and
masks when given identical state and inputs as the keras version. Metric
functions (Structured/Unstructured sparsity) are also compared directly.
"""

import os

# The keras layers are expected to run on the tensorflow backend (the long-term
# target for the keras implementation). Run this test file with
# ``KERAS_BACKEND=tensorflow`` so the keras forward paths (including those
# routed through ``ops.custom_gradient``) exercise their intended backend.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from keras import ops  # noqa: E402

from pquant.core.keras.pruning_methods.activation_pruning import (  # noqa: E402
    ActivationPruning as KerasActivationPruning,
)
from pquant.core.keras.pruning_methods.autosparse import (  # noqa: E402
    AutoSparse as KerasAutoSparse,
)
from pquant.core.keras.pruning_methods.cs import (  # noqa: E402
    ContinuousSparsification as KerasContinuousSparsification,
)
from pquant.core.keras.pruning_methods.dst import DST as KerasDST  # noqa: E402
from pquant.core.keras.pruning_methods.mdmm import MDMM as KerasMDMM  # noqa: E402
from pquant.core.keras.pruning_methods.metric_functions import (  # noqa: E402
    StructuredSparsityMetric as KerasStructuredSparsityMetric,
)
from pquant.core.keras.pruning_methods.metric_functions import (  # noqa: E402
    UnstructuredSparsityMetric as KerasUnstructuredSparsityMetric,
)
from pquant.core.keras.pruning_methods.pdp import PDP as KerasPDP  # noqa: E402
from pquant.core.keras.pruning_methods.wanda import Wanda as KerasWanda  # noqa: E402
from pquant.core.torch.pruning_methods.activation_pruning import (  # noqa: E402
    ActivationPruning as TorchActivationPruning,
)
from pquant.core.torch.pruning_methods.autosparse import (  # noqa: E402
    AutoSparse as TorchAutoSparse,
)
from pquant.core.torch.pruning_methods.cs import (  # noqa: E402
    ContinuousSparsification as TorchContinuousSparsification,
)
from pquant.core.torch.pruning_methods.dst import DST as TorchDST  # noqa: E402
from pquant.core.torch.pruning_methods.mdmm import MDMM as TorchMDMM  # noqa: E402
from pquant.core.torch.pruning_methods.metric_functions import (  # noqa: E402
    StructuredSparsityMetric as TorchStructuredSparsityMetric,
)
from pquant.core.torch.pruning_methods.metric_functions import (  # noqa: E402
    UnstructuredSparsityMetric as TorchUnstructuredSparsityMetric,
)
from pquant.core.torch.pruning_methods.pdp import PDP as TorchPDP  # noqa: E402
from pquant.core.torch.pruning_methods.wanda import Wanda as TorchWanda  # noqa: E402

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-4


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(ops.convert_to_numpy(x))


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


def keras_grad(fn, x):
    """Gradient of scalar-reducing ``fn(x).sum()`` w.r.t. keras tensor ``x``."""
    import keras

    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        xt = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(xt)
            loss = ops.sum(fn(xt))
        return tape.gradient(loss, xt)
    raise RuntimeError("keras_grad only supports the tensorflow backend")


def ap_config():
    return {
        "pruning_parameters": {
            "pruning_method": "activation_pruning",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "threshold": 0.3,
            "t_start_collecting_batch": 0,
            "threshold_decay": 0.0,
            "t_delta": 2,
        },
    }


@pytest.mark.parametrize(
    "layer_type,shape",
    [
        ("linear", (16, 8)),
        ("conv", (16, 8, 3, 3)),
    ],
)
def test_activation_pruning_matches_keras(layer_type, shape):
    cfg = ap_config()
    out_channels = shape[0]
    batch = 32

    reset_seed()
    weight_np = np.random.randn(*shape).astype(np.float32)
    # Construct outputs with distinct per-channel activity levels so the
    # resulting mask is non-trivial (some channels pct_active > threshold,
    # some below). linspace from -0.5 to 1 puts roughly 1/3 of the channels
    # below zero, guaranteeing those get pruned while the rest survive.
    per_channel = np.linspace(-0.5, 1.0, num=out_channels, dtype=np.float32)
    np.random.shuffle(per_channel)
    if layer_type == "linear":
        output_np = np.tile(per_channel[None, :], (batch, 1))
    else:
        output_np = np.tile(per_channel[None, :, None, None], (batch, 1, 4, 4))

    k_layer = KerasActivationPruning(cfg, layer_type)
    k_layer.build(shape)
    k_layer.post_pre_train_function()

    t_layer = TorchActivationPruning(cfg, layer_type)
    t_layer.build(shape)
    t_layer.post_pre_train_function()

    for _ in range(cfg["pruning_parameters"]["t_delta"]):
        k_layer.collect_output(keras_tensor(output_np), training=True)
        t_layer.collect_output(torch_tensor(output_np), training=True)

    k_layer.post_epoch_function(0, 1)
    t_layer.post_epoch_function(0, 1)

    assert_close(k_layer.mask, t_layer.mask, msg=f"AP mask ({layer_type})")

    # Sanity check: the constructed per-channel outputs put ~1/3 of the
    # channels at non-positive values, so their pct_active == 0 falls below
    # the 0.3 threshold and they should be pruned. Channels with a positive
    # value hit pct_active == 1 and survive. The expected pruned count is
    # deterministic from per_channel — verifying we actually exercise both
    # branches rather than matching a trivial all-ones mask.
    mask_np = to_numpy(t_layer.mask)
    pruned_fraction = float((mask_np == 0).sum()) / mask_np.size
    # linspace(-0.5, 1.0, 16) → 6 values <= 0 → 6/16 = 0.375 pruned channels.
    assert pruned_fraction == pytest.approx(0.375), f"AP mask ({layer_type}) pruned fraction {pruned_fraction} != 0.375"

    k_weight = keras_tensor(weight_np)
    t_weight = torch_tensor(weight_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"AP forward ({layer_type})")

    k_grad = keras_grad(k_layer, weight_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"AP backward ({layer_type})")


def pdp_config(sparsity=0.75, structured=False):
    return {
        "pruning_parameters": {
            "pruning_method": "pdp",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "epsilon": 1.0,
            "sparsity": sparsity,
            "temperature": 1e-5,
            "threshold_decay": 0.0,
            "structured_pruning": structured,
        },
    }


@pytest.mark.parametrize(
    "layer_type,shape,structured",
    [
        ("linear", (16, 8), False),
        ("linear", (16, 8), True),
        ("conv", (16, 8, 3, 3), False),
        ("conv", (16, 8, 3, 3), True),
    ],
)
def test_pdp_matches_keras(layer_type, shape, structured):
    cfg = pdp_config(structured=structured)
    target_sparsity = cfg["pruning_parameters"]["sparsity"]

    reset_seed()
    weight_np = np.random.randn(*shape).astype(np.float32)

    k_layer = KerasPDP(cfg, layer_type)
    k_layer.build(shape)
    k_layer.post_pre_train_function()

    t_layer = TorchPDP(cfg, layer_type)
    t_layer.build(shape)
    t_layer.post_pre_train_function()

    # Force the sparsity ramp to be fully complete. pre_epoch_function sets
    # r = min(1, epsilon * (epoch + 1)) * init_r; with epsilon=1.0 that already
    # puts the ramp multiplier at 1.0 on epoch 0, so r = init_r = 0.75.
    k_layer.pre_epoch_function(0, None)
    t_layer.pre_epoch_function(0, None)

    k_weight = keras_tensor(weight_np)
    t_weight = torch_tensor(weight_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"PDP forward ({layer_type}, structured={structured})")

    k_grad = keras_grad(k_layer, weight_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"PDP backward ({layer_type}, structured={structured})")

    k_layer.update_mask(keras_tensor(weight_np))
    t_layer.update_mask(torch_tensor(weight_np))
    assert_close(k_layer.mask, t_layer.mask, msg="PDP mask after update_mask")

    # Verify the produced mask hits the configured target sparsity.
    # For structured pruning the mask has shape (C, 1, ...) and encodes
    # per-channel keep/prune; its sparsity directly equals the channel-level
    # pruning fraction. For unstructured it's per-element. With temperature
    # 1e-5 the soft mask is effectively binary, so use >= 0.5 to discretize.
    t_mask_np = to_numpy(t_layer.mask)
    actual_sparsity = float((t_mask_np < 0.5).sum()) / t_mask_np.size
    assert actual_sparsity == pytest.approx(target_sparsity, abs=1e-6), (
        f"PDP {layer_type} (structured={structured}) mask sparsity " f"{actual_sparsity} != target {target_sparsity}"
    )


def cs_config(threshold_decay=1e-4):
    return {
        "pruning_parameters": {
            "pruning_method": "cs",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "threshold_init": -0.1,
            "final_temp": 200,
            "threshold_decay": threshold_decay,
        },
    }


@pytest.mark.parametrize(
    "shape",
    [(16, 8), (16, 8, 3, 3)],
)
def test_cs_matches_keras(shape):
    cfg = cs_config()
    layer_type = "linear" if len(shape) == 2 else "conv"

    reset_seed()
    s_override_np = (np.random.randn(*shape) * 0.5).astype(np.float32)
    weight_np = np.random.randn(*shape).astype(np.float32)

    k_layer = KerasContinuousSparsification(cfg, layer_type)
    k_layer.build(shape)
    k_layer.post_pre_train_function()
    k_layer.s.assign(keras_tensor(s_override_np))

    t_layer = TorchContinuousSparsification(cfg, layer_type)
    t_layer.build(shape)
    t_layer.post_pre_train_function()
    with torch.no_grad():
        t_layer.s.data.copy_(torch_tensor(s_override_np))

    k_weight = keras_tensor(weight_np)
    t_weight = torch_tensor(weight_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"CS forward ({layer_type})")

    k_grad = keras_grad(k_layer, weight_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"CS backward ({layer_type})")

    assert_close(k_layer.get_hard_mask(), t_layer.get_hard_mask(), msg="CS hard mask")
    assert_close(k_layer.calculate_additional_loss(), t_layer.calculate_additional_loss(), msg="CS additional loss")

    # post_epoch_function updates beta — trajectories should match.
    k_layer.post_epoch_function(0, 5)
    t_layer.post_epoch_function(0, 5)
    assert_close(k_layer.beta, t_layer.beta, msg="CS beta after post_epoch_function")


def dst_config(threshold_type="channelwise"):
    return {
        "pruning_parameters": {
            "pruning_method": "dst",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "alpha": 5e-6,
            "max_pruning_pct": 0.99,
            "threshold_init": 0.0,
            "threshold_type": threshold_type,
            "threshold_decay": 0.0,
        },
    }


@pytest.mark.parametrize(
    "layer_type,shape,threshold_type",
    [
        ("linear", (16, 8), "layerwise"),
        ("linear", (16, 8), "channelwise"),
        ("linear", (16, 8), "weightwise"),
        ("conv", (16, 8, 3, 3), "channelwise"),
    ],
)
def test_dst_matches_keras(layer_type, shape, threshold_type):
    cfg = dst_config(threshold_type=threshold_type)

    reset_seed()
    weight_np = (np.random.randn(*shape) * 0.5).astype(np.float32)
    if threshold_type == "layerwise":
        thr_np = np.array([[0.1]], dtype=np.float32)
    elif threshold_type == "channelwise":
        thr_np = (np.random.rand(shape[0], 1) * 0.2).astype(np.float32)
    else:  # weightwise
        thr_np = (np.random.rand(shape[0], int(np.prod(shape[1:]))) * 0.2).astype(np.float32)

    k_layer = KerasDST(cfg, layer_type)
    k_layer.build(shape)
    k_layer.post_pre_train_function()
    k_layer.threshold.assign(keras_tensor(thr_np))

    t_layer = TorchDST(cfg, layer_type)
    t_layer.build(shape)
    t_layer.post_pre_train_function()
    with torch.no_grad():
        t_layer.threshold.data.copy_(torch_tensor(thr_np))

    k_weight = keras_tensor(weight_np)
    t_weight = torch_tensor(weight_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"DST forward ({layer_type}, {threshold_type})")

    k_grad = keras_grad(k_layer, weight_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"DST backward ({layer_type}, {threshold_type})")

    assert_close(
        k_layer.get_mask(keras_tensor(weight_np)),
        t_layer.get_mask(torch_tensor(weight_np)),
        msg=f"DST get_mask ({threshold_type})",
    )
    assert_close(k_layer.calculate_additional_loss(), t_layer.calculate_additional_loss(), msg="DST additional loss")


def wanda_config(sparsity=0.75, N=None, M=None):
    return {
        "pruning_parameters": {
            "pruning_method": "wanda",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "sparsity": sparsity,
            "t_delta": 2,
            "t_start_collecting_batch": 0,
            "N": N,
            "M": M,
            "threshold_decay": 0.0,
            "calculate_pruning_budget": True,
        },
    }


@pytest.mark.parametrize(
    "layer_type,shape,N,M",
    [
        ("linear", (16, 8), None, None),
        ("conv", (16, 8, 3, 3), None, None),
        ("linear", (4, 8), 4, 8),
        ("conv", (4, 8, 3, 3), 4, 8),
    ],
)
def test_wanda_matches_keras(layer_type, shape, N, M):
    cfg = wanda_config(N=N, M=M)

    reset_seed()
    if layer_type == "linear":
        x_np = np.random.randn(32, shape[1]).astype(np.float32)
    else:
        x_np = np.random.randn(32, shape[1], shape[2], shape[3]).astype(np.float32)
    w_np = np.random.randn(*shape).astype(np.float32)

    k_layer = KerasWanda(cfg, layer_type)
    k_layer.build(shape)
    k_layer.post_pre_train_function()

    t_layer = TorchWanda(cfg, layer_type)
    t_layer.build(shape)
    t_layer.post_pre_train_function()

    for _ in range(cfg["pruning_parameters"]["t_delta"]):
        k_layer.collect_input(keras_tensor(x_np), keras_tensor(w_np), training=True)
        t_layer.collect_input(torch_tensor(x_np), torch_tensor(w_np), training=True)

    assert_close(k_layer.mask, t_layer.mask, msg=f"Wanda mask ({layer_type}, N={N}, M={M})")

    # Verify the mask hits the configured target sparsity. For N:M pruning
    # Wanda internally uses N/M as the sparsity target; for unstructured it
    # uses the configured sparsity directly. Mask values are strictly {0, 1}
    # (produced by topk + scatter), so `== 0` counts pruned entries.
    target_sparsity = (N / M) if (N is not None and M is not None) else cfg["pruning_parameters"]["sparsity"]
    mask_np = to_numpy(t_layer.mask)
    pruned_fraction = float((mask_np == 0).sum()) / mask_np.size
    assert pruned_fraction == pytest.approx(
        target_sparsity
    ), f"Wanda {layer_type} (N={N}, M={M}) pruned fraction {pruned_fraction} != target {target_sparsity}"

    k_weight = keras_tensor(w_np)
    t_weight = torch_tensor(w_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"Wanda forward ({layer_type}, N={N}, M={M})")

    k_grad = keras_grad(k_layer, w_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"Wanda backward ({layer_type}, N={N}, M={M})")


def autosparse_config(threshold_type="channelwise", threshold_init=-2.0):
    return {
        "pruning_parameters": {
            "pruning_method": "autosparse",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "alpha": 0.5,
            "alpha_reset_epoch": 100,
            "autotune_epochs": 10,
            "backward_sparsity": False,
            "threshold_init": threshold_init,
            "threshold_type": threshold_type,
            "threshold_decay": 0.0,
        },
    }


@pytest.mark.parametrize(
    "layer_type,shape,threshold_type",
    [
        ("linear", (16, 8), "layerwise"),
        ("linear", (16, 8), "channelwise"),
        ("conv", (16, 8, 3, 3), "channelwise"),
    ],
)
def test_autosparse_matches_keras(layer_type, shape, threshold_type):
    # Keras AutoSparse.call() unconditionally routes through ops.custom_gradient,
    # which under the torch backend errors because self.alpha is a keras Variable
    # (save_for_backward rejects it). The path works fine under the tensorflow
    # backend, which is the intended target for the keras implementation, so we
    # skip this test when keras is on torch to avoid spurious failures.
    import keras as _keras

    if _keras.backend.backend() == "torch":
        pytest.skip("Keras AutoSparse forward is incompatible with the torch backend.")
    cfg = autosparse_config(threshold_type=threshold_type)

    reset_seed()
    weight_np = np.random.randn(*shape).astype(np.float32)

    k_layer = KerasAutoSparse(cfg, layer_type)
    k_layer.build(shape)
    k_layer.post_pre_train_function()

    t_layer = TorchAutoSparse(cfg, layer_type)
    t_layer.build(shape)
    t_layer.post_pre_train_function()
    with torch.no_grad():
        t_layer.threshold.data.copy_(torch_tensor(to_numpy(k_layer.threshold)))

    assert_close(
        k_layer.get_mask(keras_tensor(weight_np)),
        t_layer.get_mask(torch_tensor(weight_np)),
        msg=f"AutoSparse get_mask ({layer_type}, {threshold_type})",
    )

    k_weight = keras_tensor(weight_np)
    t_weight = torch_tensor(weight_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"AutoSparse forward ({layer_type}, {threshold_type})")

    k_grad = keras_grad(k_layer, weight_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"AutoSparse backward ({layer_type}, {threshold_type})")

    # post_epoch_function updates alpha via decay; trajectories should match.
    k_layer.post_epoch_function(3, 10)
    t_layer.post_epoch_function(3, 10)
    assert_close(k_layer.alpha, t_layer.alpha, msg="AutoSparse alpha after post_epoch_function")


def mdmm_config(
    constraint_type="Equality",
    metric_type="UnstructuredSparsity",
    target_value=0.5,
    use_grad=True,
):
    return {
        "pruning_parameters": {
            "pruning_method": "mdmm",
            "disable_pruning_for_layers": [],
            "enable_pruning": True,
            "constraint_type": constraint_type,
            "target_value": target_value,
            "metric_type": metric_type,
            "target_sparsity": 0.8,
            "rf": 1,
            "epsilon": 1e-3,
            "scale": 1.0,
            "damping": 1.0,
            "use_grad": use_grad,
            "l0_mode": "coarse",
            "scale_mode": "mean",
            "constraint_lr": 1e-3,
            "threshold_decay": 0.0,
        },
    }


@pytest.mark.parametrize(
    "constraint_type,metric_type",
    [
        ("Equality", "UnstructuredSparsity"),
        ("LessThanOrEqual", "UnstructuredSparsity"),
        ("GreaterThanOrEqual", "UnstructuredSparsity"),
        ("Equality", "StructuredSparsity"),
    ],
)
def test_mdmm_matches_keras(constraint_type, metric_type):
    cfg = mdmm_config(constraint_type=constraint_type, metric_type=metric_type)
    shape = (16, 8)

    reset_seed()
    weight_np = (np.random.randn(*shape) * 0.2).astype(np.float32)

    k_layer = KerasMDMM(cfg, "linear")
    k_layer.build(shape)
    k_layer.post_pre_train_function()

    t_layer = TorchMDMM(cfg, "linear")
    t_layer.build(shape)
    t_layer.post_pre_train_function()

    k_weight = keras_tensor(weight_np)
    t_weight = torch_tensor(weight_np, requires_grad=True)
    k_out = k_layer(k_weight)
    t_out = t_layer(t_weight)
    assert_close(k_out, t_out, msg=f"MDMM forward ({constraint_type}, {metric_type})")

    k_grad = keras_grad(k_layer, weight_np)
    t_out.sum().backward()
    assert_close(k_grad, t_weight.grad, msg=f"MDMM backward ({constraint_type}, {metric_type})")

    assert_close(
        k_layer.get_hard_mask(keras_tensor(weight_np)),
        t_layer.get_hard_mask(torch_tensor(weight_np)),
        msg="MDMM hard_mask",
    )

    # Constraint penalty: read directly from the constraint layer to avoid
    # differences in how keras/torch surface accumulated losses.
    k_penalty = ops.sum(k_layer.constraint_layer(keras_tensor(weight_np)))
    t_penalty = t_layer.constraint_layer(torch_tensor(weight_np)).sum()
    assert_close(k_penalty, t_penalty, msg="MDMM constraint penalty")


def test_mdmm_finetune_returns_masked_weight():
    """In finetuning mode both layers should return weight * hard_mask."""
    cfg = mdmm_config()
    shape = (8, 6)

    reset_seed()
    weight_np = (np.random.randn(*shape) * 0.2).astype(np.float32)

    k_layer = KerasMDMM(cfg, "linear")
    k_layer.build(shape)
    k_layer.post_pre_train_function()
    k_layer.pre_finetune_function()

    t_layer = TorchMDMM(cfg, "linear")
    t_layer.build(shape)
    t_layer.post_pre_train_function()
    t_layer.pre_finetune_function()

    k_out = k_layer(keras_tensor(weight_np))
    t_out = t_layer(torch_tensor(weight_np))
    assert_close(k_out, t_out, msg="MDMM finetune forward")


@pytest.mark.parametrize("l0_mode", ["coarse", "smooth"])
@pytest.mark.parametrize("scale_mode", ["mean", "sum"])
def test_unstructured_sparsity_metric_matches_keras(l0_mode, scale_mode):
    k_layer = KerasUnstructuredSparsityMetric(l0_mode=l0_mode, scale_mode=scale_mode, target_sparsity=0.7, epsilon=1e-3)
    t_layer = TorchUnstructuredSparsityMetric(l0_mode=l0_mode, scale_mode=scale_mode, target_sparsity=0.7, epsilon=1e-3)

    reset_seed()
    w_np = (np.random.randn(16, 8) * 0.1).astype(np.float32)

    assert_close(k_layer(keras_tensor(w_np)), t_layer(torch_tensor(w_np)), msg=f"Unstructured({l0_mode},{scale_mode})")


@pytest.mark.parametrize("rf", [1, 4, 5])
def test_structured_sparsity_metric_matches_keras(rf):
    k_layer = KerasStructuredSparsityMetric(rf=rf, epsilon=1e-3)
    t_layer = TorchStructuredSparsityMetric(rf=rf, epsilon=1e-3)

    reset_seed()
    w_np = (np.random.randn(12, 7) * 0.05).astype(np.float32)

    assert_close(k_layer(keras_tensor(w_np)), t_layer(torch_tensor(w_np)), msg=f"Structured(rf={rf})")
