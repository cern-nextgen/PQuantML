import os

import numpy as np
import pytest
import torch
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant import dst_config  # noqa: E402
from pquant.activations import PQActivation  # noqa: E402
from pquant.layers import (  # noqa: E402
    PQAvgPool1d,
    PQAvgPool2d,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQLayerNorm,
    PQMultiheadAttention,
    apply_final_compression,
    post_pretrain_functions,
    pre_finetune_functions,
)

BATCH_SIZE = 2
OUT_FEATURES = 8
IN_FEATURES = 4
KERNEL_SIZE = 3
STEPS = 6


class SingleLayerModel(nn.Module):

    def __init__(self, layer, is_mha=False):
        super().__init__()
        self.layer = layer
        self.is_mha = is_mha

    def forward(self, x):
        if self.is_mha:
            x, _ = self.layer(x, x, x)
        else:
            x = self.layer(x)
        return x


def build_model_and_input(layer_type, config):
    if layer_type == "dense":
        layer = PQDense(config, IN_FEATURES, OUT_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_FEATURES)
    elif layer_type == "conv1d":
        layer = PQConv1d(config, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, padding=1)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS)
    elif layer_type == "conv2d":
        layer = PQConv2d(config, IN_FEATURES, OUT_FEATURES, KERNEL_SIZE, padding=1)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS, STEPS)
    elif layer_type == "batchnorm1d":
        layer = PQBatchNorm1d(config, IN_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS)
    elif layer_type == "batchnorm2d":
        layer = PQBatchNorm2d(config, IN_FEATURES)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS, STEPS)
    elif layer_type == "layernorm":
        layer = PQLayerNorm(config, IN_FEATURES)
        x = torch.randn(BATCH_SIZE, STEPS, IN_FEATURES)
    elif layer_type == "avgpool1d":
        layer = PQAvgPool1d(config, kernel_size=2)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS)
    elif layer_type == "avgpool2d":
        layer = PQAvgPool2d(config, kernel_size=2)
        x = torch.randn(BATCH_SIZE, IN_FEATURES, STEPS, STEPS)
    elif layer_type.startswith("activation_"):
        layer = PQActivation(config, activation=layer_type.replace("activation_", ""), quantize_output=True)
        x = torch.randn(BATCH_SIZE, IN_FEATURES)
    elif layer_type == "mha":
        layer = PQMultiheadAttention(config, embed_dim=IN_FEATURES, num_heads=2)
        x = torch.randn(STEPS, BATCH_SIZE, IN_FEATURES)
        return SingleLayerModel(layer, is_mha=True), x
    else:
        raise ValueError(f"unknown layer kind {layer_type}")

    return SingleLayerModel(layer), x


STAGE_FLAGS = ("is_pretraining", "is_finetuning", "final_compression_done")


def randomize_state(model, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, t in model.state_dict().items():
            if any(k in name for k in STAGE_FLAGS):
                continue
            if not torch.is_floating_point(t):
                continue
            t.copy_(torch.rand(t.shape, generator=gen, device="cpu") + 0.5)


def advance_to_stage(model, config, stage):
    if stage == "initial":
        return
    post_pretrain_functions(model, config)
    if stage == "post_pretrain":
        return
    pre_finetune_functions(model)
    if stage == "pre_finetune":
        return
    apply_final_compression(model)
    assert stage == "final_compression"


def make_config(use_hgq):
    config = dst_config()
    config.quantization_parameters.enable_quantization = True
    config.quantization_parameters.use_high_granularity_quantization = use_hgq
    return config


LAYER_TYPES = [
    "dense",
    "conv1d",
    "conv2d",
    "batchnorm1d",
    "batchnorm2d",
    "layernorm",
    "avgpool1d",
    "avgpool2d",
    "activation_relu",
    "activation_tanh",
    "activation_hard_tanh",
    "mha",
]
STAGES = ["initial", "post_pretrain", "pre_finetune", "final_compression"]


@pytest.mark.parametrize("use_hgq", [False, True], ids=["kif", "hgq"])
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("layer_type", LAYER_TYPES)
def test_state_dict_roundtrip(tmp_path, layer_type, stage, use_hgq):
    torch.manual_seed(0)
    config = make_config(use_hgq)

    model, x = build_model_and_input(layer_type, config)
    model(x)  # HGQ quantizers build lazily on first forward
    advance_to_stage(model, config, stage)
    randomize_state(model, seed=42)

    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), path)
    torch.manual_seed(1)
    fresh_config = make_config(use_hgq)
    fresh, _ = build_model_and_input(layer_type, fresh_config)
    fresh(x)
    advance_to_stage(fresh, fresh_config, stage)
    missing, unexpected = fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert not missing and not unexpected

    saved = model.state_dict()
    reloaded = fresh.state_dict()
    assert saved.keys() == reloaded.keys()
    for name in saved:
        np.testing.assert_array_equal(
            reloaded[name].cpu(), saved[name].cpu(), err_msg=f"state mismatch: {name}", strict=True
        )

    model.eval()
    fresh.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(fresh(x).cpu(), model(x).cpu(), strict=True)
