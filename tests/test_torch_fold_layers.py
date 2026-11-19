"""The fused BatchNorm layers: PQDenseBatchNorm (a BatchNorm on the input and / or output side of a Dense) and
PQConv1dBatchNorm / PQConv2dBatchNorm (Conv -> BN), trained on the folded weight."""

import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant.layers import (  # noqa: E402
    PQConv1dBatchNorm,
    PQConv2dBatchNorm,
    PQDenseBatchNorm,
    apply_final_compression,
)

from pquant import pdp_config, train_model  # noqa: E402
from pquant import preset_configs as presets  # noqa: E402


def _config(quantize=False, hgq=False):
    config = pdp_config()
    config.pruning_parameters.enable_pruning = False
    config.quantization_parameters.enable_quantization = quantize
    config.quantization_parameters.use_high_granularity_quantization = hgq
    return config


def _randomize(bn):
    """Non-trivial gamma, beta and running statistics of a fused norm."""
    with torch.no_grad():
        bn.weight.copy_(torch.rand(bn.weight.numel()) + 0.5)
        bn.bias.copy_(torch.randn(bn.bias.numel()))
        bn.running_mean.copy_(torch.randn(bn.running_mean.numel()))
        bn.running_var.copy_(torch.rand(bn.running_var.numel()) + 0.5)


def _bn_ref(x, bn, batch, real=None):
    """A plain batchnorm over the channel axis with the batch (masked) or the running statistics."""
    if batch:
        w = (torch.ones(x.shape[:-1]) if real is None else real.float()).unsqueeze(-1)
        n = w.sum()
        mean = (x * w).sum(dim=(0, 1)) / n
        var = ((x - mean).square() * w).sum(dim=(0, 1)) / n
    else:
        mean, var = bn.running_mean, bn.running_var
    return (x - mean) * torch.rsqrt(var + bn.eps) * bn.weight + bn.bias


def _fused(config, bn_in, bn_out, out_perm=None, out_scale=None, bias_bits=16, randomize=True, **kwargs):
    dense = PQDenseBatchNorm(
        config, 8, 6, bn_in=bn_in, bn_out=bn_out, out_perm=out_perm, out_scale=out_scale, bias_bits=bias_bits, **kwargs
    )
    dense.eval()
    dense(torch.randn(7, 3, 8))  # build the lanes without touching the running statistics
    if randomize:
        for bn in (dense.bn_in, dense.bn_out):
            if bn is not None:
                _randomize(bn)
        with torch.no_grad():
            dense._bias.copy_(torch.randn(6))
    return dense


def _reference(dense, x, batch, real=None):
    """The float BN -> Dense -> BN the layer deploys, with the batch or the running statistics."""
    if dense.bn_in is not None:
        x = _bn_ref(x, dense.bn_in, batch, real)
    y = F.linear(x, dense._weight, dense._bias)
    if dense.out_perm is not None:
        y = y[..., dense.out_perm]
    if dense.out_scale is not None:
        y = y * dense.out_scale
    if dense.bn_out is not None:
        y = _bn_ref(y, dense.bn_out, batch, real)
    return y


@pytest.mark.parametrize("bn_in,bn_out", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("train", [False, True])
def test_fold_matches_float_reference(bn_in, bn_out, train):
    torch.manual_seed(0)
    dense = _fused(_config(), bn_in, bn_out)
    x = torch.randn(7, 3, 8) * 2 + 1
    dense.train(train)
    assert torch.allclose(dense(x), _reference(dense, x, batch=train), atol=1e-5, rtol=1e-4)


def test_out_perm_and_out_scale_are_folded():
    torch.manual_seed(1)
    perm = torch.randperm(6)
    scale = torch.rand(6) + 0.5
    given = scale.clone()
    dense = _fused(_config(), False, True, out_perm=perm, out_scale=given)
    given.zero_()  # the layer holds a copy
    assert torch.equal(dense.out_scale, scale)
    x = torch.randn(5, 4, 8)
    for train in (False, True):
        dense.train(train)
        assert torch.allclose(dense(x), _reference(dense, x, batch=train), atol=1e-5, rtol=1e-4)
    assert "out_scale" in dense.state_dict() and "out_perm" not in dense.state_dict()
    # the same factor given as a callable (a parameter that trains elsewhere)
    live = nn.Parameter(scale.clone())
    dense_fn = _fused(_config(), False, True, out_perm=perm, out_scale=lambda: live)
    dense_fn.load_state_dict({k: v for k, v in dense.state_dict().items() if k != "out_scale"})
    dense.eval()
    dense_fn.eval()
    assert torch.allclose(dense_fn(x), dense(x), atol=1e-6)
    assert "out_scale" not in dense_fn.state_dict()


def test_masked_statistics_ignore_padded_tokens():
    torch.manual_seed(2)
    dense = _fused(_config(), True, True)
    x = torch.randn(7, 3, 8)
    padding_mask = torch.zeros(3, 7, dtype=torch.bool)
    padding_mask[0, 4:] = True
    padding_mask[2, 6:] = True
    real = ~padding_mask.transpose(0, 1)  # (seq, batch)
    dense.train()
    out = dense(x, padding_mask)
    ref = _reference(dense, x, batch=True, real=real)
    assert torch.allclose(out[real], ref[real], atol=1e-5, rtol=1e-4)
    noisy = x.clone()
    noisy[~real] = 100.0
    assert torch.allclose(dense(noisy, padding_mask)[real], out[real], atol=1e-5, rtol=1e-4)
    # the running statistics follow the real tokens (two updates at momentum 0.1), unbiased as nn.BatchNorm1d
    fresh = _fused(_config(), True, False, randomize=False)
    fresh.train()
    fresh(x, padding_mask)
    fresh(x, padding_mask)
    fresh.eval()
    mean = x[real].mean(dim=0)
    var = x[real].var(dim=0, unbiased=False)
    n = real.sum()
    assert torch.allclose(fresh.bn_in.running_mean, (0.1 + 0.9 * 0.1) * mean, atol=1e-5)
    unbiased = var * n / (n - 1)
    assert torch.allclose(fresh.bn_in.running_var, (1 - 0.1) ** 2 + 0.1 * unbiased * (1 + (1 - 0.1)), atol=1e-4)


def test_batch_first_takes_the_mask_as_given():
    torch.manual_seed(3)
    dense = _fused(_config(), True, True)
    dense_bf = _fused(_config(), True, True, batch_first=True)
    dense_bf.load_state_dict(dense.state_dict())
    x = torch.randn(7, 3, 8)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[1, 2:] = True
    dense.train()
    dense_bf.train()
    assert torch.allclose(dense_bf(x.transpose(0, 1), mask).transpose(0, 1), dense(x, mask), atol=1e-5)


@pytest.mark.parametrize("momentum", [0.1, 0.3, None])
def test_running_statistics_match_torch_batchnorm(momentum):
    torch.manual_seed(4)
    dense = _fused(_config(), True, False, randomize=False, eps=1e-3, momentum=momentum)
    ref = nn.BatchNorm1d(8, eps=1e-3, momentum=momentum)
    dense.train()
    ref.train()
    for step in range(3):
        x = torch.randn(7, 3, 8)
        dense(x)
        ref(x.permute(1, 2, 0))
        if step == 0:
            assert torch.equal(dense.bn_in.running_mean, torch.zeros(8))  # deferred to the next fold ...
    dense.eval()  # ... or to evaluation
    ref.eval()
    bn = dense.bn_in
    assert torch.allclose(bn.running_mean, ref.running_mean, atol=1e-6)
    assert torch.allclose(bn.running_var, ref.running_var, atol=1e-6)
    assert int(bn.num_batches_tracked) == int(ref.num_batches_tracked) == 3
    want = F.linear(ref(x.permute(1, 2, 0)).permute(2, 0, 1), dense._weight, dense._bias)
    assert torch.allclose(dense(x), want, atol=1e-6)


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("bn_in,bn_out", [(True, False), (False, True), (True, True)])
def test_final_compression_keeps_the_deployed_function(quantize, bn_in, bn_out):
    torch.manual_seed(6)
    dense = _fused(_config(quantize=quantize), bn_in, bn_out)
    x = torch.randn(7, 3, 8)
    dense.eval()
    before = dense(x)
    apply_final_compression(dense)
    assert dense._final_compression_done
    assert torch.equal(dense(x), before)
    # the stored tensors are the deployed Dense: the same output without any fold machinery
    y = dense.input_quantizer(x) if quantize else x
    assert torch.allclose(F.linear(y, dense._weight, dense._bias), before, atol=1e-6)


def test_bias_lane_is_wide_and_follows_the_deployed_bias():
    torch.manual_seed(7)
    dense = _fused(_config(quantize=True), True, False, bias_bits=20)
    x = torch.randn(7, 3, 8)
    dense.train()
    dense(x)
    k, i, f = dense.get_bias_quantization_bits()
    assert float(k + i + f) == 20.0
    deployed = dense._fold()[1]
    assert float(i) >= float(torch.ceil(torch.log2(deployed.abs().max())))
    kw, iw, fw = dense.get_weight_quantization_bits()
    assert float(kw + iw + fw) == 8.0  # the weight lane keeps the config's width


def test_fold_needs_a_bias_and_owns_its_norms():
    config = _config()
    for bn_out in (True, False):
        with pytest.raises(ValueError):
            PQDenseBatchNorm(config, 8, 6, bn_out=bn_out, bias=False)
    dense = PQDenseBatchNorm(config, 8, 6, bn_in=True)
    assert dense.bn_out is None and dense.bn_in in list(dense.modules())
    assert {"bn_in.weight", "bn_in.bias", "bn_in.running_mean", "bn_in.running_var"} <= set(dense.state_dict())
    assert all(p.requires_grad for p in dense.bn_in.parameters())


def test_state_dict_round_trip():
    torch.manual_seed(8)
    config = _config(quantize=True)
    dense = _fused(config, True, True, out_scale=torch.rand(6) + 0.5)
    x = torch.randn(7, 3, 8)
    dense.eval()
    expected = dense(x)
    torch.manual_seed(9)
    other = _fused(config, True, True, out_scale=torch.ones(6))
    other.load_state_dict(dense.state_dict())
    other.eval()
    assert torch.equal(other(x), expected)


def test_hgq_lanes_and_ebops():
    torch.manual_seed(10)
    dense = _fused(_config(quantize=True, hgq=True), True, True)
    x = torch.randn(7, 3, 8)
    dense.train()
    dense.post_pre_train_function()
    dense(x).sum().backward()
    assert torch.isfinite(dense.hgq_loss())
    assert float(dense.ebops()) > 0.0
    assert dense.bn_in.weight.grad is not None and dense.bn_out.bias.grad is not None  # the fused norms train


def _conv_fused(config, cls=PQConv2dBatchNorm, groups=1, stride=1, bias_bits=16, randomize=True):
    channels_out = 6 if groups > 1 else 5
    conv = cls(config, 6, channels_out, 3, stride=stride, padding=1, groups=groups, bias_bits=bias_bits)
    conv.eval()
    conv(_conv_input(cls))  # build the lanes without touching the running statistics
    if randomize:
        _randomize(conv.bn_out)
        with torch.no_grad():
            conv._bias.copy_(torch.randn(channels_out))
    return conv


def _conv_input(cls, batch=4):
    return torch.randn(batch, 6, 9, 9) if cls is PQConv2dBatchNorm else torch.randn(batch, 6, 11)


def _conv_reference(conv, x, batch):
    """The float convolution and batch norm the layer deploys, with the batch or the running statistics."""
    fn = F.conv2d if isinstance(conv, PQConv2dBatchNorm) else F.conv1d
    y = fn(x, conv._weight, conv._bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    shape = (1, -1) + (1,) * (y.dim() - 2)
    if batch:
        dims = (0,) + tuple(range(2, y.dim()))
        mean, var = y.mean(dims), y.var(dims, unbiased=False)
    else:
        mean, var = conv.bn_out.running_mean, conv.bn_out.running_var
    bn = conv.bn_out
    y = (y - mean.reshape(shape)) * torch.rsqrt(var.reshape(shape) + bn.eps)
    return y * bn.weight.reshape(shape) + bn.bias.reshape(shape)


@pytest.mark.parametrize(
    "cls,groups,stride",
    [(PQConv2dBatchNorm, 1, 1), (PQConv2dBatchNorm, 6, 1), (PQConv2dBatchNorm, 1, 2), (PQConv1dBatchNorm, 1, 1)],
)
@pytest.mark.parametrize("train", [False, True])
def test_conv_fold_matches_float_reference(cls, groups, stride, train):
    torch.manual_seed(11)
    conv = _conv_fused(_config(), cls, groups, stride)
    x = _conv_input(cls) * 2 + 1
    conv.train(train)
    assert torch.allclose(conv(x), _conv_reference(conv, x, batch=train), atol=1e-5, rtol=1e-4)


def test_conv_running_statistics_match_torch_batchnorm():
    torch.manual_seed(12)
    conv = _conv_fused(_config(), randomize=False)
    ref = nn.BatchNorm2d(conv.out_channels)
    x = torch.randn(4, 6, 9, 9)
    conv.train()
    ref.train()
    conv(x)
    ref(F.conv2d(x, conv._weight, conv._bias, conv.stride, conv.padding))
    assert torch.equal(conv.bn_out.running_mean, torch.zeros(conv.out_channels))  # deferred to the next fold ...
    conv.eval()  # ... or to evaluation
    assert torch.allclose(conv.bn_out.running_mean, ref.running_mean, atol=1e-6)
    assert torch.allclose(conv.bn_out.running_var, ref.running_var, atol=1e-6)


def _built(make, config, **kwargs):
    """A built fused layer of the given kind (conv, dense_in, dense_out) and an input for it."""
    if make == "conv":
        layer, x = PQConv2dBatchNorm(config, 4, 6, 3, padding=1, **kwargs), torch.randn(2, 4, 5, 5)
    else:
        layer = PQDenseBatchNorm(config, 4, 6, bn_in=make == "dense_in", bn_out=make == "dense_out", **kwargs)
        x = torch.randn(5, 2, 4)
    layer.eval()
    layer(x)
    return layer, x


@pytest.mark.parametrize("make", ["conv", "dense_out", "dense_in"])
def test_compiled_training_steps_match_eager(make):
    """The running update is deferred to the next fold, which keeps the compiled backward valid."""
    torch.manual_seed(16)
    config = _config(quantize=True)
    layers = []
    for _ in range(2):
        layer, x = _built(make, config)
        layers.append(layer.train())
    layers[1].load_state_dict(layers[0].state_dict())
    compiled = torch.compile(layers[1], backend="aot_eager_decomp_partition")
    for _ in range(3):
        for fn in (layers[0], compiled):
            fn(x).square().mean().backward()
    for layer in layers:
        layer.eval()
    a, b = layers
    assert torch.allclose(a._weight.grad, b._weight.grad, atol=1e-6)
    assert torch.allclose(a(x), b(x), atol=1e-6)


@pytest.mark.parametrize("quantize,hgq", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("groups", [1, 6])
def test_conv_final_compression_keeps_the_deployed_function(quantize, hgq, groups):
    torch.manual_seed(13)
    conv = _conv_fused(_config(quantize=quantize, hgq=hgq), groups=groups)
    x = torch.randn(4, 6, 9, 9)
    conv.eval()
    before = conv(x)
    apply_final_compression(conv)
    assert conv._final_compression_done and torch.equal(conv(x), before)
    y = conv.input_quantizer(x) if quantize else x
    deployed = F.conv2d(y, conv._weight, conv._bias, conv.stride, conv.padding, groups=groups)
    assert torch.allclose(deployed, before, atol=1e-6)


def test_folded_bias_lane_is_fixed_with_hgq_lanes():
    torch.manual_seed(14)
    config = presets.hgq(granularity="per_tensor")
    conv = _conv_fused(config, bias_bits=24)
    conv(torch.randn(4, 6, 9, 9))  # the lane's integer bits follow the deployed bias at every forward
    assert conv.weight_quantizer.use_hgq and not conv.bias_quantizer.use_hgq
    k, i, f = conv.get_bias_quantization_bits()
    assert float(k + i + f) == 24.0
    assert float(i) >= float(torch.ceil(torch.log2(conv._fold()[1].abs().max())))
    assert _conv_fused(config, bias_bits=None).bias_quantizer.use_hgq  # None keeps the configured (learned) lane
    dense = _fused(config, True, True, bias_bits=24)
    assert not dense.bias_quantizer.use_hgq and float(sum(dense.get_bias_quantization_bits())) == 24.0


def test_conv_fold_needs_a_bias_and_owns_its_norm():
    with pytest.raises(ValueError):
        PQConv2dBatchNorm(_config(), 6, 5, 3, bias=False)
    conv = PQConv2dBatchNorm(_config(), 6, 5, 3)
    assert {"bn_out.weight", "bn_out.bias", "bn_out.running_mean", "bn_out.running_var"} <= set(conv.state_dict())
    assert all(p.requires_grad for p in conv.bn_out.parameters())


def test_conv_deployed_weight_is_on_the_weight_lane():
    torch.manual_seed(27)
    conv = _conv_fused(_config(quantize=True))
    _, _, f = conv.weight_quantizer.get_quantization_bits()
    codes, raw = conv.weight * torch.exp2(f), conv._weight * torch.exp2(f)
    assert torch.equal(codes, torch.round(codes)) and not torch.equal(raw, torch.round(raw))


@pytest.mark.parametrize("make", ["conv", "dense_out", "dense_in"])
def test_pruned_weights_ship_as_zero(make):
    torch.manual_seed(26)
    config = _config()
    config.pruning_parameters.enable_pruning = True
    layer, x = _built(make, config)
    mask = (torch.rand_like(layer._weight) > 0.5).float()
    layer.pruning_layer.mask.data = mask.reshape(layer.pruning_layer.mask.shape)
    assert torch.equal(layer.weight != 0, mask.bool())
    y = layer(x)
    apply_final_compression(layer)
    assert torch.equal(layer._weight != 0, mask.bool()) and torch.allclose(layer(x), y, atol=1e-6)


@pytest.mark.parametrize("make", ["conv", "dense_out"])
def test_a_zero_gamma_trains_to_beta(make):
    torch.manual_seed(28)
    layer, x = _built(make, _config())
    with torch.no_grad():
        layer.bn_out.weight[0] = 0.0
        layer.bn_out.bias[0] = 0.5
    layer.train()
    y = layer(x).movedim(1 if make == "conv" else -1, 0)[0]
    assert torch.allclose(y, torch.full_like(y, 0.5))


@pytest.mark.parametrize("make", ["conv", "dense_out", "dense_in"])
def test_final_compression_twice_is_final_compression_once(make):
    torch.manual_seed(17)
    layer, x = _built(make, _config(quantize=True))
    _randomize(layer.bn_in if make == "dense_in" else layer.bn_out)
    before = layer(x)
    apply_final_compression(layer)
    apply_final_compression(layer)  # a second call must not fold the folded weight again
    assert torch.equal(layer(x), before)


@pytest.mark.parametrize("make", ["conv", "dense_out", "dense_in"])
def test_weight_and_bias_are_the_deployed_pair_in_every_mode(make):
    torch.manual_seed(18)
    layer, x = _built(make, _config(quantize=True))
    _randomize(layer.bn_in if make == "dense_in" else layer.bn_out)
    expected = layer(x)
    y = layer.input_quantizer(x)
    for train in (False, True):
        layer.train(train)
        w, b = layer.weight.detach(), layer.bias.detach()
        out = F.conv2d(y, w, b, padding=1) if make == "conv" else F.linear(y, w, b)
        assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("make", ["conv", "dense_out", "dense_in"])
@pytest.mark.parametrize("train", [False, True])
def test_a_second_fold_keeps_the_first_graph_valid(make, train):
    """Two forwards and a look at ``weight`` before one backward: a fold must not change a tensor autograd saved."""
    torch.manual_seed(19)
    layer, x = _built(make, _config(quantize=True))
    layer.train(train)
    out = layer(x) + layer(x)
    assert layer.weight.shape == layer._weight.shape
    out.sum().backward()
    assert layer._weight.grad is not None


def test_training_bias_stays_off_the_deployed_bias_lane():
    """The bias lane is sized for the deployed bias: the larger batch-mode bias must not go through it."""
    torch.manual_seed(20)
    config = _config(quantize=True)
    wide = dict(weight_quant_bits=(1, 3, 12), in_quant_bits=(1, 6, 9))
    # input side: a running mean that cancels the bias in the fold, so the deployed bias is small and its lane +-1
    dense = PQDenseBatchNorm(config, 8, 6, bn_in=True, **wide)
    x = torch.randn(16, 4, 8)
    dense.eval()
    dense(x)
    with torch.no_grad():
        dense._bias.fill_(5.0)
        dense.bn_in.running_mean.copy_(torch.linalg.lstsq(dense._weight, dense._bias.unsqueeze(1)).solution.squeeze(1))
    dense.train()
    out = dense(x)
    assert float(dense.get_bias_quantization_bits()[1]) == 0.0
    assert torch.allclose(out, _reference(dense, x, batch=True), atol=2e-2)
    # output side: e.g. a pretrained Conv(bias=True) + BN pair, the norm's mean carries the bias
    conv = PQConv2dBatchNorm(config, 6, 5, 3, padding=1, momentum=1.0, **wide)
    ref = nn.BatchNorm2d(5, momentum=1.0)
    x = torch.randn(16, 6, 9, 9)
    conv.eval()
    conv(x)
    with torch.no_grad():
        conv._bias.fill_(5.0)
    conv.train()
    conv(x)
    pre = F.conv2d(x, conv._weight, conv._bias, padding=1)
    ref(pre)
    conv.eval()
    ref.eval()
    assert torch.allclose(conv.bn_out.running_mean, ref.running_mean, atol=2e-2)
    assert torch.allclose(conv(x), ref(pre), atol=2e-2)


def test_learned_bias_lane_trains_on_the_deployed_bias():
    """``bias_bits=None`` with HGQ: the lane's rounding of the deployed bias reaches the loss in batch mode."""
    torch.manual_seed(21)
    conv = _conv_fused(presets.hgq(granularity="per_tensor"), bias_bits=None)
    conv.train()
    (conv(torch.randn(4, 6, 9, 9)) - 1.0).square().mean().backward()
    assert float(conv.bias_quantizer.quantizer._f.grad.abs().sum()) > 1e-6


def test_statistics_are_taken_in_full_precision_under_autocast():
    torch.manual_seed(22)
    dense = _fused(_config(), True, True, randomize=False)
    x = torch.randn(7, 3, 8).to(torch.bfloat16)
    dense.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dense(x).float().sum().backward()
    dense.eval()
    assert torch.allclose(dense.bn_in.running_mean, 0.1 * x.float().mean(dim=(0, 1)), atol=1e-6)


@pytest.mark.parametrize("make", ["conv", "dense_out", "dense_in"])
def test_frozen_statistics_train_on_the_deployed_function(make):
    torch.manual_seed(24)
    layer, x = _built(make, _config(quantize=True))
    bn = layer.bn_in if make == "dense_in" else layer.bn_out
    layer.train()
    layer(x)  # some statistics
    layer.eval()
    deployed = layer(x)
    mean, var = bn.running_mean.clone(), bn.running_var.clone()
    layer.train()
    layer.freeze_statistics = True
    out = layer(x)
    out.square().mean().backward()
    assert torch.equal(out, deployed)
    assert not bool(bn.pending) and torch.equal(bn.running_mean, mean) and torch.equal(bn.running_var, var)
    assert all(float(p.grad.abs().sum()) > 0 for p in (layer._weight, layer._bias, bn.weight, bn.bias))


def test_post_epoch_functions_freeze_the_last_epoch_after_pretraining():
    """Through train_model: the last epoch of each training round and of fine-tuning, never an epoch of pretraining."""
    config = _config()
    stages = config.training_parameters
    stages.pretraining_epochs, stages.epochs, stages.rounds, stages.fine_tuning_epochs = 2, 3, 2, 3
    model = nn.Sequential(PQDenseBatchNorm(config, 8, 6, bn_in=True, bn_out=True))
    x = torch.randn(16, 8)
    frozen = []

    def train_func(model, epoch, **kwargs):
        frozen.append(model[0].freeze_statistics)
        model(x)

    train_model(model, config, train_func, lambda model, epoch, **kwargs: None)
    assert frozen == [False, False] + [False, False, True] * 2 + [False, False, True]


def test_masked_ebops_count_the_weights_that_ship():
    torch.manual_seed(23)
    dense = PQDenseBatchNorm(_config(quantize=True), 8, 6, bn_out=True)
    dense.eval()
    dense(torch.randn(4, 8))
    full = float(dense.ebops(include_mask=True))
    with torch.no_grad():
        dense.bn_out.weight.fill_(1e-4)  # the folded weight falls under the lane's step: every weight ships as zero
    assert int(torch.count_nonzero(dense.weight)) == 0
    assert float(dense.ebops(include_mask=True)) == 16 * 6 < full  # the bias adds are all that is left


def test_conv_takes_the_convolutions_positional_arguments():
    conv = PQConv2dBatchNorm(_config(), 6, 5, 3, 2, 1, bias_bits=20)  # as nn.Conv2d(6, 5, 3, 2, 1)
    assert conv.stride == (2, 2) and conv.padding == (1, 1) and conv.bias_bits == 20
