"""HCCS (torch backend): the kernel modes against an integer re-implementation of the aie4ml ``softmax_i8`` kernel, the
score floor, and ``HCCSLearned`` (lanes, seeding, STE, export, compile)."""

import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant.activations import PQSoftmax  # noqa: E402
from pquant.layers import (  # noqa: E402
    PQMultiheadAttention,
    add_compression_layers,
    get_ebops,
    get_model_losses,
    post_pretrain_functions,
)

from pquant import dst_config  # noqa: E402
from pquant.core.constants import DynamicDataMode  # noqa: E402
from pquant.core.torch.hccs import (  # noqa: E402
    HCCSLearned,
    _floor_log2,
    default_inv_shift,
    hccs_forward_per_head,
    hccs_parameters,
    min_row_sum,
    score_floor,
)

AIE_MODES = ["int16_div", "int16_clb", "uint8_div", "uint8_clb"]


def aie_oracle(x_i8, B, S, DMAX, mode, inv_shift):
    """aie4ml ``softmax_i8`` row kernel in integers (ROUNDING = floor, SATURATION = saturate):
    distance to the row max clipped at DMAX, int8 MAC into int16 scores, int32 sum, per-mode
    reciprocal broadcast as int16, product shifted by INV_SHIFT and narrowed."""
    x = x_i8.astype(np.int64)
    d = np.minimum(x.max(-1, keepdims=True) - x, DMAX)
    score = np.clip(B - d * S, -32768, 32767)
    Z = np.maximum(score.sum(-1, keepdims=True), 1)
    k = np.floor(np.log2(Z)).astype(np.int64)
    s = inv_shift
    if mode == "int16_div":
        inv = np.minimum((32767 << s) // Z, 32767)
        return np.clip((score * inv) >> s, -32768, 32767), 32768
    if mode == "int16_clb":
        return np.clip(score * (32767 >> k), -32768, 32767), 32768
    if mode == "uint8_div":
        inv = np.minimum((255 << s) // Z, 32767)
        return np.clip((score * inv) >> s, 0, 255), 256
    inv = 255 << np.maximum(s - k, 0)
    return np.clip((score * inv) >> s, 0, 255), 256


@pytest.mark.parametrize(
    "mode,inv_shift", [(m, None) for m in AIE_MODES] + [("int16_div", 4), ("uint8_div", 12), ("uint8_clb", 8)]
)
def test_kernel_matches_aie_oracle(mode, inv_shift):
    torch.manual_seed(0)
    H = 4
    x = torch.randn(4, H, 8, 64) * 3
    B = torch.tensor([511, 255, 300, 4], dtype=torch.int32)
    S = torch.tensor([7, 3, 12, 1], dtype=torch.int32)
    D = torch.tensor([64, 80, 24, 3], dtype=torch.uint8)
    sh = torch.tensor([4, 3, 5, 2], dtype=torch.int32)
    p = hccs_forward_per_head(x, B, S, D, sh, mode=mode, inv_shift=inv_shift)
    s = default_inv_shift(mode) if inv_shift is None else inv_shift
    for h in range(H):
        x_i8 = np.clip(np.rint(x[:, h].cpu().numpy() * 2.0 ** int(sh[h])), -128, 127)
        want, unit = aie_oracle(x_i8, int(B[h]), int(S[h]), int(D[h]), mode, s)
        got = torch.round(p[:, h] * unit).long().cpu().numpy()
        assert np.array_equal(got, want), mode
    by_scale = hccs_forward_per_head(x, B, S, D, None, mode=mode, inv_shift=inv_shift, scale_per_head=torch.exp2(sh.float()))
    assert torch.equal(p, by_scale)


def test_floor_log2_is_exact():
    z = torch.arange(1, 32768, dtype=torch.int32)
    assert _floor_log2(z).tolist() == [v.bit_length() - 1 for v in range(1, 32768)]


def test_min_row_sum_and_score_floor():
    assert min_row_sum("int16_div") == 1 and min_row_sum("int16_div", 4) == 16 and min_row_sum("int16_clb") == 1
    assert min_row_sum("uint8_div") == 256 and min_row_sum("uint8_clb") == 256
    # B >= 256 covers the row sum on its own: floor 1 in every mode
    assert score_floor(64, 511) == 1 and score_floor(64, 511, "uint8_div") == 1 and score_floor(128, 255, "uint8_clb") == 1
    assert score_floor(1024, 31, "uint8_div") == 1  # ceil((256 - 31) / 1023)
    assert score_floor(4, 10, "uint8_div") == 82  # ceil((256 - 10) / 3)
    assert score_floor(64, torch.tensor([511.0, 10.0]), "uint8_div").tolist() == [1.0, 4.0]


def test_kernel_refuses_a_head_count_mismatch():
    with pytest.raises(ValueError):
        hccs_forward_per_head(
            torch.randn(2, 2, 4, 4),
            torch.tensor([10, 20, 30]),
            torch.tensor([1, 1, 1]),
            torch.tensor([8, 8, 8], dtype=torch.uint8),
            torch.tensor([4, 4, 4]),
        )


def _lane_config(mode=DynamicDataMode.TRAINING, hgq=False):
    """A config for the module's PQuant lanes: an int8-grid input lane (k1 i3 f4)."""
    config = dst_config()
    qp = config.quantization_parameters
    qp.enable_quantization, qp.use_high_granularity_quantization, qp.dynamic_data_mode = True, hgq, mode
    qp.default_data_keep_negatives, qp.default_data_integer_bits, qp.default_data_fractional_bits = 1.0, 3.0, 4.0
    return config


def _kernel_on_the_lane(m, x):
    """The integer kernel at the exported params, on the grid the input lane puts ``x`` on."""
    P = m.export_params()
    scale = torch.exp2(m.log2_scale)
    return hccs_forward_per_head(m.input_quantizer(x), P["B"], P["S"], P["DMAX"], None, scale_per_head=scale)


def test_learned_bitexact_export_and_two_forwards():
    torch.manual_seed(0)
    m = HCCSLearned(2, _lane_config())
    x = torch.randn(2, 2, 8, 64) * 2
    m.train()
    m(x)  # the input lane takes its range from the batch; seeds sigma
    assert m.initialized and m.max_keys == 64
    m.eval()
    P = m.export_params()
    with torch.no_grad():
        # the kernel runs on the input lane's grid, whose step is the int8 step (log2_scale mirrors the lane's f)
        _, _, f_lane = m.input_quantizer.get_quantization_bits()
        assert torch.equal(m.log2_scale, f_lane.expand(2))
        ref = _kernel_on_the_lane(m, x)
    assert torch.equal(m(x), ref)
    assert (
        bool((P["B"] - P["S"] * P["DMAX"] >= 1).all())
        and bool((m.logit_scale >= 2**-0.5).all())
        and bool((m.logit_scale <= 2**0.5).all())
    )
    m.train()
    x1 = torch.randn(2, 2, 8, 64, requires_grad=True)
    (m(x1).square().sum() + m(x1 * 0.5).square().sum()).backward()
    # the ramp (slope, tail) trains; the scale is the input lane's, a buffer
    assert all(float(g.abs().sum()) > 0 for g in (x1.grad, m.log2_slope.grad, m.log2_tail.grad))
    assert len(hccs_parameters(m)) == 2
    with torch.no_grad():
        m.log2_slope.add_(100.0)
    m(x1)  # a training forward projects the latents back: S = sigma / a <= 127
    assert float((m.log2_slope - m.log2_scale).max()) <= math.log2(127.0) + 1e-6


def test_learned_exports_the_expected_integers():
    """Cold start for 48 keys on a k1 i3 f4 lane: budget 682, sigma 0.35 * 682, tau 0.05 * 682, a = 16."""
    m = HCCSLearned.cold(1, _lane_config(DynamicDataMode.NEVER), 48)
    m(torch.zeros(1, 1, 1, 48))
    assert {k: int(v) for k, v in m.export_params().items()} == {"B": 679, "S": 15, "DMAX": 43, "shift_eff": 4}


def _kl_to_softmax(m, x):
    """Mean per-row KL of the float softmax against the module's renormalized output."""
    t = torch.softmax(x, dim=-1)
    p = m(x)
    p = p / p.sum(dim=-1, keepdim=True)
    return (t * (torch.log(t.clamp_min(1e-12)) - torch.log(p.clamp_min(1e-12)))).sum(dim=-1).mean()


def test_learned_slope_trains():
    """Gradient descent on the ramp improves on the seed and keeps deployable integers."""
    torch.manual_seed(0)
    rows = torch.randn(1, 1, 512, 64)
    m = HCCSLearned(1, _lane_config())
    m.train()
    m(rows)  # seeds sigma
    with torch.no_grad():
        kl0 = float(_kl_to_softmax(m, rows))
    opt = torch.optim.Adam([m.log2_slope, m.log2_tail], lr=2e-2)
    for _ in range(200):
        loss = _kl_to_softmax(m, torch.randn(4, 1, 64, 64))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        assert float(_kl_to_softmax(m, rows)) < kl0
    P = m.export_params()
    B, S, D = (int(P[k][0]) for k in ("B", "S", "DMAX"))
    assert 1 <= S <= 127 and 1 <= D <= 127 and B - S * D >= 1


@pytest.mark.parametrize("mode", list(DynamicDataMode))
def test_learned_kernel_scale_is_the_lanes_step_at_inference(mode):
    """Whatever the config's dynamic_data_mode, the input lane keeps its grid at inference."""
    torch.manual_seed(0)
    m = HCCSLearned(2, _lane_config(mode))
    m.train()
    m(1.5 * torch.tanh(torch.randn(8, 2, 8, 16)))  # a tracking lane sizes itself here: one integer bit
    tracking = mode != DynamicDataMode.NEVER
    assert m.log2_scale.tolist() == ([6.0, 6.0] if tracking else [4.0, 4.0])
    m.eval()
    x = 6.0 * torch.tanh(torch.randn(8, 2, 8, 16))  # three integer bits
    with torch.no_grad():
        y = m(x)
        step = m.input_quantizer(x).unique().diff().min()
        assert torch.equal(torch.exp2(m.log2_scale), (1.0 / step).expand(2))
        assert torch.equal(y, _kernel_on_the_lane(m, x))


def test_learned_scale_follows_the_tracked_lane():
    """With the config's dynamic_data_i_decay the kernel's scale does not jump back after an outlier batch."""
    torch.manual_seed(0)
    config = _lane_config()
    config.quantization_parameters.dynamic_data_i_decay = 0.01
    m = HCCSLearned(2, config)
    assert m.input_quantizer.i_decay == 0.01 and m.output_quantizer.i_decay is None  # the output lane is pinned
    m.train()
    m(1.5 * torch.tanh(torch.randn(8, 2, 8, 16)))
    assert m.log2_scale.tolist() == [6.0, 6.0]
    m(6.0 * torch.tanh(torch.randn(8, 2, 8, 16)))  # an outlier batch: the lane re-grids at once
    assert m.log2_scale.tolist() == [4.0, 4.0]
    m(1.5 * torch.tanh(torch.randn(8, 2, 8, 16)))  # ... and the next small batch does not take the grid back
    assert m.log2_scale.tolist() == [4.0, 4.0]


def test_learned_trains_with_hgq_lanes():
    torch.manual_seed(0)
    m = HCCSLearned(2, _lane_config(hgq=True))
    m.train()
    x = torch.randn(4, 2, 8, 16, requires_grad=True)
    m(x)  # builds the learned lanes
    for lane in (m.input_quantizer, m.output_quantizer):
        lane.post_pre_train_function()
    (m(x).square().sum() + m.hgq_loss()).backward()
    assert all(float(g.abs().sum()) > 0 for g in (x.grad, m.log2_slope.grad, m.log2_tail.grad))
    assert float(m.hgq_loss()) > float(m.input_quantizer.hgq_loss()) > 0  # both lanes
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x), m.output_quantizer(_kernel_on_the_lane(m, x)))


def test_learned_state_dict_roundtrip():
    torch.manual_seed(0)
    config = _lane_config()
    m = HCCSLearned(2, config)
    x = torch.randn(2, 2, 8, 64) * 2
    m.train()
    m(x)
    m.eval()
    y = m(x)
    m2 = HCCSLearned(2, config, auto_seed=False, max_keys=64)
    m2.load_state_dict(m.state_dict())
    m2.eval()
    assert m2.initialized and torch.equal(m2(x), y)


def test_learned_uninitialized_falls_back_and_refuses_export():
    m = HCCSLearned(2, _lane_config(), auto_seed=False)
    x = torch.randn(1, 2, 4, 16)
    assert torch.equal(m(x), m.output_quantizer(F.softmax(m.input_quantizer(x), dim=-1)))
    with pytest.raises(RuntimeError):
        m.export_params()


def test_learned_rejects_longer_rows():
    m = HCCSLearned.cold(1, _lane_config(), 64)
    with pytest.raises(ValueError):
        m(torch.randn(1, 1, 4, 128))
    assert m(torch.randn(1, 1, 4, 32)).shape[-1] == 32  # shorter rows are fine


def test_learned_masked_keys_get_no_weight_and_have_no_say():
    torch.manual_seed(0)
    m = HCCSLearned.cold(2, _lane_config(DynamicDataMode.NEVER), 32)
    mask = torch.rand(2, 2, 8, 32) > 0.3
    mask[..., 0] = True
    for training in (True, False):
        m.train(training)
        x = (torch.randn(2, 2, 8, 32) * 2).requires_grad_()
        p = m(x, mask)
        assert bool((p[~mask] == 0).all()) and float((p.sum(-1) - 1).abs().max()) < 1e-3
        assert torch.equal(p, m(torch.where(mask, x, x + 50.0), mask))  # also when a masked key holds the row max
        p.square().sum().backward()
        assert bool((x.grad[~mask] == 0).all()) and float(x.grad[mask].abs().sum()) > 0


def test_learned_ebops_count_the_kernel_products():
    config = _lane_config()
    assert HCCSLearned.cold(2, config, 8, mode="int16_div").ebops() == 2 * 8 * (8 * (49 + 256) + 256)
    assert HCCSLearned.cold(2, config, 8, mode="uint8_clb").ebops() == 2 * 8 * 8 * 49
    assert HCCSLearned(2, config).ebops() == 0.0  # no key length yet


def test_learned_out_quant_bits_narrow_the_output():
    m = HCCSLearned.cold(1, _lane_config(DynamicDataMode.NEVER), 16, out_quant_bits=(0.0, 0.0, 4.0))
    p = m(torch.randn(2, 1, 4, 16)).detach()
    assert torch.equal(p * 16, torch.round(p * 16)) and float(p.max()) > 0


def test_learned_surrogate_has_no_zero_scores():
    m = HCCSLearned.cold(1, _lane_config(), 64)
    m.eval()
    x = torch.randn(4, 1, 8, 64) * 5
    assert bool((m._soft_forward(x, None, 64) > 0).all())


def test_learned_compile_fullgraph():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    torch.manual_seed(0)
    m = HCCSLearned.cold(2, _lane_config(), 32)
    m.eval()
    x = torch.randn(2, 2, 4, 32) * 2
    torch._dynamo.reset()
    assert torch.equal(torch.compile(m, fullgraph=True)(x), m(x))


class _Scores(nn.Module):
    def __init__(self, softmax, mask=None):
        super().__init__()
        self.softmax = softmax
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self.softmax(x) if self.mask is None else self.softmax(x, self.mask)


@pytest.mark.parametrize("fx", [False, True])
@pytest.mark.parametrize("heads,mode,inv_shift", [(1, "uint8_clb", 15), (3, "uint8_div", 12)])
def test_onnx_export_carries_each_heads_integers(heads, mode, inv_shift, fx, tmp_path):
    """One ``aie4ml.HCCSSoftmax`` per head between the two lanes; onnxruntime runs its body, a float softmax."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from pquant.core.torch.convert_to_onnx import convert_to_onnx, convert_to_onnx_fx

    torch.manual_seed(0)
    m = HCCSLearned.cold(heads, _lane_config(DynamicDataMode.NEVER), 16, mode=mode, inv_shift=inv_shift)
    with torch.no_grad():
        m.log2_slope.sub_(torch.arange(heads, dtype=torch.float32))  # a different ramp per head
    m.eval()
    x = torch.randn(3, heads, 8, 16) * 2
    with torch.no_grad():
        m(x)
        want = m.output_quantizer(F.softmax(m.input_quantizer(x), dim=-1))
    path = str(tmp_path / "hccs.onnx")
    if fx:
        proto = convert_to_onnx_fx(_Scores(m), input_shape=(heads, 8, 16), output_path=path, opset=17)
    else:
        proto = convert_to_onnx(nn.Sequential(m), input_shape=(heads, 8, 16), output_path=path, opset=17)

    nodes = [n for n in proto.graph.node if n.op_type == "HCCSSoftmax"]
    got = [{a.name: onnx.helper.get_attribute_value(a) for a in n.attribute} for n in nodes]
    P = m.export_params()
    want_ints = [tuple([int(P[k][h])] for k in ("B", "S", "DMAX")) for h in range(heads)]
    assert [(g["B"], g["S"], g["Dmax"]) for g in got] == want_ints
    assert len({tuple(g["S"]) for g in got}) == heads
    assert all(n.domain == "aie4ml" for n in nodes)
    assert all((g["param_sets"], g["inv_shift"], g["use_clb"]) == (1, inv_shift, int(mode.endswith("clb"))) for g in got)
    assert [f.name for f in proto.functions] == ["HCCSSoftmax"] and proto.ir_version >= 8
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    out = ort.InferenceSession(path, options).run(None, {proto.graph.input[0].name: x.cpu().numpy()})[0]
    assert np.allclose(out, want.cpu().numpy(), atol=1e-6)


def test_onnx_export_refuses_what_the_carrier_cannot_say(tmp_path):
    pytest.importorskip("onnx")
    from pquant.core.torch.convert_to_onnx import convert_to_onnx, convert_to_onnx_fx

    config, path = _lane_config(DynamicDataMode.NEVER), str(tmp_path / "hccs.onnx")
    with pytest.raises(ValueError):  # aie4ml's softmax emits uint8
        convert_to_onnx(nn.Sequential(HCCSLearned.cold(1, config, 16)), input_shape=(1, 4, 16), output_path=path)
    with pytest.raises(RuntimeError):  # not seeded
        unseeded = HCCSLearned(1, config, mode="uint8_clb", auto_seed=False)
        convert_to_onnx(nn.Sequential(unseeded), input_shape=(1, 4, 16), output_path=path)
    masked = _Scores(HCCSLearned.cold(1, config, 16, mode="uint8_clb"), mask=torch.ones(1, 1, 4, 16, dtype=torch.bool))
    with pytest.raises(NotImplementedError):  # the node has no mask input
        convert_to_onnx_fx(masked, input_shape=(1, 4, 16), output_path=path)


def _attention_with_hccs(hgq=False, quantize=True):
    config = _lane_config(DynamicDataMode.NEVER, hgq)
    config.quantization_parameters.enable_quantization = quantize
    config.pruning_parameters.enable_pruning = False
    config.quantization_parameters.use_hccs = True
    return PQMultiheadAttention(config, embed_dim=16, num_heads=4, batch_first=True), config


def test_use_hccs_puts_the_kernel_into_attention_layers():
    mha, config = _attention_with_hccs()
    assert isinstance(mha.softmax, HCCSLearned) and mha.softmax.mode == "uint8_clb" and mha.softmax.num_heads == 4
    assert isinstance(PQMultiheadAttention(config, 16, 4, use_hccs=False).softmax, PQSoftmax)  # the argument decides
    config.quantization_parameters.use_hccs = False
    assert isinstance(PQMultiheadAttention(config, 16, 4).softmax, PQSoftmax)
    assert isinstance(PQMultiheadAttention(config, 16, 4, use_hccs=True).softmax, HCCSLearned)
    narrow = PQMultiheadAttention(config, 16, 4, use_hccs=True, attn_quant_bits=(0.0, 0.0, 6.0))
    assert float(narrow.softmax.output_quantizer.f) == 6.0  # attn_quant_bits is the kernel's output lane

    config.quantization_parameters.use_hccs = True
    model = add_compression_layers(_SelfAttention(nn.MultiheadAttention(16, 4, batch_first=True)), config, (2, 12, 16))
    assert isinstance(model.mha.softmax, HCCSLearned)
    assert model(torch.randn(2, 12, 16)).shape == (2, 12, 16)


@pytest.mark.parametrize("hgq", [False, True])
def test_attention_layer_runs_hccs_in_place_of_its_softmax(hgq):
    torch.manual_seed(0)
    mha, config = _attention_with_hccs(hgq)
    sm = mha.softmax
    x = torch.randn(3, 12, 16)
    padding = torch.zeros(3, 12, dtype=torch.bool)
    padding[:, -3:] = True
    seen = {}
    sm.register_forward_hook(
        lambda m, args, kwargs, out: seen.update(scores=args[0], mask=kwargs["mask"], out=out), with_kwargs=True
    )
    y, weights = mha(x, x, x, key_padding_mask=padding)

    P = sm.export_params()
    with torch.no_grad():
        scale, valid = torch.exp2(sm.log2_scale), seen["mask"].expand_as(seen["scores"])
        scores = sm.input_quantizer(seen["scores"])
        kernel = hccs_forward_per_head(
            scores, P["B"], P["S"], P["DMAX"], None, sm.mode, valid_mask=valid, scale_per_head=scale
        )
        assert torch.equal(seen["out"], sm.output_quantizer(kernel))
    assert float(weights[:, :, -3:].abs().max()) == 0 and float((weights.sum(-1) - 1).abs().max()) < 0.01

    post_pretrain_functions(mha, config)
    (y.square().sum() + get_model_losses(mha, torch.zeros(()))).backward()
    assert all(float(g.abs().sum()) > 0 for g in (mha.q_proj._weight.grad, sm.log2_slope.grad, sm.log2_tail.grad))
    parts = [mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj]
    assert sm.ebops() > 0 and sm.input_shape == (1, 4, 12, 12)
    assert float(get_ebops(mha)) == float(sum(p.ebops(include_mask=False) for p in parts) + mha.ebops() + sm.ebops())
    if hgq:  # in float64: the lanes' share is far below float32's resolution of the total
        others = sum(float(p.hgq_loss()) for p in parts) + float(mha.hgq_loss())
        total = float(get_model_losses(mha, torch.zeros((), dtype=torch.float64)))
        assert float(sm.hgq_loss()) > 0 and total - others == pytest.approx(float(sm.hgq_loss()), rel=1e-3)


class _SelfAttention(nn.Module):
    def __init__(self, mha):
        super().__init__()
        self.mha = mha

    def forward(self, x):
        return self.mha(x, x, x)[0]


def test_attention_layer_with_hccs_is_not_exported_yet(tmp_path):
    pytest.importorskip("onnx")
    from pquant.core.torch.convert_to_onnx import convert_to_onnx_fx

    mha, _ = _attention_with_hccs(quantize=False)
    model = _SelfAttention(mha)
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 12, 16))
    with pytest.raises(NotImplementedError):
        convert_to_onnx_fx(model, input_shape=(12, 16), output_path=str(tmp_path / "mha.onnx"))
