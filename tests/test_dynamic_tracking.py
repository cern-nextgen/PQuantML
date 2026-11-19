import copy
import math
import os

import pytest
import torch
from torch import nn

os.environ["KERAS_BACKEND"] = "torch"

from pquant.activations import PQActivation, PQSoftmax  # noqa: E402
from pquant.layers import PQAvgPool1d, PQBatchNorm1d, PQDense, PQLayerNorm  # noqa: E402

from pquant import pdp_config  # noqa: E402
from pquant.core.constants import DynamicDataMode, QuantizationGranularity  # noqa: E402
from pquant.core.torch.quantizer import Quantizer  # noqa: E402
from pquant.core.torch.tracing import check_quantization  # noqa: E402


def lane(decay=None, k=1.0, f=7.0, mode=DynamicDataMode.TRAINING, **kwargs):
    q = Quantizer(
        k=k,
        i=0.0,
        f=f,
        overflow="SAT",
        round_mode="RND",
        is_heterogeneous=False,
        is_data=True,
        dynamic_data_mode=mode,
        i_decay=decay,
        **kwargs,
    )
    q.train()
    return q


def batch(amax, n=64):
    x = torch.linspace(-1.0, 1.0, n)
    return x * amax


def bits(q):
    return float(q.i), float(q.f)


def test_batch_mode_is_unchanged():
    q = lane(None)
    assert q.i_decay is None and not hasattr(q, "i_track")
    q(batch(1.0))
    assert bits(q) == (1.0, 6.0)
    q(batch(100.0))
    assert bits(q) == (7.0, 0.0)
    q(batch(1.0))  # the next batch takes the grid back: the behaviour the tracker replaces
    assert bits(q) == (1.0, 6.0)


def test_tracked_bits_rise_at_once_and_decay_slowly():
    q = lane(0.01)
    q(batch(1.0))
    assert bits(q) == (1.0, 6.0) and float(q.i_track) == 1.0 and bool(q.track_seen)
    q(batch(100.0))  # an outlier batch: covered immediately
    assert bits(q) == (7.0, 0.0) and float(q.i_track) == 7.0
    for _ in range(99):
        q(batch(1.0))
    assert bits(q) == (7.0, 0.0)  # 6.01 tracked: the lane still uses 7 integer bits
    assert math.isclose(float(q.i_track), 7.0 - 0.99, abs_tol=1e-4)
    q(batch(1.0))
    assert bits(q) == (6.0, 1.0)  # the 100th small batch brings the tracker to 6.0
    for _ in range(500):
        q(batch(1.0))
    assert bits(q) == (1.0, 6.0)  # ... and it never goes below what the batches need
    assert float(q.i_track) == 1.0


def test_tracker_honours_the_lane_bounds():
    q = lane(0.5)
    q(batch(0.001))  # needs -9: clamped at the lane's lowest integer bits
    assert bits(q) == (-8.0, 15.0)
    q(batch(1e6))  # needs 20: clamped at b - k
    assert bits(q) == (7.0, 0.0)


def test_eval_and_final_compression_use_the_tracked_split():
    q = lane(0.01)
    q(batch(100.0))
    q(batch(1.0))
    q.eval()
    y = q(batch(1.0))
    assert bits(q) == (7.0, 0.0)  # not the batch's (1, 6)
    assert torch.equal(y, torch.round(batch(1.0)))  # 0 fractional bits: integers
    q.train()
    q.apply_final_compression()
    assert bits(q) == (7.0, 0.0) and float(q.b) == 8.0
    q(batch(0.1))
    assert bits(q) == (7.0, 0.0)  # frozen


def test_state_dict_round_trip():
    q = lane(0.01)
    q(batch(100.0))
    q(batch(1.0))
    state = q.state_dict()
    assert "i_track" in state and "track_seen" in state
    fresh = lane(0.01)
    fresh.load_state_dict(state)
    assert math.isclose(float(fresh.i_track), float(q.i_track)) and bits(fresh) == bits(q)
    # a checkpoint saved before the tracker existed seeds it from its stored bits
    old = {k: v for k, v in state.items() if k not in ("i_track", "track_seen")}
    seeded = lane(0.01)
    seeded.load_state_dict(old)
    assert float(seeded.i_track) == float(q.i) and bool(seeded.track_seen) and bits(seeded) == bits(q)


def test_weight_and_static_lanes_do_not_track():
    w = Quantizer(
        k=1.0,
        i=0.0,
        f=7.0,
        overflow="SAT",
        round_mode="RND",
        is_heterogeneous=False,
        is_data=False,
        dynamic_data_mode=DynamicDataMode.TRAINING,
        i_decay=0.02,
    )
    s = lane(0.02, mode=DynamicDataMode.NEVER)
    assert w.i_decay is None and s.i_decay is None
    assert "i_decay=0.02" in repr(lane(0.02))


def test_compiled_forward_tracks_like_eager():
    eager = lane(0.01)
    compiled = copy.deepcopy(eager)
    fn = torch.compile(compiled)
    for amax in (1.0, 100.0, 1.0, 1.0):
        x = batch(amax)
        ye, yc = eager(x), fn(x)
        assert torch.equal(ye, yc)
        assert bits(eager) == bits(compiled) and math.isclose(float(eager.i_track), float(compiled.i_track), abs_tol=1e-6)


def test_always_lanes_follow_every_batch_and_do_not_track():
    q = lane(0.01, mode=DynamicDataMode.ALWAYS)
    assert q.i_decay is None and not hasattr(q, "i_track")
    q(batch(100.0))
    q(batch(1.0))
    assert bits(q) == (1.0, 6.0)


@pytest.mark.parametrize("granularity", [QuantizationGranularity.PER_TOKEN, QuantizationGranularity.PER_BLOCK])
def test_row_wise_lanes_size_every_row_and_do_not_track(granularity):
    q = lane(0.01, granularity=granularity, block_size=8)
    assert q.i_decay is None and not hasattr(q, "i_track")
    x = torch.stack([batch(1.0, 8), batch(100.0, 8)])
    y = q(x)
    assert torch.equal(y[0], torch.round(x[0] * 64) / 64) and torch.equal(y[1], torch.round(x[1]))


def _tracking_config():
    config = pdp_config()
    config.pruning_parameters.enable_pruning = False
    qp = config.quantization_parameters
    qp.enable_quantization, qp.dynamic_data_mode, qp.dynamic_data_i_decay = True, DynamicDataMode.TRAINING, 0.02
    return config


def test_training_mode_tracks_by_default_and_null_follows_every_batch():
    config = _tracking_config()
    qp = config.quantization_parameters
    assert pdp_config().quantization_parameters.dynamic_data_i_decay == 0.01
    for decay in (0.01, None):
        qp.dynamic_data_i_decay = decay
        dense = PQDense(config, 8, 8)
        dense(torch.randn(4, 8))
        assert dense.input_quantizer.i_decay == decay


def test_config_decay_reaches_every_data_lane_the_layers_build():
    config = _tracking_config()
    layers = {
        "dense": (PQDense(config, 8, 8, quantize_output=True), torch.randn(4, 8)),
        "pool": (PQAvgPool1d(config, 2, quantize_output=True), torch.randn(4, 3, 8)),
        "batchnorm": (PQBatchNorm1d(config, 8), torch.randn(4, 8)),
        "layernorm": (PQLayerNorm(config, 8, quantize_output=True), torch.randn(4, 8)),
        "activation": (PQActivation(config, "relu", quantize_input=True, quantize_output=True), torch.randn(4, 8)),
        "softmax": (PQSoftmax(config, -1, quantize_input=True, quantize_output=True), torch.randn(4, 8)),
    }
    for name, (layer, x) in layers.items():
        layer(x)
        lanes = [q for q in layer.modules() if isinstance(q, Quantizer)]
        assert layer.input_quantizer.i_decay == 0.02, name
        assert name == "batchnorm" or layer.output_quantizer.i_decay == 0.02, name
        assert all(q.i_decay == (0.02 if q.is_data else None) for q in lanes), name  # not the weight and bias lanes


def test_config_decay_reaches_the_lanes_the_tracing_pass_adds():
    config = _tracking_config()

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = PQDense(config, 8, 8)

        def forward(self, x):
            return x + self.a(x)

    model = Residual()
    model(torch.randn(4, 8))  # built: the pass constructs the missing output lane in place
    traced = check_quantization(model, add_missing_quantizers=True, config=config)
    added = [m for name, m in traced.named_modules() if name.startswith("_auto_missing_quantizer_")]
    assert added and all(q.i_decay == 0.02 for q in added)
    assert model.a.output_quantizer.i_decay == 0.02
