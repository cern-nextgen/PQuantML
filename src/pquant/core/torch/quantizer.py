import torch
import torch.nn as nn

from pquant.core.constants import QuantizationGranularity
from pquant.core.torch.fixed_point_quantizer import get_fixed_quantizer
from pquant.core.torch.hgq_quantizer import HGQQuantizer


def _register_final_compression_flag(module):
    module.register_buffer("final_compression_done", torch.tensor(False))
    module._final_compression_done = False
    module.register_load_state_dict_post_hook(_sync_final_compression_flag)


def _sync_final_compression_flag(module, incompatible_keys=None):
    module._final_compression_done = bool(module.final_compression_done)


def _mark_final_compression_done(module):
    module.final_compression_done.fill_(True)
    module._final_compression_done = True


class Quantizer(nn.Module):
    def __init__(
        self,
        k,
        i,
        f,
        overflow,
        round_mode,
        is_heterogeneous,
        is_data=False,
        granularity=QuantizationGranularity.PER_TENSOR,
        hgq_gamma=1e-8,
        place="datalane",
        dynamic_data=True,
        shape=None,
    ):
        super().__init__()

        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.is_data = is_data
        self.dynamic_data = dynamic_data
        self.granularity = QuantizationGranularity(granularity).value

        # Params even when using HGQ, as they are used during hls4ml conversion
        param_shape = () if (is_data or self.use_hgq) else self.compute_weight_param_shape(shape)
        self.k = torch.nn.Parameter(torch.full(param_shape, float(k)), requires_grad=False)
        self.i = torch.nn.Parameter(torch.full(param_shape, float(i)), requires_grad=False)
        self.f = torch.nn.Parameter(torch.full(param_shape, float(f)), requires_grad=False)
        self.b = torch.nn.Parameter(torch.full(param_shape, float(i + k + f)), requires_grad=False)
        self.quantizer = create_quantizer(
            k,
            i,
            f,
            self.overflow,
            self.round_mode,
            self.use_hgq,
            self.is_data,
            granularity=self.granularity,
            gamma=hgq_gamma,
        )
        self.is_pretraining = True
        self.hgq_gamma = hgq_gamma
        _register_final_compression_flag(self)

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.k, self.quantizer.i, self.quantizer.f
        else:
            return self.k, self.i, self.f

    def get_total_bits(self, shape):
        if self.use_hgq:
            return self.quantizer.bits_(shape)
        else:
            b = self.i + self.f + self.k
            return torch.ones(shape).to(b.device) * b

    def _sync_hgq_mirror_bits(self):
        if not self.quantizer.built:
            return
        with torch.no_grad():
            k, i, f = self.quantizer.k.detach(), self.quantizer.i.detach(), self.quantizer.f.detach()
            self.k.data = k.clone()
            self.i.data = i.clone()
            self.f.data = f.clone()
            self.b.data = k + i + f

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.set_bits(i, f)
        else:
            self.i.data = torch.as_tensor(i, dtype=self.i.dtype, device=self.i.device).broadcast_to(self.i.shape).clone()
            self.f.data = torch.as_tensor(f, dtype=self.f.dtype, device=self.f.device).broadcast_to(self.f.shape).clone()

    def post_pre_train_function(self):
        self.is_pretraining = False

    def calculate_bits_from_abs(self, abs_x):
        m = torch.ceil(torch.log2(abs_x + 1e-6))
        int_bits = torch.clamp(m, min=0).clamp(max=self.b - self.k.to(m.device))
        frac_bits = torch.clamp(self.b - int_bits - self.k, min=0)
        return int_bits, frac_bits

    def compute_data_dynamic_bits(self, x):
        if not (self.training and self.dynamic_data):
            _, i, f = self.get_quantization_bits()
            return i, f
        abs_x = torch.amax(torch.abs(x))
        return self.calculate_bits_from_abs(abs_x)

    def compute_weight_param_shape(self, shape):
        if shape is None or self.granularity == QuantizationGranularity.PER_TENSOR or len(shape) == 1:
            return ()
        elif self.granularity == QuantizationGranularity.PER_CHANNEL:
            return (shape[0],) + (1,) * (len(shape) - 1)  # Channels first
        else:
            return shape

    def _reduce_absmax(self, x):
        if self.is_data or self.granularity == QuantizationGranularity.PER_TENSOR or x.ndim == 1:
            return torch.amax(torch.abs(x))
        if self.granularity == QuantizationGranularity.PER_CHANNEL:
            return torch.amax(torch.abs(x), dim=tuple(range(1, x.ndim)), keepdim=True)
        if self.granularity == QuantizationGranularity.PER_WEIGHT:
            return torch.abs(x)
        raise ValueError("The selected granularity is not supported.")

    def compute_weight_dynamic_bits(self, x):
        if self.granularity == QuantizationGranularity.PER_TENSOR or x.ndim == 1 or not self.training:
            _, i, f = self.get_quantization_bits()
            return i, f
        return self.calculate_bits_from_abs(self._reduce_absmax(x))

    def compute_dynamic_bits(self, x):
        if self.is_data:
            return self.compute_data_dynamic_bits(x)
        return self.compute_weight_dynamic_bits(x)

    def forward(self, x):
        if self.use_hgq:
            return self.quantizer(x, training=self.training)
        if self._final_compression_done:
            return self.quantizer(x, k=self.k, i=self.i, f=self.f, training=False)
        i, f = self.compute_dynamic_bits(x)
        with torch.no_grad():
            if self.i.shape == i.shape and self.f.shape == f.shape:
                self.i.copy_(i)
                self.f.copy_(f)
            else:
                self.i.data = i
                self.f.data = f
        return self.quantizer(x, k=self.k, i=i, f=f, training=self.training)

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        return self.quantizer.regularization_loss()

    def post_epoch_function(self):
        if self.use_hgq and self.quantizer.built:
            self.quantizer.post_epoch_constraint_apply()

    def apply_final_compression(self):
        if self.use_hgq and not self.quantizer.built:
            return
        if self.use_hgq:
            with torch.no_grad():
                self.quantizer._f.data.clamp_(self.quantizer.f_min, self.quantizer.f_max)
                if self.quantizer.overflow_mode != "WRAP":
                    self.quantizer._i.data.clamp_(self.quantizer.i_min, self.quantizer.i_max)
            self._sync_hgq_mirror_bits()
            _mark_final_compression_done(self)
            return
        _, i, f = self.get_quantization_bits()
        self.i.data = i
        self.f.data = f
        self.b.data = i + f
        _mark_final_compression_done(self)


def create_quantizer(
    k, i, f, overflow, round_mode, is_heterogeneous, is_data, granularity=QuantizationGranularity.PER_WEIGHT, gamma=1e-8
):
    if is_heterogeneous:
        return HGQQuantizer(
            k0=k,
            i0=i,
            f0=f,
            overflow_mode=overflow,
            round_mode=round_mode,
            is_data=is_data,
            granularity=granularity,
            gamma=gamma,
            # hgq's defaults: data lanes let their WRAP integer bits decay 0.01 per step, weights track exactly
            i_decay_speed=0.01 if is_data else float("inf"),
        )
    else:
        return get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow)
