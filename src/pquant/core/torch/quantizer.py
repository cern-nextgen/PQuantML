from enum import Enum

import torch
import torch.nn as nn

from pquant.core.torch.fixed_point_quantizer import get_fixed_quantizer
from pquant.core.torch.hgq_quantizer import HGQQuantizer


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
        granularity="per_tensor",
        hgq_gamma=0,
        place="datalane",
        dynamic_data=True,
    ):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(float(k)), requires_grad=False)
        self.overflow = overflow
        self.b_init = k + i + f
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.is_data = is_data
        self.dynamic_data = dynamic_data
        self.i_init = i
        self.f_init = f
        self.i = torch.nn.Parameter(torch.tensor(i), requires_grad=False)
        self.f = torch.nn.Parameter(torch.tensor(f), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor(i + k + f), requires_grad=False)
        self.granularity = granularity.value if isinstance(granularity, Enum) else granularity
        self.quantizer = create_quantizer(
            self.k,
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
        self.final_compression_done = nn.Parameter(torch.tensor(False), requires_grad=False)
        if self.granularity == "per_tensor":
            self.initialize_quantization_parameters(self.i_init, self.f_init)

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.k, self.quantizer.i, self.quantizer.f
        return self.k, self.i, self.f

    def get_total_bits(self, shape):
        if self.use_hgq:
            return self.quantizer.bits_(shape)
        b = self.i + self.f + self.k
        return torch.ones(shape).to(b.device) * b

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.set_bits(i, f)
        self.i.data = torch.tensor(i)
        self.f.data = torch.tensor(f)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def calculate_bits_from_abs(self, abs_x):
        m = torch.ceil(torch.log2(abs_x + 1e-6))
        int_bits = torch.clamp(m, min=0)
        b = self.b if hasattr(self, "b") else self.k + self.i_init + self.f_init
        int_bits = torch.clamp(m, max=b - self.k.to(m.device))
        frac_bits = torch.clamp(b - int_bits - self.k, min=0)
        return int_bits, frac_bits

    def compute_data_dynamic_bits(self, x):
        if not (self.training and self.dynamic_data):
            _, i, f = self.get_quantization_bits()
            return i, f
        abs_x = torch.amax(torch.abs(x))
        return self.calculate_bits_from_abs(abs_x)

    def compute_weight_dynamic_bits(self, x):
        if self.granularity == "per_tensor" or x.ndim == 1:
            _, i, f = self.get_quantization_bits()
            return i, f
        if self.granularity == "per_channel":
            if x.ndim == 2:
                abs_x = torch.amax(torch.abs(x), dim=1, keepdim=True)
            elif x.ndim == 3:
                abs_x = torch.amax(torch.abs(x), dim=(1, 2), keepdim=True)
            elif x.ndim == 4:
                abs_x = torch.amax(torch.abs(x), dim=(1, 2, 3), keepdim=True)
        elif self.granularity == "per_weight":
            abs_x = torch.abs(x)
        else:
            raise ValueError("The selected granularity is not supported.")
        return self.calculate_bits_from_abs(abs_x)

    def compute_dynamic_bits(self, x):
        if self.is_data:
            return self.compute_data_dynamic_bits(x)
        return self.compute_weight_dynamic_bits(x)

    def forward(self, x):
        if self.use_hgq:
            x = self.quantizer(x, training=self.training)
            _, i, f = self.get_quantization_bits()
            self.initialize_quantization_parameters(i, f)
            return x
        i, f = self.compute_dynamic_bits(x)
        self.initialize_quantization_parameters(i, f)
        self.i.data = i
        self.f.data = f
        _, i, f = self.get_quantization_bits()
        x = self.quantizer(x, k=self.k, i=i, f=f, training=self.training)
        return x

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
        _, i, f = self.get_quantization_bits()
        self.i.data = i
        self.f.data = f
        self.b.data = i + f
        self.final_compression_done.data = torch.tensor(True)

    def initialize_quantization_parameters(self, i, f):
        if hasattr(self, "f"):
            return
        # Lazy initialization
        self.i = torch.nn.Parameter(torch.tensor(i), requires_grad=False)
        self.f = torch.nn.Parameter(torch.tensor(f), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor(self.k.detach().clone() + i + f), requires_grad=False)

    def reload_from_local(self):
        if not self.use_hgq:
            return
        self.quantizer.set_bits(self.i, self.f)


def create_quantizer(k, i, f, overflow, round_mode, is_heterogeneous, is_data, granularity="per_weight", gamma=1e-8):
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
        )
    return get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow)
