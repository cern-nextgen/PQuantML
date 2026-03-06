from enum import Enum

import torch
import torch.nn as nn

from pquant.core.quantizer_functions import create_quantizer


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
        granularity='per_tensor',
        hgq_gamma=0,
        place="datalane",
    ):
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor(k), requires_grad=False)
        self.overflow = overflow
        self.b_init = k + i + f
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.is_data = is_data
        self.i_init = i
        self.f_init = f
        self.i = torch.nn.Parameter(torch.tensor(i), requires_grad=False)
        self.f = torch.nn.Parameter(torch.tensor(f), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor(i + k + f), requires_grad=False)
        self.quantizer = create_quantizer(self.k, i, f, self.overflow, self.round_mode, self.use_hgq, self.is_data, place)
        self.is_pretraining = True
        self.hgq_gamma = hgq_gamma
        if isinstance(granularity, Enum):
            self.granularity = granularity.value
        else:
            self.granularity = granularity
        self.final_compression_done = nn.Parameter(torch.tensor(False), requires_grad=False)
        if self.granularity == 'per_tensor':
            self.initialize_quantization_parameters(self.i_init, self.f_init)

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.quantizer.k, self.quantizer.quantizer.i, self.quantizer.quantizer.f
        else:
            return self.k, self.i, self.f

    def get_total_bits(self, shape):
        if self.use_hgq:
            return self.quantizer.bits_(shape)
        else:
            b = self.i + self.f + self.k
            return torch.ones(shape).to(b.device) * b

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.quantizer._i.assign(self.quantizer.quantizer._i * 0.0 + i)
            self.quantizer.quantizer._f.assign(self.quantizer.quantizer._f * 0.0 + f)
        self.i.data = torch.tensor(i)
        self.f.data = torch.tensor(f)

    def post_pre_train_function(self):
        self.is_pretraining = False

    def compute_dynamic_bits(self, x):
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

        m = torch.ceil(torch.log2(abs_x + 1e-6))
        int_bits = torch.clamp(m, min=0)
        b = self.b if hasattr(self, "b") else self.k + self.i_init + self.f_init
        frac_bits = torch.clamp(b - int_bits - self.k, min=0)
        return int_bits, frac_bits

    def forward(self, x):
        if self.use_hgq:
            x = self.quantizer(x, training=self.training)
            _, i, f = self.get_quantization_bits()
            self.initialize_quantization_parameters(i, f)
            return x
        else:
            if self.granularity == 'per_tensor':
                self.initialize_quantization_parameters(self.i_init, self.f_init)
                _, i, f = self.get_quantization_bits()
                return self.quantizer(x, k=self.k, i=i, f=f, training=self.training)
            else:
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
        loss = 0
        for layer_loss in self.quantizer.quantizer.losses:
            loss += layer_loss
        return loss

    def post_epoch_function(self):
        if self.use_hgq and self.quantizer.quantizer.built:
            constrained_i = self.quantizer.quantizer._i.constraint(self.quantizer.quantizer._i)
            self.quantizer.quantizer._i.assign(constrained_i)
            constrained_f = self.quantizer.quantizer._f.constraint(self.quantizer.quantizer._f)
            self.quantizer.quantizer._f.assign(constrained_f)

    def apply_final_compression(self):
        if self.use_hgq and not self.quantizer.built:
            return
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
        self.quantizer.quantizer._i.assign(self.i)
        self.quantizer.quantizer._f.assign(self.f)
