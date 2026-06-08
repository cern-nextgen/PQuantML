from enum import Enum

import keras
from hgq.quantizer import Quantizer as HGQQuantizer
from hgq.quantizer import QuantizerConfig
from keras import ops
from quantizers import get_fixed_quantizer


@keras.saving.register_keras_serializable(package="PQuantML")
class Quantizer(keras.layers.Layer):
    # HGQ quantizer wrapper
    def __init__(
        self,
        k=0.0,
        i=0.0,
        f=7.0,
        overflow="SAT",
        round_mode="RND",
        is_heterogeneous=False,
        is_data=False,
        granularity="per_tensor",
        hgq_gamma=0,
        place="datalane",
    ):
        super().__init__()
        self.k_init = float(k)
        self.i_init = float(i)
        self.f_init = float(f)
        self.b_init = self.k_init + self.i_init + self.f_init
        self.overflow = overflow
        self.round_mode = round_mode
        self.use_hgq = is_heterogeneous
        self.is_data = is_data
        self.place = place
        self.quantizer = create_quantizer(
            self.k_init, self.i_init, self.f_init, self.overflow, self.round_mode, self.use_hgq, self.is_data, place
        )
        self.is_pretraining = True
        self.hgq_gamma = hgq_gamma
        if isinstance(granularity, Enum):
            self.granularity = granularity.value
        else:
            self.granularity = granularity

    def compute_dynamic_bits(self, x):
        if self.granularity == "per_channel":
            if ops.ndim(x) == 2:
                abs_x = ops.max(ops.abs(x), axis=0, keepdims=True)
            elif ops.ndim(x) == 3:
                abs_x = ops.max(ops.abs(x), axis=(0, 1), keepdims=True)
            elif ops.ndim(x) == 4:
                abs_x = ops.max(ops.abs(x), axis=(0, 1, 2), keepdims=True)
            else:
                raise ValueError("Unsupported tensor rank")
        elif self.granularity == "per_weight":
            abs_x = ops.abs(x)
        else:
            raise ValueError(f"compute_dynamic_bits called for granularity={self.granularity}")
        m = ops.ceil(ops.log(abs_x + 1e-6) / ops.log(2.0))
        int_bits = ops.maximum(m, 0.0)
        b = self.b if hasattr(self, "b") else self.b_init
        frac_bits = ops.maximum(b - int_bits - self.k_init, 0.0)
        return int_bits, frac_bits

    def build(self, input_shape):
        if self.use_hgq:
            shape = tuple(input_shape) if not self.is_data else (1,) + tuple(input_shape[1:])
            self.k = self.add_weight(shape=shape, initializer=keras.initializers.Constant(self.k_init), trainable=False)
            self.i = self.add_weight(shape=shape, initializer=keras.initializers.Constant(self.i_init), trainable=False)
            self.f = self.add_weight(shape=shape, initializer=keras.initializers.Constant(self.f_init), trainable=False)
            self.b = self.add_weight(
                shape=shape,
                initializer=keras.initializers.Constant(self.k_init + self.i_init + self.f_init),
                trainable=False,
            )
            if not self.quantizer.built:
                self.quantizer.build(shape)
            self.set_quantization_bits(self.i_init, self.f_init)
        elif self.granularity == "per_tensor":
            self.k = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.k_init), trainable=False)
            self.i = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.i_init), trainable=False)
            self.f = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.f_init), trainable=False)
            self.b = self.add_weight(
                shape=(), initializer=keras.initializers.Constant(self.k_init + self.f_init + self.f_init), trainable=False
            )
        else:
            i, _ = self.compute_dynamic_bits(keras.ops.ones(input_shape))
            self.k = self.add_weight(shape=i.shape, initializer=keras.initializers.Constant(self.k_init), trainable=False)
            self.i = self.add_weight(shape=i.shape, initializer=keras.initializers.Constant(self.i_init), trainable=False)
            self.f = self.add_weight(shape=i.shape, initializer=keras.initializers.Constant(self.f_init), trainable=False)
            self.b = self.add_weight(
                shape=i.shape,
                initializer=keras.initializers.Constant(self.k_init + self.f_init + self.f_init),
                trainable=False,
            )

        super().build(input_shape)

    def get_total_bits(self, shape):
        if self.use_hgq:
            return self.quantizer.bits_(shape)
        else:
            b = self.i + self.f + self.k
            return keras.ops.ones(shape) * b

    def get_quantization_bits(self):
        if self.use_hgq:
            return self.quantizer.quantizer.k, self.quantizer.quantizer.i, self.quantizer.quantizer.f
        return self.k, self.i, self.f

    def set_quantization_bits(self, i, f):
        if self.use_hgq:
            self.quantizer.quantizer._i.assign(self.quantizer.quantizer._i * 0.0 + i)
            self.quantizer.quantizer._f.assign(self.quantizer.quantizer._f * 0.0 + f)
        self.i = i
        self.f = f

    def apply_final_compression(self):
        if self.use_hgq and not self.quantizer.built or not self.built:
            return
        k, i, f = self.get_quantization_bits()
        self.i.assign(i)
        self.f.assign(f)
        self.b.assign(k + i + f)
        self.final_compression_done = True

    def post_pre_train_function(self):
        self.is_pretraining = False

    def call(self, x, training=None):
        if self.use_hgq:
            return self.quantizer(x, training=training)
        if not training:
            return self.quantizer(x, k=self.k, i=self.i, f=self.f, training=training)
        elif self.granularity == "per_tensor":
            i, f = self.i, self.f
        else:
            i, f = self.compute_dynamic_bits(x)
            self.i.assign(i)
            self.f.assign(f)
        return self.quantizer(x, k=self.k, i=i, f=f, training=training)

    def hgq_loss(self):
        if self.is_pretraining or not self.use_hgq:
            return 0.0
        return sum(self.quantizer.losses)

    @classmethod
    def from_config(cls, config):
        use_hgq = config["is_heterogeneous"]
        instance = cls(
            k=config.pop("k"),
            i=config.pop("i"),
            f=config.pop("f"),
            round_mode=config.pop("round_mode"),
            overflow=config.pop("overflow"),
            is_heterogeneous=config.pop("is_heterogeneous"),
            is_data=config.pop("is_data"),
            granularity=config.pop("granularity"),
            place=config.pop("place"),
        )

        if use_hgq:
            quantizer_config = config.pop("quantizer")
            instance.quantizer = keras.saving.deserialize_keras_object(quantizer_config)
        return instance

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k_init,
                "i": self.i_init,
                "f": self.f_init,
                "overflow": self.overflow,
                "round_mode": self.round_mode,
                "is_data": self.is_data,
                "hgq_gamma": self.hgq_gamma,
                "is_heterogeneous": self.use_hgq,
                "granularity": self.granularity,
                "place": self.place,
            }
        )
        if self.use_hgq:
            config.update({"quantizer": keras.saving.serialize_keras_object(self.quantizer)})
        return config


def create_hgq_parameters_quantizer(k, i, f, overflow, round_mode, place, gamma=1e-8):
    quantizer_config = QuantizerConfig(
        q_type="kif", place=place, k0=k, i0=i, f0=f, overflow_mode=overflow, round_mode=round_mode, homogeneous_axis=()
    )
    return HGQQuantizer(config=quantizer_config)


def create_hgq_data_quantizer(k, i, f, overflow, round_mode, gamma=1e-8):
    quantizer_config = QuantizerConfig(
        q_type="kif",
        place="datalane",
        k0=k,
        i0=i,
        f0=f,
        overflow_mode=overflow,
        round_mode=round_mode,
        homogeneous_axis=(0,),
    )
    return HGQQuantizer(config=quantizer_config)


def create_quantizer(k, i, f, overflow, round_mode, is_heterogeneous, is_data, place="datalane", gamma=1e-8):
    if is_heterogeneous:
        if is_data:
            return create_hgq_data_quantizer(k, i, f, overflow, round_mode, gamma=gamma)
        else:
            return create_hgq_parameters_quantizer(k, i, f, overflow, round_mode, place, gamma=gamma)
    else:
        return get_fixed_quantizer(round_mode=round_mode, overflow_mode=overflow)
