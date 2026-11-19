"""
Backend-agnostic half of the PQuant ONNX converters.

Fixed-point (k, i, f) mapping
------------------------------
QONNX:
  scale      = 2^(-f)
  zero_point = 0
  bit_width  = k + i + f
  signed     = int(k)

Standard ONNX (QDQ):
  scale      = 2^(-f)
  zero_point = 0  (int8 signed, uint8 unsigned)
  clip range = [-2^i,  2^i - 2^(-f)]  signed
             = [0,     2^i - 2^(-f)]  unsigned
  Rounding is always nearest-even (QuantizeLinear behaviour).

Quantization bits are read from the layers' ``Quantizer`` objects and may be torch
tensors, Keras variables, numpy arrays or Python scalars (anything ``to_np`` converts).
"""

import dataclasses
import logging
import warnings
from collections import defaultdict
from enum import Enum

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

MIN_OPSET = 13  # Squeeze/Unsqueeze/ReduceSum take axes as an input and DequantizeLinear is per-channel from here on
IR_VERSION = 6  # ir_version stamped on the exported models
INT_CONTAINER_BITS = 8  # QuantizeLinear/DequantizeLinear use int8/uint8 containers

ROUND_MODE_MAP = {
    "TRN": "FLOOR",
    "RND": "ROUND",
    "RND_CONV": "ROUND",
    "TRN_ZERO": "TRUNCATE",
    "RND_ZERO": "ROUND",
    "RND_MIN_INF": "FLOOR",
    "RND_INF": "ROUND",
}

DEFAULT_TARGET = "reference"
DYNAMIC_BATCH = "dynamic"  # batch_size override that clears a preset's fixed batch


class QuantFormat(str, Enum):
    """How quantization is expressed in the exported graph."""

    QONNX = "qonnx"  # QONNX ``Quant`` custom nodes (hls4ml, FINN)
    QDQ = "qdq"  # QuantizeLinear/DequantizeLinear pairs, float32 weights
    QDQ_INT_WEIGHTS = "qdq_int_weights"  # QDQ pairs, weights stored as int8/uint8 + DequantizeLinear
    INTEGER_OPS = "integer_ops"  # MatMulInteger for Dense layers; implies integer weights


@dataclasses.dataclass(frozen=True)
class ExportOptions:
    """Everything that varies between ONNX consumers."""

    quant_format: QuantFormat = QuantFormat.QDQ
    include_clip: bool = True
    opset: int = MIN_OPSET
    batch_size: int | None = None
    strict: bool = False
    description: str = dataclasses.field(default="", compare=False)

    def __post_init__(self):
        object.__setattr__(self, "quant_format", QuantFormat(self.quant_format))
        if self.opset < MIN_OPSET:
            raise ValueError(f"opset {self.opset} is not supported; the ONNX export needs opset >= {MIN_OPSET}")

    @property
    def use_qonnx(self) -> bool:
        return self.quant_format is QuantFormat.QONNX

    @property
    def integer_ops(self) -> bool:
        return self.quant_format is QuantFormat.INTEGER_OPS

    @property
    def store_integer_weights(self) -> bool:
        return self.quant_format in (QuantFormat.QDQ_INT_WEIGHTS, QuantFormat.INTEGER_OPS)


_QDQ_INT8 = ExportOptions(
    quant_format=QuantFormat.QDQ_INT_WEIGHTS,
    include_clip=False,
    description="Fusable int8 QDQ graph for onnxruntime, TensorRT, QNN and OpenVINO; dynamic batch.",
)

TARGETS: dict[str, ExportOptions] = {
    "reference": ExportOptions(
        description="Bit-exact emulation of the trained fake quantization: QDQ with float weights and Clip "
        "nodes, runs on plain onnxruntime.  Validate the other targets against this one."
    ),
    "qonnx": ExportOptions(
        quant_format=QuantFormat.QONNX,
        batch_size=1,
        description="QONNX Quant nodes for hls4ml and FINN; static batch of 1.",
    ),
    "aie4ml": ExportOptions(
        quant_format=QuantFormat.QDQ_INT_WEIGHTS,
        include_clip=False,
        batch_size=1,
        description="QDQ with int8 weights and no Clip nodes for the aie4ml AI Engine flow; static batch of 1.",
    ),
    "qdq-int8": _QDQ_INT8,
    "onnxruntime": _QDQ_INT8,
    "tensorrt": _QDQ_INT8,
    "integer-ops": ExportOptions(
        quant_format=QuantFormat.INTEGER_OPS,
        description="MatMulInteger Dense layers with an int32 accumulator; onnxruntime CPU only, torch models only.",
    ),
}


def list_targets() -> dict[str, str]:
    """Return {target name: description} for every registered export target."""
    return {name: options.description for name, options in TARGETS.items()}


def resolve_export_options(target=DEFAULT_TARGET, *, include_clip=None, opset=None, batch_size=None, strict=None):
    """Look up a target preset and apply explicit overrides on top of it."""
    if isinstance(target, ExportOptions):
        options = target
    else:
        options = TARGETS.get(target) if isinstance(target, str) else None
        if options is None:
            raise ValueError(f"unknown export target {target!r}; choose one of {sorted(TARGETS)}")
    overrides = {"include_clip": include_clip, "opset": opset, "batch_size": batch_size, "strict": strict}
    changes = {name: value for name, value in overrides.items() if value is not None}
    if batch_size == DYNAMIC_BATCH:
        changes["batch_size"] = None
    return dataclasses.replace(options, **changes)


class OnnxExportWarning(UserWarning):
    """The export format cannot reproduce a trained quantizer's behaviour exactly."""


_EXACT_QDQ_ROUND_MODE = "RND_CONV"  # QuantizeLinear rounds half to even
_FLOOR_ROUND_MODES = ("TRN", "TRN_ZERO")  # differ from rounding on every off-grid value, not just on ties

# kind: (severity, message).  Listed in severity order, so that errors are raised before anything is warned about.
_ISSUES = {
    "container": (
        "error",
        "quantizers wider than the int8/uint8 QuantizeLinear container cannot be exported in the {fmt!r} format: "
        "{names}.  Use target='qonnx' (explicit bit widths) or reduce the bit width.",
    ),
    "saturation": (
        "warning",
        "without Clip nodes (include_clip=False) the graph saturates at the int8/uint8 container range instead of "
        "the trained fixed-point range for: {names}.  Outputs match only while values stay in range; set "
        "include_clip=True for an exact graph.",
    ),
    "wrap": ("warning", "WRAP overflow has no ONNX equivalent; the export saturates instead for: {names}."),
    "truncation": (
        "warning",
        "QuantizeLinear rounds half to even, but these quantizers were trained with truncation: {names}.  "
        "Exported values differ wherever an input is not already on the grid; target='qonnx' keeps the mode.",
    ),
    "ties": (
        "note",
        "QuantizeLinear breaks ties half to even; these quantizers use another tie rule and may differ on exact "
        "half-LSB inputs: {names}.",
    ),
}


def _summarize(names, limit=5):
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} (+{len(names) - limit} more)"


def cast_to_int_container(name, values, dtype):
    """Cast on-grid integer-valued floats to the int8/uint8 container, failing loudly on overflow."""
    info = np.iinfo(dtype)
    rounded = np.round(values)
    if rounded.min() < info.min or rounded.max() > info.max:
        raise ValueError(
            f"ONNX export: {name} needs integer values in [{int(rounded.min())}, {int(rounded.max())}], outside the "
            f"{np.dtype(dtype).name} container.  Use target='qonnx' or a float-weight target such as 'reference'."
        )
    return rounded.astype(dtype)


def to_np(tensor):
    """Convert a torch/Keras/TF tensor (or scalar) to a float32 numpy array."""
    if hasattr(tensor, "detach"):  # torch tensor, possibly on GPU
        tensor = tensor.detach().cpu()
    return np.asarray(tensor, dtype=np.float32)


def to_list(v, n):
    """Normalize a scalar-or-sequence layer attribute (kernel/stride/...) to an n-length list."""
    return list(v) if hasattr(v, "__iter__") else [v] * n


def _padding_attrs(padding, ndim):
    """ONNX padding attributes for a Keras/torch spec: "same"/"valid", or symmetric per-axis amounts."""
    if isinstance(padding, str):
        return {"auto_pad": "SAME_UPPER" if padding == "same" else "VALID"}
    return {"auto_pad": "NOTSET", "pads": to_list(padding, ndim) * 2}  # [begin_0, ..., end_0, ...]


def normalize_input_shapes(input_shape):
    """``input_shape`` as a list of per-input shapes; a single shape means a single input."""
    shapes = list(input_shape)
    if not (shapes and all(isinstance(s, (list, tuple)) for s in shapes)):
        shapes = [shapes]
    return [tuple(None if d is None else int(d) for d in s) for s in shapes]


def tensor_info(name, dtype, shape, batch_size=None):
    """Graph input/output description; ``shape`` excludes the batch dim (dynamic unless ``batch_size`` is given)."""
    return oh.make_tensor_value_info(name, oh.np_dtype_to_tensor_dtype(np.dtype(dtype)), [batch_size, *shape])


def quantizer_bits(quantizer):
    """(k, i, f) of a quantizer as float32 numpy arrays."""
    return tuple(to_np(bits) for bits in quantizer.get_quantization_bits())


def scalar_bits(quantizer):
    """(signed, i, f) of a quantizer as Python scalars (the first element if the bits are not per-tensor)."""
    k, i, f = (float(bits.ravel()[0]) for bits in quantizer_bits(quantizer))
    return k == 1, i, f


def _fixed_point_clip_range(signed, i, f, overflow_mode):
    """Representable [min, max] of a fixed-point grid with i integer and f fractional bits."""
    clip_max = float(2.0**i - 2.0**-f)
    if not signed:
        return 0.0, clip_max
    if overflow_mode == "SAT_SYM":
        return -clip_max, clip_max  # symmetric: excludes -2^i
    return float(-(2.0**i)), clip_max


def _per_channel_scale(f, out_channels):
    """Return the (out_channels,) scale vector, or None when f varies within a channel."""
    if f.size % out_channels != 0:
        return None
    f_per_channel = f.reshape(out_channels, -1)
    if not np.allclose(f_per_channel, f_per_channel[:, :1]):
        return None
    return (2.0 ** (-f_per_channel[:, 0])).astype(np.float32)


class GraphBuilder:
    """The ONNX graph under construction, emitted in the format that ``options`` describes."""

    def __init__(self, options: ExportOptions):
        self.options = options
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._issues: dict[str, list[str]] = defaultdict(list)  # kind (see _ISSUES) → the quantizers concerned

    def node(self, op_type, inputs, output, **attrs):
        self.nodes.append(oh.make_node(op_type, inputs=inputs, outputs=[output], **attrs))
        return output

    def initializer(self, name, array):
        """Register a constant tensor and return its name."""
        self.initializers.append(onh.from_array(np.asarray(array), name=name))
        return name

    def float_scalar(self, name, value):
        return self.initializer(name, np.array(value, dtype=np.float32))

    def int64(self, name, values):
        return self.initializer(name, np.array(values, dtype=np.int64))

    def transpose(self, prefix, x, perm):
        return self.node("Transpose", [x], f"{prefix}_transpose_{''.join(map(str, perm))}", perm=list(perm))

    def squeeze(self, prefix, x, axes):
        """Remove the given size-1 axes (a no-op when ``axes`` is empty)."""
        if not axes:
            return x
        return self.node("Squeeze", [x, self.int64(f"{prefix}_squeeze_axes", sorted(axes))], f"{prefix}_squeeze")

    def unsqueeze(self, prefix, x, axes):
        return self.node("Unsqueeze", [x, self.int64(f"{prefix}_unsqueeze_axes", axes)], f"{prefix}_unsqueeze")

    def reduce(self, op_type, x, axes, output, keepdims=1):
        """Reduce* node; ``axes`` became an input in opset 13 for ReduceSum and in opset 18 for the others."""
        if self.options.opset >= (13 if op_type == "ReduceSum" else 18):
            return self.node(op_type, [x, self.int64(f"{output}_axes", axes)], output, keepdims=keepdims)
        return self.node(op_type, [x], output, axes=axes, keepdims=keepdims)

    def quantize(self, quantizer, prefix, x):
        """Quantize ``x`` on the quantizer's fixed-point grid: a QONNX Quant node or a (Clip +) Q/DQ pair."""
        if self.options.use_qonnx:
            self._check(quantizer, prefix)
            return self._qonnx_quant(quantizer, prefix, x)
        codes, scale, zero_point = self.quantize_linear(quantizer, prefix, x)
        return self.node("DequantizeLinear", [codes, scale, zero_point], f"{prefix}_dequantized")

    def quantize_linear(self, quantizer, prefix, x):
        """(Clip +) QuantizeLinear to int8/uint8 codes.  Returns the (codes, scale, zero_point) tensor names."""
        self._check(quantizer, prefix)
        signed, i, f = scalar_bits(quantizer)
        scale = self.float_scalar(f"{prefix}_scale", 2.0**-f)
        zero_point = self.initializer(f"{prefix}_zero_point", np.array(0, dtype=np.int8 if signed else np.uint8))
        if self.options.include_clip:
            clip_min, clip_max = _fixed_point_clip_range(signed, i, f, quantizer.overflow)
            bounds = [self.float_scalar(f"{prefix}_clip_min", clip_min), self.float_scalar(f"{prefix}_clip_max", clip_max)]
            x = self.node("Clip", [x, *bounds], f"{prefix}_clipped")
        return self.node("QuantizeLinear", [x, scale, zero_point], f"{prefix}_quantized"), scale, zero_point

    def _check(self, quantizer, name):
        """Record what the export format cannot reproduce of a data quantizer's behaviour (the kinds of ``_ISSUES``)."""
        if quantizer.overflow.startswith("WRAP"):
            self._issues["wrap"].append(name)
        if self.options.use_qonnx:
            return  # Quant carries the bit width, the rounding mode and the narrow range explicitly
        k, i, f = quantizer_bits(quantizer)
        signed = int(k.ravel()[0]) == 1
        bits = int(np.max(k + i + f))
        if bits > INT_CONTAINER_BITS:
            self._issues["container"].append(f"{name} ({bits} bits)")
        elif not self.options.include_clip and (bits < INT_CONTAINER_BITS or (signed and quantizer.overflow == "SAT_SYM")):
            self._issues["saturation"].append(name)
        round_mode = quantizer.round_mode.upper().removeprefix("S_")
        if round_mode != _EXACT_QDQ_ROUND_MODE:
            kind = "truncation" if round_mode in _FLOOR_ROUND_MODES else "ties"
            self._issues[kind].append(f"{name} ({quantizer.round_mode})")

    def _report_issues(self):
        """Raise, warn about or log what ``_check`` recorded; warnings are errors with ``strict``."""
        for kind, (severity, template) in _ISSUES.items():
            names = self._issues.get(kind)
            if not names:
                continue
            message = "ONNX export: " + template.format(fmt=self.options.quant_format.value, names=_summarize(names))
            if severity == "error" or (severity == "warning" and self.options.strict):
                raise ValueError(message)
            if severity == "warning":
                warnings.warn(message, OnnxExportWarning, stacklevel=4)  # the caller of convert_to_onnx
            else:
                logging.info(message)

    def _qonnx_quant(self, quantizer, prefix, x):
        k, i, f = quantizer_bits(quantizer)
        signed = int(k.ravel()[0])
        # The Quant node holds a single scale/bit_width; widen to cover every element.
        i, f = float(i.max()), float(f.min())
        inputs = [
            x,
            self.float_scalar(f"{prefix}_scale", 2.0**-f),
            self.float_scalar(f"{prefix}_zero_point", 0.0),
            self.float_scalar(f"{prefix}_bit_width", signed + i + f),
        ]
        return self.node(
            "Quant",
            inputs,
            f"{prefix}_quantized",
            domain="qonnx.custom_op.general",
            signed=signed,
            narrow=int(signed == 1 and quantizer.overflow == "SAT_SYM"),  # SAT_SYM excludes the most-negative value
            rounding_mode=ROUND_MODE_MAP.get(quantizer.round_mode, "ROUND"),
        )

    def quantize_input(self, layer, prefix, x):
        """Apply the layer's input quantizer if the layer uses it."""
        return self._quantize_lane(layer, "input", f"{prefix}_in", x)

    def quantize_output(self, layer, prefix, x):
        """Apply the layer's output quantizer if the layer uses it."""
        return self._quantize_lane(layer, "output", f"{prefix}_out", x)

    def _quantize_lane(self, layer, lane, prefix, x):
        quantizer = getattr(layer, f"{lane}_quantizer", None)  # the layers create their quantizers conditionally
        if quantizer is not None and getattr(layer, f"quantize_{lane}") and layer.enable_quantization:
            x = self.quantize(quantizer, prefix, x)
        return x

    def param(self, layer, prefix, name, array, quantizer):
        """Emit a learnable parameter (kernel/bias/gamma/beta) of ``layer`` and return its tensor name."""
        prefix = f"{prefix}_{name}"
        if quantizer is None or not self.quantizes_params(layer):
            return self.initializer(prefix, array)
        if self.options.use_qonnx:
            return self._qonnx_quant(quantizer, prefix, self.initializer(f"{prefix}_fp", array))
        return self._integer_param(quantizer, prefix, array)

    def quantizes_params(self, layer):
        """Whether ``layer``'s parameters are exported in quantized form (Quant nodes or integer codes)."""
        return layer.enable_quantization and (self.options.use_qonnx or self.options.store_integer_weights)

    def _integer_param(self, quantizer, prefix, array):
        """Store a parameter as int8/uint8 codes + DequantizeLinear."""
        k, _, f = quantizer_bits(quantizer)
        dtype = np.int8 if int(k.ravel()[0]) == 1 else np.uint8
        if f.size == 1:
            scale, attrs = np.array(2.0 ** -float(f.ravel()[0]), dtype=np.float32), {}
        else:
            scale, attrs = _per_channel_scale(f, array.shape[0]), {"axis": 0}
            if scale is None:  # per-weight scales have no ONNX equivalent
                return self.initializer(f"{prefix}_float", array)
        codes = cast_to_int_container(prefix, array / scale.reshape((-1,) + (1,) * (array.ndim - 1)), dtype)
        inputs = [
            self.initializer(f"{prefix}_int", codes),
            self.initializer(f"{prefix}_dq_scale", scale),
            self.initializer(f"{prefix}_dq_zp", np.zeros(scale.shape, dtype=dtype)),
        ]
        return self.node("DequantizeLinear", inputs, f"{prefix}_dequantized", **attrs)

    def save(self, graph_name, inputs, outputs, output_path):
        """Assemble the ModelProto from the graph inputs/outputs (``tensor_info`` lists), validate and save it."""
        self._report_issues()
        input_names = {info.name for info in inputs}
        for idx, info in enumerate(outputs):
            if info.name in input_names:  # ONNX forbids a graph input from also being a graph output
                info.name = self.node("Identity", [info.name], f"{info.name}_identity_out{idx}")
        graph = oh.make_graph(self.nodes, graph_name, inputs, outputs, initializer=self.initializers)
        opset_imports = [oh.make_opsetid("", self.options.opset)]
        if self.options.use_qonnx:
            opset_imports.append(oh.make_opsetid("qonnx.custom_op.general", 1))
        model_proto = oh.make_model(graph, opset_imports=opset_imports)
        model_proto.ir_version = IR_VERSION
        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, output_path)
        logging.info("Saved %s model → %s", self.options.quant_format.value, output_path)
        return model_proto


def add_getitem(g, prefix, x, spec, rank):
    """Translate a constant Python indexing spec into ONNX Slice (+ Squeeze)."""
    if not isinstance(spec, tuple):
        spec = (spec,)
    ellipses = [pos for pos, s in enumerate(spec) if s is Ellipsis]
    if len(ellipses) > 1:
        raise TypeError("indexing with more than one Ellipsis is not supported in ONNX export")
    if ellipses:
        pos = ellipses[0]
        spec = spec[:pos] + (slice(None),) * (rank - len(spec) + 1) + spec[pos + 1 :]
    if len(spec) > rank:
        raise TypeError(f"indexing spec has {len(spec)} dims but tensor rank is {rank}")

    int64_max = np.iinfo(np.int64).max
    slices, squeeze_axes = [], []  # slices: (start, end, axis, step)
    for axis, s in enumerate(spec):
        if isinstance(s, slice):
            if s.start is None and s.stop is None and s.step in (None, 1):
                continue  # full slice: no-op on this axis
            step = 1 if s.step is None else int(s.step)
            if step < 1:
                raise TypeError("slice steps < 1 are not supported in ONNX export")
            start = 0 if s.start is None else int(s.start)
            slices.append((start, int64_max if s.stop is None else int(s.stop), axis, step))
        elif isinstance(s, int):
            slices.append((s, int64_max if s == -1 else s + 1, axis, 1))
            squeeze_axes.append(axis)
        else:
            raise TypeError(f"unsupported index element {s!r} for ONNX export (constant int/slice/Ellipsis only)")

    if slices:
        parts = zip(("starts", "ends", "axes", "steps"), zip(*slices))
        x = g.node("Slice", [x] + [g.int64(f"{prefix}_slice_{part}", vals) for part, vals in parts], f"{prefix}_slice")
    return g.squeeze(prefix, x, squeeze_axes)


def add_activation(g, layer, prefix, x):
    """PQActivation, in the order of its forward: relu multiplier, input quantizer, activation, output quantizer."""
    name = layer.activation_name
    if name == "relu" and layer.use_multiplier and not layer.use_hgq:
        scale = 2.0 ** round(float(to_np(layer.multiplier).ravel()[0]))
        x = g.node("Mul", [x, g.float_scalar(f"{prefix}_mul_scale", scale)], f"{prefix}_scaled")
    x = g.quantize_input(layer, prefix, x)

    out = f"{prefix}_act"
    if name in ("relu", "tanh"):
        g.node(name.capitalize(), [x], out)
    elif name == "hard_tanh":
        g.node("Clip", [x, g.float_scalar(f"{prefix}_htanh_min", -1.0), g.float_scalar(f"{prefix}_htanh_max", 1.0)], out)
    elif name == "leaky_relu":
        g.node("LeakyRelu", [x], out, alpha=layer.activation_function.negative_slope)
    elif name == "gelu":
        _add_gelu(g, prefix, x, out, getattr(layer.activation_function, "approximate", "none"))
    else:
        raise TypeError(f"PQActivation: unsupported activation {name!r} for ONNX export")
    return g.quantize_output(layer, prefix, out)


def _add_gelu(g, prefix, x, out, approximate):
    """Decompose gelu so the default opset works; ONNX added a Gelu op only in opset 20."""
    p = f"{prefix}_gelu"

    def const(name, value):
        return g.float_scalar(f"{p}_{name}", value)

    if approximate == "tanh":
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        t = g.node("Pow", [x, const("three", 3.0)], f"{p}_x3")
        t = g.node("Mul", [t, const("c1", 0.044715)], f"{p}_cx3")
        t = g.node("Add", [x, t], f"{p}_inner")
        t = g.node("Mul", [t, const("sqrt2_over_pi", np.sqrt(2.0 / np.pi))], f"{p}_scaled")
        t = g.node("Tanh", [t], f"{p}_tanh")
    else:
        # Exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        t = g.node("Mul", [x, const("inv_sqrt2", 1.0 / np.sqrt(2.0))], f"{p}_scaled")
        t = g.node("Erf", [t], f"{p}_erf")
    t = g.node("Add", [t, const("one", 1.0)], f"{p}_plus1")
    t = g.node("Mul", [x, t], f"{p}_xprod")
    g.node("Mul", [t, const("half", 0.5)], out)


def _add_weighted_op(g, layer, prefix, x, op_type, weight, bias, **attrs):
    """Input quantizer → ``op_type`` fed with the layer's weight and optional bias → output quantizer."""
    x = g.quantize_input(layer, prefix, x)
    inputs = [x, g.param(layer, prefix, "weight", weight, layer.weight_quantizer)]
    if bias is not None:
        inputs.append(g.param(layer, prefix, "bias", bias, layer.bias_quantizer))
    out = g.node(op_type, inputs, f"{prefix}_{op_type.lower()}", **attrs)
    return g.quantize_output(layer, prefix, out)


def add_dense(g, layer, prefix, x, weight, bias, rank):
    """PQDense with ``weight`` in [out, in] layout, applied to a rank-``rank`` input."""
    if rank == 2:  # Gemm only accepts rank-2 inputs
        return _add_weighted_op(g, layer, prefix, x, "Gemm", weight, bias, transB=1)

    x = g.quantize_input(layer, prefix, x)
    if g.quantizes_params(layer):
        # A quantized weight keeps its output channels on axis 0 and is transposed in the graph.
        quantized = g.param(layer, prefix, "weight", weight, layer.weight_quantizer)
        weight_t = g.node("Transpose", [quantized], f"{prefix}_weight_T", perm=[1, 0])
    else:
        weight_t = g.initializer(f"{prefix}_weight_T", weight.T)
    out = g.node("MatMul", [x, weight_t], f"{prefix}_matmul")
    if bias is not None:
        out = g.node("Add", [out, g.param(layer, prefix, "bias", bias, layer.bias_quantizer)], f"{prefix}_biased")
    return g.quantize_output(layer, prefix, out)


def add_conv(g, layer, prefix, x, kernel, bias, strides, dilations, padding, groups):
    """PQConv with ``kernel`` in ONNX [out, in/groups, *spatial] layout on a channels-first input."""
    ndim = kernel.ndim - 2
    attrs = dict(
        kernel_shape=list(kernel.shape[2:]),
        strides=to_list(strides, ndim),
        dilations=to_list(dilations, ndim),
        group=groups,
        **_padding_attrs(padding, ndim),
    )
    return _add_weighted_op(g, layer, prefix, x, "Conv", kernel, bias, **attrs)


def _add_quantized_softmax(g, softmax, prefix, x, mask=None):
    """PQSoftmax over the last axis: exp and reciprocal lookup tables, each with its own quantizers."""
    p = f"{prefix}_sm"
    exp_table, inv_table = softmax.exp_table, softmax.inv_table
    x = g.quantize_input(softmax, p, x)
    if softmax.stable:  # max(x) - x, so that the exp table only sees non-negative values
        x = g.node("Sub", [g.reduce("ReduceMax", x, [-1], f"{p}_max"), x], f"{p}_sub")

    x = g.quantize_input(exp_table, f"{p}_exp", x)
    coeff = float(softmax.input_scaler) * (-1.0 if softmax.stable else 1.0)
    if coeff != 1.0:
        x = g.node("Mul", [x, g.float_scalar(f"{p}_exp_coeff", coeff)], f"{p}_exp_arg")
    numerator = g.quantize_output(exp_table, f"{p}_exp", g.node("Exp", [x], f"{p}_exp"))
    if mask is not None:  # zero the masked positions
        mask = g.node("Cast", [mask], f"{p}_mask_f", to=onnx.TensorProto.FLOAT)
        numerator = g.node("Mul", [mask, numerator], f"{p}_masked")

    total = g.quantize_input(inv_table, f"{p}_inv", g.reduce("ReduceSum", numerator, [-1], f"{p}_sum"))
    total = g.node("Add", [total, g.float_scalar(f"{p}_eps", float(softmax.epsilon))], f"{p}_inv_add")
    divisor = g.quantize_output(inv_table, f"{p}_inv", g.node("Reciprocal", [total], f"{p}_inv"))
    return g.quantize_output(softmax, p, g.node("Mul", [numerator, divisor], f"{p}_out"))


def add_mha(g, mha, prefix, query, key, value, dense, key_padding_mask=None, attn_mask=None):
    """PQMultiheadAttention on batch-first (B, L, E) inputs.  Returns (output, head-averaged attention weights)."""

    def project(layer, name, x):
        """(B, L, E) → projection → (B, H, L, head_dim); the 0s in the Reshape keep the batch and length dims dynamic."""
        x = dense(g, layer, f"{prefix}_{name}_proj", x, 3)
        shape = g.int64(f"{prefix}_{name}_shape", [0, 0, mha.num_heads, mha.head_dim])
        x = g.node("Reshape", [x, shape], f"{prefix}_{name}_reshaped")
        return g.node("Transpose", [x], f"{prefix}_{name}_transposed", perm=[0, 2, 1, 3])

    q = project(mha.q_proj, "q", query)
    k = project(mha.k_proj, "k", key)
    v = project(mha.v_proj, "v", value)

    key_t = g.node("Transpose", [k], f"{prefix}_k_T", perm=[0, 1, 3, 2])
    scores = g.node("MatMul", [q, key_t], f"{prefix}_scores_raw")
    scores = g.node("Mul", [scores, g.float_scalar(f"{prefix}_attn_scale", float(mha.scale))], f"{prefix}_scores_scaled")
    if attn_mask is not None:
        scores = g.node("Add", [scores, attn_mask], f"{prefix}_scores_masked")
    keep = None
    if key_padding_mask is not None:  # (B, S) "is padding" → (B, 1, 1, S) "keep"
        keep = g.unsqueeze(f"{prefix}_kpm", g.node("Not", [key_padding_mask], f"{prefix}_kpm_not"), [1, 2])
    weights = _add_quantized_softmax(g, mha.softmax, f"{prefix}_attn", scores, mask=keep)

    # (B, H, T, head_dim) → (B, T, E)
    context = g.node("MatMul", [weights, v], f"{prefix}_ctx_raw")
    context = g.node("Transpose", [context], f"{prefix}_ctx_t", perm=[0, 2, 1, 3])
    context = g.node("Reshape", [context, g.int64(f"{prefix}_ctx_shape", [0, 0, mha.embed_dim])], f"{prefix}_ctx_merged")
    out = dense(g, mha.out_proj, f"{prefix}_out_proj", context, 3)
    return out, g.reduce("ReduceMean", weights, [1], f"{prefix}_avg_attn_weights", keepdims=0)
