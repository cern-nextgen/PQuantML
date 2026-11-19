"""Convert a PQuant Keras functional model to ONNX or QONNX format."""

import collections

import keras
import numpy as np
import onnx

from pquant.core.keras.activations import PQActivation
from pquant.core.keras.layers import (
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
)
from pquant.core.keras.onnx.layer_builders import (
    _add_avgpool,
    _add_batchnorm,
    _add_conv,
    _add_dense,
    _add_global_avgpool,
)
from pquant.core.onnx_common import (
    DEFAULT_TARGET,
    ExportOptions,
    GraphBuilder,
    add_activation,
    add_getitem,
    add_mha,
    normalize_input_shapes,
    resolve_export_options,
    tensor_info,
)

_ACTIVATION_OPS = {"relu": "Relu", "sigmoid": "Sigmoid", "tanh": "Tanh"}


def _add_mha(g, node, prefix, names):
    """PQMultiheadAttention called as ``mha(query | [query, key[, value]], key_padding_mask=..., attn_mask=...)``."""
    arguments = node.arguments
    qkv = arguments.args[0] if arguments.args else arguments.kwargs["inputs"]
    qkv = [names[id(tensor)] for tensor in (qkv if isinstance(qkv, (list, tuple)) else [qkv])]
    key = qkv[1] if len(qkv) > 1 else qkv[0]
    value = qkv[2] if len(qkv) > 2 else key

    def mask(name):
        """A mask is either a tensor of the graph or a constant, which is stored as an initializer."""
        given = arguments.kwargs.get(name)
        if given is None:
            return None
        if id(given) in names:
            return names[id(given)]
        given = keras.ops.convert_to_numpy(given)
        return g.initializer(f"{prefix}_{name}", given if given.dtype == bool else given.astype(np.float32))

    return add_mha(g, node.operation, prefix, qkv[0], key, value, _add_dense, mask("key_padding_mask"), mask("attn_mask"))


def _add_squeeze(g, layer, prefix, x, shape):
    """keras.ops.squeeze; axis=None squeezes every size-1 axis (the batch axis has size None, so never that one)."""
    axes = layer.axis
    if axes is None:
        axes = range(len(shape))
    elif isinstance(axes, int):
        axes = [axes]
    return g.squeeze(prefix, x, [axis % len(shape) for axis in axes if shape[axis] == 1])


def _add_standard_activation(g, layer, prefix, x):
    """keras.layers.ReLU in its plain form, or keras.layers.Activation with a supported activation."""
    if isinstance(layer, keras.layers.ReLU):
        is_plain = layer.max_value is None and not layer.negative_slope and not layer.threshold
        name = "relu" if is_plain else "ReLU with max_value/negative_slope/threshold"
    else:
        name = getattr(layer.activation, "__name__", str(layer.activation))
    if name not in _ACTIVATION_OPS:
        raise TypeError(f"Unsupported activation for ONNX export: {name}")
    return g.node(_ACTIVATION_OPS[name], [x], f"{prefix}_act")


def _emit_node(g, node, prefix, inputs, names):
    """Emit the ONNX nodes for one call of a Keras layer or operation.  Returns its output name(s)."""
    layer, x = node.operation, inputs[0]
    shape = node.input_tensors[0].shape
    rank = len(shape)
    op_name = type(layer).__name__  # keras.ops operations have no public classes to test against

    if isinstance(layer, PQMultiheadAttention):
        return _add_mha(g, node, prefix, names)
    if isinstance(layer, PQActivation):
        return add_activation(g, layer, prefix, x)
    if isinstance(layer, PQDense):
        return _add_dense(g, layer, prefix, x, rank)
    if isinstance(layer, (PQConv2d, PQConv1d, PQDepthwiseConv2d)):
        return _add_conv(g, layer, prefix, x, rank)
    if isinstance(layer, PQBatchNormalization):
        return _add_batchnorm(g, layer, prefix, x, rank)
    if isinstance(layer, (keras.layers.AveragePooling2D, keras.layers.AveragePooling1D)):  # the PQ variants included
        return _add_avgpool(g, layer, prefix, x, rank)
    if isinstance(layer, (keras.layers.GlobalAveragePooling2D, keras.layers.GlobalAveragePooling1D)):
        return _add_global_avgpool(g, layer, prefix, x, rank)
    if isinstance(layer, (keras.layers.ReLU, keras.layers.Activation)):
        return _add_standard_activation(g, layer, prefix, x)
    if isinstance(layer, keras.layers.Flatten):
        if layer.data_format == "channels_first" and rank > 2:  # Keras flattens in channels-last order
            x = g.transpose(prefix, x, [0, *range(2, rank), 1])
        return g.node("Flatten", [x], f"{prefix}_flatten", axis=1)
    if isinstance(layer, keras.layers.Reshape):
        target = g.int64(f"{prefix}_shape", [-1, *layer.target_shape])  # -1 keeps the batch dim dynamic
        return g.node("Reshape", [x, target], f"{prefix}_reshape")
    if isinstance(layer, (keras.layers.Add, keras.layers.Multiply)):
        if len(inputs) != 2:
            raise TypeError(f"{layer.name}: ONNX export supports {op_name} layers with exactly two inputs")
        op_type = "Add" if isinstance(layer, keras.layers.Add) else "Mul"
        return g.node(op_type, inputs, f"{prefix}_{op_type.lower()}")
    if isinstance(layer, keras.layers.Concatenate):
        return g.node("Concat", inputs, f"{prefix}_concat", axis=layer.axis)  # ONNX accepts negative axes too
    if isinstance(layer, keras.layers.Dropout):
        return x  # identity at inference
    if op_name == "GetItem":  # x[...] on a KerasTensor
        arguments = node.arguments
        key = arguments.args[1] if len(arguments.args) > 1 else arguments.kwargs["key"]
        return add_getitem(g, prefix, x, key, rank)
    if op_name == "ExpandDims":
        return g.unsqueeze(prefix, x, [int(layer.axis) % (rank + 1)])
    if op_name == "Squeeze":
        return _add_squeeze(g, layer, prefix, x, shape)
    raise TypeError(f"Unsupported Keras layer type for ONNX export: {op_name!r}")


def _call_nodes(model):
    """The layer/operation calls that make up the model, in execution order (a shared layer appears once per call)."""
    graph = getattr(model, "_functional", None) or model  # a built Sequential model wraps a functional one
    if not hasattr(graph, "_nodes_by_depth"):
        raise TypeError("ONNX export needs a Keras functional (or built Sequential) model")
    for depth in sorted(graph._nodes_by_depth, reverse=True):
        for node in graph._nodes_by_depth[depth]:
            if not node.is_input:
                yield node


def _input_shapes(model, input_shape):
    """Per-input shapes without the batch dim: the model's own, with ``input_shape`` pinning the dynamic dims."""
    shapes = [tuple(tensor.shape[1:]) for tensor in model.inputs]
    if input_shape is None:
        return shapes
    given = normalize_input_shapes(input_shape)
    compatible = len(given) == len(shapes) and all(
        len(new) == len(own) and all(dim is None or dim == new_dim for dim, new_dim in zip(own, new))
        for own, new in zip(shapes, given)
    )
    if not compatible:
        raise ValueError(f"input_shape {input_shape} does not match the model inputs, which have shapes {shapes}")
    return given


def convert_to_onnx(
    model: keras.Model,
    input_shape: tuple | None = None,
    output_path: str = "model.onnx",
    target: str | ExportOptions = DEFAULT_TARGET,
    *,
    opset: int | None = None,
    include_clip: bool | None = None,
    batch_size: int | str | None = None,
    strict: bool | None = None,
) -> onnx.ModelProto:
    """
    Convert a Keras functional model of PQuant layers to ONNX or QONNX.

    Args:
        model:         Functional (or built Sequential) keras.Model with apply_final_compression() called on it.
        input_shape:   Optional shape of one sample without the batch dim (one per input), pinning dynamic dims.
        output_path:   Where to save the .onnx file.
        target:        "reference" (default), "qonnx", "aie4ml", "qdq-int8", "onnxruntime" or "tensorrt".
        opset:         ONNX opset version, at least 13; overrides the target's.
        include_clip:  Whether a Clip precedes each QuantizeLinear; overrides the target's.
        batch_size:    Fixed batch dimension, or ``"dynamic"``; overrides the target's.
        strict:        Turn export-check warnings into errors.

    Returns:
        The constructed onnx.ModelProto.
    """
    options = resolve_export_options(target, include_clip=include_clip, opset=opset, batch_size=batch_size, strict=strict)
    if options.integer_ops:
        raise ValueError("The integer_ops format (MatMulInteger) is only supported by the torch ONNX converter")
    g = GraphBuilder(options)
    input_names = ["input"] if len(model.inputs) == 1 else [f"input_{i}" for i in range(len(model.inputs))]
    input_shapes = _input_shapes(model, input_shape)

    names = {id(tensor): name for tensor, name in zip(model.inputs, input_names)}  # KerasTensor → ONNX tensor name
    calls = collections.Counter()
    for node in _call_nodes(model):
        layer = node.operation
        prefix = layer.name.replace("/", "_").replace(":", "_")
        calls[prefix] += 1
        if calls[prefix] > 1:  # a shared layer is exported once per call
            prefix = f"{prefix}_call{calls[prefix]}"
        inputs = [names[id(tensor)] for tensor in node.input_tensors]
        outputs = _emit_node(g, node, prefix, inputs, names)
        for tensor, name in zip(node.outputs, outputs if isinstance(outputs, tuple) else (outputs,)):
            names[id(tensor)] = name

    batch = options.batch_size
    input_infos = [
        tensor_info(name, tensor.dtype, shape, batch) for name, tensor, shape in zip(input_names, model.inputs, input_shapes)
    ]
    output_infos = [tensor_info(names[id(tensor)], tensor.dtype, tensor.shape[1:], batch) for tensor in model.outputs]
    return g.save("pquant_keras_onnx", input_infos, output_infos, output_path)
