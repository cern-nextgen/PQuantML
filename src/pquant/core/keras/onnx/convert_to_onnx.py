"""
Convert a PQuant Keras functional model to ONNX or QONNX format.

Pass ``use_qonnx=True`` to emit QONNX ``Quant`` custom nodes (requires the
qonnx runtime).  Pass ``use_qonnx=False`` (default) to emit standard
``Clip + QuantizeLinear + DequantizeLinear`` nodes runnable with plain
onnxruntime.


Keras weight layout (kernel always stored as HWIO regardless of data_format):
  Conv2D kernel:          [kH, kW, in/g, out] → [out, in/g, kH, kW] for ONNX
  Conv1D kernel:          [kL, in/g, out]     → [out, in/g, kL]     for ONNX
  DepthwiseConv2D kernel: [kH, kW, in, dm]    → [in*dm, 1, kH, kW]  for ONNX
  Dense kernel:           [in, out]           → stored as [out, in] for Gemm (transB=1)
"""

import functools
import logging

import keras
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from keras import ops
from onnx import TensorProto

from pquant.core.keras.activations import PQActivation
from pquant.core.keras.layers import (
    PQBatchNormalization,
    PQConv1d,
    PQConv2d,
    PQDense,
    PQDepthwiseConv2d,
    PQMultiheadAttention,
)
from pquant.core.keras.onnx.helpers import (
    emit_getitem,
    emit_squeeze,
    emit_unsqueeze,
    keras_dtype_to_tp,
    qdq_node,
    quant_node,
    to_np,
)
from pquant.core.keras.onnx.layer_builders import (
    add_avgpool,
    add_batchnorm,
    add_conv,
    add_dense,
    add_depthwise_conv,
    add_global_avgpool,
    add_mha,
    add_pq_activation,
)


def resolve_mask_arg(mask, prefix, kind, tensor_to_onnx, initializers):
    if mask is None:
        return None
    if tensor_to_onnx is not None and id(mask) in tensor_to_onnx:
        return tensor_to_onnx[id(mask)]
    arr = np.asarray(to_np(mask))
    name = f"{prefix}_{kind}_const"
    initializers.append(onh.from_array(arr, name=name))
    return name


def emit_layer(
    layer,
    prefix,
    current,
    nodes,
    initializers,
    quant_fn,
    use_qonnx,
    store_integer_weights,
    input_onnx_names=None,
    tensor_to_onnx=None,
):
    """Emit ONNX nodes for a single Keras layer.  Returns the ONNX output name."""

    if isinstance(layer, PQMultiheadAttention):
        if len(input_onnx_names) >= 3:
            q_in, k_in, v_in = input_onnx_names[0], input_onnx_names[1], input_onnx_names[2]
        elif len(input_onnx_names) == 2:
            q_in, k_in, v_in = input_onnx_names[0], input_onnx_names[1], input_onnx_names[1]
        else:
            q_in = k_in = v_in = input_onnx_names[0]
        kwargs = layer._inbound_nodes[0].arguments.kwargs if layer._inbound_nodes else {}
        kpm = resolve_mask_arg(kwargs.get("key_padding_mask"), prefix, "kpm", tensor_to_onnx, initializers)
        attn_mask = resolve_mask_arg(kwargs.get("attn_mask"), prefix, "attn_mask", tensor_to_onnx, initializers)
        return add_mha(
            layer,
            prefix,
            q_in,
            k_in,
            v_in,
            nodes,
            initializers,
            quant_fn,
            use_qonnx,
            store_integer_weights,
            key_padding_mask=kpm,
            attn_mask=attn_mask,
        )

    if isinstance(layer, PQActivation):
        return add_pq_activation(layer, prefix, current, nodes, initializers, quant_fn)

    if isinstance(layer, PQDense):
        return add_dense(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, PQDepthwiseConv2d):
        return add_depthwise_conv(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, PQConv2d):
        return add_conv(
            layer,
            prefix,
            current,
            nodes,
            initializers,
            ndim=2,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )

    if isinstance(layer, PQConv1d):
        return add_conv(
            layer,
            prefix,
            current,
            nodes,
            initializers,
            ndim=1,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )

    if isinstance(layer, PQBatchNormalization):
        return add_batchnorm(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if type(layer).__name__ == "GetItem":
        # keras.ops GetItem operation recorded by ``x[...]`` KerasTensor syntax.
        node = layer._inbound_nodes[0]
        args = node.arguments.args
        spec = args[1] if len(args) > 1 else node.arguments.kwargs["key"]
        rank = len(args[0].shape)
        return emit_getitem(prefix, current, spec, rank, nodes, initializers)

    if type(layer).__name__ == "ExpandDims":
        # keras.ops.expand_dims operation; the axis is stored on the op.
        rank = len(layer._inbound_nodes[0].arguments.args[0].shape)
        return emit_unsqueeze(prefix, current, [int(layer.axis) % (rank + 1)], nodes, initializers)

    if type(layer).__name__ == "Squeeze":
        # keras.ops.squeeze operation; axis=None squeezes every size-1 axis
        # (the batch axis is None in the symbolic shape, so it is never squeezed).
        in_shape = layer._inbound_nodes[0].arguments.args[0].shape
        axis = layer.axis
        if axis is None:
            axes = [i for i, s in enumerate(in_shape) if s == 1]
        else:
            axis = axis if isinstance(axis, (list, tuple)) else (axis,)
            axes = [a for a in (int(a) % len(in_shape) for a in axis) if in_shape[a] == 1]
        return emit_squeeze(prefix, current, axes, nodes, initializers)

    if isinstance(layer, (keras.layers.ReLU, keras.layers.Activation)):
        activation = (
            layer.activation.__name__
            if isinstance(layer, keras.layers.Activation) and callable(layer.activation)
            else getattr(layer, "activation", "relu")
        )
        act_name = activation if isinstance(activation, str) else "relu"
        out = f"{prefix}_act"
        if "relu" in act_name.lower():
            nodes.append(oh.make_node("Relu", inputs=[current], outputs=[out]))
        elif "sigmoid" in act_name.lower():
            nodes.append(oh.make_node("Sigmoid", inputs=[current], outputs=[out]))
        elif "tanh" in act_name.lower():
            nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[out]))
        else:
            raise TypeError(f"Unsupported Activation for ONNX export: {act_name!r}")
        return out

    if isinstance(layer, keras.layers.Flatten):
        out = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[current], outputs=[out], axis=1))
        return out

    if isinstance(layer, keras.layers.Reshape):
        target_shape = list(layer.target_shape)
        # Prepend batch dim (-1 means dynamic)
        full_shape = [-1] + target_shape
        shape_name = f"{prefix}_shape"
        out = f"{prefix}_reshape"
        initializers.append(onh.from_array(np.array(full_shape, dtype=np.int64), name=shape_name))
        nodes.append(oh.make_node("Reshape", inputs=[current, shape_name], outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Add):
        assert input_onnx_names is not None and len(input_onnx_names) == 2
        out = f"{prefix}_add"
        nodes.append(oh.make_node("Add", inputs=input_onnx_names, outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Concatenate):
        assert input_onnx_names is not None
        axis = layer.axis
        # Negative axis: leave as-is; onnx Concat supports negative axes
        out = f"{prefix}_concat"
        nodes.append(oh.make_node("Concat", inputs=input_onnx_names, outputs=[out], axis=axis))
        return out

    if isinstance(layer, keras.layers.Multiply):
        assert input_onnx_names is not None and len(input_onnx_names) == 2
        out = f"{prefix}_mul"
        nodes.append(oh.make_node("Mul", inputs=input_onnx_names, outputs=[out]))
        return out

    if isinstance(layer, keras.layers.AveragePooling2D):
        return add_avgpool(layer, prefix, current, nodes, initializers, ndim=2, quant_fn=quant_fn)

    if isinstance(layer, keras.layers.AveragePooling1D):
        return add_avgpool(layer, prefix, current, nodes, initializers, ndim=1, quant_fn=quant_fn)

    if isinstance(layer, keras.layers.GlobalAveragePooling2D):
        return add_global_avgpool(layer, prefix, current, nodes, ndim=2)

    if isinstance(layer, keras.layers.GlobalAveragePooling1D):
        return add_global_avgpool(layer, prefix, current, nodes, ndim=1)

    if isinstance(layer, (keras.layers.Dropout,)):
        return current  # identity at inference

    raise TypeError(f"Unsupported Keras layer type for ONNX export: {type(layer).__name__!r}")


def build_tensor_onnx_map(model):
    tensor_to_onnx = {}
    for i, inp in enumerate(model.inputs):
        name = "input" if len(model.inputs) == 1 else f"input_{i}"
        tensor_to_onnx[id(inp)] = name
    return tensor_to_onnx


def inbound_input_names(layer, tensor_to_onnx):
    """Return the list of ONNX input names for this layer based on its inbound node."""
    if not layer._inbound_nodes:
        return []
    node = layer._inbound_nodes[0]
    input_tensors = node.input_tensors
    if not isinstance(input_tensors, (list, tuple)):
        input_tensors = [input_tensors]
    result = []
    for t in input_tensors:
        key = id(t)
        if key not in tensor_to_onnx:
            raise RuntimeError(
                f"Layer {layer.name!r}: input tensor not found in tensor_to_onnx map. "
                "Ensure model.layers is in topological order."
            )
        result.append(tensor_to_onnx[key])
    return result


def register_layer_output(layer, onnx_name, tensor_to_onnx):
    if not layer._inbound_nodes:
        return
    node = layer._inbound_nodes[0]
    out_tensors = node.output_tensors
    if not isinstance(out_tensors, (list, tuple)):
        out_tensors = [out_tensors]
    if isinstance(onnx_name, (list, tuple)):
        for tensor, name in zip(out_tensors, onnx_name):
            tensor_to_onnx[id(tensor)] = name
    else:
        tensor_to_onnx[id(out_tensors[0])] = onnx_name


def convert_to_onnx(
    model: keras.Model,
    input_shape: tuple,
    output_path: str = "model.onnx",
    opset: int = 13,
    use_qonnx: bool = False,
    store_integer_weights: bool = False,
    include_clip: bool = True,
    batch_size: int | None = None,
) -> onnx.ModelProto:
    """
    Convert a Keras functional model of PQuant layers to ONNX or QONNX.

    The model must have apply_final_compression() called on all PQ layers
    before calling this function.  Only inference-mode semantics are exported.

    Args:
        model:                  Trained keras.Model.  Must be a functional model
                                (built with the Keras functional API or subclassed
                                models whose layers are accessible via model.layers).
        input_shape:            Shape of a single sample excluding batch, e.g. (3, 32, 32).
                                For channels_last Conv models use e.g. (32, 32, 3).
        output_path:            Where to save the .onnx file.
        opset:                  ONNX opset version (≥13 required for per-channel
                                DequantizeLinear).
        use_qonnx:              Emit QONNX Quant custom nodes if True.
        store_integer_weights:  Store weight initializers as int8/uint8 +
                                DequantizeLinear instead of float32 (ignored when
                                use_qonnx=True).
        include_clip:           Prepend a Clip node before each QuantizeLinear when
                                True (default).  Set to False to emit bare
                                QuantizeLinear+DequantizeLinear pairs — safe when
                                values are guaranteed in-range at inference time since
                                QuantizeLinear saturates naturally.  Ignored when
                                use_qonnx=True.
        batch_size:             If not None, fix the batch dimension of all graph
                                inputs and outputs to this value.  If None (default),
                                the batch dimension is left dynamic.

    Returns:
        The constructed onnx.ModelProto.
    """
    quant_fn = quant_node if use_qonnx else functools.partial(qdq_node, include_clip=include_clip)

    onnx_nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    tensor_to_onnx = build_tensor_onnx_map(model)
    last_output_name: str = ""

    for layer in getattr(model, "operations", None) or model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        input_onnx_names = inbound_input_names(layer, tensor_to_onnx)
        if not input_onnx_names:
            continue

        current = input_onnx_names[0]
        prefix = layer.name.replace("/", "_").replace(":", "_")

        output_name = emit_layer(
            layer,
            prefix,
            current,
            onnx_nodes,
            initializers,
            quant_fn,
            use_qonnx,
            store_integer_weights,
            input_onnx_names=input_onnx_names,
            tensor_to_onnx=tensor_to_onnx,
        )

        register_layer_output(layer, output_name, tensor_to_onnx)
        last_output_name = output_name[0] if isinstance(output_name, tuple) else output_name

    n_in = len(model.inputs)
    if n_in == 1:
        input_names = ["input"]
        input_shapes = [tuple(input_shape)]
    else:
        input_names = [f"input_{i}" for i in range(n_in)]
        input_shapes = [tuple(t.shape[1:]) for t in model.inputs]
    np_dtypes = [np.dtype(str(t.dtype)) for t in model.inputs]
    tp_dtypes = [keras_dtype_to_tp(t.dtype) for t in model.inputs]

    dummies = [np.zeros((1, *shp), dtype=dt) for shp, dt in zip(input_shapes, np_dtypes)]
    dummy_out = model(dummies[0] if n_in == 1 else dummies, training=False)
    dummy_out_np = np.array(ops.convert_to_numpy(dummy_out))
    batch_dim = batch_size  # None → dynamic, int → fixed
    output_shape = [batch_dim] + list(dummy_out_np.shape[1:])

    input_vis = [
        oh.make_tensor_value_info(name, tp, [batch_dim, *shp]) for name, shp, tp in zip(input_names, input_shapes, tp_dtypes)
    ]
    output_vi = oh.make_tensor_value_info(last_output_name, TensorProto.FLOAT, output_shape)

    graph = oh.make_graph(
        nodes=onnx_nodes,
        name="pquant_keras_onnx",
        inputs=input_vis,
        outputs=[output_vi],
        initializer=initializers,
    )

    opset_imports = [oh.make_opsetid("", opset)]
    if use_qonnx:
        opset_imports.append(oh.make_opsetid("qonnx.custom_op.general", 1))
    model_proto = oh.make_model(graph, opset_imports=opset_imports)
    model_proto.ir_version = 6

    _init_names = {t.name for t in model_proto.graph.initializer}
    _data_inputs = [vi for vi in model_proto.graph.input if vi.name not in _init_names]
    del model_proto.graph.input[:]
    model_proto.graph.input.extend(_data_inputs)

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, output_path)
    fmt = "QONNX" if use_qonnx else "ONNX (QDQ)"
    logging.info("Saved %s Keras model → %s", fmt, output_path)
    return model_proto
