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
from pquant.core.keras.onnx.helpers import _keras_dtype_to_tp
from pquant.core.keras.onnx.layer_builders import (
    _add_avgpool,
    _add_batchnorm,
    _add_conv,
    _add_dense,
    _add_dense_nd,
    _add_depthwise_conv,
    _add_global_avgpool,
    _add_mha,
    _add_pq_activation,
)
from pquant.core.onnx_common import (
    add_initializer,
    add_int64_array,
    emit_getitem,
    emit_squeeze,
    emit_unsqueeze,
    qdq_node,
    quant_node,
    save_model,
    to_np,
)

_ACTIVATION_OPS = {"relu": "Relu", "sigmoid": "Sigmoid", "tanh": "Tanh"}


def _resolve_mask_arg(mask, prefix, kind, tensor_to_onnx, initializers):
    """Resolve an MHA mask call argument to an ONNX name (constant masks become initializers)."""
    if mask is None:
        return None
    if tensor_to_onnx is not None and id(mask) in tensor_to_onnx:
        return tensor_to_onnx[id(mask)]
    return add_initializer(initializers, f"{prefix}_{kind}_const", np.asarray(to_np(mask)))


def _call_arguments(layer):
    """The recorded call arguments of the layer's first inbound node."""
    return layer._inbound_nodes[0].arguments if layer._inbound_nodes else None


def _input_rank(layer):
    """Rank (batch dim included) of the layer's first symbolic input tensor."""
    tensors = layer._inbound_nodes[0].input_tensors
    tensor = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
    return len(tensor.shape)


def _add_mha_layer(layer, prefix, input_onnx_names, nodes, initializers, quant_fn, use_qonnx, store_int, tensor_to_onnx):
    if len(input_onnx_names) >= 3:
        q_in, k_in, v_in = input_onnx_names[:3]
    elif len(input_onnx_names) == 2:
        q_in, k_in, v_in = input_onnx_names[0], input_onnx_names[1], input_onnx_names[1]
    else:
        q_in = k_in = v_in = input_onnx_names[0]

    arguments = _call_arguments(layer)
    kwargs = arguments.kwargs if arguments else {}
    kpm = _resolve_mask_arg(kwargs.get("key_padding_mask"), prefix, "kpm", tensor_to_onnx, initializers)
    attn_mask = _resolve_mask_arg(kwargs.get("attn_mask"), prefix, "attn_mask", tensor_to_onnx, initializers)
    return _add_mha(
        layer,
        prefix,
        q_in,
        k_in,
        v_in,
        nodes,
        initializers,
        quant_fn,
        use_qonnx,
        store_int,
        key_padding_mask=kpm,
        attn_mask=attn_mask,
    )


def _add_getitem_op(layer, prefix, current, nodes, initializers):
    """keras.ops GetItem operation recorded by ``x[...]`` KerasTensor syntax."""
    arguments = _call_arguments(layer)
    spec = arguments.args[1] if len(arguments.args) > 1 else arguments.kwargs["key"]
    rank = len(arguments.args[0].shape)
    return emit_getitem(prefix, current, spec, rank, nodes, initializers)


def _add_expand_dims_op(layer, prefix, current, nodes, initializers):
    """keras.ops.expand_dims operation; the axis is stored on the op."""
    rank = len(_call_arguments(layer).args[0].shape)
    return emit_unsqueeze(prefix, current, [int(layer.axis) % (rank + 1)], nodes, initializers)


def _add_squeeze_op(layer, prefix, current, nodes, initializers):
    """keras.ops.squeeze operation; axis=None squeezes every size-1 axis
    (the batch axis is None in the symbolic shape, so it is never squeezed)."""
    in_shape = _call_arguments(layer).args[0].shape
    axis = layer.axis
    if axis is None:
        axes = [i for i, s in enumerate(in_shape) if s == 1]
    else:
        axis = axis if isinstance(axis, (list, tuple)) else (axis,)
        axes = [a for a in (int(a) % len(in_shape) for a in axis) if in_shape[a] == 1]
    return emit_squeeze(prefix, current, axes, nodes, initializers)


def _add_standard_activation(layer, prefix, current, nodes):
    """keras.layers.ReLU or keras.layers.Activation with a supported activation."""
    activation = (
        layer.activation.__name__
        if isinstance(layer, keras.layers.Activation) and callable(layer.activation)
        else getattr(layer, "activation", "relu")
    )
    act_name = activation if isinstance(activation, str) else "relu"
    op_type = next((op for key, op in _ACTIVATION_OPS.items() if key in act_name.lower()), None)
    if op_type is None:
        raise TypeError(f"Unsupported Activation for ONNX export: {act_name!r}")
    out = f"{prefix}_act"
    nodes.append(oh.make_node(op_type, inputs=[current], outputs=[out]))
    return out


def _emit_layer(
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
        return _add_mha_layer(
            layer, prefix, input_onnx_names, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, tensor_to_onnx
        )

    if isinstance(layer, PQActivation):
        return _add_pq_activation(layer, prefix, current, nodes, initializers, quant_fn)

    if isinstance(layer, PQDense):
        # Gemm only accepts rank-2 inputs; higher ranks (e.g. [batch, seq, dim]) go through MatMul + Add.
        add_fn = _add_dense_nd if _input_rank(layer) > 2 else _add_dense
        return add_fn(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, PQDepthwiseConv2d):
        return _add_depthwise_conv(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if isinstance(layer, (PQConv2d, PQConv1d)):
        ndim = 2 if isinstance(layer, PQConv2d) else 1
        return _add_conv(
            layer,
            prefix,
            current,
            nodes,
            initializers,
            ndim=ndim,
            quant_fn=quant_fn,
            use_qonnx=use_qonnx,
            store_integer_weights=store_integer_weights,
        )

    if isinstance(layer, PQBatchNormalization):
        return _add_batchnorm(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights)

    if type(layer).__name__ == "GetItem":
        return _add_getitem_op(layer, prefix, current, nodes, initializers)

    if type(layer).__name__ == "ExpandDims":
        return _add_expand_dims_op(layer, prefix, current, nodes, initializers)

    if type(layer).__name__ == "Squeeze":
        return _add_squeeze_op(layer, prefix, current, nodes, initializers)

    if isinstance(layer, (keras.layers.ReLU, keras.layers.Activation)):
        return _add_standard_activation(layer, prefix, current, nodes)

    if isinstance(layer, keras.layers.Flatten):
        out = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[current], outputs=[out], axis=1))
        return out

    if isinstance(layer, keras.layers.Reshape):
        full_shape = [-1] + list(layer.target_shape)  # -1 keeps the batch dim dynamic
        shape_name = add_int64_array(initializers, f"{prefix}_shape", full_shape)
        out = f"{prefix}_reshape"
        nodes.append(oh.make_node("Reshape", inputs=[current, shape_name], outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Add):
        assert input_onnx_names is not None and len(input_onnx_names) == 2
        out = f"{prefix}_add"
        nodes.append(oh.make_node("Add", inputs=input_onnx_names, outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Multiply):
        assert input_onnx_names is not None and len(input_onnx_names) == 2
        out = f"{prefix}_mul"
        nodes.append(oh.make_node("Mul", inputs=input_onnx_names, outputs=[out]))
        return out

    if isinstance(layer, keras.layers.Concatenate):
        assert input_onnx_names is not None
        out = f"{prefix}_concat"
        # Negative axes are fine: ONNX Concat supports them.
        nodes.append(oh.make_node("Concat", inputs=input_onnx_names, outputs=[out], axis=layer.axis))
        return out

    if isinstance(layer, (keras.layers.AveragePooling2D, keras.layers.AveragePooling1D)):
        ndim = 2 if isinstance(layer, keras.layers.AveragePooling2D) else 1
        return _add_avgpool(layer, prefix, current, nodes, initializers, ndim=ndim, quant_fn=quant_fn)

    if isinstance(layer, (keras.layers.GlobalAveragePooling2D, keras.layers.GlobalAveragePooling1D)):
        ndim = 2 if isinstance(layer, keras.layers.GlobalAveragePooling2D) else 1
        return _add_global_avgpool(layer, prefix, current, nodes, ndim=ndim)

    if isinstance(layer, keras.layers.Dropout):
        return current  # identity at inference

    raise TypeError(f"Unsupported Keras layer type for ONNX export: {type(layer).__name__!r}")


def _build_tensor_onnx_map(model):
    """Seed the KerasTensor-id → ONNX-name map with the model inputs."""
    return {id(inp): name for inp, name in zip(model.inputs, _model_input_names(model))}


def _model_input_names(model):
    if len(model.inputs) == 1:
        return ["input"]
    return [f"input_{i}" for i in range(len(model.inputs))]


def _inbound_input_names(layer, tensor_to_onnx):
    """Return the list of ONNX input names for this layer based on its inbound node."""
    if not layer._inbound_nodes:
        return []
    input_tensors = layer._inbound_nodes[0].input_tensors
    if not isinstance(input_tensors, (list, tuple)):
        input_tensors = [input_tensors]
    result = []
    for t in input_tensors:
        if id(t) not in tensor_to_onnx:
            raise RuntimeError(
                f"Layer {layer.name!r}: input tensor not found in tensor_to_onnx map. "
                "Ensure model.layers is in topological order."
            )
        result.append(tensor_to_onnx[id(t)])
    return result


def _register_layer_output(layer, onnx_name, tensor_to_onnx):
    if not layer._inbound_nodes:
        return
    out_tensors = layer._inbound_nodes[0].output_tensors
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
    tensor_to_onnx = _build_tensor_onnx_map(model)
    last_output_name: str = ""

    for layer in getattr(model, "operations", None) or model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        input_onnx_names = _inbound_input_names(layer, tensor_to_onnx)
        if not input_onnx_names:
            continue

        prefix = layer.name.replace("/", "_").replace(":", "_")
        output_name = _emit_layer(
            layer,
            prefix,
            input_onnx_names[0],
            onnx_nodes,
            initializers,
            quant_fn,
            use_qonnx,
            store_integer_weights,
            input_onnx_names=input_onnx_names,
            tensor_to_onnx=tensor_to_onnx,
        )

        _register_layer_output(layer, output_name, tensor_to_onnx)
        last_output_name = output_name[0] if isinstance(output_name, tuple) else output_name

    input_names = _model_input_names(model)
    if len(model.inputs) == 1:
        input_shapes = [tuple(input_shape)]
    else:
        input_shapes = [tuple(t.shape[1:]) for t in model.inputs]
    np_dtypes = [np.dtype(str(t.dtype)) for t in model.inputs]
    tp_dtypes = [_keras_dtype_to_tp(t.dtype) for t in model.inputs]

    dummies = [np.zeros((1, *shp), dtype=dt) for shp, dt in zip(input_shapes, np_dtypes)]
    dummy_out = model(dummies[0] if len(dummies) == 1 else dummies, training=False)
    dummy_out_np = np.array(ops.convert_to_numpy(dummy_out))

    batch_dim = batch_size  # None → dynamic, int → fixed
    input_vis = [
        oh.make_tensor_value_info(name, tp, [batch_dim, *shp]) for name, shp, tp in zip(input_names, input_shapes, tp_dtypes)
    ]
    output_vi = oh.make_tensor_value_info(last_output_name, TensorProto.FLOAT, [batch_dim] + list(dummy_out_np.shape[1:]))

    graph = oh.make_graph(
        nodes=onnx_nodes,
        name="pquant_keras_onnx",
        inputs=input_vis,
        outputs=[output_vi],
        initializer=initializers,
    )
    model_proto = save_model(graph, output_path, opset, use_qonnx=use_qonnx)
    logging.info("Saved %s Keras model → %s", "QONNX" if use_qonnx else "ONNX (QDQ)", output_path)
    return model_proto
