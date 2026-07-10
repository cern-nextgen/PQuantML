"""Per-layer ONNX graph builders (Dense/Conv/BN/Pool/Softmax/MHA) for the PQuant Keras converter."""

import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto

from pquant.core.keras.layers import PQBatchNormalization
from pquant.core.keras.onnx.helpers import (
    add_transpose,
    bn_transpose_info,
    channels_last,
    emit_param,
    maybe_quant_input,
    maybe_quant_output,
    to_list,
    to_np,
)


def add_dense(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = to_np(layer._kernel).T  # [out, in]
    out_units = kernel_np.shape[0]

    q_weight = emit_param(
        prefix, "weight", kernel_np, layer.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights, out_units
    )

    gemm_inputs = [current, q_weight]

    if layer._bias is not None:
        bias_np = to_np(layer._bias)
        q_bias = emit_param(
            prefix, "bias", bias_np, layer.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        gemm_inputs.append(q_bias)

    gemm_out = f"{prefix}_gemm"
    nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_out], transB=1))
    current = gemm_out

    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)
    return current


def add_conv(layer, prefix, current, nodes, initializers, ndim, quant_fn, use_qonnx, store_integer_weights):
    cl = channels_last(layer)

    if cl:
        perm_to_nchw = [0, 3, 1, 2] if ndim == 2 else [0, 2, 1]
        perm_to_nhwx = [0, 2, 3, 1] if ndim == 2 else [0, 2, 1]
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = to_np(layer._kernel)
    # Transpose kernel from Keras HWIO to ONNX OIHW
    if ndim == 2:
        kernel_onnx = np.transpose(kernel_np, (3, 2, 0, 1))  # [kH,kW,in,out] → [out,in,kH,kW]
    else:
        kernel_onnx = np.transpose(kernel_np, (2, 1, 0))  # [kL,in,out]    → [out,in,kL]

    out_channels = kernel_onnx.shape[0]

    q_weight = emit_param(
        prefix,
        "weight",
        kernel_onnx,
        layer.weight_quantizer,
        nodes,
        initializers,
        use_qonnx,
        store_integer_weights,
        out_channels,
    )

    conv_inputs = [current, q_weight]

    if layer._bias is not None:
        bias_np = to_np(layer._bias)
        q_bias = emit_param(
            prefix, "bias", bias_np, layer.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        conv_inputs.append(q_bias)

    padding = layer.padding
    if isinstance(padding, str):
        auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
        pads = None
    else:
        p = list(padding) if hasattr(padding, "__iter__") else [padding] * ndim
        pads = p + p  # ONNX format: [begin_0, begin_1, ..., end_0, end_1, ...]
        auto_pad = "NOTSET"

    conv_attrs = dict(
        kernel_shape=to_list(layer.kernel_size, ndim),
        strides=to_list(layer.strides, ndim),
        dilations=to_list(layer.dilation_rate, ndim),
        group=getattr(layer, "groups", 1),
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if cl:
        current = add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def add_depthwise_conv(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """PQDepthwiseConv2d.

    Keras kernel: [kH, kW, in, depth_mult]
    ONNX Conv with groups=in: weight [in*depth_mult, 1, kH, kW]
    """
    cl = channels_last(layer)

    if cl:
        current = add_transpose(f"{prefix}_pre", current, [0, 3, 1, 2], nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = to_np(layer._kernel)  # [kH, kW, in, depth_mult]
    in_ch, depth_mult = kernel_np.shape[2], kernel_np.shape[3]
    kernel_onnx = np.transpose(kernel_np, (2, 3, 0, 1)).reshape(in_ch * depth_mult, 1, *kernel_np.shape[:2])

    out_channels = kernel_onnx.shape[0]

    q_weight = emit_param(
        prefix,
        "weight",
        kernel_onnx,
        layer.weight_quantizer,
        nodes,
        initializers,
        use_qonnx,
        store_integer_weights,
        out_channels,
    )

    conv_inputs = [current, q_weight]

    if layer._bias is not None:
        bias_np = to_np(layer._bias)
        q_bias = emit_param(
            prefix, "bias", bias_np, layer.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        conv_inputs.append(q_bias)

    padding = layer.padding
    if isinstance(padding, str):
        auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
        pads = None
    else:
        p = list(padding) if hasattr(padding, "__iter__") else [padding, padding]
        pads = p + p
        auto_pad = "NOTSET"

    conv_attrs = dict(
        kernel_shape=to_list(layer.kernel_size, 2),
        strides=to_list(layer.strides, 2),
        dilations=to_list(layer.dilation_rate, 2),
        group=in_ch,
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if cl:
        current = add_transpose(f"{prefix}_post", current, [0, 2, 3, 1], nodes)
    return current


def add_batchnorm(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    """PQBatchNormalization (also handles plain keras BatchNormalization,
    but emit_layer currently only dispatches the PQ variant here)."""
    need_tr, perm_to_nchw, perm_to_nhwx = bn_transpose_info(layer)

    if need_tr:
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    is_pq = isinstance(layer, PQBatchNormalization)

    gamma_np = to_np(layer.gamma) if layer.gamma is not None else None
    beta_np = to_np(layer.beta) if layer.beta is not None else None

    if gamma_np is None:
        # scale=False: use ones
        n_ch = to_np(layer.moving_mean).shape[0]
        gamma_np = np.ones(n_ch, dtype=np.float32)
    if beta_np is None:
        # center=False: use zeros
        n_ch = to_np(layer.moving_mean).shape[0]
        beta_np = np.zeros(n_ch, dtype=np.float32)

    qonnx_p = use_qonnx and is_pq
    intstore_p = store_integer_weights and is_pq
    q_gamma = emit_param(
        prefix, "gamma", gamma_np, layer.weight_quantizer if is_pq else None, nodes, initializers, qonnx_p, intstore_p
    )
    q_beta = emit_param(
        prefix, "beta", beta_np, layer.bias_quantizer if is_pq else None, nodes, initializers, qonnx_p, intstore_p
    )

    mean_name = f"{prefix}_running_mean"
    var_name = f"{prefix}_running_var"
    initializers.append(onh.from_array(to_np(layer.moving_mean), name=mean_name))
    initializers.append(onh.from_array(to_np(layer.moving_variance), name=var_name))

    bn_out = f"{prefix}_bn"
    nodes.append(
        oh.make_node(
            "BatchNormalization",
            inputs=[current, q_gamma, q_beta, mean_name, var_name],
            outputs=[bn_out],
            epsilon=float(layer.epsilon),
        )
    )
    current = bn_out

    if need_tr:
        current = add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def add_dense_nd(layer, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    kernel_np = to_np(layer._kernel).T  # [out, in]
    out_units = kernel_np.shape[0]

    q_weight = emit_param(
        prefix, "weight", kernel_np, layer.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights, out_units
    )

    # Transpose [out, in] → [in, out] so MatMul(input[..., in], kernel_t[in, out]) works
    kernel_t_name = f"{prefix}_weight_t"
    nodes.append(oh.make_node("Transpose", inputs=[q_weight], outputs=[kernel_t_name], perm=[1, 0]))

    mm_out = f"{prefix}_mm"
    nodes.append(oh.make_node("MatMul", inputs=[current, kernel_t_name], outputs=[mm_out]))
    current = mm_out

    if layer._bias is not None:
        bias_np = to_np(layer._bias)
        q_bias = emit_param(
            prefix, "bias", bias_np, layer.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        add_out = f"{prefix}_bias_add"
        nodes.append(oh.make_node("Add", inputs=[current, q_bias], outputs=[add_out]))
        current = add_out

    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)
    return current


def add_quantized_softmax(sm, prefix, current, nodes, initializers, quant_fn, kpm_mask=None):
    enable = sm.enable_quantization
    scaler = float(sm.input_scaler)
    stable = bool(sm.stable)
    eps = float(sm.epsilon)

    def qdq(q, pfx, x):
        k, i, f = q.get_quantization_bits()
        q_nodes, out = quant_fn(pfx, x, q.round_mode, to_np(k), to_np(i), to_np(f), initializers, overflow_mode=q.overflow)
        nodes.extend(q_nodes)
        return out

    # 1) Softmax input quantizer.
    if sm.quantize_input and enable:
        current = qdq(sm.input_quantizer, f"{prefix}_sm_in_q", current)

    # 2) Stable max-subtract over the last axis (ReduceMax keeps axes as an attribute).
    if stable:
        m_name = f"{prefix}_sm_max"
        nodes.append(oh.make_node("ReduceMax", inputs=[current], outputs=[m_name], axes=[-1], keepdims=1))
        exp_in = f"{prefix}_sm_sub"
        nodes.append(oh.make_node("Sub", inputs=[m_name, current], outputs=[exp_in]))
    else:
        exp_in = current

    # 3) Quantized exp table: optional input QDQ (only when quantize_input==stable),
    #    Exp of (-scaler * x) for the stable branch (+scaler otherwise), output QDQ.
    exp_t = sm.exp_table
    if exp_t.quantize_input and enable:
        exp_in = qdq(exp_t.input_quantizer, f"{prefix}_sm_exp_in_q", exp_in)
    coeff = -scaler if stable else scaler
    exp_arg = exp_in
    if coeff != 1.0:
        coeff_name = f"{prefix}_sm_exp_coeff"
        initializers.append(onh.from_array(np.array(coeff, dtype=np.float32), name=coeff_name))
        exp_arg = f"{prefix}_sm_exp_arg"
        nodes.append(oh.make_node("Mul", inputs=[exp_in, coeff_name], outputs=[exp_arg]))
    exp_inp = f"{prefix}_sm_exp"
    nodes.append(oh.make_node("Exp", inputs=[exp_arg], outputs=[exp_inp]))
    if exp_t.quantize_output and enable:
        exp_inp = qdq(exp_t.output_quantizer, f"{prefix}_sm_exp_out_q", exp_inp)

    # 3b) Optional key-padding mask: zero the exp-numerator at masked positions.
    if kpm_mask is not None:
        kpm_f = f"{prefix}_sm_mask_f"
        nodes.append(oh.make_node("Cast", inputs=[kpm_mask], outputs=[kpm_f], to=TensorProto.FLOAT))
        masked = f"{prefix}_sm_masked"
        nodes.append(oh.make_node("Mul", inputs=[kpm_f, exp_inp], outputs=[masked]))
        exp_inp = masked

    # 4) Sum over the last axis (ReduceSum takes axes as an input from opset 13).
    sum_axes = f"{prefix}_sm_sum_axes"
    initializers.append(onh.from_array(np.array([-1], dtype=np.int64), name=sum_axes))
    sums = f"{prefix}_sm_sum"
    nodes.append(oh.make_node("ReduceSum", inputs=[exp_inp, sum_axes], outputs=[sums], keepdims=1))

    # 5) Quantized reciprocal table: input QDQ, 1/(x+eps), output QDQ.
    inv_t = sm.inv_table
    inv_in = sums
    if inv_t.quantize_input and enable:
        inv_in = qdq(inv_t.input_quantizer, f"{prefix}_sm_inv_in_q", inv_in)
    eps_name = f"{prefix}_sm_eps"
    initializers.append(onh.from_array(np.array(eps, dtype=np.float32), name=eps_name))
    inv_add = f"{prefix}_sm_inv_add"
    nodes.append(oh.make_node("Add", inputs=[inv_in, eps_name], outputs=[inv_add]))
    divisor = f"{prefix}_sm_inv"
    nodes.append(oh.make_node("Reciprocal", inputs=[inv_add], outputs=[divisor]))
    if inv_t.quantize_output and enable:
        divisor = qdq(inv_t.output_quantizer, f"{prefix}_sm_inv_out_q", divisor)

    # 6) Multiply numerator by reciprocal.
    out = f"{prefix}_sm_out"
    nodes.append(oh.make_node("Mul", inputs=[exp_inp, divisor], outputs=[out]))
    current = out

    # 7) Softmax output quantizer.
    if sm.quantize_output and enable:
        current = qdq(sm.output_quantizer, f"{prefix}_sm_out_q", current)
    return current


def add_mha(
    layer,
    prefix,
    q_input,
    k_input,
    v_input,
    nodes,
    initializers,
    quant_fn,
    use_qonnx,
    store_integer_weights,
    key_padding_mask=None,
    attn_mask=None,
):
    H = layer.num_heads
    head_dim = layer.head_dim
    E = layer.embed_dim
    scale_val = float(layer.scale)

    # --- Q / K / V projections: (B, L, E) → (B, L, E) ---
    q_proj_out = add_dense_nd(
        layer.q_proj, f"{prefix}_q_proj", q_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    k_proj_out = add_dense_nd(
        layer.k_proj, f"{prefix}_k_proj", k_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    v_proj_out = add_dense_nd(
        layer.v_proj, f"{prefix}_v_proj", v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    # --- Helper: (B, L, E) → (B, H, L, head_dim) using dynamic shapes ---
    def split_heads(x_name, pfx):
        shape_out = f"{pfx}_shape"
        b_scalar = f"{pfx}_b_sc"
        l_scalar = f"{pfx}_l_sc"
        b_1d = f"{pfx}_b_1d"
        l_1d = f"{pfx}_l_1d"
        h_1d_const = f"{pfx}_H_1d"
        hd_1d_const = f"{pfx}_hd_1d"
        shape_4d = f"{pfx}_shape4d"
        reshaped = f"{pfx}_reshaped"
        transposed = f"{pfx}_transposed"
        idx0 = f"{pfx}_gi0"
        idx1 = f"{pfx}_gi1"
        ax0 = f"{pfx}_ax0"

        nodes.append(oh.make_node("Shape", inputs=[x_name], outputs=[shape_out]))
        initializers.extend(
            [
                onh.from_array(np.array(0, dtype=np.int64), name=idx0),
                onh.from_array(np.array(1, dtype=np.int64), name=idx1),
                onh.from_array(np.array([0], dtype=np.int64), name=ax0),
                onh.from_array(np.array([H], dtype=np.int64), name=h_1d_const),
                onh.from_array(np.array([head_dim], dtype=np.int64), name=hd_1d_const),
            ]
        )
        nodes.append(oh.make_node("Gather", inputs=[shape_out, idx0], outputs=[b_scalar]))
        nodes.append(oh.make_node("Gather", inputs=[shape_out, idx1], outputs=[l_scalar]))
        nodes.append(oh.make_node("Unsqueeze", inputs=[b_scalar, ax0], outputs=[b_1d]))
        nodes.append(oh.make_node("Unsqueeze", inputs=[l_scalar, ax0], outputs=[l_1d]))
        nodes.append(oh.make_node("Concat", inputs=[b_1d, l_1d, h_1d_const, hd_1d_const], outputs=[shape_4d], axis=0))
        nodes.append(oh.make_node("Reshape", inputs=[x_name, shape_4d], outputs=[reshaped]))
        # (B, L, H, head_dim) → (B, H, L, head_dim)
        nodes.append(oh.make_node("Transpose", inputs=[reshaped], outputs=[transposed], perm=[0, 2, 1, 3]))
        return transposed

    q_h = split_heads(q_proj_out, f"{prefix}_q")
    k_h = split_heads(k_proj_out, f"{prefix}_k")
    v_h = split_heads(v_proj_out, f"{prefix}_v")

    k_t_name = f"{prefix}_k_T"
    nodes.append(oh.make_node("Transpose", inputs=[k_h], outputs=[k_t_name], perm=[0, 1, 3, 2]))

    raw_scores = f"{prefix}_scores_raw"
    scaled_scores = f"{prefix}_scores_scaled"
    scale_cst = f"{prefix}_attn_scale"
    nodes.append(oh.make_node("MatMul", inputs=[q_h, k_t_name], outputs=[raw_scores]))
    initializers.append(onh.from_array(np.array(scale_val, dtype=np.float32), name=scale_cst))
    nodes.append(oh.make_node("Mul", inputs=[raw_scores, scale_cst], outputs=[scaled_scores]))
    current = scaled_scores

    if attn_mask is not None:
        masked_scores = f"{prefix}_scores_masked"
        nodes.append(oh.make_node("Add", inputs=[current, attn_mask], outputs=[masked_scores]))
        current = masked_scores

    kpm_mult = None
    if key_padding_mask is not None:
        kpm_not = f"{prefix}_kpm_not"
        nodes.append(oh.make_node("Not", inputs=[key_padding_mask], outputs=[kpm_not]))
        kpm_axes = f"{prefix}_kpm_axes"
        initializers.append(onh.from_array(np.array([1, 2], dtype=np.int64), name=kpm_axes))
        kpm_mult = f"{prefix}_kpm_mask"  # (B, 1, 1, S) bool, cast to float inside the softmax
        nodes.append(oh.make_node("Unsqueeze", inputs=[kpm_not, kpm_axes], outputs=[kpm_mult]))

    current = add_quantized_softmax(
        layer.softmax, f"{prefix}_attn", current, nodes, initializers, quant_fn, kpm_mask=kpm_mult
    )
    attn_w_name = current  # softmax output = attention weights (also averaged over heads below)

    ctx_raw = f"{prefix}_ctx_raw"
    nodes.append(oh.make_node("MatMul", inputs=[current, v_h], outputs=[ctx_raw]))
    current_ctx = ctx_raw

    ctx_t = f"{prefix}_ctx_t"
    ctx_shape = f"{prefix}_ctx_shape"
    ctx_b_sc = f"{prefix}_ctx_b_sc"
    ctx_t_sc = f"{prefix}_ctx_t_sc"
    ctx_b_1d = f"{prefix}_ctx_b_1d"
    ctx_t_1d = f"{prefix}_ctx_t_1d"
    ctx_E_1d = f"{prefix}_ctx_E_1d"
    ctx_ax0 = f"{prefix}_ctx_ax0"
    ctx_gi0 = f"{prefix}_ctx_gi0"
    ctx_gi1 = f"{prefix}_ctx_gi1"
    ctx_3d = f"{prefix}_ctx_shape3d"
    ctx_merged = f"{prefix}_ctx_merged"

    nodes.append(oh.make_node("Transpose", inputs=[current_ctx], outputs=[ctx_t], perm=[0, 2, 1, 3]))
    nodes.append(oh.make_node("Shape", inputs=[ctx_t], outputs=[ctx_shape]))
    initializers += [
        onh.from_array(np.array(0, dtype=np.int64), name=ctx_gi0),
        onh.from_array(np.array(1, dtype=np.int64), name=ctx_gi1),
        onh.from_array(np.array([0], dtype=np.int64), name=ctx_ax0),
        onh.from_array(np.array([E], dtype=np.int64), name=ctx_E_1d),
    ]
    nodes.append(oh.make_node("Gather", inputs=[ctx_shape, ctx_gi0], outputs=[ctx_b_sc]))
    nodes.append(oh.make_node("Gather", inputs=[ctx_shape, ctx_gi1], outputs=[ctx_t_sc]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[ctx_b_sc, ctx_ax0], outputs=[ctx_b_1d]))
    nodes.append(oh.make_node("Unsqueeze", inputs=[ctx_t_sc, ctx_ax0], outputs=[ctx_t_1d]))
    nodes.append(oh.make_node("Concat", inputs=[ctx_b_1d, ctx_t_1d, ctx_E_1d], outputs=[ctx_3d], axis=0))
    nodes.append(oh.make_node("Reshape", inputs=[ctx_t, ctx_3d], outputs=[ctx_merged]))

    # --- Output projection: (B, T, E) → (B, T, E) ---
    out = add_dense_nd(
        layer.out_proj, f"{prefix}_out_proj", ctx_merged, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

    # --- Average attention weights over heads: (B, H, T, S) → (B, T, S) ---
    avg_attn = f"{prefix}_avg_attn_weights"
    nodes.append(oh.make_node("ReduceMean", inputs=[attn_w_name], outputs=[avg_attn], axes=[1], keepdims=0))

    return out, avg_attn


def add_avgpool(layer, prefix, current, nodes, initializers, ndim, quant_fn):
    cl = channels_last(layer)

    if cl:
        perm_to_nchw = [0, 3, 1, 2] if ndim == 2 else [0, 2, 1]
        perm_to_nhwx = [0, 2, 3, 1] if ndim == 2 else [0, 2, 1]
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    pool_out = f"{prefix}_pool"
    nodes.append(
        oh.make_node(
            "AveragePool",
            inputs=[current],
            outputs=[pool_out],
            kernel_shape=to_list(layer.pool_size, ndim),
            strides=to_list(layer.strides, ndim),
            pads=[0] * (ndim * 2),
            count_include_pad=0,
        )
    )
    current = pool_out

    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)

    if cl:
        current = add_transpose(f"{prefix}_post", current, perm_to_nhwx, nodes)
    return current


def add_global_avgpool(layer, prefix, current, nodes, ndim):
    cl = channels_last(layer)

    if cl:
        perm_to_nchw = [0, 3, 1, 2] if ndim == 2 else [0, 2, 1]
        current = add_transpose(f"{prefix}_pre", current, perm_to_nchw, nodes)

    pool_out = f"{prefix}_global_pool"
    nodes.append(oh.make_node("GlobalAveragePool", inputs=[current], outputs=[pool_out]))
    current = pool_out

    if cl:
        flatten_name = f"{prefix}_flatten"
        nodes.append(oh.make_node("Flatten", inputs=[pool_out], outputs=[flatten_name], axis=1))
        current = flatten_name

    return current


def add_pq_activation(layer, prefix, current, nodes, initializers, quant_fn):
    current = maybe_quant_input(layer, prefix, current, nodes, initializers, quant_fn)

    if layer.use_multiplier and layer.activation_name == "relu" and hasattr(layer, "multiplier"):
        m_val = float(np.array(layer.multiplier).ravel()[0])
        scale = float(2.0 ** round(m_val))
        scale_name = f"{prefix}_mul_scale"
        scaled_out = f"{prefix}_scaled"
        initializers.append(onh.from_array(np.array(scale, dtype=np.float32), name=scale_name))
        nodes.append(oh.make_node("Mul", inputs=[current, scale_name], outputs=[scaled_out]))
        current = scaled_out

    act = layer.activation_name
    act_out = f"{prefix}_act"
    if act == "relu":
        nodes.append(oh.make_node("Relu", inputs=[current], outputs=[act_out]))
    elif act == "tanh":
        nodes.append(oh.make_node("Tanh", inputs=[current], outputs=[act_out]))
    elif act == "hard_tanh":
        cmin_name = f"{prefix}_htanh_min"
        cmax_name = f"{prefix}_htanh_max"
        initializers += [
            onh.from_array(np.array(-1.0, dtype=np.float32), name=cmin_name),
            onh.from_array(np.array(1.0, dtype=np.float32), name=cmax_name),
        ]
        nodes.append(oh.make_node("Clip", inputs=[current, cmin_name, cmax_name], outputs=[act_out]))
    else:
        raise TypeError(f"PQActivation: unsupported activation {act!r} for ONNX export")
    current = act_out

    # --- optional output quantization ---
    current = maybe_quant_output(layer, prefix, current, nodes, initializers, quant_fn)
    return current
