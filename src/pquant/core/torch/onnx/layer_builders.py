"""Per-layer ONNX graph builders (Dense/Conv/BN/LN/AvgPool/Softmax/MHA) for the PQuant torch converter."""

import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto

from pquant.core.torch.onnx.helpers import (
    emit_param,
    maybe_quant_input,
    maybe_quant_output,
    qdq_node,
    to_list,
    torch_padding_to_onnx,
)


def add_dense_integer(module, prefix, current, nodes, initializers):
    if getattr(module, "input_quantizer", None) is None or not module.quantize_input:
        raise ValueError(f"{prefix}: integer_ops requires quantize_input=True on the layer")

    # --- Input: Clip + QuantizeLinear → int8 (stop before DequantizeLinear) ---
    k_x, i_x, f_x = module.input_quantizer.get_quantization_bits()
    k_x_val = int(k_x.item())
    i_x_val = float(i_x.item())
    f_x_val = float(f_x.item())
    s_x = float(2.0 ** (-f_x_val))
    signed_x = k_x_val == 1

    clip_min_x = float(-(2.0**i_x_val)) if signed_x else 0.0
    clip_max_x = float(2.0**i_x_val - 2.0 ** (-f_x_val))
    zp_x_np = np.int8(0) if signed_x else np.uint8(0)

    clip_min_name = f"{prefix}_in_clip_min"
    clip_max_name = f"{prefix}_in_clip_max"
    scale_x_name = f"{prefix}_in_scale"
    zp_x_name = f"{prefix}_in_zp"
    x_int_name = f"{prefix}_in_int"

    initializers += [
        onh.from_array(np.array(clip_min_x, dtype=np.float32), name=clip_min_name),
        onh.from_array(np.array(clip_max_x, dtype=np.float32), name=clip_max_name),
        onh.from_array(np.array(s_x, dtype=np.float32), name=scale_x_name),
        onh.from_array(np.array(zp_x_np), name=zp_x_name),
    ]
    nodes += [
        oh.make_node("Clip", inputs=[current, clip_min_name, clip_max_name], outputs=[f"{prefix}_in_clipped"]),
        oh.make_node("QuantizeLinear", inputs=[f"{prefix}_in_clipped", scale_x_name, zp_x_name], outputs=[x_int_name]),
    ]

    # --- Weights: stored pre-transposed as int8 so MatMulInteger needs no Transpose ---
    # PyTorch weight shape: [out, in].  MatMulInteger(A, B) = A @ B, so we need [in, out].
    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)
    k_w, _, f_w = module.weight_quantizer.get_quantization_bits()
    k_w_val = int(k_w.item())  # get_quantization_bits() always returns tensors
    dtype_w = np.int8 if k_w_val == 1 else np.uint8
    out_ch = weight_np.shape[0]

    f_w_t = f_w.detach().cpu()
    if f_w_t.numel() == 1:
        f_w_1d = np.array([float(f_w_t.item())])
        per_channel_w = False
    else:
        f_w_2d = f_w_t.float().numpy().reshape(out_ch, -1)
        f_w_1d = f_w_2d.min(axis=1)  # min f → max scale → covers all values
        per_channel_w = True

    s_w_1d = (2.0 ** (-f_w_1d)).astype(np.float32)  # shape [1] or [out]
    bcast_s_w = s_w_1d.reshape((out_ch,) + (1,) * (weight_np.ndim - 1)) if per_channel_w else float(s_w_1d[0])
    # Transpose before storing so MatMulInteger can use it without a runtime Transpose node
    int_weights_T = np.round(weight_np / bcast_s_w).astype(dtype_w).T  # [in, out]

    zp_w_np = np.array(dtype_w(0))  # scalar zero-point; zero for symmetric quantization
    w_int_name = f"{prefix}_weight_int"
    w_zp_name = f"{prefix}_weight_zp"
    initializers += [
        onh.from_array(int_weights_T, name=w_int_name),
        onh.from_array(zp_w_np, name=w_zp_name),
    ]

    # --- MatMulInteger([batch, in], [in, out]) → int32 [batch, out] ---
    y_int_name = f"{prefix}_matmul_int"
    nodes.append(
        oh.make_node(
            "MatMulInteger",
            inputs=[x_int_name, w_int_name, zp_x_name, w_zp_name],
            outputs=[y_int_name],
        )
    )

    current_int32 = y_int_name
    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        combined_s = s_x * s_w_1d  # shape [1] or [out]
        bias_int32 = np.round(bias_np / (combined_s if per_channel_w else float(combined_s[0]))).astype(np.int32)
        bias_int_name = f"{prefix}_bias_int"
        y_biased_name = f"{prefix}_matmul_biased"
        initializers.append(onh.from_array(bias_int32, name=bias_int_name))
        nodes.append(oh.make_node("Add", inputs=[current_int32, bias_int_name], outputs=[y_biased_name]))
        current_int32 = y_biased_name

    # --- DequantizeLinear: int32 → float32 using combined scale s_x * s_w ---
    # Per-channel: axis=1 because the output tensor is [batch, out] and out is axis 1.
    combined_scale_name = f"{prefix}_combined_scale"
    combined_zp_name = f"{prefix}_combined_zp"

    if per_channel_w:
        combined_scale_np = (s_x * s_w_1d).astype(np.float32)  # [out]
        combined_zp_np = np.zeros(out_ch, dtype=np.int32)
        dql_kwargs = {"axis": 1}
    else:
        combined_scale_np = np.array(float(s_x * s_w_1d[0]), dtype=np.float32)
        combined_zp_np = np.array(np.int32(0))
        dql_kwargs = {}

    initializers += [
        onh.from_array(combined_scale_np, name=combined_scale_name),
        onh.from_array(combined_zp_np, name=combined_zp_name),
    ]
    y_float_name = f"{prefix}_dequantized"
    nodes.append(
        oh.make_node(
            "DequantizeLinear",
            inputs=[current_int32, combined_scale_name, combined_zp_name],
            outputs=[y_float_name],
            **dql_kwargs,
        )
    )
    current = y_float_name

    # Optional output quantization (e.g. last layer with quantize_output=True)
    current = maybe_quant_output(module, prefix, current, nodes, initializers, qdq_node)
    return current


def add_dense_nd(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)  # [out, in]
    if use_qonnx or store_integer_weights:
        # Quantized/int-stored weight is emitted in native [out, in] layout, then transposed.
        q_weight_native = emit_param(
            prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        q_weight = f"{prefix}_weight_T"
        nodes.append(oh.make_node("Transpose", inputs=[q_weight_native], outputs=[q_weight], perm=[1, 0]))
    else:
        q_weight = f"{prefix}_weight_T"
        initializers.append(onh.from_array(weight_np.T, name=q_weight))  # pre-transposed [in, out]

    matmul_out = f"{prefix}_matmul"
    nodes.append(oh.make_node("MatMul", inputs=[current, q_weight], outputs=[matmul_out]))
    current = matmul_out

    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        q_bias = emit_param(
            prefix, "bias", bias_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        biased_out = f"{prefix}_biased"
        nodes.append(oh.make_node("Add", inputs=[matmul_out, q_bias], outputs=[biased_out]))
        current = biased_out

    current = maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def add_dense(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights, integer_ops=False):
    if integer_ops and not use_qonnx:
        return add_dense_integer(module, prefix, current, nodes, initializers)
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)
    q_weight = emit_param(
        prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )

    gemm_inputs = [current, q_weight]

    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        q_bias = emit_param(
            prefix, "bias", bias_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        gemm_inputs.append(q_bias)

    gemm_out = f"{prefix}_gemm"
    nodes.append(oh.make_node("Gemm", inputs=gemm_inputs, outputs=[gemm_out], transB=1))
    current = gemm_out

    current = maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def add_conv(module, prefix, current, nodes, initializers, ndim, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    weight_np = module._weight.detach().cpu().numpy().astype(np.float32)
    q_weight = emit_param(
        prefix, "weight", weight_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )

    conv_inputs = [current, q_weight]

    if module._bias is not None:
        bias_np = module._bias.detach().cpu().numpy().astype(np.float32)
        q_bias = emit_param(
            prefix, "bias", bias_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
        )
        conv_inputs.append(q_bias)

    padding = module.padding
    if isinstance(padding, str):
        auto_pad = "SAME_UPPER" if padding == "same" else "VALID"
        pads = None
    else:
        auto_pad = "NOTSET"
        pads = torch_padding_to_onnx(padding, ndim)

    conv_attrs = dict(
        kernel_shape=to_list(module.kernel_size, ndim),
        strides=to_list(module.stride, ndim),
        dilations=to_list(module.dilation, ndim),
        group=module.groups,
        auto_pad=auto_pad,
    )
    if pads is not None:
        conv_attrs["pads"] = pads

    conv_out = f"{prefix}_conv"
    nodes.append(oh.make_node("Conv", inputs=conv_inputs, outputs=[conv_out], **conv_attrs))
    current = conv_out

    current = maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def add_batchnorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    gamma_np = module._weight.detach().cpu().numpy().astype(np.float32)
    beta_np = module._bias.detach().cpu().numpy().astype(np.float32)

    q_gamma = emit_param(
        prefix, "gamma", gamma_np, module.weight_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )
    q_beta = emit_param(
        prefix, "beta", beta_np, module.bias_quantizer, nodes, initializers, use_qonnx, store_integer_weights
    )

    mean_name = f"{prefix}_running_mean"
    var_name = f"{prefix}_running_var"
    initializers.append(onh.from_array(module.running_mean.detach().cpu().numpy().astype(np.float32), name=mean_name))
    initializers.append(onh.from_array(module.running_var.detach().cpu().numpy().astype(np.float32), name=var_name))

    bn_out = f"{prefix}_bn"
    nodes.append(
        oh.make_node(
            "BatchNormalization",
            inputs=[current, q_gamma, q_beta, mean_name, var_name],
            outputs=[bn_out],
            epsilon=float(module.eps),
        )
    )
    return bn_out


def add_layernorm(module, prefix, current, nodes, initializers, quant_fn, use_qonnx, store_integer_weights):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    ns = (
        tuple(int(d) for d in module.normalized_shape)
        if hasattr(module.normalized_shape, "__iter__")
        else (int(module.normalized_shape),)
    )
    axis = -len(ns)

    has_weight = module._weight is not None
    has_bias = module._bias is not None

    gamma_np = module._weight.detach().cpu().numpy().astype(np.float32) if has_weight else np.ones(ns, dtype=np.float32)
    beta_np = module._bias.detach().cpu().numpy().astype(np.float32) if has_bias else None

    qonnx_p = use_qonnx and has_weight
    intstore_p = store_integer_weights and has_weight
    q_gamma = emit_param(
        prefix,
        "gamma",
        gamma_np,
        module.weight_quantizer if has_weight else None,
        nodes,
        initializers,
        qonnx_p,
        intstore_p,
    )
    if has_bias:
        q_beta = emit_param(
            prefix,
            "beta",
            beta_np,
            module.bias_quantizer if has_weight else None,
            nodes,
            initializers,
            qonnx_p,
            intstore_p,
        )

    ln_inputs = [current, q_gamma]
    if has_bias:
        ln_inputs.append(q_beta)
    ln_out = f"{prefix}_ln"
    nodes.append(
        oh.make_node(
            "LayerNormalization",
            inputs=ln_inputs,
            outputs=[ln_out],
            axis=axis,
            epsilon=float(module.eps),
        )
    )
    current = ln_out
    current = maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def add_avgpool(module, prefix, current, nodes, initializers, ndim, quant_fn):
    current = maybe_quant_input(module, prefix, current, nodes, initializers, quant_fn)

    pool_out = f"{prefix}_pool"
    nodes.append(
        oh.make_node(
            "AveragePool",
            inputs=[current],
            outputs=[pool_out],
            kernel_shape=to_list(module.kernel_size, ndim),
            strides=to_list(module.stride, ndim),
            pads=torch_padding_to_onnx(module.padding, ndim),
            ceil_mode=int(module.ceil_mode),
            count_include_pad=int(module.count_include_pad),
        )
    )
    current = pool_out

    current = maybe_quant_output(module, prefix, current, nodes, initializers, quant_fn)
    return current


def add_quantized_softmax(sm, prefix, current, nodes, initializers, quant_fn, kpm_mask=None):
    enable = sm.enable_quantization
    scaler = float(sm.input_scaler)
    stable = bool(sm.stable)
    eps = float(sm.epsilon)

    def qdq(q, pfx, x):
        k, i, f = q.get_quantization_bits()
        q_nodes, out = quant_fn(pfx, x, q.round_mode, k, i, f, initializers, overflow_mode=q.overflow)
        nodes.extend(q_nodes)
        return out

    if sm.quantize_input and enable:
        current = qdq(sm.input_quantizer, f"{prefix}_sm_in_q", current)

    if stable:
        m_name = f"{prefix}_sm_max"
        nodes.append(oh.make_node("ReduceMax", inputs=[current], outputs=[m_name], axes=[-1], keepdims=1))
        exp_in = f"{prefix}_sm_sub"
        nodes.append(oh.make_node("Sub", inputs=[m_name, current], outputs=[exp_in]))
    else:
        exp_in = current

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

    if kpm_mask is not None:
        kpm_f = f"{prefix}_sm_mask_f"
        nodes.append(oh.make_node("Cast", inputs=[kpm_mask], outputs=[kpm_f], to=TensorProto.FLOAT))
        masked = f"{prefix}_sm_masked"
        nodes.append(oh.make_node("Mul", inputs=[kpm_f, exp_inp], outputs=[masked]))
        exp_inp = masked

    sum_axes = f"{prefix}_sm_sum_axes"
    initializers.append(onh.from_array(np.array([-1], dtype=np.int64), name=sum_axes))
    sums = f"{prefix}_sm_sum"
    nodes.append(oh.make_node("ReduceSum", inputs=[exp_inp, sum_axes], outputs=[sums], keepdims=1))

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

    out = f"{prefix}_sm_out"
    nodes.append(oh.make_node("Mul", inputs=[exp_inp, divisor], outputs=[out]))
    current = out

    if sm.quantize_output and enable:
        current = qdq(sm.output_quantizer, f"{prefix}_sm_out_q", current)
    return current


def add_mha(
    module,
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
    H = module.num_heads
    head_dim = module.head_dim
    E = module.embed_dim
    scale_val = float(module.scale)

    if not module.batch_first:
        q_t = f"{prefix}_q_in_t"
        k_t = f"{prefix}_k_in_t"
        v_t = f"{prefix}_v_in_t"
        nodes.append(oh.make_node("Transpose", inputs=[q_input], outputs=[q_t], perm=[1, 0, 2]))
        nodes.append(oh.make_node("Transpose", inputs=[k_input], outputs=[k_t], perm=[1, 0, 2]))
        nodes.append(oh.make_node("Transpose", inputs=[v_input], outputs=[v_t], perm=[1, 0, 2]))
        q_input, k_input, v_input = q_t, k_t, v_t

    # --- Q / K / V projections: (B, L, E) → (B, L, E) via MatMul (input is rank-3) ---
    q_proj_out = add_dense_nd(
        module.q_proj, f"{prefix}_q_proj", q_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    k_proj_out = add_dense_nd(
        module.k_proj, f"{prefix}_k_proj", k_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    v_proj_out = add_dense_nd(
        module.v_proj, f"{prefix}_v_proj", v_input, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )

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
        module.softmax, f"{prefix}_attn", current, nodes, initializers, quant_fn, kpm_mask=kpm_mult
    )
    attn_w_name = current  # softmax output = attention weights (also averaged over heads below)

    ctx_raw = f"{prefix}_ctx_raw"
    nodes.append(oh.make_node("MatMul", inputs=[current, v_h], outputs=[ctx_raw]))
    current_ctx = ctx_raw

    ctx_t = f"{prefix}_ctx_t"  # after Transpose → (B, T, H, head_dim)
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

    out = add_dense_nd(
        module.out_proj, f"{prefix}_out_proj", ctx_merged, nodes, initializers, quant_fn, use_qonnx, store_integer_weights
    )
    avg_attn = f"{prefix}_avg_attn_weights"
    nodes.append(oh.make_node("ReduceMean", inputs=[attn_w_name], outputs=[avg_attn], axes=[1], keepdims=0))

    if not module.batch_first:
        out_final = f"{prefix}_out_seq_first"
        nodes.append(oh.make_node("Transpose", inputs=[out], outputs=[out_final], perm=[1, 0, 2]))
        return out_final, avg_attn

    return out, avg_attn
