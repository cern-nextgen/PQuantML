import logging

import torch
import torch.nn.functional as F

from pquant.core.torch.activations import PQActivation
from pquant.core.torch.layers import (
    PQAvgPoolBase,
    PQBatchNorm1d,
    PQBatchNorm2d,
    PQWeightBiasBase,
)
from pquant.core.torch.quantizer import Quantizer

_PQUANTML_LAYER_TYPES = (PQWeightBiasBase, PQAvgPoolBase, PQBatchNorm1d, PQBatchNorm2d, PQActivation)


def _analyze_quantization(model):
    """Trace model with torch.fx and return (traced, node_issues, edges_to_quantize).

    node_issues maps each problematic Node to a list of human-readable
    descriptions of what quantization is missing at that node.

    edges_to_quantize is a set of (producer_node, consumer_node) tuples
    identifying graph edges on which a data quantizer should be inserted to
    fix the missing quantization.
    """
    import operator
    from collections import defaultdict

    from torch.fx import GraphModule, Tracer

    class _PQTracer(Tracer):
        def is_leaf_module(self, m, module_qualified_name):
            if isinstance(m, (_PQUANTML_LAYER_TYPES, Quantizer)):
                return True
            return super().is_leaf_module(m, module_qualified_name)

    tracer = _PQTracer()
    graph = tracer.trace(model)
    traced = GraphModule(tracer.root, graph)
    modules = dict(traced.named_modules())

    arith_functions = {
        operator.add,
        operator.iadd,
        operator.sub,
        operator.isub,
        operator.mul,
        operator.imul,
        operator.matmul,
        operator.imatmul,
        operator.truediv,
        operator.itruediv,
        operator.floordiv,
        operator.ifloordiv,
        operator.pow,
        operator.ipow,
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        torch.divide,
        torch.true_divide,
        torch.floor_divide,
        torch.matmul,
        torch.bmm,
        torch.mm,
        torch.einsum,
        torch.pow,
        torch.cat,
        torch.stack,
    }
    for _name in ("concat", "concatenate"):
        _fn = getattr(torch, _name, None)
        if _fn is not None:
            arith_functions.add(_fn)

    _nonlin_torch_names = (
        "sigmoid",
        "tanh",
        "exp",
        "log",
        "log2",
        "log10",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "softmax",
        "log_softmax",
        "sin",
        "cos",
        "tan",
    )
    _nonlin_F_names = (
        "sigmoid",
        "tanh",
        "softmax",
        "log_softmax",
        "gelu",
        "silu",
        "elu",
        "selu",
        "softplus",
        "mish",
        "hardsigmoid",
        "hardswish",
        "leaky_relu",
    )
    nonlin_functions = set()
    for _name in _nonlin_torch_names:
        _fn = getattr(torch, _name, None)
        if _fn is not None:
            nonlin_functions.add(_fn)
    for _name in _nonlin_F_names:
        _fn = getattr(F, _name, None)
        if _fn is not None:
            nonlin_functions.add(_fn)

    quant_sensitive_functions = arith_functions | nonlin_functions
    quant_sensitive_methods = {
        "add",
        "add_",
        "__add__",
        "__radd__",
        "__iadd__",
        "sub",
        "sub_",
        "__sub__",
        "__rsub__",
        "__isub__",
        "mul",
        "mul_",
        "__mul__",
        "__rmul__",
        "__imul__",
        "matmul",
        "__matmul__",
        "__rmatmul__",
        "__imatmul__",
        "div",
        "div_",
        "__truediv__",
        "__rtruediv__",
        "__itruediv__",
        "floor_divide",
        "floor_divide_",
        "__floordiv__",
        "__rfloordiv__",
        "__ifloordiv__",
        "true_divide",
        "true_divide_",
        "pow",
        "pow_",
        "__pow__",
        "__rpow__",
        "__ipow__",
        "bmm",
        "mm",
        "einsum",
        "sigmoid",
        "sigmoid_",
        "tanh",
        "tanh_",
        "exp",
        "exp_",
        "log",
        "log_",
        "log2",
        "log2_",
        "log10",
        "log10_",
        "sqrt",
        "sqrt_",
        "rsqrt",
        "rsqrt_",
        "reciprocal",
        "reciprocal_",
        "softmax",
        "log_softmax",
        "sin",
        "sin_",
        "cos",
        "cos_",
        "tan",
        "tan_",
    }

    def is_quant_sensitive(node):
        if node.op == "call_function":
            return node.target in quant_sensitive_functions
        if node.op == "call_method":
            return node.target in quant_sensitive_methods
        return False

    def get_module(node):
        if node.op == "call_module":
            return modules.get(node.target)
        return None

    quantized: dict = {}
    node_issues = defaultdict(list)
    edges_to_quantize = set()

    for node in traced.graph.nodes:
        input_nodes = node.all_input_nodes
        all_inputs_quantized = bool(input_nodes) and all(quantized.get(n, False) for n in input_nodes)

        if node.op in ("placeholder", "get_attr"):
            quantized[node] = False
        elif node.op == "call_module":
            mod = get_module(node)
            if isinstance(mod, _PQUANTML_LAYER_TYPES):
                if not getattr(mod, "quantize_input", False) and not all_inputs_quantized:
                    node_issues[node].append(
                        f"PQuantML layer '{node.target}' has quantize_input=False but receives unquantized input"
                    )
                    for n in input_nodes:
                        if not quantized.get(n, False):
                            edges_to_quantize.add((n, node))
                # A relu activation is a grid-preserving clip (its optional multiplier is a
                # power-of-two scale), so if the value reaching it is already quantized the
                # output stays on-grid and needs no output quantizer.
                grid_preserving = isinstance(mod, PQActivation) and mod.activation_name == "relu"
                input_quantized = bool(getattr(mod, "quantize_input", False)) or all_inputs_quantized
                quantized[node] = bool(getattr(mod, "quantize_output", False)) or (grid_preserving and input_quantized)
            elif isinstance(mod, Quantizer):
                quantized[node] = True
            else:
                quantized[node] = all_inputs_quantized
        elif node.op in ("call_function", "call_method"):
            if is_quant_sensitive(node):
                for n in input_nodes:
                    if not quantized.get(n, False):
                        node_issues[node].append(f"input '{n.name}' is not quantized")
                        edges_to_quantize.add((n, node))
                quantized[node] = False
            else:
                quantized[node] = all_inputs_quantized
        elif node.op == "output":
            for n in input_nodes:
                if not quantized.get(n, False):
                    node_issues[node].append(f"model output '{n.name}' is not quantized")
                    edges_to_quantize.add((n, node))
            quantized[node] = all_inputs_quantized
        else:
            quantized[node] = False

    return traced, node_issues, edges_to_quantize


def _insert_missing_quantizers(traced, edges_to_quantize, config):
    from collections import defaultdict

    qp = config.quantization_parameters

    def _make_quantizer():
        return Quantizer(
            k=qp.default_data_keep_negatives,
            i=qp.default_data_integer_bits,
            f=qp.default_data_fractional_bits,
            overflow=qp.overflow_mode_data,
            round_mode=qp.round_mode,
            is_heterogeneous=False,
            is_data=True,
            granularity="per_tensor",
            hgq_gamma=qp.hgq_gamma,
        )

    def _enable_pquantml_output_quantization(layer):
        layer.quantize_output = True
        if isinstance(layer, PQWeightBiasBase) and getattr(layer, "built", False) and not hasattr(layer, "output_quantizer"):
            device = next(layer.parameters()).device
            layer.output_quantizer = Quantizer(
                k=torch.tensor(layer.k_output),
                i=torch.tensor(layer.i_output),
                f=torch.tensor(layer.f_output),
                overflow=layer.overflow_mode_data,
                round_mode=layer.round_mode,
                is_heterogeneous=layer.use_hgq,
                is_data=True,
                hgq_gamma=layer.hgq_gamma,
                place="datalane",
            ).to(device)

    modules = dict(traced.named_modules())

    by_producer = defaultdict(list)
    for producer, consumer in edges_to_quantize:
        by_producer[producer].append(consumer)

    graph = traced.graph
    idx = 0
    for producer, consumers in by_producer.items():
        pqml_layer = None
        if producer.op == "call_module":
            mod = modules.get(producer.target)
            if isinstance(mod, _PQUANTML_LAYER_TYPES):
                pqml_layer = mod
        if pqml_layer is not None:
            _enable_pquantml_output_quantization(pqml_layer)
            continue
        q_name = f"_auto_missing_quantizer_{idx}"
        idx += 1
        traced.add_module(q_name, _make_quantizer())
        with graph.inserting_after(producer):
            qnode = graph.call_module(q_name, (producer,))
        for consumer in consumers:
            consumer.replace_input_with(producer, qnode)

    graph.lint()
    traced.recompile()
    return traced


def check_quantization(model, add_missing_quantizers=False, config=None):
    """Verify quantization is present everywhere in the model's forward graph.

    The model's input is assumed to be unquantized. The model is traced with
    torch.fx; PQuantML layers and Quantizer modules are treated as leaves.
    For each node the output's quantization state is propagated forward:
      - PQuantML layers (PQWeightBiasBase, PQAvgPoolBase, PQBatchNorm1d,
        PQBatchNorm2d, PQActivation): output is quantized iff quantize_output
        is True. If quantize_input is False and the incoming data is not
        already quantized, that is reported as missing quantization.
      - Quantizer modules always produce quantized output.
      - Non-PQuantML quant-sensitive ops — arithmetic/combining
        (add/sub/mul/div/matmul/bmm/mm/einsum/cat/stack/pow) and off-grid
        nonlinearities (sigmoid/tanh/softmax/exp/log/sqrt/rsqrt/reciprocal/
        gelu/silu/elu/selu/softplus/mish/hardsigmoid/hardswish/leaky_relu/
        sin/cos/tan), whether used as torch functions, torch.nn.functional
        functions, operator.* functions, or tensor methods — require each
        input to already be quantized; their own output is marked unquantized
        so the next consumer's per-edge input check flags it if needed.
      - The model's output node(s) are required to be quantized: any
        unquantized return value is flagged.
      - Other ops (shape-only views, indexing, non-arithmetic torch calls)
        propagate the quantization state of their inputs.

    When `add_missing_quantizers` is False (default), returns True if every
    location in the graph has the required quantization, otherwise a list of
    strings describing each missing quantization.

    When `add_missing_quantizers` is True, `config` must be provided. For
    every producer whose output feeds an edge with missing quantization:
    - if the producer is a PQuantML layer, its `quantize_output` flag is
      set to True (and, for built `PQWeightBiasBase` subclasses that lacked
      an output_quantizer, one is constructed in-place from the layer's
      own k_output/i_output/f_output and mode settings);
    - otherwise, a fresh `Quantizer` module is instantiated with the
      config's default data k/i/f and data round/overflow modes, attached
      to the transformed module as `_auto_missing_quantizer_<idx>`, and a
      `call_module` to that new quantizer is inserted on each affected
      edge. Using one Quantizer per insertion site (rather than sharing
      one) allows heterogeneous quantization to specialize per site.
    Returns the transformed torch.fx GraphModule.
    """
    traced, node_issues, edges_to_quantize = _analyze_quantization(model)

    if add_missing_quantizers:
        if config is None:
            raise ValueError("check_quantization(add_missing_quantizers=True) requires config")
        if edges_to_quantize:
            _insert_missing_quantizers(traced, edges_to_quantize, config)
        return traced

    if not node_issues:
        return True
    messages = []
    for node, msgs in node_issues.items():
        for msg in msgs:
            messages.append(f"'{node.name}' ({node.op}): {msg}")
    return messages


def print_quantization_check(model, use_color=True):
    """Print the traced model's graph with missing-quantization nodes flagged.

    Runs `check_quantization(model)` and prints the torch.fx graph line by
    line. Nodes with missing quantization are highlighted in red (git-diff
    style) and followed by one annotated line per issue.

    Set use_color=False to disable ANSI escape codes (e.g. when redirecting
    to a file or a terminal that does not support colors).

    Returns the same value as `check_quantization(model)`.
    """
    traced, node_issues, _ = _analyze_quantization(model)

    RED = "\033[31m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    for node in traced.graph.nodes:
        line = node.format_node() or f"{node.name} = {node.op} {node.target}"
        if node in node_issues:
            logging.info(f"{RED}{BOLD}- {line}{RESET}")
            for msg in node_issues[node]:
                logging.info(f"{RED}    ! {msg}{RESET}")
        else:
            logging.info(f"  {line}")

    if not node_issues:
        return True
    messages = []
    for node, msgs in node_issues.items():
        for msg in msgs:
            messages.append(f"'{node.name}' ({node.op}): {msg}")
    return messages
