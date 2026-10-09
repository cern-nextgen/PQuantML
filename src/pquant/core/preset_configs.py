from pquant.core.constants import QuantizationGranularity
from pquant.core.hyperparameter_optimization import PQConfig, dst_config, pdp_config, wanda_config

_WEIGHT_BITS = (1, 0, 7)
_FIXED_DATA_BITS = (0, 0, 8)
_HGQ_DATA_BITS = (0, 3, 5)
_FIXED_OVERFLOW_MODE_DATA = "SAT"
_HGQ_OVERFLOW_MODE_DATA = "WRAP"
_OVERFLOW_MODE_PARAMETERS = "SAT_SYM"
_ROUND_MODE = "RND"


def _quantization(
    granularity,
    data_bits,
    weight_bits,
    overflow_mode_data,
    overflow_mode_parameters,
    round_mode,
    learned,
    hgq_beta=None,
    hgq_gamma=None,
    dynamic_data=False,
):
    granularity = QuantizationGranularity(granularity)
    if learned and granularity == QuantizationGranularity.PER_CHANNEL:
        raise ValueError("HGQ supports 'per_weight' and 'per_tensor' granularity, not 'per_channel'")
    k_data, i_data, f_data = data_bits
    k_weight, i_weight, f_weight = weight_bits
    section = {
        "use_high_granularity_quantization": learned,
        "granularity": granularity.value,
        "default_data_keep_negatives": k_data,
        "default_data_integer_bits": i_data,
        "default_data_fractional_bits": f_data,
        "default_weight_keep_negatives": k_weight,
        "default_weight_integer_bits": i_weight,
        "default_weight_fractional_bits": f_weight,
        "overflow_mode_data": overflow_mode_data,
        "overflow_mode_parameters": overflow_mode_parameters,
        "round_mode": round_mode,
        "dynamic_data_quantization": dynamic_data,
    }
    if learned:
        section.update(hgq_beta=hgq_beta, hgq_gamma=hgq_gamma)
    return section


def _structured_pdp(target_sparsity, epochs):
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    return {"structured_pruning": True, "sparsity": target_sparsity, "epsilon": 1.0 / (0.9 * epochs)}


def _nm_wanda(n, m, prune_at_epoch, calibration_batches):
    if not (isinstance(n, int) and isinstance(m, int)) or m < 2 or not 0 < n < m:
        raise ValueError(f"n:m must be integers with 0 < n < m and m >= 2, got {n}:{m}")
    return {
        "N": n,
        "M": m,
        "sparsity": n / m,
        "calculate_pruning_budget": False,
        "t_start_collecting_batch": prune_at_epoch,
        "t_delta": calibration_batches,
    }


def _no_pruning():
    return PQConfig.load_from_config({"pruning_parameters": {"pruning_method": None}})


def _build(base, **sections):
    config = base.get_dict()
    for name, values in sections.items():
        config[name].update(values)
    return PQConfig.load_from_config(config)


def quantized(
    granularity="per_tensor",
    data_bits=_FIXED_DATA_BITS,
    weight_bits=_WEIGHT_BITS,
    overflow_mode_data=_FIXED_OVERFLOW_MODE_DATA,
    overflow_mode_parameters=_OVERFLOW_MODE_PARAMETERS,
    round_mode=_ROUND_MODE,
    epochs=100,
    dynamic_data=True,
):
    return _build(
        _no_pruning(),
        quantization_parameters=_quantization(
            granularity,
            data_bits,
            weight_bits,
            overflow_mode_data,
            overflow_mode_parameters,
            round_mode,
            learned=False,
            dynamic_data=dynamic_data,
        ),
        training_parameters={"pretraining_epochs": 0, "epochs": epochs, "fine_tuning_epochs": 0},
    )


def hgq(
    granularity="per_weight",
    data_bits=_HGQ_DATA_BITS,
    weight_bits=_WEIGHT_BITS,
    overflow_mode_data=_HGQ_OVERFLOW_MODE_DATA,
    overflow_mode_parameters=_OVERFLOW_MODE_PARAMETERS,
    round_mode=_ROUND_MODE,
    hgq_beta=1e-5,
    bitwidth_regularization_l1=1e-8,
    epochs=100,
):

    return _build(
        _no_pruning(),
        quantization_parameters=_quantization(
            granularity,
            data_bits,
            weight_bits,
            overflow_mode_data,
            overflow_mode_parameters,
            round_mode,
            learned=True,
            hgq_beta=hgq_beta,
            hgq_gamma=bitwidth_regularization_l1,
        ),
        training_parameters={"pretraining_epochs": 0, "epochs": epochs, "fine_tuning_epochs": 0},
    )


def quantized_unstructured_pruning(
    alpha=1e-7,
    granularity="per_tensor",
    data_bits=_FIXED_DATA_BITS,
    weight_bits=_WEIGHT_BITS,
    overflow_mode_data=_FIXED_OVERFLOW_MODE_DATA,
    overflow_mode_parameters=_OVERFLOW_MODE_PARAMETERS,
    round_mode=_ROUND_MODE,
    epochs=100,
    fine_tuning_epochs=0,
    dynamic_data=True,
):
    return _build(
        dst_config(),
        pruning_parameters={"alpha": alpha, "threshold_type": "weightwise"},
        quantization_parameters=_quantization(
            granularity,
            data_bits,
            weight_bits,
            overflow_mode_data,
            overflow_mode_parameters,
            round_mode,
            learned=False,
            dynamic_data=dynamic_data,
        ),
        training_parameters={"pretraining_epochs": 0, "epochs": epochs, "fine_tuning_epochs": fine_tuning_epochs},
    )


def quantized_structured_pruning(
    target_sparsity,
    granularity="per_tensor",
    data_bits=_FIXED_DATA_BITS,
    weight_bits=_WEIGHT_BITS,
    overflow_mode_data=_FIXED_OVERFLOW_MODE_DATA,
    overflow_mode_parameters=_OVERFLOW_MODE_PARAMETERS,
    round_mode=_ROUND_MODE,
    pretraining_epochs=10,
    epochs=100,
    fine_tuning_epochs=10,
    dynamic_data=True,
):
    return _build(
        pdp_config(),
        pruning_parameters=_structured_pdp(target_sparsity, epochs),
        quantization_parameters=_quantization(
            granularity,
            data_bits,
            weight_bits,
            overflow_mode_data,
            overflow_mode_parameters,
            round_mode,
            learned=False,
            dynamic_data=dynamic_data,
        ),
        training_parameters={
            "pretraining_epochs": pretraining_epochs,
            "epochs": epochs,
            "fine_tuning_epochs": fine_tuning_epochs,
        },
    )


def quantized_nm_pruning(
    n=2,
    m=4,
    granularity="per_tensor",
    data_bits=_FIXED_DATA_BITS,
    weight_bits=_WEIGHT_BITS,
    overflow_mode_data=_FIXED_OVERFLOW_MODE_DATA,
    overflow_mode_parameters=_OVERFLOW_MODE_PARAMETERS,
    round_mode=_ROUND_MODE,
    pretraining_epochs=10,
    epochs=100,
    prune_at_epoch=10,
    calibration_batches=100,
    dynamic_data=True,
):
    return _build(
        wanda_config(),
        pruning_parameters=_nm_wanda(n, m, prune_at_epoch, calibration_batches),
        quantization_parameters=_quantization(
            granularity,
            data_bits,
            weight_bits,
            overflow_mode_data,
            overflow_mode_parameters,
            round_mode,
            learned=False,
            dynamic_data=dynamic_data,
        ),
        training_parameters={"pretraining_epochs": pretraining_epochs, "epochs": epochs, "fine_tuning_epochs": 0},
    )


def hgq_structured_pruning(
    target_sparsity,
    granularity="per_tensor",
    data_bits=_HGQ_DATA_BITS,
    weight_bits=_WEIGHT_BITS,
    overflow_mode_data=_HGQ_OVERFLOW_MODE_DATA,
    overflow_mode_parameters=_OVERFLOW_MODE_PARAMETERS,
    round_mode=_ROUND_MODE,
    hgq_beta=1e-5,
    bitwidth_regularization_l1=1e-8,
    pretraining_epochs=10,
    epochs=100,
    fine_tuning_epochs=10,
):

    return _build(
        pdp_config(),
        pruning_parameters=_structured_pdp(target_sparsity, epochs),
        quantization_parameters=_quantization(
            granularity,
            data_bits,
            weight_bits,
            overflow_mode_data,
            overflow_mode_parameters,
            round_mode,
            learned=True,
            hgq_beta=hgq_beta,
            hgq_gamma=bitwidth_regularization_l1,
        ),
        training_parameters={
            "pretraining_epochs": pretraining_epochs,
            "epochs": epochs,
            "fine_tuning_epochs": fine_tuning_epochs,
        },
    )
