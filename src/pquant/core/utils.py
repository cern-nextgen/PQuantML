from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def get_default_config(pruning_method: str):
    assert pruning_method in [
        "autosparse",
        "ap",
        "cs",
        "dst",
        "pdp",
        "wanda",
        "mdmm",
    ], "Unkown pruning method. Acceptable pruning methods: autosparse, ap, cs, dst, pdp, wanda"
    yaml_name = f"config_{pruning_method}.yaml"
    return get_pruning_config(CONFIG_DIR / yaml_name)


def get_pruning_config(config_path):
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def write_config_to_yaml(config, output_path, sort_keys=True):
    with Path(output_path).open("w") as f:
        yaml.dump(config, f, sort_keys=sort_keys)


def validate_pruning_parameters(config):
    pruning_method = config.pruning_parameters.pruning_method
    if pruning_method == "dst":
        valid_keys = [
            "alpha",
            "disable_pruning_for_layers",
            "enable_pruning",
            "max_pruning_pct",
            "pruning_method",
            "threshold_decay",
            "threshold_init",
            "threshold_type",
        ]
    elif pruning_method == "autosparse":
        valid_keys = [
            "alpha",
            "alpha_reset_epoch",
            "backward_sparsity",
            "disable_pruning_for_layers",
            "enable_pruning",
            "pruning_method",
            "threshold_decay",
            "threshold_init",
            "threshold_type",
        ]
    elif pruning_method == "cs":
        valid_keys = [
            "disable_pruning_for_layers",
            "enable_pruning",
            "final_temp",
            "pruning_method",
            "threshold_decay",
            "threshold_init",
        ]
    elif pruning_method == "pdp":
        valid_keys = [
            "disable_pruning_for_layers",
            "enable_pruning",
            "epsilon",
            "sparsity",
            "temperature",
            "threshold_decay",
            "structured_pruning",
        ]
    elif pruning_method == "activation_pruning":
        valid_keys = [
            "disable_pruning_for_layers",
            "enable_pruning",
            "pruning_method",
            "threshold",
            "threshold_decay",
            "t_delta",
        ]
    elif pruning_method == "wanda":
        valid_keys = [
            "disable_pruning_for_layers",
            "enable_pruning",
            "M",
            "N",
            "pruning_method",
            "threshold_decay",
            "t_delta",
            "t_start_collecting",
            "sparsity",
        ]
    for k in valid_keys:
        assert k in config.pruning_parameters, f"missing pruning parameter: {k}"


def validate_quantization_parameters(config):
    valid_keys = [
        "default_integer_bits",
        "default_fractional_bits",
        "enable_quantization",
        "hgq_gamma",
        "layer_specific",
        "use_high_granularity_quantization",
        "use_real_tanh",
        "use_symmetric_quantization",
    ]
    for k in valid_keys:
        assert k in config.quantization_parameters, f"missing quantization parameter: {k}"


def validate_training_parameters(config):
    valid_keys = [
        "epochs",
        "fine_tuning_epochs",
        "pretraining_epochs",
        "pruning_first",
        "rewind",
        "rounds",
        "save_weights_epoch",
    ]
    for k in valid_keys:
        assert k in config.training_parameters, f"missing training parameter: {k}"


def validate_config(config):
    validate_pruning_parameters(config)
    validate_quantization_parameters(config)
    validate_training_parameters(config)
