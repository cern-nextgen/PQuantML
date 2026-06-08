import importlib
import os
import sys

# flake8: noqa
backend = os.getenv("KERAS_BACKEND", "tensorflow")
if backend == "torch":
    from . import configs
    from .core.hyperparameter_optimization import (
        PQConfig,
        ap_config,
        autosparse_config,
        cs_config,
        dst_config,
        fitcompress_config,
        load_from_dictionary,
        load_from_file,
        mdmm_config,
        pdp_config,
        wanda_config,
    )
    from .core.torch import activations, layers, optimizers, pruning_methods, quantizer
    from .core.torch.layers import (
        add_compression_layers,
        apply_final_compression,
        get_ebops,
        get_layer_keep_ratio,
        get_model_losses,
        load_torch_hgq_model,
        post_training_prune,
    )
    from .core.torch.train import train_model

    _forwards = ["activations", "layers", "quantizer", "optimizers"]

    for name in _forwards:
        mod = importlib.import_module(f".core.torch.{name}", package="pquant")
        sys.modules[f"{__name__}.{name}"] = mod
        setattr(sys.modules[__name__], name, mod)

    _forwards.append("train_model")
    _forwards.append("add_compression_layers")
    _forwards.append("configs")
    _forwards.append("get_layer_keep_ratio")
    _forwards.append("get_model_losses")
    _forwards.append("pruning_methods")
    _forwards.append("post_training_prune")
    _forwards.append("ap_config")
    _forwards.append("autosparse_config")
    _forwards.append("cs_config")
    _forwards.append("dst_config")
    _forwards.append("mdmm_config")
    _forwards.append("pdp_config")
    _forwards.append("wanda_config")
    _forwards.append("fitcompress_config")
    _forwards.append("load_from_file")
    _forwards.append("load_from_dictionary")
    _forwards.append("get_ebops")
    _forwards.append("load_torch_hgq_model")
    _forwards.append("PQConfig")
    __all__ = _forwards

else:
    from . import configs
    from .core.hyperparameter_optimization import (
        PQConfig,
        ap_config,
        autosparse_config,
        cs_config,
        dst_config,
        load_from_dictionary,
        load_from_file,
        mdmm_config,
        pdp_config,
        wanda_config,
    )
    from .core.keras import activations, layers, pruning_methods, quantizer
    from .core.keras.layers import (
        add_compression_layers,
        apply_final_compression,
        get_ebops,
        get_layer_keep_ratio,
        get_model_losses,
        post_training_prune,
    )
    from .core.keras.train import train_model

    _forwards = ["activations", "layers", "quantizer"]

    for name in _forwards:
        mod = importlib.import_module(f".core.keras.{name}", package="pquant")
        sys.modules[f"{__name__}.{name}"] = mod
        setattr(sys.modules[__name__], name, mod)

    _forwards.append("train_model")
    _forwards.append("add_compression_layers")
    _forwards.append("configs")
    _forwards.append("get_layer_keep_ratio")
    _forwards.append("get_model_losses")
    _forwards.append("pruning_methods")
    _forwards.append("post_training_prune")
    _forwards.append("ap_config")
    _forwards.append("autosparse_config")
    _forwards.append("cs_config")
    _forwards.append("dst_config")
    _forwards.append("mdmm_config")
    _forwards.append("pdp_config")
    _forwards.append("wanda_config")
    _forwards.append("load_from_file")
    _forwards.append("load_from_dictionary")
    _forwards.append("PQConfig")
    __all__ = _forwards
