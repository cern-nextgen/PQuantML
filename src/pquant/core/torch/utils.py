from pquant.core.torch.pruning_methods.activation_pruning import ActivationPruning
from pquant.core.torch.pruning_methods.autosparse import AutoSparse
from pquant.core.torch.pruning_methods.cs import ContinuousSparsification
from pquant.core.torch.pruning_methods.dst import DST
from pquant.core.torch.pruning_methods.mdmm import MDMM
from pquant.core.torch.pruning_methods.pdp import PDP
from pquant.core.torch.pruning_methods.wanda import Wanda


def get_pruning_layer(config, layer_type):
    pruning_method = config.pruning_parameters.pruning_method
    if pruning_method == "dst":
        return DST(config, layer_type)
    elif pruning_method == "autosparse":
        return AutoSparse(config, layer_type)
    elif pruning_method == "cs":
        return ContinuousSparsification(config, layer_type)
    elif pruning_method == "pdp":
        return PDP(config, layer_type)
    elif pruning_method == "activation_pruning":
        return ActivationPruning(config, layer_type)
    elif pruning_method == "wanda":
        return Wanda(config, layer_type)
    elif pruning_method == "mdmm":
        return MDMM(config, layer_type)
