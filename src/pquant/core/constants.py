from enum import Enum

import optuna

from pquant.data_models.pruning_model import (
    ActivationPruningModel,
    AutoSparsePruningModel,
    CSPruningModel,
    DSTPruningModel,
    FITCompressPruningModel,
    MDMMPruningModel,
    PDPPruningModel,
    WandaPruningModel,
)


class QuantizationGranularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_WEIGHT = "per_weight"


PRUNING_MODEL_REGISTRY = {
    "cs": CSPruningModel,
    "dst": DSTPruningModel,
    "fitcompress": FITCompressPruningModel,
    "pdp": PDPPruningModel,
    "wanda": WandaPruningModel,
    "autosparse": AutoSparsePruningModel,
    "activation_pruning": ActivationPruningModel,
    "mdmm": MDMMPruningModel,
}

SAMPLER_REGISTRY = {
    "GridSampler": optuna.samplers.GridSampler,
    "RandomSampler": optuna.samplers.RandomSampler,
    "TPESampler": optuna.samplers.TPESampler,
    "CmaEsSampler": optuna.samplers.CmaEsSampler,
    "GPSampler": optuna.samplers.GPSampler,
    "NSGAIISampler": optuna.samplers.NSGAIISampler,
    "NSGAIIISampler": optuna.samplers.NSGAIIISampler,
    "QMCSampler": optuna.samplers.QMCSampler,
    "BruteForceSampler": optuna.samplers.BruteForceSampler,
}

TRACKING_URI = "http://0.0.0.0:5000/"
DB_STORAGE = "sqlite:///optuna_study.db"

TORCH_BACKEND = "torch"
TF_BACKEND = 'tensorflow'

FINETUNING_DIRECTION = {"maximize", "minimize"}
CONFIG_FILE = "config.yaml"

N_JOBS = 1

# --- Hardware-aware pruning metric constants ---
# Conv-kernel layout -> axis index, used to canonicalise weight layouts in the PACA
# pattern utilities. Keras conv weights are HWIO, Torch conv weights are OIHW.
CONV_LAYOUT_AXES = {"H": 0, "W": 1, "I": 2, "O": 3}
CANONICAL_CONV_LAYOUT = "OIHW"

# PACAPatternMetric pattern-distance metrics
DISTANCE_HAMMING = "hamming"
DISTANCE_VALUED_HAMMING = "valued_hamming"
DISTANCE_COSINE = "cosine"
PACA_DISTANCE_METRICS = (DISTANCE_HAMMING, DISTANCE_VALUED_HAMMING, DISTANCE_COSINE)

# FPGAAwareSparsityMetric target hardware resources
TARGET_RESOURCE_DSP = "DSP"
TARGET_RESOURCE_BRAM = "BRAM"
FPGA_TARGET_RESOURCES = (TARGET_RESOURCE_DSP, TARGET_RESOURCE_BRAM)
