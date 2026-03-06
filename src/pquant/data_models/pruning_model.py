from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BasePruningModel(BaseModel):
    disable_pruning_for_layers: List[str] = Field(default_factory=list)
    enable_pruning: bool = Field(default=True)
    threshold_decay: float = Field(default=0.0)


class CSPruningModel(BasePruningModel):
    pruning_method: Literal["cs"] = "cs"
    final_temp: int = Field(default=200)
    threshold_init: float = Field(default=0)


class DSTPruningModel(BasePruningModel):
    pruning_method: Literal["dst"] = "dst"
    alpha: float = Field(default=5.0e-06)
    max_pruning_pct: float = Field(default=0.99)
    threshold_init: float = Field(default=0.0)
    threshold_type: str = Field(default="channelwise")


class FITCompressPruningModel(BasePruningModel):
    pruning_method: Literal["fitcompress"] = "fitcompress"
    min_frac_bits: float = Field(default=2.0)


class PDPPruningModel(BasePruningModel):
    pruning_method: Literal["pdp"] = "pdp"
    epsilon: float = Field(default=0.015)
    sparsity: float = Field(default=0.8)
    temperature: float = Field(default=1.0e-05)
    structured_pruning: bool = Field(default=False)


class WandaPruningModel(BasePruningModel):
    pruning_method: Literal["wanda"] = "wanda"
    M: Optional[int] = (Field(default=None),)
    N: Optional[int] = (Field(default=None),)
    sparsity: float = Field(default=0.9)
    t_delta: int = Field(default=100)
    t_start_collecting_batch: int = Field(default=100)
    calculate_pruning_budget: bool = Field(default=True)


class AutoSparsePruningModel(BasePruningModel):
    pruning_method: Literal["autosparse"] = "autosparse"
    alpha: float = Field(default=0.5)
    alpha_reset_epoch: int = Field(default=90)
    autotune_epochs: int = Field(default=10)
    backward_sparsity: bool = Field(default=False)
    threshold_init: float = Field(default=-5.0)
    threshold_type: str = Field(default="channelwise")


class ActivationPruningModel(BasePruningModel):
    pruning_method: Literal["activation_pruning"] = "activation_pruning"
    threshold: float = Field(default=0.3)
    t_delta: int = Field(default=50)
    t_start_collecting_batch: int = Field(default=50)


class MetricType(str, Enum):
    UNSTRUCTURED = "UnstructuredSparsity"
    STRUCTURED = "StructuredSparsity"


class ConstraintType(str, Enum):
    EQUALITY = "Equality"
    LEQ = "LessThanOrEqual"
    GEQ = "GreaterThanOrEqual"


class MDMMPruningModel(BasePruningModel):
    pruning_method: Literal["mdmm"] = "mdmm"
    constraint_type: ConstraintType = Field("Equality")
    target_value: float = Field(default=0.0)
    metric_type: MetricType = Field(default="UnstructuredSparsity")
    target_sparsity: float = Field(default=0.9)
    rf: int = Field(default=1)
    epsilon: float = Field(default=1.0e-03)
    scale: float = Field(default=10.0)
    damping: float = Field(default=1.0)
    use_grad: bool = Field(default=False)
    l0_mode: Literal["coarse", "smooth"] = Field(default="coarse")
    scale_mode: Literal["mean", "sum"] = Field(default="mean")
    constraint_lr: float = Field(default=1.0e-3)
