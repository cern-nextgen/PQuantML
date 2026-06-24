from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    FPGA_AWARE = "FPGAAwareSparsity"
    PACA_PATTERN = "PACAPatternSparsity"


class ConstraintType(str, Enum):
    EQUALITY = "Equality"
    LEQ = "LessThanOrEqual"
    GEQ = "GreaterThanOrEqual"


class MDMMPruningModel(BasePruningModel):
    pruning_method: Literal["mdmm"] = "mdmm"
    # Defaults must be the enum members, not bare strings: Pydantic does not validate
    # defaults, so a str default leaves the field holding a str at runtime and triggers
    # `PydanticSerializationUnexpectedValue` ("Expected enum") on model_dump — which in
    # turn makes the discriminated pruning_parameters union fall back to per-member
    # serialization warnings. Using the enum members keeps serialization clean.
    constraint_type: ConstraintType = Field(default=ConstraintType.EQUALITY)
    target_value: float = Field(default=0.0)
    metric_type: MetricType = Field(default=MetricType.UNSTRUCTURED)
    target_sparsity: float = Field(default=0.9)
    rf: int = Field(default=1)
    epsilon: float = Field(default=1.0e-03)
    scale: float = Field(default=10.0)
    damping: float = Field(default=1.0)
    use_grad: bool = Field(default=False)
    l0_mode: Literal["coarse", "smooth"] = Field(default="coarse")
    scale_mode: Literal["mean", "sum"] = Field(default="mean")
    constraint_lr: float = Field(default=1.0e-3)
    # --- Hardware-aware metric parameters ---
    # Each group is consumed only when the matching metric_type is selected. The MDMM
    # layer filters these against the chosen metric's __init__ signature, so leaving a
    # value as None falls back to that metric's own default. Validation lives here
    # (ge/le + Literal) rather than as asserts inside the metric classes.
    # FPGAAwareSparsityMetric  (metric_type == "FPGAAwareSparsity")
    precision: Optional[int] = Field(default=None, ge=1)
    target_resource: Optional[Literal["DSP", "BRAM"]] = Field(default=None)
    bram_width: Optional[int] = Field(default=None, ge=1)
    # PACAPatternMetric  (metric_type == "PACAPatternSparsity")
    num_patterns_to_keep: Optional[int] = Field(default=None, ge=1)
    beta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    distance_metric: Optional[Literal["hamming", "valued_hamming", "cosine"]] = Field(default=None)

    @model_validator(mode="after")
    def _enforce_paca_constraint(self):
        # PACA pattern pruning is defined as driving the pattern-distance metric to
        # zero, so it always pairs with an equality constraint at target 0. Enforced
        # here (in the data model) so the MDMM layer can stay metric-agnostic.
        if self.metric_type == MetricType.PACA_PATTERN:
            self.constraint_type = ConstraintType.EQUALITY
            self.target_value = 0.0
        return self
