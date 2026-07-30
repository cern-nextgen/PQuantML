from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

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


class BaseMetricModel(BaseModel):
    """Per-metric parameter block nested under MDMMPruningModel.metric.

    Subclasses carry only the parameters exclusive to one metric and validate them
    themselves; knobs shared across metrics (epsilon, rf, l0_mode, ...) stay on the
    parent. Hooks the parent consults are overridden here, so MDMMPruningModel never
    branches on the metric type.
    """

    def constraint_overrides(self):
        """(constraint_type, target_value) this metric mandates, or None."""
        return None


class UnstructuredSparsityModel(BaseMetricModel):
    metric_type: Literal["UnstructuredSparsity"] = "UnstructuredSparsity"


class StructuredSparsityModel(BaseMetricModel):
    metric_type: Literal["StructuredSparsity"] = "StructuredSparsity"


class FPGAAwareSparsityModel(BaseMetricModel):
    metric_type: Literal["FPGAAwareSparsity"] = "FPGAAwareSparsity"
    precision: int = Field(default=16, ge=1)
    target_resource: Literal["DSP", "BRAM"] = Field(default="DSP")
    bram_width: int = Field(default=36, ge=1)

    @model_validator(mode="after")
    def _validate_bram_packing(self):
        # The metric packs c = bram_width // precision DSP groups per BRAM block
        # (2*bram_width // precision when not divisible); c < 1 exactly when
        # precision > 2*bram_width, which would make BRAM packing impossible.
        if self.target_resource == "BRAM" and self.precision > 2 * self.bram_width:
            raise ValueError(
                f"BRAM packing needs precision <= 2*bram_width "
                f"(got precision={self.precision}, bram_width={self.bram_width})."
            )
        return self


class PACAPatternModel(BaseMetricModel):
    metric_type: Literal["PACAPatternSparsity"] = "PACAPatternSparsity"
    num_patterns_to_keep: int = Field(default=16, ge=1)
    beta: float = Field(default=0.75, ge=0.0, le=1.0)
    distance_metric: Literal["hamming", "valued_hamming", "cosine"] = Field(default="valued_hamming")

    def constraint_overrides(self):
        # PACA drives the pattern-distance metric to zero: always Equality at 0.
        return ConstraintType.EQUALITY, 0.0


MetricModel = Annotated[
    Union[UnstructuredSparsityModel, StructuredSparsityModel, FPGAAwareSparsityModel, PACAPatternModel],
    Field(discriminator="metric_type"),
]

# Legacy flat layout: these keys used to live directly on MDMMPruningModel.
_FLAT_METRIC_KEYS = ("precision", "target_resource", "bram_width", "num_patterns_to_keep", "beta", "distance_metric")


class MDMMPruningModel(BasePruningModel):
    pruning_method: Literal["mdmm"] = "mdmm"
    # Defaults must be the enum members, not bare strings: Pydantic does not validate
    # defaults, so a str default leaves the field holding a str at runtime and triggers
    # `PydanticSerializationUnexpectedValue` ("Expected enum") on model_dump — which in
    # turn makes the discriminated pruning_parameters union fall back to per-member
    # serialization warnings. Using the enum members keeps serialization clean.
    constraint_type: ConstraintType = Field(default=ConstraintType.EQUALITY)
    target_value: float = Field(default=0.0)
    metric: MetricModel = Field(default_factory=UnstructuredSparsityModel)
    target_sparsity: float = Field(default=0.9)
    rf: int = Field(default=1)
    epsilon: float = Field(default=1.0e-03)
    scale: float = Field(default=10.0)
    damping: float = Field(default=1.0)
    use_grad: bool = Field(default=False)
    l0_mode: Literal["coarse", "smooth"] = Field(default="coarse")
    scale_mode: Literal["mean", "sum"] = Field(default="mean")
    constraint_lr: float = Field(default=1.0e-3)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_metric_layout(cls, values):
        # Back-compat with the flat layout, where metric_type and the per-metric
        # parameters were siblings of the shared fields: lift them into the nested
        # `metric` block. A config that already has `metric` is passed through as-is.
        if not isinstance(values, dict) or "metric" in values:
            return values
        metric = {}
        metric_type = values.pop("metric_type", None)
        if metric_type is not None:
            metric["metric_type"] = getattr(metric_type, "value", metric_type)
        for key in _FLAT_METRIC_KEYS:
            if values.get(key) is not None:
                metric[key] = values.pop(key)
        if metric:
            values["metric"] = metric
        return values

    @property
    def metric_type(self) -> MetricType:
        return MetricType(self.metric.metric_type)

    @model_validator(mode="after")
    def _apply_metric_constraint_overrides(self):
        # A metric that mandates its constraint (e.g. PACA: Equality at 0) declares it
        # by overriding constraint_overrides(); no metric-type branching here.
        overrides = self.metric.constraint_overrides()
        if overrides is not None:
            self.constraint_type, self.target_value = overrides
        return self
