from pydantic import BaseModel, Field, model_validator

from pquant.core.constants import DynamicDataMode, QuantizationGranularity


class BaseQuantizationModel(BaseModel):
    default_weight_keep_negatives: float = Field(default=1.0)
    default_weight_integer_bits: float = Field(default=0.0)
    default_weight_fractional_bits: float = Field(default=7.0)
    default_data_keep_negatives: float = Field(default=0.0)
    default_data_integer_bits: float = Field(default=0.0)
    default_data_fractional_bits: float = Field(default=7.0)
    quantize_input: bool = Field(default=True)
    quantize_output: bool = Field(default=False)
    granularity: QuantizationGranularity = Field(default=QuantizationGranularity.PER_TENSOR)
    block_size: int = Field(default=128, gt=0)
    dynamic_data_mode: DynamicDataMode = Field(default=DynamicDataMode.NEVER)
    enable_quantization: bool = Field(default=True)
    hgq_gamma: float = Field(default=1e-8, ge=0.0)
    hgq_beta: float = Field(default=1e-5)
    hgq_heterogeneous: bool = Field(default=True)
    layer_specific: dict[str, dict] = Field(default_factory=dict)
    use_high_granularity_quantization: bool = Field(default=False)
    use_real_tanh: bool = Field(default=False)
    overflow_mode_parameters: str = Field(default="SAT")
    overflow_mode_data: str = Field(default="SAT")
    round_mode: str = Field(default="RND")
    use_relu_multiplier: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def _translate_legacy_dynamic_data_flags(cls, data):
        """Accept configs written with the removed dynamic_data_quantization / dynamic_data_inference booleans."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        dynamic = data.pop("dynamic_data_quantization", None)
        at_inference = data.pop("dynamic_data_inference", None)
        if "dynamic_data_mode" not in data and dynamic is not None:
            if not dynamic:
                data["dynamic_data_mode"] = DynamicDataMode.NEVER
            else:
                data["dynamic_data_mode"] = DynamicDataMode.ALWAYS if at_inference else DynamicDataMode.TRAINING
        return data
