from pydantic import BaseModel, Field

from pquant.core.constants import QuantizationGranularity


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
    dynamic_data_quantization: bool = Field(default=False)
    enable_quantization: bool = Field(default=True)
    hgq_gamma: float = Field(default=0.0003)
    hgq_beta: float = Field(default=1e-5)
    hgq_heterogeneous: bool = Field(default=True)
    layer_specific: dict[str, dict] = Field(default_factory=dict)
    use_high_granularity_quantization: bool = Field(default=False)
    use_real_tanh: bool = Field(default=False)
    overflow_mode_parameters: str = Field(default="SAT")
    overflow_mode_data: str = Field(default="SAT")
    round_mode: str = Field(default="RND")
    use_relu_multiplier: bool = Field(default=True)
