
from pydantic import BaseModel, Field


class PruningSchedule(BaseModel):
    start: int = Field(default=0)
    end: int = Field(default=-3)
    steps: int = Field(default=40)


class BaseFitCompressModel(BaseModel):
    enable_fitcompress: bool = Field(default=False)
    optimize_quantization: bool = Field(default=True)
    quantization_schedule: list[float] = Field(default_factory=lambda: [7.0, 4.0, 3.0, 2.0])
    pruning_schedule: PruningSchedule = Field(default_factory=PruningSchedule)
    compression_goal: float = Field(default=0.10)
    optimize_pruning: bool = Field(default=False)
    greedy_astar: bool = Field(default=True)
    approximate: bool = Field(default=True)
    f_lambda: float = Field(default=1)
