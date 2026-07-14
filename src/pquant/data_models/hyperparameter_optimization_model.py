from typing import Any

from pydantic import BaseModel, Field


class HyperparameterSearch(BaseModel):
    numerical: dict[str, list[int | float]] = Field(default_factory=dict)
    categorical: dict[str, list[str]] | None = Field(default_factory=dict)


class Sampler(BaseModel):
    type: str = Field(default="TPESampler")
    params: dict[str, Any] = Field(default_factory=dict)


class BaseHyperparameterOptimizationModel(BaseModel):
    experiment_name: str = Field(default="experiment_1")
    model_name: str = Field(default="example_model")
    sampler: Sampler = Field(default_factory=Sampler)
    num_trials: int = Field(default=0)
    hyperparameter_search: HyperparameterSearch = Field(default_factory=HyperparameterSearch)
