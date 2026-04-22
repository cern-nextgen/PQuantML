from pydantic import BaseModel, ConfigDict, Field


class BaseTrainingModel(BaseModel):
    model_config = ConfigDict(extra='allow')
    epochs: int = Field(default=200)
    fine_tuning_epochs: int = Field(default=0)
    pretraining_epochs: int = Field(default=50)
    rewind: str = Field(default="never")
    rounds: int = Field(default=1)
    save_weights_epoch: int = Field(default=-1)
    pruning_first: bool = Field(default=False)
