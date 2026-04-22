# FAQs

## What models formats does PQuantML currently support?
PQuantML primarily supports PyTorch and TensorFlow/Keras models and supports both direct construction and automatic layer replacement using `add_compression_layers(...)`.

## What are requirements to use PQuantML?
Install PyTorch with the correct CUDA version that matches your system and other frameworks, like TensorFlow. This prevents version mismatches and GPU compatibility issues.

An example to install PyTorch with CUDA 13.0:

```python
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
## Can I use MLflow locally?
Yes.

PQuantML integrates with MLflow for experiment tracking and model logging and local usage is fully supported.


### Start local MLFlow UI:
```python
mlflow ui --host 0.0.0.0 --port 5000
```

### Use a local or remote database for Optuna tuning:
```python
from pquant.core.finetuning import TuningTask
tuner = TuningTask(config)
tuner.set_storage_db("sqlite:///optuna_study.db")
```
