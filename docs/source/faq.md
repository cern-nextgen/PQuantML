# FAQs

## What models formats does PQuantML currently support?
PQuantML primarily supports PyTorch and TensorFlow/Keras models and supports both direct construction and automatic layer replacement using `add_compression_layers(...)` method.


## What are the requirements for using PQuantML?
PQuantML supports two backends. If you are using the PyTorch backend, make sure to install a version of PyTorch built for the CUDA version installed on your system. If you are also using frameworks such as TensorFlow, ensure that all frameworks are compatible with the same CUDA version to avoid version conflicts and GPU compatibility issues.

An example to install PyTorch with CUDA 13.0:

```python
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Can I export models to ONNX?

Yes. PQuantML supports exporting compatible models to the **ONNX** format, making it easy to deploy quantized models across different inference runtimes and hardware platforms.


## Can I use MLflow locally?
Yes.

PQuantML integrates with MLflow for experiment tracking and model logging and local usage is fully supported.


### Start local MLFlow UI:
```python
mlflow ui --host 0.0.0.0 --port 5000
```
By default, the MLflow UI is available at `http://localhost:5000`.

### Use a local or remote Optuna database:
PQuantML also supports storing Optuna studies in either a local SQLite database or a remote database server.

```python
from pquant.core.finetuning import TuningTask
tuner = TuningTask(config)
tuner.set_storage_db("sqlite:///optuna_study.db")
```
