![alt text](docs/source/_static/pquant_white_font.png)

## Prune and Quantize ML models
PQuant is a library for training compressed machine learning models, developed at CERN as part of the [Next Generation Triggers](https://nextgentriggers.web.cern.ch/t13/) project.

### Installation

PQuant requires **Python 3.10 or newer** (tested on 3.10–3.13).

PQuant is built on [Keras 3](https://keras.io/), which needs a backend to run. Install PQuant together
with the backend you intend to use:

```bash
pip install pquant-ml[tensorflow]   # TensorFlow backend
pip install pquant-ml[torch]        # PyTorch backend
```

Installing `pquant-ml` on its own will not give you a usable install — a backend is required.

Optional extras:

```bash
pip install pquant-ml[onnx]         # ONNX export support
pip install -e ".[dev]"             # contributors: linting, typing, tests, docs
```

### Selecting the backend

Keras chooses its backend from the `KERAS_BACKEND` environment variable, and it must be set **before**
importing PQuant. It defaults to `tensorflow`, so PyTorch users must set it explicitly:

```bash
export KERAS_BACKEND=torch          # or: tensorflow
```

Or from Python, before the first import:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import pquant
```

### Quick start

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

from pquant import add_compression_layers, pdp_config, train_model

# 1. Load a default configuration for a pruning method (pdp, cs, dst, wanda, ap, mdmm, autosparse).
config = pdp_config()

# 2. Replace the model's layers with their compressed/quantized variants.
model = add_compression_layers(model, config, input_shape)

# 3. Train. You supply the per-epoch train/validate functions; PQuant drives the
#    pre-training, pruning and fine-tuning phases described by the config.
model = train_model(model, config, train_func, valid_func, input_shape)
```

See the [example notebooks](https://github.com/cern-nextgen/PQuantML/tree/main/examples) for complete,
runnable versions.

PQuant replaces the layers and activations it finds with a Compressed (in the case of layers) or Quantized (in the case of activations) variant. These automatically handle the quantization of the weights, biases and activations, and the pruning of the weights.
Both PyTorch and TensorFlow models are supported.

### Layers that can be compressed

* **PQConv*D**: Convolutional layers
* **PQAvgPool*D**: Average pooling layers
* **PQBatchNorm*D**: BatchNorm layers
* **PQDense**: Linear layer
* **PQActivation**: Activation layers (ReLU, Tanh)

The various pruning methods have different training steps, such as a pre-training step and fine-tuning step. PQuant provides a training function, where the user provides the functions to train and validate an epoch, and PQuant handles the training while triggering the different training steps.


![alt text](docs/source/_static/overview_pquant.png)



### Example
Example notebook can be found [here](https://github.com/cern-nextgen/PQuantML/tree/main/examples). It handles the
  1. Creation of a torch model and data loaders.
  2. Creation of the training and validation functions.
  3. Loading a default pruning configuration of a pruning method.
  4. Using the configuration, the model, and the training and validation functions, call the training function of PQuant to train and compress the model.
  5. Creating a custom quantization and pruning configuration for a given model (disable pruning for some layers, different quantization bitwidths for different layers).
  6. Direct layers usage and layers replacement approaches.
  7. Usage of fine-tuning platform.

### Pruning methods
A description of the pruning methods and their hyperparameters can be found [here](docs/source/reference.md#pruning-methods).

### Quantization parameters
A description of the quantization parameters can be found [here](docs/source/reference.md#quantization-parameters).


For detailed documentation check this page: [PQuantML documentation](https://pquantml.readthedocs.io/en/latest/)

### Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, and
[CHANGELOG.md](CHANGELOG.md) for the release history. To set up a development environment:

```bash
git clone https://github.com/cern-nextgen/PQuantML.git
cd PQuantML
pip install -e ".[dev,torch]"   # or ".[dev,tensorflow]"
pre-commit install
```

### License

PQuant is released under the [Apache License 2.0](LICENSE).

### Authors
 - Roope Niemi (CERN)
 - Anastasiia Petrovych (CERN)
 - Arghya Das (Purdue University)
 - Enrico Lupi (CERN)
 - Chang Sun (Caltech)
 - Dimitrios Danopoulos (CERN)
 - Marlon Joshua Helbing
 - Mia Liu (Purdue University)
 - Michael Kagan (SLAC National Accelerator Laboratory)
 - Vladimir Loncar (CERN)
 - Maurizio Pierini (CERN)
