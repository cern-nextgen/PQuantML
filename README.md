![alt text](docs/source/_static/pquant_white_font.png)

## Prune and Quantize ML models
PQuant is a library for training compressed machine learning models, developed at CERN as part of the [Next Generation Triggers](https://nextgentriggers.web.cern.ch/t13/) project.

Installation via pip: ```pip install pquant-ml```.

With TensorFlow ```pip install pquant-ml[tensorflow]```.

With PyTorch ```pip install pquant-ml[torch]```.

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
A description of the pruning methods and their hyperparameters can be found [here](docs/pruning_methods.md).

### Quantization parameters
A description of the quantization parameters can be found [here](docs/quantization_parameters.md).


For detailed documentation check this page: [PQuantML documentation](https://pquantml.readthedocs.io/en/latest/)


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
