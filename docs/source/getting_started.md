# Quick User Guide

```{note}
This section provides an overview of how to use the PQuantML library: defining models with pruning and quantization, running fine-tuning, and optionally converting the final model to hls4ml.
```

## Model definition & training


To compress a model with PQuantML, all layers must be replaced with their PQuantML equivalents. For example, replace `Dense` by `PQDense`, `ReLU` by `PQActivation`, etc.


Model compression behaviour such as pruning strength, quantization bit-widths, training parameters, etc. is controlled through the configuration object, which is a Pydantic model.

### Load a default configuration
``` python
from pquant import dst_config

# Upload a default DST config
config = dst_config()

config.training_parameters.epochs = 1000
config.quantization_parameters.default_data_integer_bit = 3.
config.quantization_parameters.default_data_fractional_bits = 2.
config.quantization_parameters.default_weight_fractional_bits = 3.
config.quantization_parameters.use_relu_multiplier = False
```

### Building a model
PQuantML supports two ways of defining compressed models. Below we illustrate both approaches using a simple jet-tagging architecture.

### Direct layer usage

```python
from pquant.layers import PQDense
from pquant.activations import PQActivation

def build_model(config):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = PQDense(config, 16, 64,
                                  in_quant_bits = (1, 3, 3))
            self.relu = PQActivation(config, "relu")
            self.dense2 = PQDense(config, 64, 32)
            self.dense3 = PQDense(config, 32, 32)
            self.dense4 = PQDense(config, 32, 5,
                                  quantize_output=True,
                                  out_quant_bits=(1, 3, 3))

        def forward(self, x):
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            x = self.relu(self.dense3(x))
            x = self.dense4(x)
            return x

    return Model(config)
```


### Layer-replacement usage
```python

def build_model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = nn.Linear(16, 64)
            self.relu = nn.ReLU()
            self.dense2 = nn.Linear(64, 32)
            self.dense3 = nn.Linear(32, 32)
            self.dense4 = nn.Linear(32, 5)

        def forward(self, x):
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            x = self.relu(self.dense3(x))
            x = self.dense4(x)
            return x


    return Model()

# Convert to PQuantML-compressed model
model = add_compression_layers(model, config)
```

### Fine-Tuning with PQuantML
PQuantML provides an automated fine-tuning and hyperparameter-optimization workflow through the `TuningTask API`. This allows you to search for optimal pruning, quantization, and training parameters using your own training, validation, and objective functions.

```python
from pquant.core.finetuning import TuningTask, TuningConfig

# Convert defined yaml file into the object
config = TuningConfig.load_from_file(CONFIG_PATH)

# Create finetuning task class
tuner = TuningTask(config)

# (Optional) Enable mlflow logging
tuner.set_enable_mlflow()
tuner.set_tracking_uri("https://ngt.cern.ch/models")
tuner.set_user("your_email@cern.ch", "your_access_token")

# Register training, validation and objective functions
tuner.set_training_function(train_resnet)
tuner.set_validation_function(validate_resnet)
tuner.set_objective_function(name="accuracy", fn=calculate_accuracy, direction="maximize")

# Set optimizer, scheduler and hyperparameters
tuner.set_hyperparameters()
tuner.set_optimizer_function(get_optimizer)
tuner.set_scheduler_function(get_scheduler)
```

To run optimization:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

best_params = tuner.run_optimization(model,
                        trainloader=...,
                        testloader=...,
                        loss_func=...)
```
```{note}
`tuner.run_optimization()` automatically runs multiple compression–fine-tuning cycles, evaluates each trial using your objective function, and returns the best hyperparameters.
```

All other training code remains unchanged.

### Train a model

```python
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=1e-2, weight_decay=1e-4, params=model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 800], gamma=0.1
```
Training is handled through the `train_model(...)` wrapper:

```python
from pquant import train_model

trained_model = train_model(model = model,
                                config = config,
                                train_func = ...,
                                valid_func = ...,
                                trainloader = ...,
                                device="cuda",
                                testloader = ...,
                                loss_func = loss_func,
                                optimizer = optimizer,
                                scheduler=scheduler
                                )
```

### Using different quantization settings per layer
```{note}
For FITCompress, HGQ, or architectures, where activations require different quantization bit-widths, each activation layer must be instantiated separately.
```
```python
def build_model(config):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = PQDense(config, 16, 64,
                                  in_quant_bits = (1, 3, 3))
            self.relu1 = PQActivation(config, "relu")
            self.relu2 = PQActivation(config, "relu")
            self.relu3 = PQActivation(config, "relu")
            self.dense2 = PQDense(config, 64, 32)
            self.dense3 = PQDense(config, 32, 32)
            self.dense4 = PQDense(config, 32, 5,
                                  quantize_output=True,
                                  out_quant_bits=(1, 3, 3))

        def forward(self, x):
            x = self.relu1(self.dense1(x))
            x = self.relu2(self.dense2(x))
            x = self.relu3(self.dense3(x))
            x = self.dense4(x)
            return x

    return Model(config)
```


## Conversion to hls4ml
After training, the PQuantML model can be exported to hls4ml for HLS synthesis.

```python
from hls4ml.converters import convert_from_pytorch_model
from hls4ml.utils import config_from_pytorch_model

hls_config = config_from_pytorch_model(
        model,
        input_shape=input_shape,
        )

hls_model = convert_from_pytorch_model(
        model,
        io_type=""io_parallel"",
        output_dir=...,
        backend="vitis",
        hls_config=hls_config,
        )
hls_model.compile()
```

For a complete example, please refer to this [notebook](https://github.com/nroope/PQuant/blob/dev/examples/example_jet_tagging.ipynb).
