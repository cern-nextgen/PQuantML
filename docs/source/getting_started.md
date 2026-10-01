# Quick User Guide

```{note}
This section provides an overview of how to use the PQuantML library: defining models with pruning and quantization, running hyperparameters optimization, and optionally converting the final model to hls4ml.
```


## Model definition & training
To enable pruning and quantization, a model must use PQuantML layers. This can be done in one of two ways:

- Direct layer definition, by building the model with PQuantML layers such as PQDense and PQActivation.
- Automatic layer replacement, by converting an existing PyTorch model using add_compression_layers(...).

Model compression behaviour such as pruning strength, quantization bit-widths, training parameters, etc. is controlled through the configuration object, which is a Pydantic model to provide an automatic type checking.

### Load the default DST configuration
```python
from pquant import dst_config

# Upload a default DST config
config = dst_config()

config.training_parameters.epochs = 1000
config.quantization_parameters.default_data_integer_bit = 3.
config.quantization_parameters.default_data_fractional_bits = 2.
config.quantization_parameters.default_weight_fractional_bits = 3.
config.quantization_parameters.use_relu_multiplier = False
```

### Use-case presets
`dst_config()`, `pdp_config()` and the other `*_config()` functions load the full default configuration of one pruning method. The presets below (`from pquant import preset_configs as presets`) instead start from what you want to achieve and take a handful of keyword arguments. Each returns an ordinary `PQConfig` that you can keep editing.
| **Preset** | **Pruning** | **Default bits** | **Training** |
|---|---|---|---|
| `quantized()` | none (`pruning_method: None`) | Fixed bitwidths at `granularity` (`per_tensor` by default): data (0, 0, 8) with integer bits set from the data (`dynamic_data=True`), weights and biases (1, 0, 7) | `epochs=100`, no pretraining or fine-tuning stage |
| `hgq(granularity="per_weight")` | none (`pruning_method: None`) | HGQ learns the bitwidths: one per weight and activation element with `per_weight` (the default), one per tensor with `per_tensor`; data lanes start at (0, 3, 5), weights and biases at (1, 0, 7) | `epochs=100`, no pretraining or fine-tuning stage |
| `quantized_unstructured_pruning(alpha=1e-7)` | DST with a weight-wise threshold | As in `quantized()` | `epochs=100`, `fine_tuning_epochs=0` (a stage with the pruning mask fixed) |
| `quantized_nm_pruning(n=2, m=4)` | Wanda N:M: `n` of every `m` consecutive weights pruned (the N:M literature usually counts kept weights; same thing for 2:4). Mask computed once at `prune_at_epoch` (10) of the main stage from `calibration_batches` (100) batches of input statistics, then fixed | Same as above | `pretraining_epochs=10`, `epochs=100` |
| `quantized_structured_pruning(target_sparsity)` | Structured PDP towards `target_sparsity` (fraction of weights removed, in [0, 1)); `epsilon` is derived so the sparsity ramp reaches the target after about 90% of `epochs` | Same as above | `pretraining_epochs=10`, `epochs=100`, `fine_tuning_epochs=10` |
| `hgq_structured_pruning(target_sparsity, granularity="per_tensor")` | Structured PDP as above | HGQ-learned bitwidths as in `hgq`, one per tensor by default | `pretraining_epochs=10`, `epochs=100`, `fine_tuning_epochs=10` |

```python
from pquant import preset_configs as presets

config = presets.quantized_structured_pruning(target_sparsity=0.5, epochs=30, weight_bits=(1, 0, 5))
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
This approach is recommended when developing a new architecture from scratch.

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

If you already have a model, it can be converted automatically by replacing supported layers with their PQuantML equivalents.

### Hyperparameters optimization with PQuantML
PQuantML provides an automated hyperparameter-optimization workflow through the TuningTask API. This allows you to search for optimal pruning, quantization, and training parameters using your own training, validation, and objective functions.

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

Run optimization:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

best_params = tuner.run_optimization(model,
                        trainloader=...,
                        testloader=...,
                        loss_func=...)
```

```{note}
`tuner.run_optimization()` automatically runs multiple compression cycles, evaluates each trial using your objective function, and returns the best hyperparameter configuration.
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

#### Training with `torch.compile`

`train_model` accepts a model wrapped with `torch.compile`:

- Run one forward pass on real-shaped data before compiling. PQ layers build their quantizers lazily on the first call.
- Pass `dynamic=True`, so a smaller last batch does not trigger a recompile.
- Raise `torch._dynamo.config.cache_size_limit`. Each stage (pretrain, train, finetune) and mode (train, eval) compiles its own graph, more than the default limit of 8 allows. Past the limit, dynamo silently falls back to eager execution.

```python
model(next(iter(trainloader))[0].to(device))
torch._dynamo.config.cache_size_limit = 64
compiled = torch.compile(model, dynamic=True)
compiled = train_model(model=compiled, config=config, ...)
trained_model = compiled._orig_mod
```
### Using different quantization settings per layer
```{note}
If different activation layers require different quantization settings (for example when using FITCompress or HGQ), instantiate each `PQActivation` layer separately instead of reusing a single activation module.
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
