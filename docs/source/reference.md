# Usage Reference

## Config file

The most important part of the library is a user-defined `config.yaml` file. It has five separate sections: **training, pruning, quantization, hpo, and fitcompress**. By default, the parameters in the config are the following:

### Training parameters
The following table outlines the description of default parameters used to configure the training process:

| **Field**               | **Type**                                   | **Default**   | **Description**                                            |
|------------------------|---------------------------------------------|---------------|------------------------------------------------------------|
| `epochs`               | int                                         | `200`         | Number of epochs during the main training stage.           |
| `fine_tuning_epochs`   | int                                         | `0`           | Number of epochs during the fine-tuning stage.             |
| `pretraining_epochs`   | int                                         | `50`          | Number of epochs during the pretraining stage.             |
| `rewind`               | str                                         | `"never"`     | When to rewind the weights: never, every-round, or posttraining-stage.                               |
| `rounds`               | int                                         | `1`           | Number of pruning/quantization rounds during training.                         |
| `save_weights_epoch`   | int                                         | `-1`          | Epoch at which weights are saved for rewinding during the first round.           |
| `pruning_first`   | bool                                         | `False`          | Whether to prune before quantization. If false, pruning occurs after quantization.|

```{note}
If you require additional parameters for the training or optimization loops, please define them directly in the config.yaml file.
```

### Quantization parameters

| **Field**                         | **Type** | **Default** | **Description**                                                        |
|----------------------------------|----------|-------------|------------------------------------------------------------------------|
| `default_data_keep_negatives`    | bool     | `0`         | Default **k** value for data quantization (0 = clamp negatives, 1 = keep). |
| `default_data_integer_bits`      | int      | `0`         | Default integer bitwidth **i** for data quantization.                  |
| `default_data_fractional_bits`   | int      | `0`         | Default fractional bitwidth **f** for data quantization.               |
| `default_weight_keep_negatives`  | bool     | `0`         | Default **k** value for weight quantization (0 or 1).                  |
| `default_weight_integer_bits`    | int      | `0`         | Default integer bitwidth **i** for weight quantization.                |
| `default_weight_fractional_bits` | int      | `0`         | Default fractional bitwidth **f** for weight quantization.             |
| `quantize_input`                 | bool     | `true`      | Whether inputs to layers are quantized by default.                     |
| `quantize_output`                | bool     | `true`      | Whether outputs of layers are quantized by default.                    |
| `enable_quantization`            | bool     | `true`      | Global switch to enable or disable quantization.                       |
| `granularity` | str | `"per_tensor"` | Whether bitwidths are shared across the whole tensor, per-channel, or per-weight. |
| `hgq_gamma`                      | float    | `0.0`       | HGQ regularization coefficient for bitwidth stability.                 |
| `hgq_beta`                       | float    | `0.0`       | HGQ loss coefficient scaling EBOPs.                                    |
| `layer_specific`                 | dict     | `{}`        | Dictionary for per-layer quantization overrides.                       |
| `use_high_granularity_quantization`                        | bool     | `false`     | Enable or disable High Granularity Quantization (HGQ).                 |
| `use_real_tanh`                  | bool     | `false`     | Use a real `tanh` instead of hard/approximate `tanh`.                  |
| `overflow_mode_data`                       | str      | `"SAT"`     | Overflow handling mode for input and output quantizers(`SAT`, `SAT_SYM`, `WRAP`, `WRAP_SM`).          |
| `overflow_mode_parameters`                       | str      | `"SAT"`     | Overflow handling mode for weight and biases quantizers(`SAT`, `SAT_SYM`, `WRAP`, `WRAP_SM`).          |
| `round_mode`                     | str      | `"RND"`     | Rounding mode (`TRN`, `RND`, `RND_CONV`, `RND_ZERO`, etc.).            |
| `use_relu_multiplier`            | bool     | `true`      | Enable a learned bit-shift multiplier inside ReLU layers.              |


### Hyperparameters optimization parameters

| **Field**                | **Type**                | **Default**       | **Description**                               |
|-------------------------|--------------------------|--------------------|-----------------------------------------------|
| `experiment_name`       | str                      | `"experiment_1"`  | Name of the study.                            |
| `model_name`                | str                                         | `"resnet18"`  | Model architecture name.                            |
| `sampler`               | str                      | `GridSampler`     | Sampler selection for the search space.       |
| `num_trials`            | int                      | `0`               | Number of trials.                             |
| `hyperparameter_search` | HyperparameterSearch     | `{}`              | Ranges for non-grid samplers.                 |

#### Samplers

| **Field** | **Type**          | **Default**       | **Description**                                                 |
|-----------|-------------------|--------------------|-----------------------------------------------------------------|
| `type`    | str               | `"TPESampler"`     | Sampler class name (e.g., `TPESampler`, `GridSampler`).        |
| `params`  | Dict[str, Any]    | `{}`              | Sampler-specific kwargs (e.g., `seed`, `search_space`).        |

More about samplers can be found in {[optuna documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)}


#### HyperparameterSearch

| **Field**        | **Type**                                 | **Default** | **Description**                        |
|------------------|------------------------------------------|-------------|----------------------------------------|
| `numerical`      | Dict[str, List[Union[int, float]]]       | `{}`        | Numeric ranges `[low, high, step]`.   |
| `categorical`    | Optional[Dict[str, List[str]]]           | `{}`        | Categorical choices.                   |


### Pruning methods
PQuantML supports seven different pruning methods.
#### Method Overview

| **Method**            | **Model**                  |
|----------------------|----------------------------|
| `cs`                 | `CSPruningModel`           |
| `dst`                | `DSTPruningModel`          |
| `pdp`                | `PDPPruningModel`          |
| `wanda`              | `WandaPruningModel`        |
| `autosparse`         | `AutoSparsePruningModel`   |
| `activation_pruning` | `ActivationPruningModel`   |
| `mdmm`               | `MDMMPruningModel`         |


There are the parameters shared by all methods:

| **Field**                     | **Type**     | **Default** | **Description**                               |
|------------------------------|--------------|-------------|-----------------------------------------------|
| `disable_pruning_for_layers` | List[str]    | `[]`        | Layer names to exclude from pruning.          |
| `enable_pruning`             | bool         | `true`      | Master pruning on/off switch.                 |
| `threshold_decay`            | float        | `0.0`       | Optional pruning threshold decay term.        |


```{note}
Layer names in `disable_pruning_for_layers` field  must match your framework’s naming (e.g., Keras `layer.name`).
```

There are more details about every pruning method:
####  CS Pruning

| **Field**         | **Type** | **Default** | **Description**                                  |
|-------------------|----------|-------------|--------------------------------------------------|
| `pruning_method`  | str      | `cs`        | Selects this pruning schema.                     |
| `final_temp`      | int      | `200`       | Target temperature at the end of the schedule.   |
| `threshold_init`  | int      | `0`         | Initial sparsification threshold.                |


#### DST Pruning

| **Field**          | **Type** | **Default**     | **Description**                            |
|--------------------|----------|------------------|--------------------------------------------|
| `pruning_method`   | str      | `dst`            | Selects this pruning schema.               |
| `alpha`            | float    | `5.0e-06`        | Mask dynamics update coefficient.          |
| `max_pruning_pct`  | float    | `0.99`           | Upper bound on total pruning ratio.        |
| `threshold_init`   | float    | `0.0`            | Initial threshold value.                   |
| `threshold_type`   | str      | `"channelwise"`  | Thresholding granularity.                  |

#### PDP Pruning

| **Field**            | **Type** | **Default** | **Description**                                   |
|----------------------|----------|-------------|---------------------------------------------------|
| `pruning_method`     | str      | `pdp`           | Selects this pruning schema.                      |
| `epsilon`            | float    | `0.015`     | Smoothing/regularization factor for gating.       |
| `sparsity`           | float    | `0.8`       | Target sparsity level (0–1).                      |
| `temperature`        | float    | `1.0e-05`   | Annealing temperature.                            |
| `structured_pruning` | bool     | `false`     | Enable structured pruning.                        |

#### Wanda Pruning

| **Field**                   | **Type**         | **Default** | **Description**                                  |
|-----------------------------|------------------|-------------|--------------------------------------------------|
| `pruning_method`            | str              | `wanda`     | Selects this pruning schema.                     |
| `M`                         | Optional[int]    | `null`      | Optional grouping constant.                      |
| `N`                         | Optional[int]    | `null`      | Optional grouping constant.                      |
| `sparsity`                  | float            | `0.9`       | Target sparsity level (0–1).                     |
| `t_delta`                   | int              | `100`       | Window size / steps for stats collection.        |
| `t_start_collecting_batch`  | int              | `100`       | Warm-up steps before collecting statistics.      |
| `calculate_pruning_budget`  | bool             | `true`      | Auto-compute pruning budget from data.           |

#### Autosparse Pruning

| **Field**             | **Type** | **Default**  | **Description**                                   |
|-----------------------|----------|--------------|---------------------------------------------------|
| `pruning_method`      | str      | `autosparse` | Selects this pruning schema.                      |
| `alpha`               | float    | `0.5`        | Weight/penalty coefficient.                       |
| `alpha_reset_epoch`   | int      | `90`         | Epoch at which `alpha` is reset/tuned.            |
| `autotune_epochs`     | int      | `10`         | Number of epochs in the tuning window.            |
| `backward_sparsity`   | bool     | `false`      | Apply sparsity in backward pass (if supported).   |
| `threshold_init`      | float    | `-5.0`       | Initial threshold (often in logit space).         |
| `threshold_type`      | str      | `"channelwise"` | Thresholding granularity.                       |

#### Activation Pruning
| **Field**                  | **Type** | **Default** | **Description**                            |
|----------------------------|----------|-------------|--------------------------------------------|
| `pruning_method`           | str      | `activation_pruning`           | Selects this pruning schema.               |
| `threshold`                | float    | `0.3`       | Activation magnitude cutoff.                |
| `t_delta`                  | int      | `50`        | Steps used to aggregate statistics.         |
| `t_start_collecting_batch` | int      | `50`        | Steps to skip before collecting statistics. |

#### MDMM Pruning
| **Field**          | **Type**              | **Default**                | **Description**                                                                 |
|--------------------|-----------------------|----------------------------|---------------------------------------------------------------------------------|
| `pruning_method`   | str                   | `"mdmm"`                   | Selects this pruning schema.                                                    |
| `constraint_type`  | ConstraintType        | `"Equality"`               | Constraint form: `Equality`, `LessThanOrEqual`, or `GreaterThanOrEqual`.        |
| `target_value`     | float                 | `0.0`                      | Target value for the chosen metric (used for the general constraint).           |
| `metric_type`      | MetricType            | `"UnstructuredSparsity"`   | Metric being constrained: `UnstructuredSparsity` or `StructuredSparsity`.       |
| `target_sparsity`  | float                 | `0.9`                      | Target sparsity when constraining sparsity.                                     |
| `rf`               | int                   | `1`                        | Regularization / frequency parameter.                                           |
| `epsilon`          | float                 | `1.0e-03`                  | Feasibility tolerance.                                                          |
| `scale`            | float                 | `10.0`                     | Penalty scaling for constraint violation.                                       |
| `damping`          | float                 | `1.0`                      | Damping term for numerical stability.                                           |
| `use_grad`         | bool                  | `false`                    | Use gradient information during updates.                                        |
| `l0_mode`          | `"coarse"` \| `"smooth"` | `"coarse"`              | L0 approximation mode.                                                          |
| `scale_mode`       | `"mean"` \| `"sum"`   | `"mean"`                   | Aggregation mode for penalties.                                                 |
| `constraint_lr`    | float                 | `1.0e-03`                  | Learning rate for the Lagrange multiplier (dual variable).                      |


Optionally, there is also FITCompress method implemented for PyTorch-only:
### FitCompress method
| **Field**                 | **Type** | **Default** | **Description**                                                                 |
|---------------------------|----------|-------------|---------------------------------------------------------------------------------|
| `enable_fitcompress`      | bool     | `false`     | Master switch that enables or disables FITCompress.                             |
| `optimize_quantization`   | bool     | `true`      | Whether FITCompress searches over quantization bit-width candidates.            |
| `quantization_schedule`   | List[float] | `[7., 4., 3., 2.]` | Candidate bit-widths evaluated during quantization search.                    |
| `pruning_schedule`        | dict     | `{start: 0, end: -3, steps: 40}` | Logarithmic pruning curve (base 10) with defined start, end, and step count. |
| `compression_goal`        | float    | `0.10`      | Target compression ratio for the search procedure.                              |
| `optimize_pruning`        | bool     | `false`     | Whether FITCompress searches over pruning ratios.                               |
| `greedy_astar`            | bool     | `true`      | Disable fallback in A* search: once a candidate is selected, all others discarded. |
| `approximate`             | bool     | `true`      | Use Fisher Trace approximations to speed up FIT score estimation.               |
| `f_lambda`                | float    | `1`         | Multiplicative factor λ in the distance function (g + λf).                      |


### Quantization layers in PQuantML

- `PQConv*D`:  Convolutional layers.
- `PQAvgPool*D`: Average pooling layers.
- `PQBatchNorm*D`: BatchNorm layers.
- `PQDense`: Linear layer.
- `PQActivation`: Activation layers (ReLU, Tanh, Leaky Relu, Gelu, Hard Tanh, or a user-provided activation function(Torch only)).
- `MultiHeadAttention`: Multi-head attention layer.
- `LayerNorm`: Layer normalization layer. (Currently Torch only).

```{note}
PQuantML supports two quantization modes, each with several granularity options:

**Fixed-point quantization** (for weights):
- per-weight
- per-channel
- per-tensor

**HGQ (High Granularity Quantization)**:
- per-weight (learned bit-widths per weight; pure HGQ)
- per-tensor (learned bit-widths per tensor)
```