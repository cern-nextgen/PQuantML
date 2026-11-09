# MDMM pruning guide

MDMM (Modified Differential Method of Multipliers) trains a model with a constraint on its weights alongside the task loss. In PQuantML, each enabled pruning layer evaluates a metric and adds a penalty for violating its configured target. The same metric choices are available in the Keras and native PyTorch implementations.

Start with `mdmm_config()`, `mdmm_fpga_config()`, or `mdmm_paca_config()` before building or converting a model. The [Quick User Guide](getting_started.md) covers model construction and the training wrapper; the [Usage Reference](mdmm-pruning) lists configuration defaults.

Metric-specific settings belong in the nested `metric` block; flat fields are no longer migrated into that block. When supplying a metric block, include its `metric_type` discriminator. Other metric fields retain their documented defaults and validation bounds. Omitting the whole block selects the default unstructured metric.

## Choose the metric and target together

| Metric | What it measures | Constraint choice and practical limitation |
| --- | --- | --- |
| `UnstructuredSparsity` | A product of an L1 weight sum and the difference between squared desired and measured zero-weight fractions. | Use `Equality` with `target_value: 0`; set the desired fraction with `target_sparsity`. The metric is not itself a sparsity fraction. |
| `StructuredSparsity` | Fraction of groups of `rf` weights with L2 norm at most `epsilon`. | A sparsity lower bound uses `GreaterThanOrEqual` and `target_value`. Its hard threshold has no useful weight gradient; it does not provide a smooth training surrogate. |
| `FPGAAwareSparsity` | Fraction of near-zero DSP or BRAM weight groups, or its smooth approximation. | Use `GreaterThanOrEqual` with the desired fraction in `target_value`, and `l0_mode: smooth` for training. |
| `PACAPatternSparsity` | Mean distance from each convolution kernel to its closest dominant binary support pattern. | Configuration validation always overrides the constraint to `Equality` at `0`. It operates only on 4D weights. |

For unstructured pruning, the metric is
`(target_sparsity**2 - zero_fraction**2) * sum(abs(weights))`, divided by the number of weights when `scale_mode: mean`. Coarse mode counts weights with magnitude at most `epsilon`; smooth mode averages `exp(-100 * weight**2)`. An all-zero tensor also makes this metric zero, so a zero constraint value alone does not certify the requested sparsity.

For a metric value `g` and `target_value` `t`, PQuantML uses these nonnegative violations:

| `constraint_type` | Violation |
| --- | --- |
| `Equality` | `abs(g - t)` |
| `LessThanOrEqual` | `max(g - t, 0)` |
| `GreaterThanOrEqual` | `max(t - g, 0)` |

The added loss is `scale * (multiplier * violation + damping * violation**2 / 2)`. These are the implemented violations, including the absolute value for equality; use them when interpreting training logs. The original MDMM derivation combines multiplier updates with quadratic damping; see [Platt and Barr, section 5](https://proceedings.neurips.cc/paper_files/paper/1987/file/a1126573153ad7e9f44ba80e99316482-Paper.pdf).

## FPGA resource groups

```python
from pquant import mdmm_fpga_config

config = mdmm_fpga_config()
config.pruning_parameters.metric.target_resource = "BRAM"
```

This helper sets `rf: 4`, `target_value: 0.9`, `scale: 50.0`, and smooth counting. A corresponding pruning section is:

```yaml
pruning_parameters:
  pruning_method: mdmm
  constraint_type: GreaterThanOrEqual
  target_value: 0.9
  rf: 4
  epsilon: 1.0e-3
  l0_mode: smooth
  scale: 50.0
  damping: 1.0
  use_grad: false
  constraint_lr: 1.0e-3
  metric:
    metric_type: FPGAAwareSparsity
    precision: 16
    target_resource: BRAM
    bram_width: 36
```

The implementation flattens trailing weight dimensions into rows and pads each row to a multiple of `rf`. Each DSP group holds `rf` weights. BRAM groups combine `c` consecutive DSP groups, where `c = bram_width // precision` if divisible and `c = (2 * bram_width) // precision` otherwise. BRAM configurations must allow at least one DSP group per block. Vectors are treated as one row, so this metric also supports dense layers and biases.

Coarse counting tests a group's L2 norm against `epsilon`. Smooth counting uses `exp(-100 * norm**2)` and therefore differs from the measured fraction of pruned groups. Inspect both the surrogate and the thresholded groups when evaluating a trained model. `precision` describes the packing model; it does not set the quantizer's bit-width.

The grouping is a resource proxy. Actual DSP/BRAM savings depend on the deployed weight layout and synthesis results. [Ramhorst, Loncar, and Constantinides](https://arxiv.org/abs/2308.05170) describe FPGA resource-aware weight grouping and a pruning algorithm based on a knapsack formulation; PQuantML's MDMM constraint is a separate training formulation.

## PACA kernel patterns

```python
from pquant import mdmm_paca_config

config = mdmm_paca_config()
```

The helper uses the following pruning section. Its `beta` and `distance_metric` differ from the model defaults shown in the reference table.

```yaml
pruning_parameters:
  pruning_method: mdmm
  constraint_type: Equality
  target_value: 0.0
  epsilon: 1.0e-3
  scale: 1.0
  damping: 1.0
  use_grad: false
  constraint_lr: 1.0e-3
  metric:
    metric_type: PACAPatternSparsity
    num_patterns_to_keep: 16
    beta: 0.85
    distance_metric: cosine
```

For every input/output-channel pair, PACA defines a kernel's support as `abs(weight) > epsilon`. It ranks distinct patterns by frequency and retains enough to reach `beta` coverage, subject to `num_patterns_to_keep`. The pattern set is recomputed from the current weights on every call. Keras encodes patterns in integers and limits kernel spatial size to 62 positions; native PyTorch uses `torch.unique` without that limit. Stay within 62 positions for portability.

`hamming` counts support mismatches and has no useful weight gradient. `valued_hamming` weights each mismatch by the corresponding weight magnitude. `cosine` compares a kernel with its projection onto each candidate support. The latter two depend on weight values, while support selection remains discrete. Non-4D weights have zero PACA penalty and an all-ones projection mask.

```{note}
PACA does not prescribe a desired fraction of zeros: `target_sparsity`, `rf`, and `l0_mode` do not control this metric. When kernels start with fully dense support, the dominant pattern can also be fully dense, leaving no useful pressure to create sparse support. The configuration does not implement a separate sparsification schedule. Choose and validate that schedule for the experiment before relying on PACA to prune a dense initialization.
```

The related accelerator work is [Wang et al., *PACA: A Pattern Pruning Algorithm and Channel-Fused High PE Utilization Accelerator for CNNs* (2022)](https://doi.org/10.1109/TCAD.2022.3140730). The behavior above describes PQuantML's metric and projection; it does not imply reproduction of the paper's complete training or hardware pipeline.

## Training stages and losses

`train_model(...)` manages the phase transitions in both implementations. For Keras `model.fit`, use `PQuantCallback` from `pquant.core.keras.train` and train for `callback.total_epochs`.

1. **Pretraining:** the MDMM contribution to the loss is zero and weights pass through without an MDMM mask.
2. **Main training:** the metric penalty is active. The layer tracks a magnitude mask, but its forward pass returns the unmasked weights.
3. **Fine-tuning:** the constraint is turned off and the forward pass applies a mask. PACA uses its nearest-pattern projection; other metrics use the magnitude cutoff. The current implementation computes this mask from the current weights on each call, rather than freezing one projection at the phase boundary.

Keras `model.fit` includes the MDMM penalty through `add_loss()`. In native PyTorch custom training functions, include the collected losses after the forward pass:

```python
from pquant import get_model_losses

prediction = model(inputs)
loss = loss_fn(prediction, targets)
loss = get_model_losses(model, loss)
loss.backward()
```

Keras custom loops can also use `get_model_losses`; do not add the same MDMM contribution again through `model.losses`. Custom loops still need the phase transitions supplied by `train_model` or the corresponding layer helpers. Calling a newly created MDMM layer alone leaves it in pretraining mode.

## Adjusting parameters

- `scale` multiplies the complete constraint loss; `damping` weights its quadratic term. Compare the constraint contribution with the task loss when adjusting them. The [reference PyTorch MDMM implementation](https://github.com/crowsonkb/mdmm) discusses scaling and damping, but its constraint API and update rules differ from PQuantML's.
- With `use_grad: false`, the multiplier uses explicit updates based on the previous violation and `constraint_lr * scale`. With `use_grad: true`, gradient reversal makes the model optimizer perform multiplier ascent; `constraint_lr` is not used. Build the layers before creating a PyTorch optimizer so it includes the multiplier parameters.
- `epsilon` is a weight/support or coarse group cutoff, not a tolerance on constraint violation. Changing it changes the masks and PACA pattern inventory. It does not tune the smooth zero-count formula.
- Configure the complete nested metric before constructing the model. PACA's equality/zero override is applied when the pruning configuration is validated, taking precedence over supplied parent values.

The current hyperparameter search applies sampled values to top-level configuration sections only. It does not descend into `pruning_parameters.metric`; nested fields such as `beta`, `precision`, and `num_patterns_to_keep` are not applied through that search path. Set these values explicitly in the configuration until nested search support is available.
