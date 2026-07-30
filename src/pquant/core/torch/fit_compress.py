import logging
import math
import random
import string
import typing

import numpy as np
import torch
import torch.nn as nn
from quantizers import get_fixed_quantizer

if typing.TYPE_CHECKING:
    from pquant.core.torch.activations import PQActivation  # noqa: F401
    from pquant.core.torch.layers import (  # noqa: F401
        PQAvgPoolBase,
        PQConv1d,
        PQConv2d,
        PQDense,
        PQWeightBiasBase,
    )


quantizer = get_fixed_quantizer(overflow_mode="SAT", round_mode="RND")


def call_fitcompress(config, trained_uncompressed_model, train_loader, loss_func, input_shape):
    """
    Calls the path-finding algorithm of FITcompress to find an optimal configuration for quantization
    (layer-wise) and pruning (global sparsity value) of weights for the uncompressed network.


    Args:
            config : configuration settings
            trained_uncompressed_model : pre-trained, uncompressed model
            train_loader : training data loader
            loss_func : loss function

    Returns:
            config : configuration settings , but with updated pruning sparsity and
                            layer-wise quantization bits for weights and activations.

    """

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check that we have a pruning method active which has a global pruning sparsity target

    def enable_quantization(model):
        """

        Helper function to enable quantization for weights, activations and pooling.

        Args :
                model - current model

        Returns :
                model - current model with quantization enabled

        """
        from pquant.activations import PQActivation  # noqa: F811
        from pquant.layers import PQAvgPoolBase, PQWeightBiasBase  # noqa: F811

        for m in model.modules():
            if isinstance(m, PQWeightBiasBase):
                m.enable_quantization = True
                m.enable_pruning = True
            if m.__class__ in [PQActivation, PQAvgPoolBase]:
                m.enable_quantization = True
        return model

    def add_quantization_settings_to_config(model, quant_info_weights, config):
        """


        Helper function to add the found optimal configuration for quantization - regarding weights, activations and
        pooling - to the config file.


        Args :
                model - current model
                quant_info_weights - the found optimal quantization settings for weights, layerwise
                config - current configuration file
                activ_int_bits - the found optimal integer bits for activations
                activ_frac_bits - the found optimal fractional bits for activations
                pool_int_bits - the found optimal integer bits for pooling layer(s)
                pool_frac_bits - the found optimal fractional bits for pooling layer(s)


        Notes :
                1. This logic needs to be changed such that it can work with any model, i.e. by tracing the model beforehand


        """

        from pquant.core.torch.activations import PQActivation  # noqa: F401, F811
        from pquant.core.torch.layers import (  # noqa: F401, F811
            PQAvgPoolBase,
            PQBatchNorm2d,
            PQConv2d,
            PQDense,
        )

        # Since in config currently a list, but dictionary makes it easier
        config.quantization_parameters.layer_specific = {}
        for name, layer in model.named_modules():
            # For weights
            if isinstance(layer, (PQDense, PQConv2d)):
                config.quantization_parameters.layer_specific[name] = {
                    "weight": {"integer_bits": quant_info_weights[name][0], "fractional_bits": quant_info_weights[name][1]},
                }
                layer.weight_quantizer.set_quantization_bits(quant_info_weights[name][0], quant_info_weights[name][1])

    def print_info_bits(model):
        """
        logging.info integer bits and fractional bits for all weight layers and activations and pooling layers.

        Args:
        model - current model

        """
        from pquant.core.torch.activations import PQActivation  # noqa: F811
        from pquant.core.torch.layers import (  # noqa: F811
            PQAvgPoolBase,
            PQConv1d,
            PQConv2d,
            PQDense,
        )

        for n, m in model.named_modules():
            if isinstance(m, (PQConv2d, PQConv1d, PQDense)):
                logging.info(f"Layer {n}: {m.get_weight_quantization_bits()} bits")
            elif isinstance(m, (PQActivation, PQAvgPoolBase)):
                logging.info(f"Layer {n}: {m.get_input_quantization_bits()} bits")

    # Save the this model's state dict (i.e. uncompressed version)
    trained_uncompressed_model_state_dict = trained_uncompressed_model.state_dict()

    logging.info("Starting FITcompress ...")

    # Instantiate FITcompress
    fit_compress_computer = FITcompress(
        model=trained_uncompressed_model,
        device=device,
        dataloader=train_loader,
        criterion=loss_func,
        config=config,
        layerwise_pruning=False,
        input_shape=input_shape,
    )

    # Start A* (path-finding through compression space)
    (
        optimal_node,
        quant_prune_config,
        trained_uncompressed_model,
        optimal_node_pruning_mask,
    ) = fit_compress_computer.astar(config)

    logging.info("Finished FITcompress")

    # Reset the model's state dict to the uncompressed version (for the next training phases), since FITcompress
    # only finds optimal pruning and quantization settings, but shouldn't change the model's weights/quantization settings
    trained_uncompressed_model.load_state_dict(trained_uncompressed_model_state_dict)

    # Enable quantization for the model
    trained_uncompressed_model = enable_quantization(trained_uncompressed_model)

    if config.fitcompress_parameters.optimize_quantization:
        # Set layer specific quantization in config file
        add_quantization_settings_to_config(
            trained_uncompressed_model,
            quant_prune_config["quant_config"],
            config,
        )
        # Now add the layer specific configuration to the model
        # add_layer_specific_quantization_to_model(trained_uncompressed_model, config)

    logging.info("Layerwise quantization bits after FITcompress : ", config.quantization_parameters.layer_specific)

    print_info_bits(trained_uncompressed_model)

    return config, optimal_node_pruning_mask


class node:

    def __init__(
        self,
        matrices_params_layerwise,
        FeM,
        quant_config,
        pruning_metrics,
        gscore,
        fscore,
        full_dist,
        state,
        curr_compression_rate,
        unquantized_weights,
        int_bits,
        frac_bits,
    ):
        """
        Setup a node (i.e. a current model) in the compression space ; we can then go from node-to-node in
        the compression space.

        Args:
                matrices_params_layerwise - current parameters of the model, layerwise
                FeM - FeM, layerwise
                quant_config - the quantization config to use, layerwise
                pruning_metrics - the pruning metrics used based on the pruning method (sparsity, only for PDP and Wanda)
                gscore - the current score of g (as in paper), path cost/distance between initial model and current model
                fscore - the current score of f (as in paper), heuristic cost/distance between current model and goal model
                full_dist - the full distance : gscore + lambda * fscore (as in paper)
                state - the current "state" of the model (this is important for the schedulers to know what value to use)
                compression - the current compression rate (alpha_j in the papers pseudo code) of the model
                unquantized_weights - refers to the original weights of the model that are only affected by pruning.
                These are needed such that we can always quantize based on 32 bit values and not loose precision.
                int_bits - the current integer bits for quantization of the weights, layerwise
                frac_bits - the current fractional bits for quantization of the weights, layerwise
        """

        self.parameters = matrices_params_layerwise
        self.FeM = FeM
        self.quant_config = quant_config
        self.pruning_metrics = pruning_metrics
        self.gscore = gscore
        self.fscore = fscore
        self.full_dist = full_dist
        self.state = state
        self.curr_compression_rate = curr_compression_rate
        self.unquantized_weights = unquantized_weights
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

    def extract_config_from_node(self, layer_names):
        """
        Extract the quantization and pruning configuration from the node.
        This is used to get the configuration of the model after the path finding process.

        Returns :
                config - A dictionary containing the quantization and pruning configuration of the node.
        """

        assert len(self.quant_config) == len(layer_names), "Quantization config length does not match number of layers"

        # Create a dictionary to store the quantization config w.r.t layer names
        quant_config = {
            layer_name: [i_bits, f_bits] for layer_name, i_bits, f_bits in zip(layer_names, self.int_bits, self.frac_bits)
        }

        config = {'quant_config': quant_config, 'pruning_metrics': self.pruning_metrics}

        return config


class FITcompress:

    def __init__(self, model, device, dataloader, criterion, config, layerwise_pruning=False, input_shape=None):
        """
        Calculate initial EF of the uncompressed model and set up quantization &
        pruning schedules, as well as the initial node in the compression space.

        Notes:
            1) We use the FeM value for FIT calculation (as in the original code),
               not the EF trace directly (as in the paper).

        Args:
            model: Pre-trained model (uncompressed, but loaded with compression layers).
            device: Device to use for the model.
            dataloader: Dataloader for training data.
            criterion: Loss function (e.g., CrossEntropyLoss).
            config: Configuration object.
            layerwise_pruning: Whether to find layerwise pruning targets
                (NOTE: not tested).

        Config Args:
            compression_goal: Desired compression constraint (alpha in the paper).
            quantization_schedule: Bit-widths to try during the search.
            pruning_schedule: A schedule with (start, end, steps) describing the
                sparsity progression.
        """
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.criterion = criterion
        self.config = config
        self.layerwise_pruning = layerwise_pruning

        # Initialize an instance of the FIT class based on the uncompressed model.
        # This marks which weights & activations can be pruned/quantized.
        # We can then reuse this instance and its corresponding .get_EF() function
        # and get_FIT() functions, passing the appropriate empirical Fisher traces.
        self.fit_computer = FIT(self.model, self.device, input_spec=input_shape)

        # Calculate the EF trace of the uncompressed model (i.e. initial EF trace), only based on weights
        self.FeM, self.EF_trace_params_layerwise_uncompressed, _, _, _ = self.fit_computer.get_EF(
            self.model, self.dataloader, self.criterion, tolerance=0.01, min_iterations=100, max_iterations=100
        )

        # Get the number of layers in the model
        self.n_layers = len(self.EF_trace_params_layerwise_uncompressed)

        # Get the current model weights
        matrices_params_layerwise, _, self.layer_names = self.fit_computer.get_model_weights(self.model)

        # Store a copy of the original weights
        import copy

        self.original_matrices_params_layerwise = copy.deepcopy(matrices_params_layerwise)

        self.compression_goal = config.fitcompress_parameters.compression_goal
        self.quant_schedule = config.fitcompress_parameters.quantization_schedule
        self.pruning_schedule = 1 - np.logspace(
            config.fitcompress_parameters.pruning_schedule.start,
            config.fitcompress_parameters.pruning_schedule.end,
            base=10,
            num=config.fitcompress_parameters.pruning_schedule.steps,
        )

        # for N:M pruning in Wanda, use 50% pruning cap during FITcompress
        if self.config.pruning_parameters.pruning_method == 'wanda' and type(self.config.pruning_parameters.N) is int:
            self.pruning_schedule = 0.5 * (
                1
                - np.logspace(
                    config.fitcompress_parameters.pruning_schedule.start,
                    config.fitcompress_parameters.pruning_schedule.end,
                    base=10,
                    num=config.fitcompress_parameters.pruning_schedule.steps,
                )
            )

        # Dictionary structure allows us to possibly iterate over multiple different pruning metrics
        # but currently only one as in FITcompress, the target pruning sparsity, i.e. percentage
        pruning_metrics = {'percentage': 0}

        # If we want to find sparsity targets per layer (not part of FITcompress paper)
        if layerwise_pruning:
            self.pruning_schedulers_layerwise = self.get_pruning_schedulers_layer_specific(
                matrices_params_layerwise, None, mode='fit'
            )
            # Add the layer-specific starting pruning percentages to the current metric
            pruning_metrics = pruning_metrics | {f'{self.layer_names[i]}_percentage': 0 for i in range(self.n_layers)}

        # Initialize the first node in the compression space
        self.initial_node = node(
            matrices_params_layerwise=matrices_params_layerwise,
            FeM=self.FeM.copy(),
            quant_config=[31 for _ in range(self.n_layers)],  # 31 bits for each layer (since one goes to sign)
            pruning_metrics=pruning_metrics,  # Initial pruning defined in config.yaml
            gscore=0.0,  # Initial gscore
            fscore=np.inf,  # Initial fscore
            full_dist=np.inf,  # Initial full distance (gscore + lambda * fscore)
            # -1 such that 0 will be the first for first neighbouring nodes
            state=[-1 for _ in range(self.n_layers)]
            + [0],  # Initial state for all layers (for quantization) and one for global pruning
            curr_compression_rate=1.0,  # Initial compression rate (no compression yet)
            unquantized_weights=self.original_matrices_params_layerwise,  # Unquantized weights
            int_bits=[
                0 for _ in range(self.n_layers)
            ],  # Initial integer bits (although this could be set to any number, just need the list structure)
            frac_bits=[
                31 for _ in range(self.n_layers)
            ],  # Initial fractional bits ((although this could be set to any number, just need the list structure))
        )

        # Intialize a list to store nodes that can be traversed during the path finding process
        self.potential_nodes = [self.initial_node]

    def get_pruning_schedulers_layer_specific(self, matrices_params_layerwise, global_sparsity_scheduler, mode='fit'):
        """
        Calculates layer-specific pruning schedulers. The idea is that layers with weights
        that are not that much affected by pertubation should be pruned more/faster than layers with weights
        that are more affected by pertubation.
        The "speed" of schedulers are calculated depending on the mode.
        For mode 'fit' : via the layer-wise FIT score of the initial
        model.

        Returns :
                List of pruning schedulers (one per layer)

        """

        schedulers = {}
        if mode == 'fit':
            # Get the layer-wise FIT scores of the initial model
            _, FIT_layerwise = self.fit_computer.get_FIT_old(
                FeM=self.FeM, params_after=matrices_params_layerwise, same_theta=True
            )

            # For each layer, first take the sum over the FIT values (of each weight)
            FIT_layerwise_summed = [torch.sum(layer).item() for layer in FIT_layerwise]
            # Find min and max importance
            min_importance = min(FIT_layerwise_summed)
            max_importance = max(FIT_layerwise_summed)

            for layer_idx, importance in enumerate(FIT_layerwise_summed):

                # Scale importance between 0 and 1
                importance_ratio = (importance - min_importance) / (max_importance - min_importance)

                # Decides "strength" of difference between flattest/steepest curves -higher the value, more difference)
                scale_factor = 5

                # The higher the importance score, the bigger the exponent, the flatter the curve
                exponent = scale_factor * importance_ratio

                scheduler_curve = []
                for step in range(40):
                    # Normalize to [0,1] range
                    t = step / 39
                    pruning_at_step = t**exponent
                    scheduler_curve.append(pruning_at_step)

                schedulers[self.layer_names[layer_idx]] = np.array(scheduler_curve)

            if global_sparsity_scheduler is not None:
                for step in range(len(global_sparsity_scheduler)):
                    if step == 0:  # So we do not get a problem with division by 0
                        continue
                    # Calculate the sum of all layer-specific scheduler values at that step
                    sum_layer_schedulers = sum(curr_scheduler[step] for curr_scheduler in schedulers.values())
                    for layer_name in self.layer_names:
                        schedulers[layer_name][step] *= global_sparsity_scheduler[step] / sum_layer_schedulers

                    assert np.isclose(
                        sum(sched[step] for sched in schedulers.values()),
                        global_sparsity_scheduler[step],
                    ), (
                        f"Sum of layer-specific schedulers at step {step} does not match "
                        f"global sparsity scheduler value {global_sparsity_scheduler[step]}"
                    )

        return schedulers

    def assign_parameters(self, model, params):
        """
        Update model's actual parameters with new values from the compression search process.

        Args:
                model : The model to update the parameters of.
                params : Array of new parameters to assign to the model's layers.

        Notes:
                1. This only deals with weights. We do not look at any activations & bias, as
                this is not done in the original code.
        """
        i = 0
        from pquant.layers import PQConv2d, PQDense  # noqa: F811

        for _, module in model.named_modules():
            if isinstance(module, (PQDense, PQConv2d)):
                for name_param, matrix_param in list(module.named_parameters()):
                    if name_param.endswith('_weight'):
                        matrix_param.data = nn.parameter.Parameter(params[i].to(self.device))
                        matrix_param.collect = True
                        i += 1

    def add_quantization(self, model, params, quant_config, reset=False):
        """
        Quantizes weights of the model based on fixed-point quantization. Given the
        bit-width of the current quantization configuration of a layer, bits are
        distributed between integer and fractional part based on the maximum absolute
        value of the weights of that layer.

        Args :
                model : The model to quantize
                params : List of unquantized (but possibly pruned) parameters of the model, layerwise
                quant_config : List of current quantization configuration, layerwise
                reset : Flag that allows to quantize based on unpruned weights

        Returns :
                neighbour_node_parameters_layerwise : List of quantized parameters of the model, layerwise
                all_int_bits : List of integer bits for parameters, layerwise
                all_frac_bits : List of fractional bits for parameters, layerwise

        """
        neighbour_node_parameters_layerwise = []
        all_int_bits = []
        all_frac_bits = []

        # Get integer & fractional bits based on max abs value of each layer's weights
        for idx, param_layer in enumerate(params):
            max_abs = torch.max(torch.abs(param_layer.detach().cpu()))
            eps = 1e-12
            int_bits = max(0, math.ceil(math.log2(max_abs + eps)))
            fractional_bits = max(1, quant_config[idx] - int_bits)

            all_int_bits.append(int_bits)
            all_frac_bits.append(fractional_bits)
            logging.info(f"Weights int bits={int_bits}, fractional bits={fractional_bits}")

        for idx, param_layer in enumerate(params):
            # If reset is inactive, we quantize weights given the unquantized, but possibly pruned weights
            # of the current node
            # Note that this doesn't quantize the actual module, but just returns us the weights
            if not reset:
                new_weight = quantizer(
                    param_layer,
                    k=torch.tensor(1.0),
                    i=torch.tensor(all_int_bits[idx]),
                    f=torch.tensor(all_frac_bits[idx]),
                    training=True,
                )

            # If reset active, we quantized based on unpruned weights
            else:
                new_weight = quantizer(
                    self.original_matrices_params_layerwise[idx],
                    k=torch.tensor(1.0),
                    i=torch.tensor(all_int_bits[idx]),
                    f=torch.tensor(all_frac_bits[idx]),
                    training=True,
                )  # module_f_weights[idx]

            neighbour_node_parameters_layerwise.append(new_weight)

        return neighbour_node_parameters_layerwise, all_int_bits, all_frac_bits

    def add_pruning(self, current_node, params, importance_score, pruning_percentage):
        """
        Unstructured, global pruning based on a pruning percentage.
        This provides the pruning mask P as described in the FITCompress paper.

        Notes :
        1. Pruning importance scores are calculated based on the node that
        we are currently searching neighbours for : We prune the current node's weights
        to create the neighbour node based on pruning.

        Args :
                current_node : The current best model
                params : parameters after quantization
                importance_score : importance scores for each weight of the current best model
                pruning_percentage : The percentage of weights to prune from the current node's weights.

        Returns :
                current_node_matrices_params_layerwise : List of pruned parameters, layerwise.
                current_node_matrices_unquantized_params_layerwise : List of unquantized pruned parameters, layerwise.
        """

        # Get shape of each layers weight matrix
        current_node_matrices_params_shapes_layerwise = [layer.shape for layer in params]
        # Get total number of each layers weights
        current_node_matrices_params_numel_layerwise = [layer.numel() for layer in params]
        # Get cumulative sum of weights per layer (for indexing purposes)
        current_node_matrices_params_cumsum_layerwise = list(np.cumsum(current_node_matrices_params_numel_layerwise))
        # Add 0 at the beginning for easier indexing
        current_node_matrices_params_cumsum_layerwise.insert(0, 0)

        # Flatten parameters & importance scores and concatenate to have single vectors containing everything
        current_node_params_flat = torch.cat([layer.view(-1) for layer in params]).detach().cpu()
        # Same for importance scores
        current_node_importance_scores_flat = torch.cat([layer.view(-1) for layer in importance_score]).detach().cpu()

        # Also create an instance for the unquantized weights
        current_node_unquantized_params_flat = (
            torch.cat([layer.view(-1) for layer in current_node.unquantized_weights]).detach().cpu()
        )

        # Calculate the number of parameters to prune (percentage * nums of all parameters in the model)
        num_params_to_prune = int(pruning_percentage * len(current_node_params_flat))
        # Based on the negative importance scores of all the weights, find the indices of weights with the
        # smallest importance scores (closest to 0) (i.e. the ones that are not affected by pertubation as much)
        _, indices_params_to_prune = torch.topk(-current_node_importance_scores_flat, num_params_to_prune)
        # Set those parameters/weights to 0
        current_node_pruned_params_flat = torch.scatter(current_node_params_flat, -1, indices_params_to_prune, 0.0)

        # Also for unquantized weights
        current_node_pruned_unquantized_params_flat = torch.scatter(
            current_node_unquantized_params_flat, -1, indices_params_to_prune, 0.0
        )

        # Now reconstruct the correct shape
        current_node_matrices_params_layerwise = []

        for i in range(self.n_layers):
            current_node_matrices_params_layerwise.append(
                torch.reshape(
                    current_node_pruned_params_flat[
                        current_node_matrices_params_cumsum_layerwise[i] : current_node_matrices_params_cumsum_layerwise[
                            i + 1
                        ]
                    ],
                    current_node_matrices_params_shapes_layerwise[i],
                )
            )

        current_node_matrices_unquantized_params_layerwise = []
        for i in range(self.n_layers):
            current_node_matrices_unquantized_params_layerwise.append(
                torch.reshape(
                    current_node_pruned_unquantized_params_flat[
                        current_node_matrices_params_cumsum_layerwise[i] : current_node_matrices_params_cumsum_layerwise[
                            i + 1
                        ]
                    ],
                    current_node_matrices_params_shapes_layerwise[i],
                )
            )

        # Put everything on GPU again
        current_node_matrices_params_layerwise = [layer.to(self.device) for layer in current_node_matrices_params_layerwise]
        current_node_matrices_unquantized_params_layerwise = [
            layer.to(self.device) for layer in current_node_matrices_unquantized_params_layerwise
        ]

        return current_node_matrices_params_layerwise, current_node_matrices_unquantized_params_layerwise

    def add_pruning_layer_specific(self, current_node, pruning_metrics, layer_idx=None):
        """
        NOTE: Experimental (untested).

        This function aims to estimate per-layer pruning sparsity targets instead of a
        single global target. It could, for example, initialize layerwise targets for
        PDP.

        Per-layer pruning is performed via importance pruning based on the layer’s FIT
        value (currently using the FIT of the corresponding layer).

        Notes:
                A) Create a new node for each candidate pruned layer (similar to the
                quantization step). This is expensive because, per the paper, FIT would
                need to be recomputed each time.
                B) Create a single node for pruning, but apply per-layer sparsity targets
                (e.g., from PDP_setup() or FIT(theta_i, theta_i)) instead of one global
                target, and prune each layer according to its current target.
        """

        # As in the original code, the importance scores are calculated
        # as FIT(theta_i, theta_i) of the current model, i.e. the current node.
        _, FIT_layerwise = self.fit_computer.get_FIT_old(
            FeM=current_node.FeM, params_after=current_node.parameters, same_theta=True
        )

        # Get shape of each layers weight matrix
        current_node_matrices_params_shapes_layerwise = [layer.shape for layer in current_node.parameters]
        # Flatten parameters & importance scores
        current_node_params_flat_layerwise = [layer.view(-1).detach().cpu() for layer in current_node.parameters]
        # Same for importance scores
        current_node_importance_scores_flat_layerwise = [layer.view(-1).detach().cpu() for layer in FIT_layerwise]

        current_node_matrices_params_layerwise = []
        # Now iterate through all layers
        for idx, curr_pruning_percentage in enumerate(pruning_metrics.values()):

            if idx == 0:  # Global pruning percentage
                continue

            # Calculate number of parameters to prune in this layer, based on the current pruning percentage
            num_params_to_prune = int(curr_pruning_percentage * len(current_node_params_flat_layerwise[idx - 1]))

            # Based on the negative importance scores of all the weights, find the indices of weights with the
            # smallest importance scores (closest to 0) (i.e. the ones that are not affected by pertubation as much)
            _, indices_params_to_prune = torch.topk(
                -current_node_importance_scores_flat_layerwise[idx - 1], num_params_to_prune
            )
            # Set those parameters/weights to 0
            current_node_pruned_params_flat = torch.scatter(
                current_node_params_flat_layerwise[idx - 1], -1, indices_params_to_prune, 0.0
            )
            # Now reconstruct the correct shape
            current_node_matrices_params_layerwise.append(
                torch.reshape(current_node_pruned_params_flat, current_node_matrices_params_shapes_layerwise[idx - 1])
            )

        # Put everything on GPU again
        current_node_matrices_params_layerwise = [layer.to(self.device) for layer in current_node_matrices_params_layerwise]

        return current_node_matrices_params_layerwise

    def set_activation_bits(self, layer):
        if layer.quantize_input:
            max_abs = torch.max(torch.tensor([torch.max(torch.abs(e)) for e in layer.saved_inputs]))
            k, i, f = layer.get_input_quantization_bits()
            bits = k + i + f
            int_bits = math.ceil(math.log2(max_abs))
            frac_bits = max(self.config.pruning_parameters.min_frac_bits, bits - int_bits - k)
            layer.saved_inputs = []
            layer.input_quantizer.set_quantization_bits(int_bits, frac_bits)
            logging.info(f"Set input quantization bits from {i}, {f} to {int_bits}, {frac_bits}")

        if layer.quantize_output:
            max_abs = torch.max(torch.tensor([torch.max(torch.abs(e)) for e in layer.saved_outputs]))
            k, i, f = layer.get_output_quantization_bits()
            bits = k + i + f
            int_bits = math.ceil(math.log2(max_abs))
            frac_bits = max(self.config.pruning_parameters.min_frac_bits, bits - int_bits - k)
            layer.saved_outputs = []
            layer.output_quantizer.set_quantization_bits(int_bits, frac_bits)
            logging.info(f"Set output quantization bits from {i}, {f} to {int_bits}, {frac_bits}")

    def post_fitcompress_calibration(self, best_node_quant_config, config, calibration_epochs=50):
        """
        Calibrate integer/fractional bit allocation for activations, pooling layers,
                and model inputs *after* the FITcompress path search.

                We run forward passes through the **quantized but unpruned** model (weights
                use the FITcompress-found quant settings; pruning is not applied). For each
                module, we collect input ranges during these calibration passes and derive
                bit splits accordingly.

                - Activations: integer/fractional bits are chosen from the max absolute
                activation input. Total bit-width equals that of the corresponding weight
                layer selected by FITcompress for that activation unit.
                - Pooling & inputs: computed similarly from their input ranges. Since they
                have no corresponding weight layer, the total bit-width is fixed at 7
                bits (1 bit is the sign).

                Args:
                        best_node_quant_config: Per-layer quantization configuration produced
                                by FITcompress (mapping from layer name to [int_bits, frac_bits]).
                        calibration_epochs: Number of calibration forward passes to run.

                Returns:
                        activ_int_bits: List of integer bits per activation unit.
                        activ_frac_bits: List of fractional bits per activation unit.
                        pool_int_bits: Integer bits for the (single) pooling layer (res20).
                        pool_frac_bits: Fractional bits for the (single) pooling layer (res20).
        """
        from pquant.core.torch.activations import PQActivation  # noqa: F811
        from pquant.core.torch.layers import (  # noqa: F811
            PQAvgPoolBase,
            PQWeightBiasBase,
        )

        # To avoid numerical issues
        # Store input data, as we also need to quantize input (which is currently done in resnet.py of pquant-dev)
        data_input = []
        for m in self.model.modules():
            if isinstance(m, (PQAvgPoolBase, PQWeightBiasBase)):
                m.post_fitcompress_calibration = True
            elif m.__class__ == PQActivation and m.activation_name == "relu":
                m.post_fitcompress_calibration = True

        # Trigger forward pass through model
        self.model.eval()
        counter = 0

        for _, data in enumerate(self.dataloader):
            while counter < calibration_epochs:
                self.model.zero_grad()
                data_batch, _ = data[0].to(self.device), data[1].to(self.device)
                data_input.append(data_batch)
                _ = self.model(data_batch)
                counter += 1

        for m in self.model.modules():
            if isinstance(m, PQAvgPoolBase):
                m.post_fitcompress_calibration = False
                self.set_activation_bits(m)
            elif m.__class__ == PQActivation and m.activation_name == "relu":
                m.post_fitcompress_calibration = False
                self.set_activation_bits(m)
            elif isinstance(m, PQWeightBiasBase):
                m.post_fitcompress_calibration = False
                self.set_activation_bits(m)

    def astar(self, config):
        """
        The actual search algorithm of FITcompress, which is based on the A* algorithm.
        Find either the node that has an optimal configuration (i.e. compression rate lower than the goal) and break or find
        the node with the lowest distance between initial & optimal model among the potential best nodes.
        Config Args :
        greedy_astar : If set to True, remove all other current neighbour nodes and only keep the current best node
        Returns :
                A node descriptor of the best node found in the compression space
        """
        iterations = 0
        while len(self.potential_nodes) > 0 and iterations < 1000:
            logging.info(f'Iteration : {iterations} ')

            next_best_node = None

            logging.info(f"Finding the next best node among the  {len(self.potential_nodes)} neighbour nodes...")
            # Iterate through all potential next nodes to visit in the compression space
            for p_node in self.potential_nodes:
                # If we find a node with wanted compression rate, we can return it and stop the A* algorithm
                if p_node.curr_compression_rate < self.compression_goal:

                    logging.info(
                        f"Optimal node found with full distance {p_node.full_dist}, "
                        f"compression rate {p_node.curr_compression_rate}, "
                        f"quantization config {p_node.quant_config} and "
                        f"pruning metrics {p_node.pruning_metrics}"
                    )
                    # Based on the unquantized, but pruned weights, get the pruning mask based on the importance scores
                    p_node_pruning_mask_layerwise = [
                        (p_node.unquantized_weights[i] != 0).float() for i in range(self.n_layers)
                    ]

                    # Reset model's pruning, keep quantization, such that we can do post fitcompress calibration
                    params_quantized_unpruned, _, _ = self.add_quantization(
                        model=self.model,
                        params=p_node.unquantized_weights.copy(),
                        quant_config=p_node.quant_config.copy(),
                        reset=True,
                    )

                    self.assign_parameters(self.model, params_quantized_unpruned)

                    self.post_fitcompress_calibration(
                        p_node.extract_config_from_node(self.layer_names)['quant_config'], config
                    )

                    return (
                        p_node,
                        p_node.extract_config_from_node(self.layer_names),
                        self.model,
                        p_node_pruning_mask_layerwise,
                    )

                # Find the node with lowest distance between initial & optimal model
                if next_best_node is None or p_node.full_dist < next_best_node.full_dist:
                    next_best_node = p_node

            logging.info(
                f"Next best node found with full distance {next_best_node.full_dist}, "
                f"compression rate {next_best_node.curr_compression_rate}, "
                f"quantization config {next_best_node.quant_config} and "
                f"pruning metrics {next_best_node.pruning_metrics}"
            )

            # Keep only the found best neighbouring node to our current node and remove all other nodes in greedy search
            if self.config.fitcompress_parameters.greedy_astar:
                self.potential_nodes = [next_best_node]

            # After the next best node was found, set the model parameters
            self.assign_parameters(self.model, next_best_node.parameters.copy())
            self.create_neighbours(next_best_node)

            iterations += 1

    def create_neighbours(self, current_node):
        """
        Create neighbour nodes in the compression space for A* exploration.

        This expands the current node by advancing **either** quantization (per
        layer) **or** pruning (global, optionally layerwise) one step according to
        the configured schedules and the node's current state. The transformed
        parameters are passed to `create_new_node()`, and the resulting neighbours
        are added to `self.potential_nodes`.

        Specifically:
        • Quantization: create L neighbours (one per weight layer), each moving
                that layer to the next bit-width in `quantization_schedule`.
        • Pruning: create one neighbour that advances global sparsity to the next
                value in `pruning_schedule` (importance based on FIT(theta_i, theta_i)).

        Args:
                current_node: The node to expand (i.e., the current best model state).

        Config flags:
                fitcompress_parameters.optimize_quantization: enable quantisation neighbours.
                fitcompress_parameters.optimize_pruning: enable pruning neighbours.
                fitcompress_parameters.approximate: reuse previous FeM for quantisation
                        neighbours (computational shortcut).

        Notes:
                1) Unlike the original code (which combined pruning during quantisation
                neighbour creation), we **separate** the two steps to match the
                paper’s formulation.
                2) With `approximate=True`, we reuse the current node’s FeM when
                building quantisation neighbours (per the paper’s computational
                details). We do **not** use this shortcut for pruning.
        """

        if self.config.fitcompress_parameters.approximate:

            # Update FeM for the best node and use it when creating the neighbours for quantization.
            # This leads to num_layers less FIT calculations, as we do not need to calculate the FeM again,
            # which reduces runtime
            # Assign the current parameters to the model (i.e. the ones of the current node)
            self.assign_parameters(self.model, current_node.parameters)
            # Calculate the FeM of the current model
            curr_FeM, _, _, _, _ = self.fit_computer.get_EF(
                self.model, self.dataloader, self.criterion, min_iterations=100, max_iterations=100
            )
            # Set the current node's FeM to the calculated one
            current_node.FeM = curr_FeM.copy()

        # Calculate importance score of the current node (FIT(theta_i, theta_i))
        _, FIT_layerwise = self.fit_computer.get_FIT_old(
            FeM=current_node.FeM, params_after=current_node.parameters, same_theta=True
        )

        current_node_state = current_node.state.copy()

        logging.info("Current node states for quantization & pruning: ", current_node_state)
        if self.config.fitcompress_parameters.optimize_quantization:
            for layer_idx in range(self.n_layers):

                # Set neighbour state to current state
                neighbour_node_state = current_node_state.copy()

                # Check that we do not go out of bounds; if the scheduler ends, don't create a new node for this layer
                if neighbour_node_state[layer_idx] < len(self.quant_schedule) - 1:
                    # Move one step forward in the state for the current layer
                    neighbour_node_state[layer_idx] += 1
                    # Set neighbour quant config to current quant config
                    neighbour_node_quant_config = current_node.quant_config.copy()
                    # Update the neighbour quant config for the current layer based
                    # on the quantization schedule and the neighbour node state
                    neighbour_node_quant_config[layer_idx] = self.quant_schedule[neighbour_node_state[layer_idx]]

                    # If we want to skip a layer's quantization
                    if neighbour_node_quant_config[layer_idx] == 0:
                        continue

                    # Get the current node's pruning metrics (will not be changed during quantization)
                    neighbour_node_pruning_metrics = current_node.pruning_metrics.copy()
                    neighbour_node_parameters_layerwise, neighbour_node_int_bits, neighbour_node_frac_bits = (
                        self.add_quantization(
                            model=self.model,
                            params=current_node.unquantized_weights.copy(),
                            quant_config=neighbour_node_quant_config,
                        )
                    )

                    # Create node structure based on applied quantization setting for the current layer
                    neighbour_node = self.create_new_node(
                        current_node=current_node,
                        neighbour_node_parameters_layerwise=neighbour_node_parameters_layerwise,
                        neighbour_node_quant_config=neighbour_node_quant_config,
                        neighbour_node_state=neighbour_node_state,
                        neighbour_node_pruning_metrics=neighbour_node_pruning_metrics,
                        neighbour_node_unquantized_parameters_layerwise=current_node.unquantized_weights.copy(),
                        neighbour_node_int_bits=neighbour_node_int_bits,
                        neighbour_node_frac_bits=neighbour_node_frac_bits,
                        approximate=self.config.fitcompress_parameters.approximate,
                    )

                    # Add the neighbour node to the potential nodes list
                    self.potential_nodes.append(neighbour_node)

        if self.config.fitcompress_parameters.optimize_pruning:

            # Set neighbour state to current state
            neighbour_node_state = current_node_state.copy()

            # Check that we do not go out of bounds ; if the scheduler ends, we don't create new nodes for pruning anymore
            if neighbour_node_state[-1] + 1 < len(self.pruning_schedule):
                # Move one step forward in the state for pruning
                neighbour_node_state[-1] += 1

                # Get current quantization config of the current node (will not be changed during pruning)
                neighbour_node_quant_config = current_node.quant_config.copy()
                neighbour_node_int_bits = current_node.int_bits.copy()
                neighbour_node_frac_bits = current_node.frac_bits.copy()

                neighbour_node_pruning_percentage = self.pruning_schedule[neighbour_node_state[-1]]

                neighbour_node_pruning_percentages = [neighbour_node_pruning_percentage]

                # layerwise pruning (works, but not tested for performance) ; probably outdated
                if self.layerwise_pruning:
                    neighbour_node_pruning_percentage_layerwise = [
                        layer_scheduler[neighbour_node_state[-1]]
                        for layer_scheduler in self.pruning_schedulers_layerwise.values()
                    ]  # Layerwise pruning percentage
                    neighbour_node_pruning_percentages += (
                        neighbour_node_pruning_percentage_layerwise  # Add the layerwise pruning percentages to the list
                    )

                neighbour_node_pruning_metrics = current_node.pruning_metrics.copy()

                # Update the pruning metrics for the neighbour node
                for idx, key in enumerate(neighbour_node_pruning_metrics.keys()):
                    neighbour_node_pruning_metrics[key] = neighbour_node_pruning_percentages[idx]

                if self.layerwise_pruning:
                    neighbour_node_parameters_layerwise = self.add_pruning_layer_specific(
                        current_node=current_node, pruning_metrics=neighbour_node_pruning_metrics
                    )
                else:

                    neighbour_node_parameters_layerwise, neighbour_node_unquantized_parameters_layerwise = self.add_pruning(
                        current_node=current_node,
                        params=current_node.parameters.copy(),
                        importance_score=FIT_layerwise,
                        pruning_percentage=neighbour_node_pruning_percentage,
                    )

                # Create node structure based on applied pruning
                neighbour_node = self.create_new_node(
                    current_node=current_node,
                    neighbour_node_parameters_layerwise=neighbour_node_parameters_layerwise,
                    neighbour_node_quant_config=neighbour_node_quant_config,
                    neighbour_node_state=neighbour_node_state,
                    neighbour_node_pruning_metrics=neighbour_node_pruning_metrics,
                    neighbour_node_unquantized_parameters_layerwise=neighbour_node_unquantized_parameters_layerwise,
                    neighbour_node_int_bits=neighbour_node_int_bits,
                    neighbour_node_frac_bits=neighbour_node_frac_bits,
                    approximate=False,
                )

                # Add the neighbour node to the potential nodes list
                self.potential_nodes.append(neighbour_node)

        # Remove the current node from the potential nodes list, as we have now created all its neighbours
        current_node_key = current_node.key
        for idx, node in enumerate(self.potential_nodes):
            if node.key == current_node_key:
                del self.potential_nodes[idx]
                break

    def create_new_node(
        self,
        current_node,
        neighbour_node_parameters_layerwise,
        neighbour_node_quant_config,
        neighbour_node_state,
        neighbour_node_pruning_metrics,
        neighbour_node_unquantized_parameters_layerwise,
        neighbour_node_int_bits,
        neighbour_node_frac_bits,
        approximate=False,
    ):
        """
        Create a new node in the compression space based on the current node and the new parameters.
        Args :
                current_node : The current node, i.e. the current model.
                neighbour_node_parameters_layerwise : The parameters of the neighbour node, layerwise.
                neighbour_node_quant_config : The quantization configuration of the neighbour node, layerwise
                neighbour_node_state : The state of the neighbour node.
                neighbour_node_pruning_metrics : Sparsity goal for pruning of the neighbour node
                neighbour_node_unquantized_parameters_layerwise: Unquantized parameters of the neighbour node, layerwise.
                neighbour_node_int_bits : The integer bits for weights of the neighbour node, layerwise.
                neighbour_node_frac_bits : The fractional bits for weights of the neighbour node, layerwise.
                approximate : If set to True, we use the previous FeM to calculate FIT values for Quantization.
        Returns :
        The newly created neighbour node.
        """

        if approximate:
            # If approximate is set to True, we do not recalculate the FeM, but use the one from the current node
            neighbour_node_FeM = current_node.FeM.copy()

        else:
            # First, assign the new parameters of the neighbour node to the model
            # This is done in order that we can calculate the FeM based on these new parameters
            self.assign_parameters(self.model, neighbour_node_parameters_layerwise)

            # Then, compute the new FeM based on the new parameters of the neighbour node
            neighbour_node_FeM, _, _, _, _ = self.fit_computer.get_EF(
                model=self.model,
                data_loader=self.dataloader,
                loss_func=self.criterion,
                tolerance=1e-3,
                min_iterations=100,
                max_iterations=100,
            )

        # Calculate the gscore, fscore, full distance and current compression rate
        neighbour_node_gscore, neighbour_node_fscore, neighbour_node_full_dist, neighbour_node_compression_rate = (
            self.calculate_path_cost(
                current_node=current_node,
                neighbour_node_parameters_layerwise=neighbour_node_parameters_layerwise,
                neighbour_node_FeM=neighbour_node_FeM,
                neighbour_node_quant_config=neighbour_node_quant_config,
            )
        )

        # Create the instance for the neighbour node
        neighbour_node = node(
            matrices_params_layerwise=neighbour_node_parameters_layerwise,
            FeM=neighbour_node_FeM,
            quant_config=neighbour_node_quant_config,
            pruning_metrics=neighbour_node_pruning_metrics,
            gscore=neighbour_node_gscore,
            fscore=neighbour_node_fscore,
            full_dist=neighbour_node_full_dist,
            state=neighbour_node_state,
            curr_compression_rate=neighbour_node_compression_rate,
            unquantized_weights=neighbour_node_unquantized_parameters_layerwise,
            int_bits=neighbour_node_int_bits,
            frac_bits=neighbour_node_frac_bits,
        )

        return neighbour_node

    def calculate_path_cost(
        self, current_node, neighbour_node_parameters_layerwise, neighbour_node_FeM, neighbour_node_quant_config
    ):
        """
        Calculates the g and f score to evaluate the cost of the path from initial model to current model (g score)
        and the heuristic cost to the goal model (f score). Furthermore, calculates the full distance based on both scores.
        Additionally, the compression rate of the neighbour node (i.e. model) is calculated.
        Args :
                current_node : The current node in the compression space, i.e. the current model.
                neighbour_node_parameters_layerwise : The parameters of the neighbour node, layerwise.
                neighbour_node_FeM : The FeM of the neighbour node, layerwise.
                neighbour_node_quant_config : The quantization configuration of the neighbour node, layerwise.
        Config Args:
                f_lambda : The lambda value to use for the full distance calculation.
        Returns:
                neighbour_node_gscore : The g score of the neighbour node, the cost from initial to neighbour nodes.
                neighbour_node_fscore : The score of the neighbour node, cost to the goal model from the neighbour node.
                neighbour_node_full_dist : The full distance from initial node to final node,
                given we use the neighbour node, i.e. g score + f_lambda * f score.
                neighbour_node_compression_rate: Compression rate of the neighbour node, % of the original model is active
        """
        # curr_g_score + sqrt(FIT(params_current_node, params_neighbour_node))
        neighbour_node_gscore = current_node.gscore + np.sqrt(
            self.fit_computer.get_FIT_old(
                params_before=current_node.parameters,
                FeM=current_node.FeM,
                params_after=neighbour_node_parameters_layerwise,
                same_theta=False,
            )
        )

        # abs(neighbour_node_compression_rate - compression_goal) * sqrt(FIT(params_neighbour_node, params_neighbour_node))
        # First, calculate the compression rate of the neighbour node (i.e. model)
        neighbour_node_compression_rate = self.calculate_current_compression_rate(
            params_layerwise=neighbour_node_parameters_layerwise, quant_config=neighbour_node_quant_config
        )
        # Then get FIT(params_neighbour_node,params_neighbour_node)
        neighbour_node_FIT, _ = self.fit_computer.get_FIT_old(
            params_after=neighbour_node_parameters_layerwise, FeM=neighbour_node_FeM, same_theta=True
        )
        # Finally, calculate the f score
        neighbour_node_fscore = np.sqrt(
            (np.abs(neighbour_node_compression_rate - self.compression_goal) ** 2) * neighbour_node_FIT
        )

        # g_score + lambda * f_score
        neighbour_node_full_dist = (
            neighbour_node_gscore + self.config.fitcompress_parameters.f_lambda * neighbour_node_fscore
        )

        return neighbour_node_gscore, neighbour_node_fscore, neighbour_node_full_dist, neighbour_node_compression_rate

    def calculate_current_compression_rate(self, params_layerwise, quant_config):
        """
        Calculates the compression ratio of the model (what fraction of the original model in bytes is
        still "active" after applying compression)

        Args:
                params_layerwise : The current parameters (theta) of the model, layer-wise.
                quant_config : The current (per layer) quantization config.

        Returns:
                The compression ratio, i.e. how many bytes are still active after pruning and quantization
                (= active bytes / uncompressed bytes)
        """

        active_bytes = 0.0
        uncompressed = 0.0
        for params_layer, quant_conf_layer in zip(params_layerwise, quant_config):
            # Count which parameters are non-zero, non_zero is simply the number of non-zero parameters in the current layer
            int_bits = max(0, math.ceil(math.log2(torch.max(torch.abs(params_layer)))))
            frac_bits_round_threshold = max(
                1, quant_conf_layer - int_bits - 2
            )  # - sign bit and rounding causes the second one
            non_zero = (
                torch.sum(torch.where(torch.abs(params_layer) < 2**-frac_bits_round_threshold, 0, 1)).detach().cpu().numpy()
            )
            logging.info(f"With {quant_conf_layer}, fractional bits is {frac_bits_round_threshold}")
            active_bytes += (
                non_zero * quant_conf_layer / 8
            )  # Gives us the number of total bytes needed to store the parameters in the current layer

            # For the uncompressed version, we simply look at ALL parameters that are in that layer
            # and simply multiply by 4, since each original parameter is 32 bits
            uncompressed += params_layer.numel() * 4

        return active_bytes / uncompressed


class FIT:

    def __init__(self, model, device, input_spec):
        """
        Initialize the FIT class, which is used to compute the FIT values for quantization and pruning.
        Args:
                model (torch.nn.Module): The model for which to compute the FIT values.
                device (torch.device): The device on which the model is located.
                input_spec (tuple): The input specification for the model, e.g. (3, 32, 32) for CIFAR-10.
        """
        self.hooks = []
        self.device = device

        self.matrices_params_layerwise, self.matrices_params_sizes_layerwise, _ = self.get_model_weights(model)
        self.hook_layers(model)

        # Dummy Forward Pass to trigger hooks & collect activations
        _ = model(torch.randn(input_spec)[None, ...].to(self.device))

        # List of sizes of tensors of activation inputs
        self.matrices_activs_sizes_layerwise = []

        for _, module in model.named_modules():
            if module.act_quant:
                self.matrices_activs_sizes_layerwise.append(module.activ_in[0].size())

    def get_model_weights(self, model):
        """
        Set collect flag to True for all weights of the layers of interest in the model.
        This will give us easy access to the weights that we want to quantize/prune later on.
        Furthermore, we can also access the weights, which we need for the FIT calculation.

        Notes:
                1.This is only called once initially. Its main purpose is to set the .collect flag to True,
                such that we can then later on access the weights easily.

        Args:
                model (torch.nn.Module): The model from which to get the weights.
        Returns:
                matrices_params_layerwise (list): A list of the weight matrices for each layer of interest.
                matrices_params_sizes_layerwise (list): A list of sizes of the weight matrices for each layer of interest.
                layer_names (list): A list of the names of the layers of interest.
        """
        from pquant.layers import PQConv2d, PQDense  # noqa: F811

        matrices_params_layerwise = []
        layer_names = []
        # Iterate through all modules in the model
        for name, module in model.named_modules():

            if isinstance(module, (PQDense, PQConv2d)):
                layer_names.append(name)
                for name_param, matrix_param in list(module.named_parameters()):
                    # Search for the weights
                    if name_param.endswith('_weight'):
                        matrices_params_layerwise.append(matrix_param)
                        # Set their collect flag to True (later on we can then access them easily like this)
                        matrix_param.collect = True
                    else:
                        matrix_param.collect = False
                continue

            # For Batch Normalization layers etc. we do not collect any weights
            for matrix_param in list(module.parameters()):
                if matrix_param.requires_grad:
                    matrix_param.collect = False

        # Collect the sizes of the weight matrices
        matrices_params_sizes_layerwise = [param.size() for param in matrices_params_layerwise]

        return matrices_params_layerwise, matrices_params_sizes_layerwise, layer_names

    def hook_layers(self, model):
        """
        Used to get the activation inputs during the forward pass, which are
        needed for computing the FIT (if calculated with the noise model) w.r.t activations.

        Args :
                model (torch.nn.Module): The model to hook the layers of.
        """
        from pquant.layers import PQConv2d, PQDense  # noqa: F811

        def hook_inp(module, inp, outp):
            """
            Store activation input of the module.

            """
            module.activ_in = inp

        for _, module in model.named_modules():
            if isinstance(module, (PQDense, PQConv2d)):
                # Forward Hook to get inputs into activation function
                hook = module.register_forward_hook(hook_inp)
                self.hooks.append(hook)  # Store hooks so we can remove them later
                module.act_quant = True  # mark it
            else:
                module.act_quant = False

    def hook_removal(self):
        """
        Remove all hooks that were registered to the model.
        """
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()
        assert len(self.hooks) == 0, "Hooks were not removed properly!"

    def get_loss(self, model, data_batch, target_batch, loss_func, mode='mini-batch'):
        """
        This function triggers the loss calcuation of a model.
        We use it such that we can then calculate gradients which are
        needed for EF trace (which is one part of the FIT).

        Notes:
                1. I here give the idea how we could do it for both mini-batch and sample loss calculation.

        Args:
                model (torch.nn.Module): The (trained) model to calculate the loss for.
                data_batch (Tensor): Current input data mini-batch for the model.
                target_batch (Tensor): Current target data mini-batch for the model.
                loss_func (callable): Loss function to use for the calculation.
                mode (str): Mode of loss calculation, either 'mini-batch' or 'sample'.
                'mini-batch' calculates loss for each mini-batch (by summing losses
                and averaging over mini-batch),
                'sample' calculates loss for each sample (which should be more close
                to the actual paper's definition).
        Returns:
                loss :  Loss for current mini-batch (either averaged over mini-batch or per sample).

        """

        output = model(data_batch)

        if mode == 'mini-batch':
            # Check which loss_func instance is active
            if isinstance(loss_func, torch.nn.CrossEntropyLoss):
                # Calculate loss based on mini-batch and averaged over it
                loss_func = torch.nn.CrossEntropyLoss()

        if mode == 'sample':
            if isinstance(loss_func, torch.nn.CrossEntropyLoss):
                # Calculate loss for each sample
                loss_func = torch.nn.CrossEntropyLoss(reduce=False, reduction='none')

        loss = loss_func(output, target_batch)

        return loss

    def get_gradients(self, model, loss, matrices_layerwise, batch_size):
        """
        This function calculates the gradients & squared gradients.
        These are then used to calculate the EF trace down the line,
        either for parameters or activations.

        Args:
                model (torch.nn.Module): The model to calculate the loss for.
                loss (Tensor): The loss tensor for which to calculate gradients.
                matrices_layerwise (list): List of parameters/activation matrices for which to calculate gradients.
                batch_size (int): Size of the mini-batch used for the loss calculation.
        Returns:
                squared_grad : Squared gradients for the passed parameters/activations, layer-wise.
        """
        grads = torch.autograd.grad(loss, [*matrices_layerwise], retain_graph=True)
        squared_grads = [batch_size * g**2 for g in grads]

        return squared_grads

    def get_EF(self, model, data_loader, loss_func, tolerance=1e-3, min_iterations=100, max_iterations=100):
        """
        Calculate the approximate Empirical Fisher (EF) trace / approximate Fisher
        Information Metric (FIM) for the model.

        Notes:
                1) Computed per mini-batch: we accumulate EF over mini-batches. This
                differs from the paper, which defines EF per sample and then
                accumulates.
                2) FeM vs. EF_trace_params_layerwise_cpu: FeM keeps layer tensors
                (no per-layer sum), while EF_trace_params_layerwise_cpu stores
                per-layer scalar traces (sum over parameters).
                3) The returned min/max ranges are only needed when using the noise
                model for FIT.

        Args:
                model (torch.nn.Module): Model to evaluate.
                data_loader (DataLoader): Training data loader.
                loss_func (Callable): Loss function used to compute gradients.
                tolerance (float): Convergence tolerance for EF accumulation.
                min_iterations (int): Minimum mini-batches before checking convergence.
                max_iterations (int): Maximum mini-batches to process.

        Returns:
                FeM (list[Tensor]): Layer-wise EF tensors accumulated over mini-batches.
                EF_trace_params_layerwise_cpu (np.ndarray): Layer-wise EF traces
                        (parameters), accumulated over mini-batches.
                EF_trace_activs_layerwise_cpu (np.ndarray): Layer-wise EF traces
                        (activations), accumulated over mini-batches.
                per_batch_layerwise_minmax_range_params (list[list[float]]): Per-batch
                        min–max ranges for parameters per layer.
                per_batch_layerwise_minmax_range_activs (list[list[float]]): Per-batch
                        min–max ranges for activations per layer.
        """

        # Convergence flag based on variance of change in EF between current mini-batch estimation and accumulated EF trace
        convergence_flag = False
        total_batches = 0
        model.eval()

        # Hook layers again, needed for when we recalculate EF traces during FITcompress
        self.hook_layers(model)

        # Initialize list to store accumulated EF of parameters (weights) over mini-batches
        batch_accum_EF_matrices_params_layerwise = [
            torch.zeros(size).to(self.device) for size in self.matrices_params_sizes_layerwise
        ]
        # Initialize list to store accumulated EF of activations over mini-batches
        batch_accum_EF_matrices_activs_layerwise = [
            torch.zeros(size).to(self.device) for size in self.matrices_activs_sizes_layerwise[1:]
        ]
        # NOTE:layerwise means that each element itself is a list of the ranges for each layer for the current mini-batch
        per_batch_layerwise_minmax_range_params = []
        per_batch_layerwise_minmax_range_activs = []
        # These will be needed for the convergence check
        # NOTE: layerwise means that each element itself is a list of the ranges for each layer for the current mini-batch
        per_batch_layerwise_grad_sum_squared_params = []
        per_batch_layerwise_grad_sum_squared_activs = []
        # Iterate over mini-batches in the data loader untill we reach the max_iterations or convergence flag is not set
        batch_size = None
        while total_batches < max_iterations and not convergence_flag:
            for _, data in enumerate(data_loader):
                model.zero_grad()
                data_batch, target_batch = data[0].to(self.device), data[1].to(self.device)
                if batch_size is None:
                    batch_size = data_batch.size(0)  # Only save once
                if data_batch.size(0) != batch_size:
                    continue  # Uneven batches break loop

                loss = self.get_loss(model, data_batch, target_batch, loss_func, mode='mini-batch')
                curr_batch_matrices_params_layerwise = []
                curr_batch_minmax_range_params_layerwise = []
                for weights in model.parameters():
                    if hasattr(weights, "collect"):
                        if weights.collect:
                            curr_batch_matrices_params_layerwise.append(weights)
                            curr_batch_minmax_range_params_layerwise.append(
                                (torch.max(weights.data) - torch.min(weights.data)).detach().cpu().numpy()
                            )

                per_batch_layerwise_minmax_range_params.append(curr_batch_minmax_range_params_layerwise)

                curr_batch_matrices_activs_layerwise = []
                curr_batch_minmax_range_activs_layerwise = []
                for _, module in model.named_modules():
                    if module.act_quant:
                        curr_batch_matrices_activs_layerwise.append(module.activ_in[0])
                        curr_batch_minmax_range_activs_layerwise.append(
                            (torch.max(module.activ_in[0]) - torch.min(module.activ_in[0])).detach().cpu().numpy()
                        )

                per_batch_layerwise_minmax_range_activs.append(curr_batch_minmax_range_activs_layerwise)

                # Calculate squared gradients for current mini-batch
                curr_batch_squared_grads_params_layerwise = self.get_gradients(
                    model, loss, curr_batch_matrices_params_layerwise, batch_size
                )
                curr_batch_squared_grads_activs_layerwise = self.get_gradients(
                    model, loss, curr_batch_matrices_activs_layerwise[1:], batch_size
                )  # skip first layer activations, since it is the input layer

                # NOTE : We need this for early stopping based on convergence , it is not necessary for the EF calculation
                # Take the sum of squared gradients for parameters/activations of each layer
                curr_batch_summed_squared_grads_params_layerwise = np.array(
                    [
                        torch.sum(param_matrix).detach().cpu().numpy()
                        for param_matrix in curr_batch_squared_grads_params_layerwise
                    ]
                )
                curr_batch_summed_squared_grads_activs_layerwise = np.array(
                    [
                        torch.sum(activ_matrix).detach().cpu().numpy()
                        for activ_matrix in curr_batch_squared_grads_activs_layerwise
                    ]
                )
                # Append the current mini-batch squared gradients to the list of per-batch squared gradients
                per_batch_layerwise_grad_sum_squared_params.append(curr_batch_summed_squared_grads_params_layerwise)
                per_batch_layerwise_grad_sum_squared_activs.append(curr_batch_summed_squared_grads_activs_layerwise)
                batch_accum_EF_matrices_params_layerwise = [
                    curr_val_layer + curr_squared_grad_layer + 0.0
                    for curr_val_layer, curr_squared_grad_layer in zip(
                        batch_accum_EF_matrices_params_layerwise, curr_batch_squared_grads_params_layerwise
                    )
                ]
                batch_accum_EF_matrices_activs_layerwise = [
                    acc + grad + 0.0
                    for acc, grad in zip(
                        batch_accum_EF_matrices_activs_layerwise,
                        curr_batch_squared_grads_activs_layerwise,
                    )
                ]
                total_batches += 1
                # NOTE: Only when we iterated over all mini-batches and would stop there, it is 1/N !
                batch_accum_EF_matrices_params_normalized_layerwise = [
                    accum_grad_layer / float(total_batches) for accum_grad_layer in batch_accum_EF_matrices_params_layerwise
                ]
                batch_accum_EF_matrices_activs_normalized_layerwise = [
                    accum_grad_layer / float(total_batches) for accum_grad_layer in batch_accum_EF_matrices_activs_layerwise
                ]

                # FeM of the original code and in usage for the current FITcompress implementation
                FeM = [value.detach().cpu() for value in batch_accum_EF_matrices_params_normalized_layerwise]

                EF_trace_params_layerwise = [
                    torch.sum(value) for value in batch_accum_EF_matrices_params_normalized_layerwise
                ]
                EF_trace_activs_layerwise = [
                    torch.sum(value) for value in batch_accum_EF_matrices_activs_normalized_layerwise
                ]

                EF_trace_params_layerwise_cpu = np.array(
                    [value.detach().cpu().numpy() for value in EF_trace_params_layerwise]
                )
                EF_trace_activs_layerwise_cpu = np.array(
                    [value.detach().cpu().numpy() for value in EF_trace_activs_layerwise]
                )

                # Convergence check
                if total_batches >= 2:
                    # Calculate variance of the change in EF trace for parameters and activations
                    params_var = (
                        np.var(
                            (per_batch_layerwise_grad_sum_squared_params - EF_trace_params_layerwise_cpu)
                            / EF_trace_params_layerwise_cpu
                        )
                        / total_batches
                    )
                    activs_var = (
                        np.var(
                            (per_batch_layerwise_grad_sum_squared_activs - EF_trace_activs_layerwise_cpu)
                            / EF_trace_activs_layerwise_cpu
                        )
                        / total_batches
                    )

                    if activs_var < tolerance and params_var < tolerance and total_batches > min_iterations:
                        convergence_flag = True

                if convergence_flag or total_batches >= max_iterations:
                    break

        # Remove hooks after the forward pass
        self.hook_removal()

        self.FeM = FeM
        self.EF_trace_params_layerwise_cpu = EF_trace_params_layerwise_cpu
        self.EF_trace_activs_layerwise_cpu = EF_trace_activs_layerwise_cpu
        self.per_batch_layerwise_minmax_range_params = per_batch_layerwise_minmax_range_params
        self.per_batch_layerwise_minmax_range_activs = per_batch_layerwise_minmax_range_activs

        return (
            FeM,
            EF_trace_params_layerwise_cpu,
            EF_trace_activs_layerwise_cpu,
            per_batch_layerwise_minmax_range_params,
            per_batch_layerwise_minmax_range_activs,
        )

    def squared_step_width_quantization(self, ranges_layerwise, quant_bit_precision_layerwise):
        """
        Calculate the squared step width of the quantization (reference can be found in FIT paper (Appendix E),
        this is the formula for Delta).
        This is needed for the uniform noise model, which will calculate delta_theta

        Notes:
                Since this was part of the FIT paper, the noise model is based solely on quantization, not pruning.

        Args:
                ranges: min-max range of the weights/activations, layer-wise
                quant_bit_precision: quantization bit precision, layer-wise
        Returns:
                squared step width of the quantization, layer-wise

        """

        return (ranges_layerwise / (2**quant_bit_precision_layerwise - 1)) ** 2

    def get_FIT_noise_model(
        self,
        EF_trace_params_layerwise_cpu,
        quant_bit_precision_params_layerwise,
        quant_bit_precision_activs_layerwise=None,
        EF_trace_activs_layerwise_cpu=None,
        use_activations=False,
    ):
        """
        Calculate the FIT for the model, based on the empirical Fisher trace and the squared step width of the quantization
        that comes from the noise model. This implements the FIT formula from the FITCompress paper.
        Notes:
                1. The original FITCompress code does not use the noise model, but rather the actual parameter values.
                2. The original FIT was implemented only based on activations. I here also include parameters.
                3. TODO : FOR FITCOMPRESS : Since FITCompress works only with parameters
                since they will not change during the path-finding process, i.e. they are not quantized
                4. TODO : FOR FITCOMPRESS : How to deal with the FIT(theta_i, theta_i) calculation ?
        Args:
                EF_trace_params_layerwise_cpu (list): empirical Fisher trace values  parameters (weights), layer-wise
                EF_trace_activs_layerwise_cpu (list): empirical Fisher trace values activations, layer-wise
                quant_bit_precision_params (list): quantization bit precision for the parameters (weights), layer-wise
                quant_bit_precision_activs (list): quantization bit precision for the activations, layer-wise
                use_activations (bool): whether to include activations in the FIT calculation or not

        Returns:
                FIT_full (float): FIT value
        """

        # Get the mean across all stored mini-batches for each layer
        mean_range_params_layerwise = np.mean(self.per_batch_layerwise_minmax_range_params, axis=0)

        # Calculate the squared step width of the quantization for parameters and activations
        # TODO : here we need to deal with the FIT(theta_i, theta_i) calculation
        squared_step_width_quant_params_layerwise = self.squared_step_width_quantization(
            mean_range_params_layerwise, np.array(quant_bit_precision_params_layerwise)
        )

        FIT_params_layerwise = squared_step_width_quant_params_layerwise * EF_trace_params_layerwise_cpu

        # Calculate full FIT by summing over all layers
        # TODO : missing is the 1/n(l) normalization, but cancels itself out according to Adrian
        FIT_full = np.sum(FIT_params_layerwise)

        if use_activations:
            mean_range_activs_layerwise = np.mean(self.per_batch_layerwise_minmax_range_activs, axis=0)[
                1:
            ]  # # skip first layer activations, since it is the input layer
            squared_step_width_quant_activs_layerwise = self.squared_step_width_quantization(
                mean_range_activs_layerwise, np.array(quant_bit_precision_activs_layerwise[1:])
            )  # 1: depends on the setting (whether first layer is included in config or not)
            FIT_activs_layerwise = squared_step_width_quant_activs_layerwise * EF_trace_activs_layerwise_cpu
            FIT_full += np.sum(FIT_activs_layerwise)

        self.FIT_full = FIT_full

        return FIT_full

    def get_FIT_real_values(self, params_before, EF_trace_params_layerwise, params_after=None, same_theta=False):
        """
        Calculate the actual FIT for the model, based on the parameter values between two
        consecutive models during the FITCompress path-finding process.

        Notes:
                1. This deviates from the code for FITCompress. In the original code, they use the still
        layer-wise EF and not the trace and do element-wise multiplication.
                Here I implement the actual FIT formula from the theoretical paper. This means it still has to be tested.
                3. Does not include any activations until now since original FITCompress code doesn't include
                                them, so would need to be extended.
        Args:
                params_before (list): List of parameters (weights) before the path-finding step.
                params_after (list): List of parameters (weights) after the path-finding step.
                EF_trace_params_layerwise (list): List of empirical Fisher trace values for each layer.
                same_theta (bool) : When we need to calculate FIT(theta,theta) for the f heuristic
        Returns:
        curr_FIT (float): FIT value
        """

        curr_FIT = 0
        if not same_theta:
            for theta_before, theta_after, EF_trace in zip(params_before, params_after, EF_trace_params_layerwise):
                # Calculate the squared difference between the parameters before and after
                delta_theta = torch.sum((theta_before.detach().cpu() - theta_after.detach().cpu()) ** 2).numpy()
                # Calculate the FIT for the current layer
                curr_FIT += EF_trace * delta_theta

        else:

            for theta, EF_trace in zip(params_before, EF_trace_params_layerwise):
                # Calculate the squared difference between the parameters before and after
                delta_theta = torch.sum(theta.detach().cpu() ** 2)
                # Calculate the FIT for the current layer
                curr_FIT += EF_trace * delta_theta

        return curr_FIT

    def get_FIT_old(self, params_before=None, FeM=None, params_after=None, same_theta=False):
        """
        Implementation of the original FIT code (compute_fake_FIT_params()), with the addition that we add the
        same_theta mode). Calculates the FIT of the model.

        Args:
                params_before (list): List of parameters (weights) before the path-finding step.
                FeM (list) : The EF of parameters stored as tensors and accumulated over mini-batches, layer-wise
                params_after (list): List of parameters (weights) after the path-finding step.
                same_theta (bool) : When we need to calculate FIT(theta,theta), the importance score
        Returns:
                curr_FIT (float) : FIT value
                FIT_layerwise (list) : FIT value per layer (only returned when same_theta is True)


        """

        curr_FIT = 0

        if not same_theta:

            # Taken from compute_fake_FIT_params()
            for theta_before, theta_after, layer_FeM in zip(params_before, params_after, FeM):

                curr_FIT_layer = torch.sum(
                    layer_FeM * (theta_before.detach().cpu() - theta_after.detach().cpu()) ** 2
                ).numpy()
                curr_FIT += curr_FIT_layer

            return curr_FIT

        else:
            FIT_layerwise = []

            # Taken from generate_FIT_pruning_importance()
            for theta_after, layer_FeM in zip(params_after, FeM):

                curr_FIT_layer = layer_FeM * (theta_after.detach().cpu() ** 2)
                FIT_layerwise.append(curr_FIT_layer)

            # Taken from renorm_heuristic()
            final_FIT = torch.sum(torch.cat([FIT_score.view(-1) for FIT_score in FIT_layerwise])).detach().cpu().numpy()

            curr_FIT += final_FIT

            return curr_FIT, FIT_layerwise
