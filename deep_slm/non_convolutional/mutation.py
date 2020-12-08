from copy import copy

from numpy import zeros, empty, empty_like, concatenate, sum

from deep_slm.utils import FLOAT_TYPE

from .common.neural_network_builder import NeuralNetworkBuilder
from .components.hidden_neuron import HiddenNeuron
from .components.input_neuron import InputNeuron

from .lbfgs import LBFGS


############################
#### Mutation operators ####
############################
def mutate_hidden_layers(added_input_layer_X, added_test_input_layer_X, X, y, neural_network, random_state, learning_step, sparseness, maximum_new_neurons_per_layer=3, maximum_neuron_connection_weight=1.0, maximum_bias_weight=1.0,
                         delta_target=None, global_preds=None, hidden_activation_functions_ids=None, prob_activation_hidden_layers=None, time_print=False, params=False, minimum_new_neurons_per_layer=1):
    """
    Function that changes a given neural network's topology by possibly
    adding neurons on each one of the hidden layers.

    Parameters
    ----------
    neural_network : NeuralNetwork
        Neural network to be changed.

    random_state : RandomState instance
        A Numpy random number generator.

    learning_step : float or {'optimized'}
        Strategy to calculate the weight for connections in the last
        hidden layer's neurons to the output layer.

    sparseness : dict
        Dictionary containing information regarding neurons' connections,
        namely sparseness and the existence of skip connections (keys:
        'sparse', 'minimum_sparseness', 'maximum_sparseness', and
        'prob_skip_connection').

    maximum_new_neurons_per_layer : int
        Maximum number of neurons that can be added to a hidden layer.

    maximum_neuron_connection_weight : float
        Maximum value a weight connection between two neurons can have.

    maximum_bias_weight : float
        Maximum value a bias of a neuron can have.

    delta_target : array of shape (num_samples,), optional
        Array containing the distance of neural network's predictions
        to target values. Required if learning_step is set as 'optimized'.

    learning_step_function : function, optional, default None
        Optional function used to calculate optimized learning step
        for the neuron. The available functions are: pinv and lstsq
        from numpy package, and pinv, pinv2 and lstsq from scipy package. ,

    hidden_activation_functions_ids : array of shape (num_functions,), optional
        Names of activation functions that can be used in hidden layers.

    prob_activation_hidden_layers : float, optional
        Probability of the neurons on hidden layers have an activation
        function associated.

    Returns
    -------
    mutated_neural_network : NeuralNetwork
    """
    
    """ measuring time in mutate_hidden_layers """
    import timeit
    start_time = timeit.default_timer()
    hidden_semantics_time = 0
    
    # Add between 0 up to 'maximum_new_neurons_per_layer' neurons in each layer:
    #===========================================================================
    # new_neurons_per_layer = [random_state.randint(50, 50 + 1) for _ in range(neural_network.get_number_hidden_layers())]
    #===========================================================================

    if params['custom_ncp']:
        new_neurons_per_layer = params['ncp_mutation_nodes_per_layer']
    else:
        new_neurons_per_layer = [random_state.randint(minimum_new_neurons_per_layer, maximum_new_neurons_per_layer + 1)
                                 for _ in range(neural_network.get_number_hidden_layers())]

    #===========================================================================
    # new_neurons_per_layer = [random_state.randint(1, maximum_new_neurons_per_layer) for _ in range(neural_network.get_number_hidden_layers())]
    #===========================================================================

    # The last hidden layer needs to receive at least 1 new neuron:
    if new_neurons_per_layer[-1] == 0:
        new_neurons_per_layer[-1] = 1

    # Auxiliary list containing neurons to be filtered in skip connections:
    if sparseness.get('prob_skip_connection') > 0:
        neurons_for_skip_connections = copy(neural_network.input_layer)
        neurons_for_skip_connections.extend(neural_network.get_hidden_neurons())
    else:
        neurons_for_skip_connections = None

    # Auxiliary list that will contain references for new hidden neurons created for each layer:
    added_new_neurons = list()

    start_create_connect_neurons = timeit.default_timer()
    for i, number_neurons in enumerate(new_neurons_per_layer):
        new_hidden_neurons = NeuralNetworkBuilder.create_hidden_neurons(number_neurons=number_neurons,
                                                                        random_state=random_state, level_layer=i,
                                                                        maximum_bias_weight=maximum_bias_weight,
                                                                        hidden_activation_functions_ids=hidden_activation_functions_ids,
                                                                        prob_activation_hidden_layers=prob_activation_hidden_layers,
                                                                        neuron_id_start=len(neural_network.hidden_layers[i]))
        neural_network.extend_hidden_layer(layer_index=i, new_neurons=new_hidden_neurons)
        added_new_neurons.append(new_hidden_neurons)
#       for i, number_neurons in enumerate(new_neurons_per_layer):
#
#         if number_neurons > 0:
#
#             # Get new hidden layer to extend:
#
#             start_create_neurons = timeit.default_timer()
#             new_hidden_neurons = NeuralNetworkBuilder.create_hidden_neurons(number_neurons=number_neurons,
#                                                                             random_state=random_state, level_layer=i,
#                                                                             maximum_bias_weight=maximum_bias_weight,
#                                                                             hidden_activation_functions_ids=hidden_activation_functions_ids,
#                                                                             prob_activation_hidden_layers=prob_activation_hidden_layers,
#                                                                             neuron_id_start=len(neural_network.hidden_layers[i]))
#             create_neurons_time = "{:.4f}".format(timeit.default_timer() - start_create_neurons)
#             start_connect_neurons = timeit.default_timer()
#             # Establish connections with previous layer:
#             if i == 0:
#                 # Note: Previous layer is the input layer, so there are no skipped connections (although we might have sparseness):
#                 if any([params['ncp_only_mutation_nodes'], params['ncp_fully_connect_mutation']]) and not params['only_ncp']:
#                     if params['ncp_fully_connect_mutation']:
#                         #TO DO: ncp_only_mutation_nodes
#                         sparseness_copy = sparseness.copy()
#                         sparseness_copy['sparse'] = False
#
#                     else:
#                         sparseness_copy = sparseness.copy()
#
#                     NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
#                                                         previous_layer=neural_network.mutation_input_layer,
#                                                         maximum_neuron_connection_weight=maximum_neuron_connection_weight,
#                                                         random_state=random_state, sparseness=sparseness_copy,
#                                                         neurons_for_skip_connections=None)
#
#                     if not params['ncp_only_mutation_nodes']:
#                         not_new_mutation_input_nodes = list(
#                             set(neural_network.input_layer) - set(neural_network.mutation_input_layer))
#
#                         NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
#                                                             previous_layer=not_new_mutation_input_nodes,
#                                                             maximum_neuron_connection_weight=maximum_neuron_connection_weight,
#                                                             random_state=random_state, sparseness=sparseness,
#                                                             neurons_for_skip_connections=None)
#
#                 else:
#                     NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
#                                                         previous_layer=neural_network.input_layer,
#                                                         maximum_neuron_connection_weight=maximum_neuron_connection_weight,
#                                                         random_state=random_state, sparseness=sparseness,
#                                                         neurons_for_skip_connections=None)
#             else:
#                 # Filter neurons for skip connections:
#                 if neurons_for_skip_connections:
#                     skip_connections_set = list(
#                         filter(lambda x: isinstance(x, InputNeuron), neurons_for_skip_connections) if i == 1 else
#                         list(filter(lambda x: isinstance(x, InputNeuron) or
#                                               (isinstance(x, HiddenNeuron) and
#                                                x.level_layer < i - 1), neurons_for_skip_connections)))
#                 else:
#                     skip_connections_set = None
#
#                 if params['ncp_only_mutation_nodes'] or params['ncp_fully_connect_mutation']:
#                     if params['ncp_fully_connect_mutation']:
#                         #TO DO: ncp_only_mutation_nodes
#                         sparseness_copy = sparseness.copy()
#                         sparseness_copy['sparse'] = False
#
#                     else:
#                         sparseness_copy = sparseness.copy()
#
#                     NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
#                                                         previous_layer=added_new_neurons[-1],
#                                                         maximum_neuron_connection_weight=maximum_neuron_connection_weight,
#                                                         random_state=random_state, sparseness=sparseness_copy,
#                                                         neurons_for_skip_connections=skip_connections_set)
#
#                     if not params['ncp_only_mutation_nodes']:
#                         not_new_mutation_nodes = list(set(neural_network.hidden_layers[i - 1]) - set(added_new_neurons[-1]))
#                         NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
#                                                             previous_layer=not_new_mutation_nodes,
#                                                             maximum_neuron_connection_weight=maximum_neuron_connection_weight,
#                                                             random_state=random_state, sparseness=sparseness,
#                                                             neurons_for_skip_connections=skip_connections_set)
#                 else:
#                     NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
#                                                         previous_layer=neural_network.hidden_layers[i - 1],
#                                                         maximum_neuron_connection_weight=maximum_neuron_connection_weight,
#                                                         random_state=random_state, sparseness=sparseness,
#                                                         neurons_for_skip_connections=skip_connections_set)
#
#                 # Starting at second hidden layer (i > 0), all neurons must be connected with at least 1 new neuron added to the previous layer:
#                 for neuron in new_hidden_neurons:
#                     if not any(connection.is_from_previous_layer for connection in neuron.input_connections):
#                         print("PREVIOUS LAYER CONNECTION MISSING")
#                         # Warning: previous_hidden_layer_index can be None if none of previous layers received new neurons during this mutation.
#                         #=======================================================
#                         # previous_hidden_layer_index = get_closest_positive_number_index(new_neurons_per_layer, i - 1)
#                         #=======================================================
#
#                         """TO DO: Shouldnt this be previous_hidden_layer_index=i-1??"""
#                         previous_hidden_layer_index = 0
#
#                         if previous_hidden_layer_index:
#                             NeuralNetworkBuilder.connect_consecutive_mutated_layers(neuron, added_new_neurons[previous_hidden_layer_index],
#                                                                                     random_state, maximum_neuron_connection_weight)
#
# #===============================================================================
# #             # Calculate semantics for new hidden neurons:
# #             hidden_semantics_time_start_time = timeit.default_timer()
# #             [hidden_neuron.calculate_semantics() for hidden_neuron in new_hidden_neurons]
# #
# #             hidden_semantics_time = "{:.4f}".format(timeit.default_timer() - hidden_semantics_time_start_time)
# #===============================================================================
#
#             # Extend new hidden neurons to the respective hidden layer:
#             neural_network.extend_hidden_layer(layer_index=i, new_neurons=new_hidden_neurons)
#
#             # Store references of new hidden neurons:
#             added_new_neurons.append(new_hidden_neurons)
#
#         else:  # No new hidden neurons were added to this hidden layer:
#             added_new_neurons.append(list())
#
#         connect_neurons_time = "{:.4f}".format(timeit.default_timer() - start_connect_neurons)
#
#     create_connect_neurons_time = "{:.4f}".format(timeit.default_timer() - start_create_connect_neurons)
#     start_connect_layers = timeit.default_timer()
#
#     # Connect new hidden neurons from last hidden layer with output layer:
#     NeuralNetworkBuilder.connect_layers(layer_to_connect=neural_network.output_layer, previous_layer=neural_network.get_last_n_neurons(new_neurons_per_layer[-1]),
#                                         maximum_neuron_connection_weight=maximum_neuron_connection_weight, random_state=random_state,
#                                         sparseness={'sparse': False}, neurons_for_skip_connections=None)
#
#     connect_layers_time = "{:.4f}".format(timeit.default_timer() - start_connect_layers)

    ls_start_time = timeit.default_timer()

    # Calculate learning step for new neurons added in the last hidden layer:
    neural_network.new_neurons = added_new_neurons
    neural_network.update_hidden_semantics(added_input_layer_X, added_test_input_layer_X, sparseness=sparseness)

    #===========================================================================
    # mutation = mutation_lbfgs_all_neurons
    #===========================================================================
    mutation = mutation_lbfgs_new_neurons
    mutation(X, y, neural_network, new_neurons_per_layer[-1], random_state)
    
    ls_time = "{:.4f}".format(timeit.default_timer() - ls_start_time)

    # apply sparseness to output connections
    if params.get('ncp_sparseness_output') is True:
        min_sparseness = sparseness['min_output_sparseness']
        max_sparseness = sparseness['max_output_sparseness']
        array_to_mask = neural_network.coefs[-1][neural_network.added_neurons[-1]:, :]
        masked_array = neural_network._mask_array(array_to_mask, min_sparseness, max_sparseness)
        neural_network.coefs[-1][neural_network.added_neurons[-1]:, :] = masked_array



    incremental_start_time = timeit.default_timer()
    # Sum previous semantics to output layer:
    if mutation == mutation_lbfgs_all_neurons:
        neural_network.calculate_output_semantics()
    else:
        neural_network.update_output_semantics()
    incremental_time = "{:.4f}".format(timeit.default_timer() -  incremental_start_time)

    current_time = timeit.default_timer() - start_time
    #others_time = "{:.4f}".format(current_time - float(create_connect_neurons_time) - float(ls_time) - float(incremental_time) - float(hidden_semantics_time))
    if time_print:
        print('\n\t\tmutate_hidden_layers total time = %.3f seconds\n\t\t\tcalculate_learning_step = %.3f seconds\n\t\t\tincremental_output_semantics_update = %.3f seconds\n\t\t\thidden semantics computation = %.3f seconds\n\t\t\tothers = %.3f seconds' % (time, ls_time, incremental_time, hidden_semantics_time, others_time))
        print('\n\t\t\tcalculate_learning_step = %% of total mutation time %.2f\n\t\t\tincremental_output_semantics_update = %% of total mutation time %.2f\n\t\t\thidden semantics computation = %% of total mutation time %.2f\n\t\t\tothers = %% of total mutation time %.2f' % (ls_time / time * 100, incremental_time / time * 100, hidden_semantics_time / time * 100, others_time / time * 100))

    #times = [create_connect_neurons_time, create_neurons_time, connect_neurons_time, hidden_semantics_time, connect_layers_time, ls_time, incremental_time]

    # Return mutated neural_network:
    return neural_network, None #times


def mutation_lbfgs_new_neurons(X, y, nn, n_new_neurons, random_state):
    
    n_samples = y.shape[0]
    #===========================================================================
    # new_neurons = nn.get_last_n_neurons(n_new_neurons)
    # hidden_semantics = zeros((n_samples, n_new_neurons))
    # for i, hidden_neuron in enumerate(new_neurons):
    #     hidden_semantics[:, i] = hidden_neuron.get_semantics()
    #===========================================================================
    hidden_semantics = nn.last_hidden_layer_semantics[:, -n_new_neurons:]
    
    layer_units = [n_new_neurons, y.shape[1]]
    activations = []
    activations.extend([hidden_semantics])
    activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
    deltas = [empty_like(a_layer) for a_layer in activations]
    coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
    intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
    
    solver = LBFGS()
    
    """ zero-weight initialization for new neurons """
    coef_init = zeros((layer_units[0], layer_units[1]))

    intercept_init = nn.intercepts[-1].copy()

    # intercept_init = zeros(layer_units[1])
    # for output_index, output_neuron in enumerate(nn.output_layer):
    #     intercept_init[output_index] = output_neuron.bias

    fixed_weighted_input = zeros((n_samples, layer_units[1]))
    for output_index, output_bias in enumerate(nn.intercepts[-1]):
        fixed_weighted_input[:, output_index] = nn.predictions[:, output_index] - output_bias

    # fixed_weighted_input = zeros((n_samples, layer_units[1]))
    # for output_index, output_neuron in enumerate(nn.output_layer):
    #     fixed_weighted_input[:, output_index] = nn.predictions[:, output_index] - output_neuron.bias
    
    coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state, coef_init=coef_init, intercept_init=intercept_init, fixed_weighted_input=fixed_weighted_input)
    
    coefs = coefs[-1].astype(FLOAT_TYPE)
    intercepts = intercepts[-1].astype(FLOAT_TYPE)

    nn.incremental_output_intercepts = intercepts - nn.intercepts[-1]
    nn.intercepts[-1] = intercepts
    nn.coefs[-1] = coefs

    # for output_index, output_neuron in enumerate(nn.output_layer):
    #     for i in range(n_new_neurons):
    #         # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
    #         output_neuron.input_connections[-n_new_neurons + i].weight = coefs[i, output_index]
    #
    #     # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
    #     #output_neuron.bias = intercepts[output_index]
    #     output_neuron.increment_bias(intercepts[output_index] - output_neuron.bias)


def mutation_lbfgs_all_neurons(X, y, nn, n_new_neurons, random_state, init_bound=1):
    
    n_samples = y.shape[0]
    n_neurons = len(nn.hidden_layers[-1])
    #===========================================================================
    # hidden_semantics = zeros((n_samples, n_neurons))
    # for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
    #     hidden_semantics[:, i] = hidden_neuron.get_semantics()
    #===========================================================================
    hidden_semantics = nn.last_hidden_layer_semantics
    
    layer_units = [n_neurons, y.shape[1]]
    activations = []
    activations.extend([hidden_semantics])
    activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
    deltas = [empty_like(a_layer) for a_layer in activations]
    coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
    intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
    
    """ zero-weight initialization for new neurons """
    coef_init = zeros((layer_units[0], layer_units[1]))
    for output_index, output_neuron in enumerate(nn.output_layer):
        for i, connection in enumerate(output_neuron.input_connections[:-n_new_neurons]):
            coef_init[i][output_index] = connection.weight
            # print("i =", i, ", output_index =", output_index, ", weight =", connection.weight)
    
    intercept_init = zeros(layer_units[1])
    for output_index, output_neuron in enumerate(nn.output_layer):
        intercept_init[output_index] = output_neuron.bias
    
    solver = LBFGS()
    coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state, coef_init=coef_init, intercept_init=intercept_init)
    coefs = coefs[-1]
    intercepts = intercepts[-1]
    for output_index, output_neuron in enumerate(nn.output_layer):
        for i in range(n_neurons):
            # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
            output_neuron.input_connections[-n_neurons + i].weight = coefs[i, output_index]
        
        # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
        output_neuron.bias = intercepts[output_index]
