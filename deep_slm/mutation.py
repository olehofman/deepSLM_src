import timeit

from numpy import concatenate

from .cnn import ConvolutionalNeuralNetwork
from .convolutional.convolutional_part import ConvolutionalPart
from .convolutional.mutation import mutate_cnn
from .non_convolutional.common.neural_network_builder import NeuralNetworkBuilder
from .non_convolutional.components.input_neuron import InputNeuron
from .non_convolutional.mutation import mutate_hidden_layers
from .non_convolutional.non_convolutional_part import NonConvolutionalPart


def mutate(parent, parameters):
    """
    Mutates a given CNN.
    """
    if parameters['only_ncp']:
        if parameters['reporter']:
            print('Mutate Non Convolutional Part...')
        start_time = timeit.default_timer()
        child_ncp, times = mutate_ncp(parent, None, parameters)
        ncp_time = "{:.4f}".format(timeit.default_timer() - start_time)
        child_cp, cp_time = None, None
        #print("NCP Time:\t{}".format(ncp_time))

    else:
        if parameters['reporter']:
            print('\nMutate Convolutional Part...')
        start_time = timeit.default_timer()
        child_cp = mutate_cp(parent, parameters)
        cp_time = "{:.2f}".format(timeit.default_timer() - start_time)

        if parameters['reporter']:
            print('Mutate Non Convolutional Part...')
        start_time = timeit.default_timer()
        child_ncp, times = mutate_ncp(parent, child_cp, parameters)
        ncp_time = "{:.2f}".format(timeit.default_timer() - start_time)
        #print("CP Time:\t{}\tNCP Time:\t{}".format(cp_time, ncp_time))

    # if False:
    #     '''In depth profile of Mutation Times'''
    #     profile_times = [ncp_time] + times
    #     profile_times = ",".join(profile_times)
    #     file_name = "ncp_mutation_time_05.csv"
    #     file = open(file_name, 'a+')
    #     file.write(profile_times + '\n')
    #     file.close()

    return ConvolutionalNeuralNetwork(child_cp, child_ncp, feed_original_X=parameters['feed_original_X'],
                                      only_ncp=parameters['only_ncp']), cp_time, ncp_time


def mutate_cp(parent, parameters):
    child_conv_network = mutate_cnn(parent.cp.conv_network, parameters)
    return ConvolutionalPart(child_conv_network)


def mutate_ncp(parent, child_cp, parameters):
    """
    output_layer is the custom output layer object
    
    output_layer.nodes return a list with all node objects (and thus the semantics of each node can be accessed by node.semantics )
    
    output_layer.tensors returns a list with all TensorFlow.Tensors, if you want to implement a predict call for the deep-SLM to asses the generalization ability, 
        this will be the output where the new values will be outputted. 
    
    output_layer.last_mutation_nodes return is list with all output nodes added by the last mutation (with this keep track of the last added nodes), 
        with this attribute you can determine for the deep-SLM how many nodes have been added by the previous mutation, and from these nodes the input semantics 
        can be obtained for incremental evaluation of the non-deep SLM 
    
    output_layer.last_mutation_tensors returns the corresponding tensors for these nodes (we need this for constructing the predict call)
    """

    if not parameters['one_child_keep_child']:
        child_nn = NeuralNetworkBuilder.clone_neural_network(parent.ncp.nn)
    else:
        child_nn = parent.ncp.nn

    if parameters['ncp_fully_connect_mutation'] or parameters['ncp_only_mutation_nodes']:
        child_nn.mutation_input_layer = []

    added_input_layer_X = None

    if not parameters['only_ncp']:
        # neuron_id = len(child_nn.input_layer)

        added_input_layer_X = concatenate(child_cp.conv_network.output_layer.mutation_semantics, axis=1)

        if parameters['recompute']:
            del child_cp.conv_network.output_layer.mutation_semantics
            for node, tensor in zip(child_cp.conv_network.output_layer.mutation_nodes,  child_cp.conv_network.output_layer.mutation_tensors):
                del node
                del tensor
            del child_cp.conv_network.output_layer.mutation_nodes
            del child_cp.conv_network.output_layer.mutation_tensors

        # for node in child_cp.conv_network.output_layer.mutation_nodes:
        #
        #     if added_input_layer_X is None:
        #         added_input_layer_X = node.semantics
        #     else:
        #         added_input_layer_X = concatenate((added_input_layer_X, node.semantics), axis=1)
        #
        #     for _ in node.semantics.T:
        #         new_input_neuron = InputNeuron(neuron_id, None)
        #         #===============================================================
        #         # new_input_neuron = InputNeuron(neuron_id, input_data)
        #         #===============================================================
        #         neuron_id += 1
        #         if parameters['ncp_fully_connect_mutation'] or parameters['ncp_only_mutation_nodes']:
        #             child_nn.mutation_input_layer.append(new_input_neuron)
        #         child_nn.input_layer.append(new_input_neuron)
        #     if parameters['recompute']:
        #         del node.semantics

    if parameters['only_ncp']:
        added_test_input_layer_X = None
        original_test_X = parameters['X_test']
        channels = original_test_X.shape[3]
        for i in range(channels):
            X = original_test_X[:, :, :, i]
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
            if added_test_input_layer_X is None:
                added_test_input_layer_X = X
            else:
                added_test_input_layer_X = concatenate((added_test_input_layer_X, X), axis=1)
    else:
        added_test_input_layer_X = concatenate(child_cp.conv_network.output_layer.mutation_test_semantics, axis=1)

    if parameters['recompute'] and not parameters['only_ncp']:
        del child_cp.conv_network.output_layer.mutation_test_semantics


    random_state = parameters['random_state']
    learning_step = 'optimized'
    sparseness = { 'sparse': parameters['ncp_sparseness'],
                   'minimum_sparseness': parameters['ncp_min_sparseness'],
                   'maximum_sparseness': parameters['ncp_max_sparseness'],
                   'fully_connect_mutation_nodes' : parameters['ncp_fully_connect_mutation'],
                   'only_mutation_nodes' : parameters['ncp_only_mutation_nodes'],
                   'min_output_sparseness' : parameters['min_output_sparseness'],
                   'max_output_sparseness' : parameters['max_output_sparseness'],
                   'prob_skip_connection': 0}

    #===========================================================================
    # sparseness = { 'sparse': False, 'minimum_sparseness': 0, 'maximum_sparseness': 1, 'prob_skip_connection': 0}
    #===========================================================================

    maximum_new_neurons_per_layer = parameters['ncp_max_mutation_nodes']
    minimum_new_neurons_per_layer = parameters['ncp_min_mutation_nodes']

    maximum_bias_weight = parameters['ncp_max_bias_weight']
    maximum_neuron_connection_weight = parameters['ncp_max_connection_weight']

    X = None
    #X = parameters['X']
    y = parameters['y']
    global_preds = parent.get_predictions()
    delta_target = y - global_preds
    hidden_activation_functions_ids = [parameters['ncp_activation']]
    prob_activation_hidden_layers = 1
    child_nn, times = mutate_hidden_layers(added_input_layer_X, added_test_input_layer_X, X, y, child_nn, random_state, learning_step, sparseness, maximum_new_neurons_per_layer, maximum_neuron_connection_weight, maximum_bias_weight, delta_target, global_preds, hidden_activation_functions_ids, prob_activation_hidden_layers, params=parameters, minimum_new_neurons_per_layer=minimum_new_neurons_per_layer)

    if parameters['ncp_clear_semantics']:
        child_nn.clear_hidden_semantics()
        if parameters['ncp_only_mutation_nodes'] and not parameters['only_ncp']:
            # [input_neuron.clear_semantics() for input_neuron in child_nn.input_layer]
            [input_neuron.clear_semantics() for input_neuron in child_nn.mutation_input_layer]

    return NonConvolutionalPart(child_nn), times
