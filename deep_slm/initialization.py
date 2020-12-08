from numpy import zeros, ones, empty, empty_like, where, concatenate, append as np_append
from sklearn.linear_model import LinearRegression

from numpy import concatenate

from deep_slm.non_convolutional.lbfgs import LBFGS

from deep_slm.utils import FLOAT_TYPE

from .cnn import ConvolutionalNeuralNetwork
from .convolutional.common.convolutional_network_builder import ConvolutionalNetworkBuilder
from .convolutional.convolutional_part import ConvolutionalPart
from .non_convolutional.common.neural_network_builder import NeuralNetworkBuilder
from .non_convolutional.components.input_neuron import InputNeuron
from .non_convolutional.non_convolutional_part import NonConvolutionalPart


def create(parameters):
    """
    Creates a random CNN.
    """

    if parameters['only_ncp']:

        ncp = create_ncp(None, parameters, feed_original_X=True)

        return ConvolutionalNeuralNetwork(None, ncp,
                                          feed_original_X=parameters['feed_original_X'],
                                          only_ncp=parameters['only_ncp'])

    else:

        cp = create_cp(parameters)

        ncp = create_ncp(cp, parameters, feed_original_X=parameters['feed_original_X'])

        return ConvolutionalNeuralNetwork(cp, ncp,
                                          feed_original_X=parameters['feed_original_X'],
                                          only_ncp=parameters['only_ncp'])


def create_cp(params):

    if params['reporter']:
        print('Initialize Convolutional Part...')

    conv_builder = ConvolutionalNetworkBuilder(params['random_state'],
                                               sparseness=params['sparseness'],
                                               only_mutation_nodes=params['cp_only_mutation_nodes'],
                                               layer_types=params['layer_types'],
                                               layer_probs=params['layer_probs'],
                                               reporter=params['reporter'])

    if params['custom_cn']:
        conv_network = conv_builder.build_custom_conv_network(params['input_shape'], params['cn_structure'],
                                                              params['cp_nodes_per_layer'], params=params,
                                                              layer_params=params['layer_params'])

    else:
        conv_network = conv_builder.build_random_conv_network(params['min_layers'], params['max_layers'],
                                                              params['min_nodes'], params['max_nodes'],
                                                              params['input_shape'], params)

    conv_network.evaluate_network(params['X'], params['X_test'], mutation=False)

    return ConvolutionalPart(conv_network)


def create_ncp(cp, parameters, feed_original_X=True):

    if parameters['reporter']:
        print('Initialize Non Convolutional Part...')

    # input_neuron_id = 0
    input_layer_X = None
    #
    # input_layer = list()
    if feed_original_X:
        if parameters['reporter']:
            print('\tFeeding original X')
        original_X = parameters['X']
        channels = original_X.shape[3]
        for i in range(channels):
            X = original_X[:, :, :, i]
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
            
            if input_layer_X is None:
                input_layer_X = X
            else:
                input_layer_X = concatenate((input_layer_X, X), axis=1)
            
            # for _ in X.T:
            #     input_layer.append(InputNeuron(input_neuron_id, None))
            #     #===============================================================
            #     # input_layer.append(InputNeuron(input_neuron_id, input_data))
            #     #===============================================================
            #     input_neuron_id += 1

    if not parameters['only_ncp']:
        input_layer_X = concatenate(cp.conv_network.output_layer.semantics, axis=1)

        # for node in cp.conv_network.output_layer.nodes:
        #     if input_layer_X is None:
        #         input_layer_X = node.semantics
        #     else:
        #         input_layer_X = concatenate((input_layer_X, node.semantics), axis=1)
        #
        #     for _ in node.semantics.T:
        #         input_layer.append(InputNeuron(input_neuron_id, None))
        #         #===============================================================
        #         # input_layer.append(InputNeuron(input_neuron_id, input_data))
        #         #===============================================================
        #         input_neuron_id += 1
        #
        #     if parameters['recompute']:
        #         del node.semantics

        if parameters['recompute']:
            del cp.conv_network.output_layer.semantics
            for node, tensor in zip(cp.conv_network.output_layer.nodes,  cp.conv_network.output_layer.tensors):
                del node
                del tensor
            del cp.conv_network.output_layer.nodes
            del cp.conv_network.output_layer.tensors

    if parameters['only_ncp']:
        test_input_layer_X = None
        original_test_X = parameters['X_test']
        channels = original_test_X.shape[3]
        for i in range(channels):
            X = original_test_X[:, :, :, i]
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
            if test_input_layer_X is None:
                test_input_layer_X = X
            else:
                test_input_layer_X = concatenate((test_input_layer_X, X), axis=1)
    else:
        test_input_layer_X = concatenate(cp.conv_network.output_layer.test_semantics, axis=1)

        if parameters['recompute']:
            del cp.conv_network.output_layer.test_semantics

    random_state = parameters['random_state']
    if parameters['custom_ncp']:
        number_hidden_layers = parameters['ncp_n_hidden_layers']
        number_hidden_neurons = parameters['ncp_nodes_per_layer']
    else:
        number_hidden_layers = random_state.randint(parameters['ncp_min_layers'], parameters['ncp_max_layers'] + 1)
        number_hidden_neurons = [random_state.randint(parameters['ncp_min_nodes'], parameters['ncp_max_nodes'] + 1)
                                 for _ in range(number_hidden_layers)]

    y = parameters['y']
    number_output_neurons = y.shape[1]

    maximum_bias_weight = parameters['ncp_max_bias_weight']
    maximum_neuron_connection_weight = parameters['ncp_max_connection_weight']

    activation_function = 'identity'
    sparseness = {'sparse': parameters['ncp_sparseness'],
                  'minimum_sparseness': parameters['ncp_min_sparseness'],
                  'maximum_sparseness': parameters['ncp_max_sparseness'],
                  'prob_skip_connection': 0}
    #===========================================================================
    # sparseness = { 'sparse': False, 'minimum_sparseness': 0, 'maximum_sparseness': 1, 'prob_skip_connection': 0}
    #===========================================================================
    hidden_activation_functions_ids = ['relu']
    nn_activation = parameters['ncp_activation']
    #===========================================================================
    # hidden_activation_functions_ids = ['tanh']
    #===========================================================================
    prob_activation_hidden_layers = 1
    if parameters['reporter']:
        print('\tCall to NeuralNetworkBuilder.generate_new_neural_network')

    input_layer = None
    nn = NeuralNetworkBuilder.generate_new_neural_network(input_layer_X, test_input_layer_X, number_hidden_layers, number_hidden_neurons,
                                                          number_output_neurons, maximum_neuron_connection_weight,
                                                          maximum_bias_weight, activation_function, input_layer,
                                                          random_state, sparseness, hidden_activation_functions_ids,
                                                          prob_activation_hidden_layers, nn_activation=nn_activation,
                                                          reporter=parameters['reporter'])

    nn.compute_hidden_semantics()
    
    if parameters['reporter']:
        print('\tCall to init_lbfgs')

    # init_lbgs appends output coefs and inercept to nn
    init_lbfgs(None, y, nn, random_state)
    
    if parameters['reporter']:
        print('\tCall to nn.calculate_output_semantics')
    
    nn.calculate_output_semantics()

    if parameters['ncp_clear_semantics']:
        nn.clear_hidden_semantics()
        if hasattr(nn, 'mutation_input_layer'):
            [input_neuron.clear_semantics() for input_neuron in nn.mutation_input_layer]

    return NonConvolutionalPart(nn)


def init_lbfgs(X, y, nn, random_state):
    
    n_samples = y.shape[0]
    n_neurons = nn.get_number_last_hidden_neurons()
    
    if nn.last_hidden_layer_semantics is None:
        hidden_semantics = zeros((n_samples, n_neurons))
        for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
            hidden_semantics[:, i] = hidden_neuron.get_semantics()
    else:
        hidden_semantics = nn.last_hidden_layer_semantics
    
    layer_units = [n_neurons, y.shape[1]]
    #===========================================================================
    # activations = [X]
    #===========================================================================
    activations = []
    activations.extend([hidden_semantics])
    activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
    deltas = [empty_like(a_layer) for a_layer in activations]
    coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
    intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
    
    solver = LBFGS()
    
    coef_init = zeros((layer_units[0], layer_units[1]))
    intercept_init = zeros(layer_units[1])
    coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state, coef_init=coef_init, intercept_init=intercept_init)

    coefs = coefs[-1].astype(FLOAT_TYPE)
    intercepts = intercepts[-1].astype(FLOAT_TYPE)

    nn.coefs.append(coefs)
    nn.intercepts.append(intercepts)

    '''for output_index, output_neuron in enumerate(nn.output_layer):
        for i in range(n_neurons):
            # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
            output_neuron.input_connections[-n_neurons + i].weight = coefs[i, output_index]
        
        # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
        output_neuron.bias = intercepts[output_index]
        # output_neuron.increment_bias(intercepts[output_index])'''


def calculate_ols(nn, num_last_neurons, target, compute_intercept=True):
        
        instances = target.shape[0]
        weights_to_compute = num_last_neurons
        partial_semantics = zeros((instances, weights_to_compute))
        
        # Get semantics of last hidden neurons (the same number of last hidden neurons as the number of learning steps to be computed)
        last_hidden_neurons = nn.get_last_n_neurons(num_last_neurons)
        for i, hidden_neuron in enumerate(last_hidden_neurons):
            partial_semantics[:, i] = hidden_neuron.get_semantics()
        
        for output_index, output_neuron in enumerate(nn.output_layer):
            
            output_neuron_delta_target = target[:, output_index]
            partial_semantics_n = partial_semantics.copy()
            output_neuron_delta_target_n = output_neuron_delta_target

            neuron_targets = target[:, output_index]
            neuron_target_class_1 = where(neuron_targets == 1)
            neuron_target_class_1_indices = neuron_target_class_1[0]
            neuron_target_class_1_count = neuron_target_class_1_indices.shape[0]
            neuron_target_class_0 = where(neuron_targets == 0)
            neuron_target_class_0_indices = neuron_target_class_0[0]
            neuron_target_class_0_count = neuron_target_class_0_indices.shape[0]
             
            sample_weights = ones(instances)
            class_1_weight = neuron_target_class_0_count / neuron_target_class_1_count
            sample_weights[neuron_target_class_1_indices] = class_1_weight
           
            reg = LinearRegression().fit(partial_semantics_n, output_neuron_delta_target_n, sample_weights)
            optimal_weights = np_append(reg.coef_.T, reg.intercept_)

            # Update connections with the learning step value:
            for i in range(num_last_neurons):
                output_neuron.input_connections[-num_last_neurons + i].weight = optimal_weights[i]
            
            if compute_intercept:
                output_neuron.increment_bias(optimal_weights[-1])
