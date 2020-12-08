import timeit
import os

from deep_slm.convolutional.common.convolutional_network_builder import ConvolutionalNetworkBuilder

def mutate_cnn(conv_network, params):
    start_mutation = timeit.default_timer()

    add_neurons, add_layers = select_mutation_type(params['p_mutate_nodes'],
                                                   params['p_mutate_layers'],
                                                   params['random_state'])

    conv_builder = ConvolutionalNetworkBuilder(params['random_state'],
                                               sparseness=params['sparseness'],
                                               only_mutation_nodes=params['cp_only_mutation_nodes'],
                                               layer_types=params['layer_types'],
                                               layer_probs=params['layer_probs'],
                                               reporter=params['reporter'])
    if add_neurons:
        if add_layers:
            start_add_elements = timeit.default_timer()
            mutated_conv_network = conv_builder.add_mutation_nodes(conv_network,
                                                                   params['min_mutation_nodes'],
                                                                   params['max_mutation_nodes'],
                                                                   params)

            mutated_conv_network = conv_builder.add_mutation_layers(mutated_conv_network,
                                                                    params['min_mutation_layers'],
                                                                    params['max_mutation_layers'],
                                                                    params['min_mutation_nodes'],
                                                                    params['max_mutation_nodes'],
                                                                    params,
                                                                    double_mutation=True)

            add_elements_time = "{:.6f}".format(timeit.default_timer() - start_add_elements)
        else:
            start_add_elements = timeit.default_timer()
            mutated_conv_network = conv_builder.add_mutation_nodes(conv_network,
                                                                   params['min_mutation_nodes'],
                                                                   params['max_mutation_nodes'],
                                                                   params,
                                                                   n_custom_nodes_per_layer=
                                                                   params['cp_mutation_nodes_per_layer'])

            add_elements_time = "{:.6f}".format(timeit.default_timer() - start_add_elements)
    else:
        start_add_elements = timeit.default_timer()
        mutated_conv_network = conv_builder.add_mutation_layers(conv_network,
                                                                params['min_mutation_layers'],
                                                                params['max_mutation_layers'],
                                                                params['min_mutation_nodes'],
                                                                params['max_mutation_nodes'],
                                                                params,
                                                                double_mutation=False)

        add_elements_time = "{:.6f}".format(timeit.default_timer() - start_add_elements)
    start_evaluate_network = timeit.default_timer()

    mutated_conv_network.evaluate_network(params['X'], params['X_test'], mutation=True, recompute=params['recompute'])

    '''evaluation_time = "{:.6f}".format(timeit.default_timer() - start_evaluate_network)
    mutation_time = "{:.6f}".format(timeit.default_timer() - start_mutation)
    file_name = "cp_mutation_logger.csv"
    mode = 'a+' if os.path.exists(file_name) else 'w'
    file = open(file_name, mode)
    if mode == 'w':
        file.write(','.join(['mutation_time', 'evaluation', 'add_nodes_time\n']))
    file.write(','.join([mutation_time, evaluation_time, add_nodes_time, '\n']))
    file.close()'''

    return mutated_conv_network

def select_mutation_type(p_add_neurons, p_add_layers, random_state):

    # selects the mutation type according to p_add_layers and p_add_neurons
    add_neurons = random_state.rand() < p_add_neurons
    add_layers = random_state.rand() < p_add_layers

    if add_neurons == False and add_layers == False:
        if p_add_neurons == 0 and p_add_layers != 0:
            add_layers = True

        elif p_add_neurons != 0 and p_add_layers == 0:
            add_neurons = True

        else:
            random_int = random_state.randint(1,4)
            if random_int == 1:
                add_layers = True

            elif random_int == 2:
                add_neurons == True

            else:
                add_neurons = True
                add_layers = True

    return add_neurons, add_layers

