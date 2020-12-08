import os
from sys import argv, exit
import time

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import column_or_1d
from tensorflow.keras.datasets import mnist

from deep_slm.deep_slm import DeepSLM
from deep_slm.utils import secure_random_state
import numpy as np


def get_parameters(X_train, y_train, random_state, seed):
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = column_or_1d(y_train.ravel(), warn=True)

    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_train)
    classes = label_binarizer.classes_

    y_train_original = y_train.copy()
    y_train = label_binarizer.transform(y_train)


    parameters = {
        'use_subset' : True,
        'log': False,
        'reporter': True,
        'iterations': 2,
        'X': X_train,
        'y': y_train.astype(float),
        'input_shape': X_train.shape[1:],
        'random_state': random_state,
        'seed': seed,
        ################################
        ################################
        'only_ncp': False,
        'ncp_only_mutation_nodes': False,
        'cp_only_mutation_nodes': True,  # if True -> connect only to previous mutation_nodes
        'feed_original_X': True,
        'ncp_fully_connect_mutation': True,
        ################################
        'recompute': True,  # Not implemented yet
        'ncp_clear_semantics': False,  # Only possible when ncp_n_hidden_layers == 1
        ################################
        ################################
        # Initialize with custom CN ctructure
        'custom_cn': True,
        'cn_structure': ['Conv2D'],
        'nodes_per_layer': [1],
        'layer_params': None,
        ################################

        # Initialization Parameters
        'min_layers': 9,
        'max_layers': 9,
        'min_nodes': 1,
        'max_nodes': 1,

        # Mutation Parameters
        'p_mutate_nodes': 1,
        'p_mutate_layers': 0,
        'min_mutation_nodes': 10,
        'max_mutation_nodes': 10,
        'min_mutation_layers': 1,
        'max_mutation_layers': 1,


        # Conv Parameters
        'conv_kernel_dims': (3, 3, 1),
        'conv_stride': (1, 1, 1),
        'conv_padding': 'valid',  # random #same
        'conv_activation': 'tanh',  # random #tanh #sigmoid #linear
        'conv_p_identity': 0,
        'conv_init': 'Random_Uniform',  # Glorot_Uniform #LeCun_Uniform #He_Uniform
        'bias_init': 'Random_Uniform',  # Glorot_Uniform #LeCun_Uniform #He_Uniform
        'conv_init_max': 0.1,
        'bias_init_max': 0.1,

        # Pool Parameters
        'pool_kernel_dims': (2, 2, 1),
        'pool_stride': (2, 2, 2),
        'pool_padding': 'valid',  # random #same

        # Conv Network Builder Params
        'sparseness': False,
        'sparseness_range': (0.99, 0.6),
        'skip_connections': False,  # Not implemented yet
        'skip_connections_range': 1,
        'layer_types': ['Conv2D', 'MaxPool2D', 'AvgPool2D'],
        'layer_probs': [0.5, 0.25, 0.25],
        'only_2D': True,  # Not implemented yet
        'merge_mode': True,  # Not implemented yet
        'vary_params': False,  # Not implemented yet
        'homogeneous_mode': True,  # Not implemented yet
        # # # ###########################
        # # # ###########################

        # NCP Parameters
        ################################

        # Initialization Parameters
        'ncp_min_layers': 1,
        'ncp_max_layers' : 1,
        'ncp_min_nodes': 10,
        'ncp_max_nodes': 10,

        # Mutation Parameters
        'ncp_max_mutation_layers': None,
        'ncp_min_mutation_nodes': 1,
        'ncp_max_mutation_nodes': 1,

        # Sparseness & weights initialization
        'ncp_activation': 'relu',
        'ncp_sparseness': True,
        'ncp_min_sparseness': 0.50,
        'ncp_max_sparseness': 0.51,
        'ncp_max_connection_weight': 0.1,
        'ncp_max_bias_weight': 0.1,

        # Initialize with custom NCP ctructure
        'custom_ncp': True,
        'ncp_n_hidden_layers': 1,
        'ncp_nodes_per_layer':[100],
        'ncp_mutation_nodes_per_layer': [100],

        'one_child_keep_child': True
    }
    parameters['y_original'] = y_train_original.astype(float)
    parameters['label_binarizer'] = label_binarizer
    parameters['classes'] = classes

    exit_code = 0
    if sum(parameters['layer_probs']) != 1:
        exit_code = 1
    elif parameters['only_ncp'] and not parameters['feed_original_X']:
        exit_code = 2
    #elif parameters['only_ncp'] and parameters['ncp_fully_connect_mutation']:
        #exit_code = 3
    elif parameters['sparseness'] and parameters['cp_only_mutation_nodes']:
        exit_code = 4
    elif len(parameters['cn_structure']) != len(parameters['nodes_per_layer']):
        exit_code = 5
    elif parameters['ncp_n_hidden_layers'] != len(parameters['ncp_nodes_per_layer']):
        exit_code = 6
    elif parameters['ncp_n_hidden_layers'] > 1 and parameters['ncp_clear_semantics'] and not parameters['ncp_only_mutation_nodes']:
        exit_code = 7
    elif parameters['ncp_clear_semantics'] and not parameters['ncp_only_mutation_nodes']:
        exit_code = 8
    elif any([True for layer in parameters['cn_structure'] if layer not in parameters['layer_types']]):
        exit_code = 9
    elif parameters['ncp_n_hidden_layers'] != len(parameters['ncp_mutation_nodes_per_layer']):
        exit_code = 10

    if exit_code > 0:
        print('Invalid Parameter Settings: Exit Code {}'.format(exit_code))
        exit()

    return parameters


def write_csv_header(file_name, parameters):
    file = open(file_name, 'w')
    for key in parameters.keys():
        if all([key != 'X', key != 'y', key != 'random_state', key != 'label_binarizer', key != 'classes',
                 key != 'y_original', key != 'use_subset', key != 'log', key != 'iterations', key != 'only_2D',
                 key !=  'recompute', key != 'merge_mode', key != 'vary_params', key != 'homogeneous_mode']):

            file.write(
                "{}, {}\n".format(key, str(parameters[key]) if type(parameters[key]) != str else parameters[key]))

        else:
            continue

    file.write(
        ','.join(
            ['seed', 'iteration', 'time', 'cp_time', 'ncp_time', 'cnn_rmse', 'cnn_log', 'cnn_acc', 'neuron_1_rmse',
             'neuron_2_rmse', 'neuron_3_rmse', 'neuron_4_rmse', 'neuron_5_rmse', 'neuron_6_rmse', 'neuron_7_rmse',
             'neuron_8_rmse', 'neuron_9_rmse', 'neuron_10_rmse', 'neuron_1_acc', 'neuron_2_acc', 'neuron_3_acc',
             'neuron_4_acc', 'neuron_5_acc', 'neuron_6_acc', 'neuron_7_acc', 'neuron_8_acc', 'neuron_9_acc',
             'neuron_10_acc\n']))

    file.close()

    return


def dataset_from_csv(base_path, dataset, header=None):
    from os.path import join, dirname
    from pandas import read_csv

    dataset_path = join(dirname(__file__), base_path + dataset + '.csv')
    dataset_path = dataset_path.replace('\\', '/')
    
    data = read_csv(filepath_or_buffer=dataset_path, header=header)
    data.dropna(inplace=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    X = X.to_numpy()
    y = y.to_numpy()
    
    return X, y


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    parameters = get_parameters(X_train, y_train, None, None)

    use_subset = parameters['use_subset']
    subset_size = 20

    if use_subset:
        # select subset if needed
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]

    scale = True
    if scale:
        X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

        X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1))
        X_test = scaler.transform(X_test)
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    try:
        initial_seed = int(argv[1])
    except IndexError:
        initial_seed = 95
        print("ARGV SEED NOT USED: using seed = {}".format(initial_seed))
    nr_seeds = 1

    log = parameters['log']
    if log:

        time_stamp = time.strftime("%d-%m-%y_%H:%M")
        file_name = "benchmark_test.csv"

        mode = 'a+' if os.path.exists(file_name) else 'w'
        if mode == 'w':
            write_csv_header(file_name, parameters)

        for seed in range(initial_seed, initial_seed + nr_seeds):
            # set all random seeds generators
            random_state = secure_random_state(seed, use_gpu=parameters['log'])

            parameters = get_parameters(X_train, y_train, random_state, seed)

            dslm = DeepSLM(parameters, filename=file_name, log=True)

            dslm.fit(X_train, parameters['y'])


    else:

        if False:
            '''In depth profile of Mutation Times'''

            file_name = "ncp_mutation_time_05.csv"

            file = open(file_name, 'w')

            file.write(','.join(['ncp_total_time, create_connect_neurons_time, create_neurons_time, connect_neurons_time, hidden_semantics_time, connect_layers_time, ls_time, incremental_time\n']))

            file.close()

        for seed in range(initial_seed, initial_seed + nr_seeds):
            # set all random seeds generators

            random_state = secure_random_state(seed, use_gpu=parameters['log'])

            parameters = get_parameters(X_train, y_train, random_state, seed)

            dslm = DeepSLM(parameters, reporter=parameters['reporter'])

            dslm.fit(X_train, parameters['y'])
