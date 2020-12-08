import os
from sys import argv, exit
import time

import tracemalloc

from sklearn.neural_network._base import log_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import column_or_1d
from tensorflow.keras.datasets import cifar10

from deep_slm.deep_slm import DeepSLM
from deep_slm.utils import secure_random_state, check_parameter_settings, write_csv_header, FLOAT_TYPE, \
    generate_LeNet5_params
import numpy as np


def get_parameters(X_train, y_train, X_test, y_test, random_state, seed):
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = column_or_1d(y_train.ravel(), warn=True)

    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_train)
    classes = label_binarizer.classes_

    y_train_original = y_train.copy().astype('int8')
    y_train = label_binarizer.transform(y_train).astype('int8')

    y_test_original = y_test.copy().astype('int8')
    y_test = label_binarizer.transform(y_test).astype('int8')

    parameters = {
        'use_subset': False,
        'log': True,
        'reporter': False,
        'iterations': 200,
        'population_size': 1,
        'X': X_train,
        'y': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'input_shape': X_train.shape[1:],
        'random_state': random_state,
        'seed': seed,
        'one_child_keep_child': True,
        ################################
        ################################
        'only_ncp': False,
        'ncp_only_mutation_nodes': True,
        'cp_only_mutation_nodes': False,  # if True -> connect only to previous mutation_nodes
        'feed_original_X': False,
        'ncp_fully_connect_mutation': False,
        ################################
        'recompute': True,  # Not implemented yet
        'ncp_clear_semantics': False,  # not relevant anymore # Only possible when ncp_n_hidden_layers == 1
        ################################
        ################################
        # Initialize with custom CN ctructure
        # 'custom_cn': True,
        # # 'cn_structure': ['Conv2D'],
        # # 'cp_nodes_per_layer': [10],
        # # 'layer_params': [generate_LeNet5_params([10, 10, 20, 20, 60], random_state)[0]],
        # # 'cp_mutation_nodes_per_layer': [10],
        'custom_cn': True,
        'cn_structure': ['Conv2D', 'MaxPool2D'],
        'cp_nodes_per_layer': [10, 10],
        'layer_params': generate_LeNet5_params([10, 10, 20, 20, 60], random_state)[0:2], # or None
        'cp_mutation_nodes_per_layer': [10, 10],  # or set to None to use 'min_mutation_nodes'''
        ################################

        # Initialization Parameters
        'min_layers': 9,
        'max_layers': 9,
        'min_nodes': 10,
        'max_nodes': 10,

        # Mutation Parameters
        'p_mutate_nodes': 1,
        'p_mutate_layers': 0,
        'min_mutation_nodes': 1,
        'max_mutation_nodes': 1,
        'min_mutation_layers': 1,
        'max_mutation_layers': 1,

        # Conv Parameters
        'conv_kernel_dims': (3, 3, 1),
        'conv_stride': (1, 1, 1),
        'conv_padding': 'valid',  # random #same
        'conv_activation': 'relu',  # random #tanh #sigmoid #linear
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
        'sparseness_range': (0.499, 0.5001),
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
        'ncp_max_layers': 1,
        'ncp_min_nodes': 10,
        'ncp_max_nodes': 10,

        # Mutation Parameters
        'ncp_max_mutation_layers': None,
        'ncp_min_mutation_nodes': 1,
        'ncp_max_mutation_nodes': 1,

        # Sparseness & weights initialization
        'ncp_activation': 'relu',
        'ncp_sparseness': False,
        'ncp_min_sparseness': 0.499,
        'ncp_max_sparseness': 0.501,
        'ncp_sparseness_output': False,
        'min_output_sparseness': 0.499,
        'max_output_sparseness': 0.501,
        'ncp_max_connection_weight': 0.1,
        'ncp_max_bias_weight': 0.1,

        # Initialize with custom NCP ctructure
        'custom_ncp': True,
        # 'ncp_n_hidden_layers': 2,
        # 'ncp_nodes_per_layer': [50, 25],
        # 'ncp_mutation_nodes_per_layer': [50, 25],

        'ncp_n_hidden_layers': 1,
        'ncp_nodes_per_layer': [50],
        'ncp_mutation_nodes_per_layer': [50],
    }

    '''parameters['y_original'] = y_train_original.astype(float)
    parameters['y_test_original'] = y_test_original.astype(float)'''

    parameters['y_original'] = y_train_original
    parameters['y_test_original'] = y_test_original
    parameters['label_binarizer'] = label_binarizer
    parameters['classes'] = classes

    """ TODO: testing some parameters 
    parameters['use_subset'] = True
    parameters['iterations'] = 200
    parameters['ncp_nodes_per_layer'] = [400]
    parameters['ncp_mutation_nodes_per_layer'] = [50]"""

    exit_code = check_parameter_settings(parameters)
    if exit_code > 0:
        '''check in utils.py what parameter settings combination caused the exit'''
        print('Invalid Parameter Settings: Exit Code {}'.format(exit_code))
        exit()

    return parameters


#### dataset_from_csv is in deep_slm.utils

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    parameters = get_parameters(X_train, y_train, X_test, y_test, np.random.RandomState(10), None)

    subset_size = 100
    use_subset = parameters['use_subset']
    if use_subset:
        # select subset if needed
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]

        X_test = X_test[:subset_size]
        y_test = y_test[:subset_size]

    select_first_n_classes = False
    if select_first_n_classes:
        '''selects n times one instance from each class for the test set'''
        n = 1

        X_train = np.concatenate((X_train, X_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)

        y_test = np.zeros((10 * n, 1))
        X_test = np.zeros((10 * n, 32, 32, 3))

        for i in range(n):
            class_ = 0
            while class_ != 9:
                # iterates over y_train until first instance of a class
                for idx, j in enumerate(y_train):
                    if j == class_:
                        y_test[class_] = j
                        X_test[class_] = X_train[idx]

                        y_train = np.delete(y_train, idx, axis=0)
                        X_train = np.delete(X_train, idx, axis=0)

                        class_ += 1
                        break

    scale = True
    if scale:

        X_train = X_train.reshape((X_train.shape[0], 32 * 32 * 3))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(FLOAT_TYPE)
        X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))

        X_test = X_test.reshape((X_test.shape[0], 32 * 32 * 3))
        X_test = scaler.transform(X_test).astype(FLOAT_TYPE)
        X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))

    try:
        initial_seed = int(argv[1])
    except IndexError:
        initial_seed = 95
        print("ARGV SEED NOT USED: using seed = {}".format(initial_seed))
    nr_seeds = 1

    log = parameters['log']
    if log:

        time_stamp = time.strftime("%d-%m-%y_%H:%M")
        file_name = "/content/gdrive/My Drive/master_thesis_colabs/benchmark_csv/30seed_benchmark/csv/DSLM/......csv"
        # file_name = "test_logger.csv"

        mode = 'a+' if os.path.exists(file_name) else 'w'
        if mode == 'w':
            write_csv_header(file_name, parameters)

        for seed in range(initial_seed, initial_seed + nr_seeds):
            # set all random seeds generators
            random_state = secure_random_state(seed, use_gpu=parameters['log'])

            del parameters
            parameters = get_parameters(X_train, y_train, X_test, y_test, random_state, seed)

            dslm = DeepSLM(parameters, filename=file_name, log=True, reporter=parameters['reporter'])

            dslm.fit(X_train, parameters['y'])

    else:
        # if False:
        #     '''In depth profile of Mutation Times'''
        #     file_name = "ncp_mutation_time_05.csv"
        #     file = open(file_name, 'w')
        #     file.write('ncp_total_time, ''create_connect_neurons_time, create_neurons_time, '
        #                'connect_neurons_time, hidden_semantics_time, connect_layers_time, '
        #                'ls_time, incremental_time\n')
        #     file.close()

        for seed in range(initial_seed, initial_seed + nr_seeds):
            # set all random seed generators
            random_state = secure_random_state(seed, use_gpu=parameters['log'])

            del parameters
            parameters = get_parameters(X_train, y_train, X_test, y_test, random_state, seed)

            dslm = DeepSLM(parameters, reporter=parameters['reporter'])

            tracemalloc.start()
            dslm.fit(X_train, parameters['y'])
            current, peak = tracemalloc.get_traced_memory()
            print(f"FINAL memory usage: {current / 10 ** 6}MB; Peak: {peak / 10 ** 6}MB")
            tracemalloc.stop()

            nn = dslm.best.ncp.nn
            loss = nn.get_loss()

            # '''y_pred_train = dslm.predict(X_train)
            # ce_loss = log_loss(parameters['y'], y_pred_train)
            # print('\nTrain CE: %.5f' % (ce_loss))
            #
            # y_pred_test = dslm.predict(parameters['X_test'], reporter=parameters['reporter'])
            #
            # ce_loss = log_loss(parameters['y_test'], y_pred_test)
            # # ce_loss = log_loss(parameters['label_binarizer'].transform(y_test), y_pred_test)
            # print('Test CE: %.5f' % (ce_loss))'''
