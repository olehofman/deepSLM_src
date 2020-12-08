import os
import random
import warnings
from tensorflow.keras.initializers import RandomUniform

from numpy import sqrt, where, argmax
from sklearn.metrics import mean_squared_error, accuracy_score

import numpy as np

# Numpy's float dtype to use for storing connections weights and biasses
FLOAT_TYPE = 'float32'

def secure_random_state(seed, use_gpu=False):
    warnings.filterwarnings('ignore')
    # tf.disable_v2_behavior()
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.autograph.set_verbosity(1)

    """ TODO: tf.compat.v1 """
    # ===========================================================================
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # ===========================================================================

    random.seed(seed)
    np.random.seed(seed)
    # TENSORFLOW 2.0 tf.random.set_seed(seed)

    """ TODO: tf.compat.v1 """
    # ===========================================================================
    # tf.compat.v1.set_random_seed(seed)
    # if use_gpu:
    #     session_conf = tf.compat.v1.ConfigProto()
    # else:
    #     session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # with sess:
    #     pass
    # tf.compat.v1.keras.backend.set_session(sess)
    # tf.compat.v1.keras.backend.set_learning_phase(0)
    # ===========================================================================

    # print("GPU availability: {}".format(tf.config.experimental.list_physical_devices('GPU')))
    return np.random.RandomState(seed)

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def accuracy(y_true, y_pred):
    if y_true.ndim == 1:
        y_pred = where(y_pred >= 0.5, 1, 0)
    else:
        if y_true.shape[1] > 1:
            y_true = argmax(y_true, axis=1)

            y_pred_original = y_pred
            y_pred = argmax(y_pred, axis=1)
            from numpy import zeros
            d = where(y_true != y_pred)[0]
            incorrect_count = d.shape[0]
            correct_vs_incorrect = zeros((y_pred_original.shape[1], y_pred_original.shape[1]))
            for i in d:
                # ===============================================================
                # print('\ty_true[i] =', y_true[i])
                # ===============================================================
                correct = y_true[i]
                incorrect = y_pred[i]
                correct_vs_incorrect[correct, incorrect] += 1
                # ===============================================================
                # print('\ty_pred[i] =', y_pred_original[i])
                # print('\ty_pred incorrect target =', y_pred_original[i, correct])
                # print('\ty_pred other =', y_pred_original[i, incorrect])
                # print('\ty_pred dif =', y_pred_original[i, correct] - y_pred_original[i, incorrect])
                # print('\ti =', i)
                # ===============================================================
            # ===================================================================
            # print(correct_vs_incorrect)
            # correct_vs_incorrect /= incorrect_count
            # correct_vs_incorrect *= 100
            # print(' ', end='')
            # for k in range(y_pred_original.shape[1]):
            #     print('\t%d' % (k), end='')
            # print()
            # for i in range(y_pred_original.shape[1]):
            #     print(i, end='')
            #     for k in range(y_pred_original.shape[1]):
            #         print('\t%.2f' % (correct_vs_incorrect[i, k]), end='')
            #     print()
            # print()
            # ===================================================================
        else:
            y_true = y_true.ravel()
            y_pred = y_pred.ravel()
            y_pred = where(y_pred >= 0.5, 1, 0)

    return accuracy_score(y_true, y_pred, normalize=True)

def check_parameter_settings(parameters):
    if sum(parameters['layer_probs']) != 1:
        '''CP layer probabalities have to sum up to exactly 1'''
        return 1
    elif parameters['only_ncp'] and not parameters['feed_original_X']:
        '''if only NCP part is used, the original data / input has to feed into the NCP part'''
        return 2
    elif parameters['sparseness'] and parameters['cp_only_mutation_nodes']:
        '''CP sparseness cannot be applied when the CP mutation nodes have to fully connected'''
        return 3
    elif len(parameters['cn_structure']) != len(parameters['cp_nodes_per_layer']):
        '''the CP number nodes per layer has to be in accordance with the length of the custom CP network structure '''
        return 4
    elif parameters['ncp_n_hidden_layers'] != len(parameters['ncp_nodes_per_layer']):
        '''the NCP number nodes per layer has to be correspondent with the length of the custom NCP network structure'''
        return 5
    elif parameters['ncp_clear_semantics'] and not parameters['ncp_only_mutation_nodes']:
        '''the NCP semantics cannot be cleared when the NCP mutation nodes can receive incoming connections not only from the previous NCP mutation nodes, but also from the NCP parent network'''
        return 6
    elif any([True for layer in parameters['cn_structure'] if layer not in parameters['layer_types']]):
        '''the custom CP structure must solely be defined with layer types contained in CP layer types '''
        return 7
    elif parameters['ncp_n_hidden_layers'] != len(parameters['ncp_mutation_nodes_per_layer']):
        '''the NCP number of mutatio nodes per layer must be in accordance with the length of the NCP custom network structure'''
        return 8
    elif parameters['cp_mutation_nodes_per_layer'] is not None and len(parameters['cp_nodes_per_layer']) != len(
            parameters['cp_mutation_nodes_per_layer']):
        '''the CP number of mutation nodes per layer must be in accordance with the length of the custom CP network structure'''
        return 9
    elif parameters['cp_mutation_nodes_per_layer'] is not None and parameters['p_mutate_nodes'] != 1 and parameters[
        'p_mutate_layers'] > 0:
        "if the number of CP mutation nodes is defined, then no CP layers must be added to the CP part"
        return 10
    elif parameters['population_size'] > 1 and parameters['one_child_keep_child'] is True:
        "Cannot have 'keep child' and population size > 1 "
        return 11

    else:
        return 0

def write_csv_header(file_name, parameters):
    file = open(file_name, 'w')
    for key in parameters.keys():
        if all([key != 'X', key != 'y', key != 'random_state', key != 'label_binarizer', key != 'classes',
                key != 'X_test', key != 'y_test', key != 'y_original', key != 'y_test_original', key != 'use_subset',
                key != 'log', key != 'iterations', key != 'only_2D', key != 'recompute', key != 'merge_mode',
                key != 'vary_params', key != 'homogeneous_mode']):

            file.write(
                "{}, {}\n".format(key, str(parameters[key]) if type(parameters[key]) != str else parameters[key]))
        else:
            continue

    # column_names = ','.join(['seed', 'iteration', 'time', 'cp_time', 'ncp_time', 'train_rmse',
    #                          'test_rmse', 'train_ce_loss', 'train_accuracy', 'test_ce_loss', 'test_accuracy',
    #                          'train_binary_log_loss', 'train_binary_log_loss_fac10', 'train_log_loss_xtreme',
    #                          'train_hinge_loss_func', 'test_binary_log_loss', 'test_binary_log_loss_fac10',
    #                          'test_log_loss_xtreme', 'test_hinge_loss_func'])

    column_names = ','.join(['seed', 'iteration', 'time', 'cp_time', 'ncp_time', 'train_rmse',
                             'train_ce_loss', 'train_accuracy', 'test_ce_loss', 'test_accuracy'])


    file.write(column_names + '\n')
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

def generate_LeNet5_params(n_nodes_per_layer, random_state):
    params = [
        {'layer_type': 'Conv2D',
         'n_nodes': n_nodes_per_layer[0],
         'node_params': {'kernel_size': (5, 5),
                         'stride': (1, 1),
                         'padding': 'valid',
                         'activation': 'relu',
                         'kernel_init': RandomUniform(minval=-0.1, maxval=0.1, seed=random_state.randint(0, 2 ** 31)),
                         'bias_init': RandomUniform(minval=-0.1, maxval=0.1, seed=random_state.randint(0, 2 ** 31))}},

        {'layer_type': 'AvgPool2D',
         'n_nodes': n_nodes_per_layer[1],
         'node_params': {'kernel_size': (2, 2),
                         'stride': (2, 2),
                         'padding': 'valid'}},

        {'layer_type': 'Conv2D',
         'n_nodes': n_nodes_per_layer[2],
         'node_params': {'kernel_size': (5, 5),
                         'stride': (1, 1),
                         'padding': 'valid',
                         'activation': 'relu',
                         'kernel_init': RandomUniform(minval=-0.1, maxval=0.1, seed=random_state.randint(0, 2 ** 31)),
                         'bias_init': RandomUniform(minval=-0.1, maxval=0.1, seed=random_state.randint(0, 2 ** 31))}},

        {'layer_type': 'AvgPool2D',
         'n_nodes': n_nodes_per_layer[3],
         'node_params': {'kernel_size': (2, 2),
                         'stride': (2, 2),
                         'padding': 'valid'}},

        {'layer_type': 'Conv2D',
         'n_nodes': n_nodes_per_layer[4],
         'node_params': {'kernel_size': (5,5),
                         'stride': (1, 1),
                         'padding': 'valid',
                         'activation': 'relu',
                         'kernel_init': RandomUniform(minval=-0.1, maxval=0.1, seed=random_state.randint(0, 2 ** 31)),
                         'bias_init': RandomUniform(minval=-0.1, maxval=0.1, seed=random_state.randint(0, 2 ** 31))}}]

    return params


























