from numpy import zeros, matmul, concatenate
from sklearn.neural_network._base import softmax

from deep_slm.non_convolutional.common.activation_functions import ACTIVATION_FUNCTIONS_DICT
from deep_slm.utils import FLOAT_TYPE
import numpy as np


class NeuralNetwork:

    def __init__(self, input_layer_X, test_input_layer_X, input_layer, hidden_layers, output_layer, random_state, max_weight=0.1, max_bias=0.1, activation='relu'):
        self.input_layer_X = input_layer_X
        self.test_input_layer_X = test_input_layer_X
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        self.loss = None
        self.predictions = None
        self.test_predictions = None
        
        self.coefs = None
        self.intercepts = None
        self.last_hidden_layer_semantics = None

        self.max_bias_value = max_bias
        self.max_weight_value = max_weight

        self.random_state = random_state

        self.activation_func = activation

    def __repr__(self):
        return "NeuralNetwork"

    ####################################################################################################################

    def _build_hidden_coefs(self):

        max_weight = self.max_weight_value
        max_bias = self.max_bias_value

        # Initialize weights and baisses for the first hidden layer
        coefs_hl_0 = self.random_state.uniform(low=-max_weight,
                                               high=max_weight,
                                               size=(self.input_layer_X.shape[1], len(self.hidden_layers[0]))).astype(FLOAT_TYPE)

        intercepts_hl_0 = self.random_state.uniform(low=-max_bias,
                                                    high=max_bias,
                                                    size=(len(self.hidden_layers[0], ))).astype(FLOAT_TYPE)
        # save weights & biasses
        self.coefs = [coefs_hl_0]
        self.intercepts = [intercepts_hl_0]

        # Initialize weights and biasses for the other hidden layer
        for i in range(1, len(self.hidden_layers)):
            coefs_hl_i = self.random_state.uniform(low=-max_weight,
                                                   high=max_weight,
                                                   size=(
                                                   len(self.hidden_layers[i - 1]), len(self.hidden_layers[i]))).astype(
                FLOAT_TYPE)
            intercepts_hl_i = self.random_state.uniform(low=-max_bias,
                                                        high=max_bias,
                                                        size=(len(self.hidden_layers[i], ))).astype(FLOAT_TYPE)
            # save weights & biasses
            self.coefs.append(coefs_hl_i)
            self.intercepts.append(intercepts_hl_i)

    def _forward_pass_hidden_semantics(self):
        semantics = matmul(self.input_layer_X, self.coefs[0], dtype=FLOAT_TYPE) + self.intercepts[0]
        for i, neuron in enumerate(self.hidden_layers[0]):
            semantics[:, i] = neuron.activation_function(semantics[:, i]).astype(FLOAT_TYPE)

        for i in range(1, len(self.hidden_layers)):
            semantics = matmul(semantics, self.coefs[i], dtype=FLOAT_TYPE) + self.intercepts[i]
            for i, neuron in enumerate(self.hidden_layers[i]):
                semantics[:, i] = neuron.activation_function(semantics[:, i]).astype(FLOAT_TYPE)

        self.last_hidden_layer_semantics = semantics

    def compute_hidden_semantics(self):
        self._build_hidden_coefs()
        self._forward_pass_hidden_semantics()

    ####################################################################################################################
    
    def _update_input_layer_X(self, input_layer_X, test_input_layer_X):
        if input_layer_X is not None:
            self.input_layer_X = input_layer_X
            self.test_input_layer_X = test_input_layer_X

    def _update_hidden_coefs(self, sparseness=None):
        min_sparseness = sparseness['minimum_sparseness']
        max_sparseness = sparseness['maximum_sparseness']

        max_weight = self.max_weight_value
        max_bias = self.max_bias_value
        self.added_neurons = [len(i) for i in self.new_neurons]

        # Initialize mutation weights and biasses for the first hidden layer and concattenate
        coefs_hl_0 = self.random_state.uniform(low=-max_weight, high=max_weight, size=(self.input_layer_X.shape[1], self.added_neurons[0])).astype(FLOAT_TYPE)
        intercepts_kl_0 = self.random_state.uniform(low=-max_bias, high=max_bias, size=(self.added_neurons[0],)).astype(FLOAT_TYPE)

        dif = 0
        # if sparseness, apply sparseness to the coefs_hl_0 matrix
        if sparseness.get('sparse') is True:
            # apply sparseness only to parent connections
            if sparseness.get('fully_connect_mutation_nodes') is True:
                if dif > 0:
                    # slice parent connections and mask parent connections only
                    array_to_mask = coefs_hl_0[:-dif, :]
                    masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                    coefs_hl_0[:-dif, :] = masked_array
                else:
                    # if difference == 0, the only previous nodes are the parent connections -> skip this action then
                    pass

            # apply sparseness only within mutation nodes
            elif sparseness.get('only_mutation_nodes') is True:
                if dif > 0:
                    # set parent connections to zero (if there are any) and mask within mutation nodes
                    coefs_hl_0[:-dif, :] = zeros(coefs_hl_0[:-dif, :].shape, dtype=FLOAT_TYPE)
                    array_to_mask = coefs_hl_0[dif:, :]
                    masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                    coefs_hl_0[dif:, :] = masked_array
                else:
                    # if no previous mutation nodes, mask all incoming connections
                    array_to_mask = coefs_hl_0
                    masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                    coefs_hl_0 = masked_array

            # apply sparseness to all connections
            else:
                array_to_mask = coefs_hl_0
                masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                coefs_hl_0 = masked_array

        # else if no sparseness but forced only to connect with mutation nodes -> insert a zeros matrix for the parent connections
        elif sparseness.get('only_mutation_nodes') is True:
            if dif > 0:
                # insert zero matrix for parent connections
                coefs_hl_0[:-dif, :] = zeros(coefs_hl_0[:-dif, :].shape, dtype=FLOAT_TYPE)
            else:
                # if difference == 0, the only previous nodes are the parent connections -> skip this action then
                pass


        self.coefs[0] = coefs_hl_0
        self.intercepts[0] = intercepts_kl_0
        
        for i in range(1, len(self.hidden_layers)):
            new_neurons_previous_layer = self.added_neurons[i - 1]

            # create mutation nodes weights matrix and concatenate
            coefs_hl_i = self.random_state.uniform(low=-max_weight,high=max_weight, size=(self.coefs[i].shape[0], self.added_neurons[i])).astype(FLOAT_TYPE)
            # if sparseness, apply sparseness to the coefs_hl_i matrix
            if sparseness.get('sparse') is True:
                # apply sparseness only to the parent connections
                if sparseness.get('fully_connect_mutation_nodes') is True:
                    '''n new_neurons_previous_layer has to be > 0, otherwise there will be an error here'''
                    # slice parent connections and mask parent connections only
                    array_to_mask = coefs_hl_i[:-new_neurons_previous_layer, :]
                    masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                    coefs_hl_i[:-new_neurons_previous_layer, :] = masked_array

                # apply sparseness only within mutation nodes
                elif sparseness.get('only_mutation_nodes') is True:
                    # set parent connections to zero (if there are any) and apply sparseness only within mutation nodes
                    coefs_hl_i[:-new_neurons_previous_layer, :] = zeros(coefs_hl_i[:-new_neurons_previous_layer, :].shape, dtype=FLOAT_TYPE)
                    array_to_mask = coefs_hl_i[new_neurons_previous_layer:, :]
                    masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                    coefs_hl_i[new_neurons_previous_layer:, :] = masked_array

                # apply sparseness to all incoming connections
                else:
                    array_to_mask = coefs_hl_i
                    masked_array = self._mask_array(array_to_mask, min_sparseness, max_sparseness)
                    coefs_hl_i = masked_array

            # else if no sparseness but forced only to connect with mutation nodes -> insert a zeros matrix for the parent connections
            elif sparseness.get('only_mutation_nodes') is True:
                coefs_hl_i[:-new_neurons_previous_layer, :] = zeros(coefs_hl_i[:-new_neurons_previous_layer, :].shape, dtype=FLOAT_TYPE)

            # create mutation nodes biasses and concatenate
            intercepts_hl_i = self.random_state.uniform(low=-max_bias, high=max_bias, size=(self.added_neurons[i],)).astype(FLOAT_TYPE)

            self.coefs[i] = coefs_hl_i
            self.intercepts[i] = intercepts_hl_i

    def _forward_pass_update_hidden_semantics(self):
        new_neurons = self.added_neurons[0]
        semantics = matmul(self.input_layer_X, self.coefs[0], dtype=FLOAT_TYPE) + self.intercepts[0]
        for i, neuron in enumerate(self.hidden_layers[0][-new_neurons:]):
            semantics[:, i] = neuron.activation_function(semantics[:, i]).astype(FLOAT_TYPE)

        for i in range(1, len(self.hidden_layers)):
            new_neurons = self.added_neurons[i]
            semantics = matmul(semantics, self.coefs[i], dtype=FLOAT_TYPE) + self.intercepts[i]
            for i, neuron in enumerate(self.hidden_layers[i][-new_neurons:]):
                semantics[:, i] = neuron.activation_function(semantics[:, i]).astype(FLOAT_TYPE)

        self.last_hidden_layer_semantics = semantics

    def update_hidden_semantics(self, added_input_layer_X, added_test_input_layer_X, sparseness=None):
        self._update_input_layer_X(added_input_layer_X, added_test_input_layer_X)
        self._update_hidden_coefs(sparseness=sparseness)
        self._forward_pass_update_hidden_semantics()

    ####################################################################################################################

    def _forward_pass_output_semantics(self):
        self.predictions = matmul(self.last_hidden_layer_semantics, self.coefs[-1], dtype=FLOAT_TYPE) + self.intercepts[-1]
        self.last_hidden_layer_semantics = None
        return

    def _forward_pass_output_test_semantics(self):
        semantics = matmul(self.test_input_layer_X, self.coefs[0], dtype=FLOAT_TYPE) + self.intercepts[0]
        for i, neuron in enumerate(self.hidden_layers[0]):
            semantics[:, i] = neuron.activation_function(semantics[:, i]).astype(FLOAT_TYPE)

        for i in range(1, len(self.hidden_layers)):
            semantics = matmul(semantics, self.coefs[i], dtype=FLOAT_TYPE) + self.intercepts[i]
            for i, neuron in enumerate(self.hidden_layers[i]):
                semantics[:, i] = neuron.activation_function(semantics[:, i]).astype(FLOAT_TYPE)

        self.test_predictions = matmul(semantics, self.coefs[-1], dtype=FLOAT_TYPE) + self.intercepts[-1]
        semantics = None

    def _forward_pass_update_output_semantics(self):
        predictions = matmul(self.last_hidden_layer_semantics, self.coefs[-1], dtype=FLOAT_TYPE) + self.incremental_output_intercepts
        self.predictions = np.sum((predictions, self.predictions), axis=0)
        self.last_hidden_layer_semantics = None
        return

    def _forward_pass_update_test_output_semantics(self):
        new_neurons = self.added_neurons[0]
        incremental_semantics = matmul(self.test_input_layer_X, self.coefs[0], dtype=FLOAT_TYPE) + self.intercepts[0]
        for i, neuron in enumerate(self.hidden_layers[0][-new_neurons:]):
            incremental_semantics[:, i] = neuron.activation_function(incremental_semantics[:, i]).astype(FLOAT_TYPE)

        for i in range(1, len(self.hidden_layers)):
            new_neurons = self.added_neurons[i]
            incremental_semantics = matmul(incremental_semantics, self.coefs[i], dtype=FLOAT_TYPE) + self.intercepts[i]
            for i, neuron in enumerate(self.hidden_layers[i][-new_neurons:]):
                incremental_semantics[:, i] = neuron.activation_function(incremental_semantics[:, i]).astype(FLOAT_TYPE)

        incremental_semantics = matmul(incremental_semantics, self.coefs[-1], dtype=FLOAT_TYPE) + self.incremental_output_intercepts
        self.test_predictions = np.sum((incremental_semantics, self.test_predictions), axis=0)
        incremental_semantics = None
        return

    def update_output_semantics(self):
        ''''''
        '''The output coefs and intercepts were already assigned in the mutation_lbfgs() function'''
        #self._update_output_coefs()
        self._forward_pass_update_output_semantics()
        self._forward_pass_update_test_output_semantics()

    def calculate_output_semantics(self):
        ''''''
        '''The output coefs and intercepts were already assigned in the init_lbfgs() function'''
        #self._build_output_coefs()
        self._forward_pass_output_semantics()

        self._forward_pass_output_test_semantics()

    ####################################################################################################################

    def _mask_array(self, array_to_mask, min_sparsensss, max_sparseness):
        # store original size and shape
        _shape = array_to_mask.shape
        _size = array_to_mask.size # equivalent to n number of connections in matrix

        # define n connections to be mask
        sp = self.random_state.uniform(low=min_sparsensss, high=max_sparseness)
        n_conn = max(round((1 - sp) * _size), 1)
        n_to_mask = _size - n_conn

        # mask with zeros
        flattened_arr = array_to_mask.flatten()
        indices = self.random_state.choice(flattened_arr.size, size=n_to_mask, replace=False)
        flattened_arr[indices] = zeros((indices.size, ), dtype=FLOAT_TYPE)
        masked_array = flattened_arr.reshape(_shape)
        return masked_array

    ####################################################################################################################
    ####################################################################################################################
    ####### Not Used Methods
    ####################################################################################################################
    ####################################################################################################################

    def _update_output_coefs(self):
        self.coefs[-1] = concatenate((self.coefs[-1], zeros((self.added_neurons[-1], self.coefs[-1].shape[1]), dtype=FLOAT_TYPE)), axis=0)

        for neuron in self.output_layer:
            self.intercepts[-1][neuron.neuron_id] = neuron.bias
            #===================================================================
            # if neuron.bias_increment is not None:
            #     self.intercepts[-1][neuron.neuron_id] += neuron.bias_increment
            # else:
            #     self.intercepts[-1][neuron.neuron_id] = neuron.bias
            #
            #===================================================================
            for c in neuron.input_connections[-self.added_neurons[-1]:]:
                self.coefs[-1][c.from_neuron.neuron_id, neuron.neuron_id] = c.weight

    def _build_output_coefs(self):
        self.coefs.append(zeros((len(self.hidden_layers[-1]), len(self.output_layer)), dtype=FLOAT_TYPE))
        self.intercepts.append(zeros(len(self.output_layer), dtype=FLOAT_TYPE))

        for neuron in self.output_layer:
            self.intercepts[-1][neuron.neuron_id] = neuron.bias
            for c in neuron.input_connections:
                self.coefs[-1][c.from_neuron.neuron_id, neuron.neuron_id] = c.weight

    def calculate_semantics(self):
        """Calculate semantics of all hidden neurons and output neurons. At the end,
        it stores the obtained predictions in the neural network itself."""

        self.compute_hidden_semantics()
        self.calculate_output_semantics()

    def predict(self, X):
        """
        CIFAR-10 with iterations = 0 (just initialization), 100 neurons initialization:
        50k instances: speed-up of 44.787, 0.6046923 vs. 27.0823756
        10k instances: speed-up of 46.734, 0.10204330000000184 vs. 4.768940199999996

        CIFAR-10 with iterations = 2, 100 neurons initialization + 100 neurons mutation:
        50k instances: speed-up of 83.776, 2.002351499999989 vs. 167.750337
        10k instances: speed-up of 53.970, 0.34136030000000517 vs. 18.423392199999995

        CIFAR-10 with iterations = 0 (just initialization), 500 neurons initialization:
        50k instances: speed-up of 50.832, 2.652495999999985 vs. 134.83273770000002
        10k instances: speed-up of 48.729, 0.5089987000000065 vs. 24.80340590000003
        """

        predictions = matmul(X, self.coefs[0], dtype=FLOAT_TYPE) + self.intercepts[0]
        for i, neuron in enumerate(self.hidden_layers[0]):
            predictions[:, i] = neuron.activation_function(predictions[:, i]).astype(FLOAT_TYPE)

        for i in range(1, len(self.hidden_layers)):
            predictions = matmul(predictions, self.coefs[i], dtype=FLOAT_TYPE) + self.intercepts[i]
            for i, neuron in enumerate(self.hidden_layers[i]):
                predictions[:, i] = neuron.activation_function(predictions[:, i]).astype(FLOAT_TYPE)

        predictions = matmul(predictions, self.coefs[-1], dtype=FLOAT_TYPE) + self.intercepts[-1]

        return softmax(predictions).astype(FLOAT_TYPE)

        # self.input_layer_X = X
        # self._forward_pass_hidden_semantics()
        # self._forward_pass_output_semantics()
        # self.input_layer_X = None
        # return softmax(self.predictions)

    def incremental_output_semantics_update(self, num_last_connections):
        """Sums a given partial_semantics' value to current semantics' value in the
        output layer and, consequently, updates the predictions emitted by the neural
        network. These partial_semantics result from the addition of new (usually
        hidden) neurons to the neural network.

        Parameters
        ----------
        partial_semantics : array of shape (num_samples,)
            Partial semantics to be added arising from the addition of new neurons to
            the neural network.
        """
        
        for i, output_neuron in enumerate(self.output_layer):
            self.predictions[:, i] = output_neuron.incremental_semantics_update(num_last_connections)
 
    def load_input_neurons(self, X):
        """Loads new input data on the input layer"""
        for neuron, input_data in zip(self.input_layer, X.T):
            neuron.semantics = input_data

    def get_hidden_neurons(self):
        """Returns a list containing all hidden neurons."""
        neurons = list()
        [neurons.extend(hidden_neurons) for hidden_neurons in self.hidden_layers]
        return neurons

    def count_connections(self):
        """Determines the number of connections on the neural network.

        Returns
        -------
        number_connections : int
            The number of connections this neural network contains.
        """

        def _count_connections(layer):
            return sum([len(neuron.input_connections) for neuron in layer])

        counter = _count_connections(self.get_hidden_neurons())
        # Count connections between last hidden layer and the output layer:
        counter += _count_connections(self.output_layer)

        return counter

    def get_topology(self):
        """Creates a dictionary containing the number of hidden layers, hidden
        neurons and connections on the neural network.

        Returns
        -------
        topology : dict
            Dictionary containing information regarding the neural network's
            architecture.
        """
        return {
            "number of input neurons": len(self.input_layer),
            "number of hidden_layers": len(self.hidden_layers),
            "number of hidden_neurons": len(self.get_hidden_neurons()),
            'number of hidden_neurons per layer': [len(layer) for layer in self.hidden_layers],
            "number of output neurons": len(self.output_layer),
            "number of connections": self.count_connections()
        }

    def get_number_hidden_layers(self):
        """Determines the number of hidden layers of the neural network.

        Returns
        -------
        number_hidden_layers : int
            Number of hidden layers present in the neural network.
        """
        return len(self.hidden_layers)

    def clear_semantics(self):
        """Clears semantics from entire neural network."""
        [input_neuron.clear_semantics() for input_neuron in self.input_layer]
        if hasattr(self, 'mutation_input_layer'):
            [input_neuron.clear_semantics() for input_neuron in self.mutation_input_layer]
        [hidden_neuron.clear_semantics() for hidden_neuron in self.get_hidden_neurons()]
        [output_neuron.clear_semantics() for output_neuron in self.output_layer]

        self.predictions = None
        self.input_layer_X = None
        self.last_hidden_layer_semantics = None
    
    def clear_hidden_semantics(self):
        for hl in self.hidden_layers:
            for hn in hl:
                hn.clear_semantics()

    def get_loss(self):
        """Returns current neural network's loss value."""
        return self.loss

    def update_loss(self, loss):
        """Overrides current neural network's loss value. Usually, this occurs when new
        input data enters the neural network and semantics are recalculated.

        Parameters
        ----------
        loss : float
            New loss value for input data.
        """
        self.loss = loss

    def override_predictions(self, predictions):
        self.predictions = predictions.copy()

    def get_predictions(self):
        """Returns the predictions currently stored in the neural network."""
        return self.predictions.copy()

    def get_last_n_neurons(self, num_last_neurons):
        """Returns the last N neurons from the last hidden layer.

        Parameters
        ----------
        num_last_neurons : int
            Number of neurons to be retrieved from last hidden layer.
        """
        last_layer = self.hidden_layers[-1]
        return last_layer[-num_last_neurons:]

    def get_number_last_hidden_neurons(self):
        """Returns the number of neurons in the last hidden layer."""
        return len(self.hidden_layers[-1])

    def extend_hidden_layer(self, layer_index, new_neurons):
        """Adds a set of newly created neurons to a given hidden layer of the
        neural network.

        Parameters:
        ----------
        layer_index : int
            Index of hidden layer that will receive the new neurons.

        new_neurons : array of shape (num_neurons,)
            Neurons to be added on the specified hidden layer.
        """
        self.hidden_layers[layer_index].extend(new_neurons)
