from numpy import zeros, concatenate


class ConvolutionalNeuralNetwork:
    """
    Class representing a Convolutional Neural Network (CNN).
    
    CNN = Convolutional Part (CP) + Non-Convolutional Part (NCP)
    CNN = CP + NCP
    """

    def __init__(self, cp, ncp, feed_original_X=True, only_ncp=False):
        self.cp = cp
        self.ncp = ncp

        self.feed_original_X = feed_original_X
        self.only_ncp = only_ncp

    def __repr__(self):
        return "ConvolutionalNeuralNetwork"
    
    def set_feed_original_X(self, feed_original_X):
        self.feed_original_X = feed_original_X

    def _get_output_shape(self):
        instances = self.input_layer[0].get_semantics().shape[0]
        outputs = len(self.output_layer)
        return instances, outputs
    
    def calculate_output_semantics(self):
        """Calculate semantics of output layer without recalculating the semantics for
        all hidden layers."""
        
        # Compute semantics for output neuron(s) and store predictions
        self.predictions = zeros(self._get_output_shape())
        for i, output_neuron in enumerate(self.output_layer):
            self.predictions[:, i] = output_neuron.calculate_semantics()

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
        self.ncp.clear_semantics()

    def get_loss(self):
        """Returns current neural network's loss value."""
        return self.ncp.nn.get_loss()

    def update_loss(self, loss):
        """Overrides current neural network's loss value. Usually, this occurs when new
        input data enters the neural network and semantics are recalculated.

        Parameters
        ----------
        loss : float
            New loss value for input data.
        """
        self.ncp.nn.loss = loss

    def override_predictions(self, predictions):
        self.predictions = predictions.copy()

    def get_predictions(self):
        """Returns the predictions currently stored in the neural network."""
        return self.ncp.nn.get_predictions()

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

    def predict(self, X, reporter=False):
        import timeit
        method_start = timeit.default_timer()
        if reporter:
            print('\tCP predict start')
        start = timeit.default_timer()

        if not self.only_ncp:
            ncp_input = self.cp.predict(X)
        else:
            ncp_input = None

        time = timeit.default_timer() - start
        if reporter:
            print('\tCP predict end')
            print('\tCP predict time =', time)
        start = timeit.default_timer()

        if self.feed_original_X:
            original_input = X[:, :, :, 0].reshape((-1, X.shape[1] * X.shape[2]))
            channels = X.shape[3]
            for i in range(1, channels):
                channel_X = X[:, :, :, i]
                channel_X = channel_X.reshape((-1, channel_X.shape[1] * channel_X.shape[2]))
                original_input = concatenate((original_input, channel_X), axis=1)

            if ncp_input is not None:
                ncp_input = concatenate((original_input, ncp_input), axis=1)
            else:
                ncp_input = original_input

        time = timeit.default_timer() - start
        if reporter:
            print('\tFeed original X time =', time)
            print('\tNCP predict start')
        start = timeit.default_timer()

        predictions = self.ncp.predict(ncp_input)

        time = timeit.default_timer() - start
        if reporter:
            print('\tNCP predict end')
            print('\tNCP predict time =', time)
            print('\tClear semantics start')
        '''the nn.predict() does not store the semantics anymore (nn._forwardpass() does),
        the semantics must therefore NOT be cleared after this predict call'''
        # self.clear_semantics()
        return predictions
