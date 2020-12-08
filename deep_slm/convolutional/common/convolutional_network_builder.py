import sys

from tensorflow.keras.initializers import RandomUniform, glorot_uniform, lecun_uniform, he_uniform
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, concatenate

from deep_slm.convolutional.components.layers import HiddenLayer, OutputLayer
from deep_slm.convolutional.common.convolutional_network import Convolutional_Network
from deep_slm.convolutional.components.neurons import InputNeuron, HiddenNeuron, OutputNeuron


class ConvolutionalNetworkBuilder(object):

    def __init__(self, random_state, skip_connections=False, sparseness=False, vary_params=False, only_2D=True,
                 homogeneous=True, layer_types=None, layer_probs=None, only_mutation_nodes=False, reporter=True):
        """"
        Parameters that define what type of convolutional network will be build
        Sparseness, 3D, skip-cpnnections and homogeneous are not yet implemented but they are here anyway for the idea..
        """
        self.only_2D = only_2D
        self.sparseness = sparseness
        self.homogeneous = homogeneous
        self.vary_params = vary_params
        self.random_state = random_state
        self.skip_connections = skip_connections
        self.only_mutation_nodes = only_mutation_nodes
        self.reporter = reporter

        if layer_types is None:
            if self.only_2D:
                # Set default layer types with probabilities
                self.layer_types = ['Conv2D', 'MaxPool2D', 'AvgPool2D']
                self.layer_probs = [0.5, 0.25, 0.25]
            else:
                # if 3D convolution and pooling
                pass
        else:
            # Set custom layer types with probabilities
            self.layer_types = layer_types
            self.layer_probs = layer_probs

        self.max_value = 2 ** 31

    ####################################################################################################################
    #### Building Functions
    ####################################################################################################################

    def build_random_conv_network(self, min_layers, max_layers, min_nodes, max_nodes, input_shape, params,
                                  mutation=False, mutation_tensor=None):

        if self.reporter:
            print('\tCall to ConvolutionalNetworkBuilder.build_random_conv_network')

        ################################################################################################################
        #### Define n hidden layers and nodes per layers
        ################################################################################################################

        n_hidden_layers = self.random_state.randint(min_layers, max_layers + 1)

        n_nodes_per_layer = [self.random_state.randint(min_nodes, max_nodes + 1) for _ in range(n_hidden_layers)]

        ################################################################################################################
        ### Define the structure of the network
        ################################################################################################################

        network_structure = self.generate_random_structure(n_hidden_layers, mutation=mutation)

        ################################################################################################################
        ### Define layer params
        ################################################################################################################

        layer_params = [self.generate_random_layer_params(
            layer, n_nodes_per_layer[n], params) for n, layer in enumerate(network_structure)]

        ################################################################################################################
        ### Build Custom Convolutional Network
        ################################################################################################################

        custom_network = self.build_custom_conv_network(input_shape,
                                                        network_structure,
                                                        n_nodes_per_layer,
                                                        layer_params=layer_params,
                                                        mutation=mutation,
                                                        mutation_tensor=mutation_tensor)
        return custom_network

    def build_custom_conv_network(self, input_shape, structure_, nodes_per_layer, layer_params=None, params=None,
                                  mutation=False, mutation_tensor=None):
        if self.reporter:
            print('\tCall to ConvolutionalNetworkBuilder.build_custom_conv_network')

        if layer_params is None:
            layer_params = [self.generate_random_layer_params(
                layer, nodes_per_layer[n], params) for n, layer in enumerate(structure_)]
        else:
            nodes_per_layer = [dict['n_nodes'] for dict in layer_params]

        if mutation:
            incoming_tensor = mutation_tensor
        else:
            input_neuron = self.initialize_input_neuron(input_shape)
            incoming_tensor = input_neuron.tensor

        hidden_layers = []
        if self.reporter:
            print('\t\tConvolutionalNetworkBuilder.build_hidden_layers()')
        if not self.skip_connections:
            for idx, layer_ in enumerate(structure_):
                layer_params[idx]['node_params']['kernel_size'] = self.check_input_shape(
                    layer_params[idx]['node_params']['kernel_size'], incoming_tensor.shape.as_list())

                new_hidden_layer = self.build_custom_layer(layer_, nodes_per_layer[idx], layer_params[idx])

                self.connect_layer(new_hidden_layer, incoming_tensor)

                incoming_tensor = new_hidden_layer.tensor

                hidden_layers.append(new_hidden_layer)
        else:
            # skip connetions implementation here
            pass

        if mutation:
            return hidden_layers
        else:

            output_layer = self.connect_output_layer(hidden_layers[-1].nodes)

            convolutional_network = Convolutional_Network(input_neuron, hidden_layers, output_layer,
                                                          reporter=self.reporter)

            return convolutional_network

    def build_custom_layer(self, layer_type, n_nodes, layer_params):

        if layer_type == 'Conv2D':

            new_layer = self.initialize_conv2d_layer(n_nodes, layer_params)

        elif layer_type == 'MaxPool2D':

            new_layer = self.initialize_maxpool2d_layer(layer_params)

        elif layer_type == 'AvgPool2D':

            new_layer = self.initialize_avgpool2d_layer(layer_params)

        else:

            pass

        return new_layer

    ####################################################################################################################
    #### Mutation Functions
    ####################################################################################################################

    def add_mutation_nodes(self, conv_network, min_nodes, max_nodes, params, n_custom_nodes_per_layer=None):

        if self.reporter:
            print('\tCall to ConvolutionalNetworkBuilder.add_mutation_nodes')

        if n_custom_nodes_per_layer is None:
            n_nodes_per_layer = [
                self.random_state.randint(min_nodes, max_nodes + 1) for _ in range(len(conv_network.hidden_layers))]
        else:
            n_nodes_per_layer = n_custom_nodes_per_layer

        if not self.skip_connections:

            for idx, layer in enumerate(conv_network.hidden_layers):

                previous_layer = conv_network.hidden_layers[idx - 1] if idx > 0 else conv_network.input

                if layer.type == 'Conv2D':
                    # define incoming connections

                    if self.sparseness and idx > 0:

                        incoming = self.select_sparse_connections(previous_layer, params['sparseness_range'],
                                                                  mutation=True)

                    else:

                        incoming = previous_layer.tensor if idx == 0 or not self.only_mutation_nodes else previous_layer.mutation_tensor

                    if self.homogeneous and not self.vary_params:

                        mutation_nodes = self.initialize_conv2d_neurons(
                            n_nodes_per_layer[idx], layer.params['node_params'], mutation=True)

                    else:
                        # heterogeneous and vary params implementation here
                        pass

                else:
                    # define incoming connections
                    incoming = previous_layer.tensor if idx == 0 else previous_layer.mutation_tensor

                    # incoming = previous_layer.mutation_tensor if previous_layer else conv_network.input_tensor

                    if self.homogeneous and not self.vary_params:

                        if layer.type == 'MaxPool2D':

                            mutation_nodes = self.initialize_maxpool2d_neurons(layer.params['node_params'])

                        elif layer.type == 'AvgPool2D':

                            mutation_nodes = self.initialize_avgpool2d_neurons(layer.params['node_params'])

                    else:
                        # heterogeneous and vary params implementation here
                        pass

                self.connect_mutation_nodes([mutation_nodes], incoming, layer)

                layer.mutation_nodes = [mutation_nodes]

                layer.nodes.append(mutation_nodes)

        else:
            # sparseness and skip connections implementation here
            pass

        self.connect_output_layer(layer.mutation_nodes, mutation=True, conv_network=conv_network)

        return conv_network

    def add_mutation_layers(self, conv_network, min_layers, max_layers, min_nodes, max_nodes, params,
                            double_mutation=False):
        print('\tCall to ConvolutionalNetworkBuilder.add_mutation_layers')

        '''n_layers = self.random_state.randint(min_layers, max_layers + 1)

        n_nodes_per_layer = [self.random_state.randint(min_nodes, max_nodes + 1) for _ in range(n_layers)]

        mutation_structure_ = self.generate_random_structure(n_layers, mutation=True)

        layer_params = [self.generate_random_layer_params(
            layer, n_nodes_per_layer[n], params) for n, layer in enumerate(mutation_structure_)]

        new_layers = self.build_custom_conv_network(None, mutation_structure_, n_nodes_per_layer,
                                                      layer_params=layer_params, mutation=True, incoming_tensor=incoming_)'''

        if not self.skip_connections:
            # define incoming connection
            if self.sparseness:

                incoming_ = self.select_sparse_connections(
                    conv_network.hidden_layers[-1], params['sparseness_range'], mutation=double_mutation)

            elif self.only_mutation_nodes:

                incoming_ = conv_network.hidden_layers[-1].mutation_tensor

            else:

                incoming_ = conv_network.hidden_layers[-1].tensor

        else:
            # skip conections implmentation here
            pass

        mutation_layers = self.build_random_conv_network \
            (min_layers, max_layers, min_nodes, max_nodes, None, params, mutation=True, mutation_tensor=incoming_)

        conv_network.hidden_layers.extend(mutation_layers)

        self.connect_output_layer(
            mutation_layers[-1].nodes, mutation=True, conv_network=conv_network, double_mutation=double_mutation)

        return conv_network

    ####################################################################################################################
    #### Connect Functions
    ####################################################################################################################

    def connect_node(self, node, incoming):

        tensor = node.node(incoming)

        node.tensors.append(tensor)

        return tensor

    def connect_layer(self, to_layer, incoming):
        if self.reporter:
            print('\t\tConvolutionalNetworkBuilder.connect_hidden_layer()')

        connected_tensors = [self.connect_node(node, incoming) for node in to_layer.nodes]

        tensor = concatenate(connected_tensors, axis=-1) if len(connected_tensors) > 1 else connected_tensors[0]

        to_layer.tensor = tensor

        return

    def connect_mutation_nodes(self, mutation_nodes, incoming, layer):
        if self.reporter:
            print('\t\tConvolutionalNetworkBuilder.connect_mutation_node()')
        if self.only_2D and not self.vary_params and self.homogeneous:
            tensors_ = [self.connect_node(node, incoming) for node in mutation_nodes]

            tensor_ = concatenate(tensors_, axis=-1) if len(tensors_) > 1 else tensors_[0]

            layer.tensor = concatenate([layer.tensor, tensor_], axis=-1)

            layer.mutation_tensor = tensor_

        return

    def connect_output_layer(self, output_nodes, mutation=False, conv_network=None, double_mutation=False):

        if self.reporter:
            print('\t\tConvolutionalNetworkBuilder.connect_output_layer()')
        # create output neurons
        output_tensors = [Flatten()(tensor) for node in output_nodes for tensor in node.tensors]

        output_neurons = [OutputNeuron(tensor) for tensor in output_tensors]

        if mutation:
            if double_mutation:
                # extend all
                conv_network.output_layer.nodes.extend(output_neurons)
                conv_network.output_layer.tensors.extend(output_tensors)

                conv_network.output_layer.mutation_nodes.extend(output_neurons)
                conv_network.output_layer.mutation_tensors.extend(output_tensors)
            else:
                # extend and assig new
                if hasattr(conv_network.output_layer, 'nodes'):
                    conv_network.output_layer.nodes.extend(output_neurons)
                    conv_network.output_layer.tensors.extend(output_tensors)

                conv_network.output_layer.mutation_nodes = output_neurons
                conv_network.output_layer.mutation_tensors = output_tensors
            return

        else:
            # if not mutation, initialize new output layer
            output_tensors = [Flatten()(tensor) for node in output_nodes for tensor in node.tensors]

            output_neurons = [OutputNeuron(tensor) for tensor in output_tensors]

            output_layer = OutputLayer(output_neurons, output_tensors)

            return output_layer

    ####################################################################################################################
    #### Sparse connections
    ####################################################################################################################

    def select_sparse_connections(self, incoming_layer, sparseness_range, mutation=False):

        if mutation:
            incoming_nodes = incoming_layer.nodes[:-len(incoming_layer.mutation_nodes)]

        else:
            incoming_nodes = incoming_layer.nodes

        min_sparse, max_sparse = sparseness_range[0], sparseness_range[1]

        sp = self.random_state.uniform(min_sparse, max_sparse)

        ec = max(round((1 - sp) * len(incoming_nodes)), 1)

        sparse_selection = self.random_state.choice(incoming_nodes, size=ec, replace=False).tolist()

        incoming_tensors = [tensor for node in sparse_selection for tensor in node.tensors]

        if mutation:
            incoming_tensors.extend([tensor for node in incoming_layer.mutation_nodes for tensor in node.tensors])

        return concatenate(incoming_tensors, axis=-1) if len(incoming_tensors) > 1 else incoming_tensors[0]

    ####################################################################################################################
    #### Layer Initializations
    ####################################################################################################################

    def initialize_maxpool2d_layer(self, layer_params):

        pool_neurons = [self.initialize_maxpool2d_neurons(layer_params['node_params'])]

        pool_layer = HiddenLayer(pool_neurons, 'MaxPool2D', layer_params)

        return pool_layer

    def initialize_avgpool2d_layer(self, layer_params):

        pool_neurons = [self.initialize_avgpool2d_neurons(layer_params['node_params'])]

        pool_layers = HiddenLayer(pool_neurons, 'AvgPool2D', layer_params)

        return pool_layers

    def initialize_conv2d_layer(self, n_nodes, layer_params):

        conv_neurons = [self.initialize_conv2d_neurons(n_nodes, layer_params['node_params'])]

        conv_layer = HiddenLayer(conv_neurons, 'Conv2D', layer_params)

        return conv_layer

    ####################################################################################################################
    #### Neurons Initializations
    ####################################################################################################################

    def initialize_input_neuron(self, input_shape):

        input_tensor = Input(input_shape)

        return InputNeuron(input_shape, input_tensor)

    def initialize_maxpool2d_neurons(self, node_params):

        pool_node = MaxPooling2D(node_params['kernel_size'], strides=node_params['stride'],
                                 padding=node_params['padding'])

        pool_neuron = HiddenNeuron(pool_node, 'MaxPool2D', node_params)

        return pool_neuron

    def initialize_avgpool2d_neurons(self, node_params):

        pool_node = AveragePooling2D(node_params['kernel_size'], strides=node_params['stride'],
                                     padding=node_params['padding'])

        pool_neuron = HiddenNeuron(pool_node, 'AvgPool2D', node_params)

        return pool_neuron

    def initialize_conv2d_neurons(self, n_nodes, node_params, mutation=False):

        if mutation:
            node_params['bias_init'].seed = self.random_state.randint(0, self.max_value)
            node_params['kernel_init'].seed = self.random_state.randint(0, self.max_value)

        if node_params['activation'] == 'random':
            activation_functions = ['relu', 'tanh', 'sigmoid']
            activation = self.random_state.choice(activation_functions)
        else:
            activation = node_params['activation']

        conv_node = Conv2D(n_nodes, node_params['kernel_size'], strides=node_params['stride'],
                           padding=node_params['padding'], kernel_initializer=node_params['kernel_init'],
                           activation=activation, use_bias=True,
                           bias_initializer=node_params['bias_init'])

        conv_neuron = HiddenNeuron(conv_node, 'Conv2D', node_params)

        return conv_neuron

    ####################################################################################################################
    #### Random Parameter Sampling
    ####################################################################################################################

    def generate_random_conv2d_params(self, params):

        kernel_size = (
            self.random_state.randint(3, params['conv_kernel_dims'][0] + 1),
            self.random_state.randint(3, params['conv_kernel_dims'][1] + 1))

        stride = (self.random_state.randint(1, params['conv_stride'][0] + 1),
                  self.random_state.randint(1, params['conv_stride'][1] + 1))

        if params['conv_p_identity'] > 0 and self.random_state.rand() < params['conv_p_identity']:
            activation = 'linear'
        elif params['conv_activation'] == 'random':
            activation = 'random'
            # activation_functions = ['relu',  'tanh', 'sigmoid']
            # activation = self.random_state.choice(activation_functions)
        else:
            activation = params['conv_activation']

        if params['conv_padding'] == 'random':
            padding_types = ['valid', 'same']
            padding = self.random_state.choice(padding_types)
        else:
            padding = params['conv_padding']

        if params['conv_init'] == 'random':
            pass
        elif params['conv_init'] == 'Random_Uniform':
            kernel_init = RandomUniform(
                minval=-params['conv_init_max'],
                maxval=params['conv_init_max'],
                seed=self.random_state.randint(0, self.max_value))
        elif params['conv_init'] == 'Glorot_Uniform':
            kernel_init = glorot_uniform(seed=self.random_state(0, self.max_value))
        elif params['conv_init'] == 'LeCun_Uniform':
            kernel_init = lecun_uniform(seed=self.random_state(0, self.max_value))
        elif params['conv_init'] == 'He_Uniform':
            kernel_init = he_uniform(seed=self.random_state(0, self.max_value))

        if params['bias_init'] == 'random':
            pass
        elif params['bias_init'] == 'Random_Uniform':
            bias_init = RandomUniform(
                minval=-params['bias_init_max'],
                maxval=params['bias_init_max'],
                seed=self.random_state.randint(0, self.max_value))
        elif params['bias_init'] == 'Glorot_Uniform':
            bias_init = glorot_uniform(seed=self.random_state.randint(0, self.max_value))
        elif params['bias_init'] == 'LeCun_Uniform':
            bias_init = lecun_uniform(seed=self.random_state.randint(0, self.max_value))
        elif params['bias_init'] == 'He_Uniform':
            bias_init = he_uniform(seed=self.random_state.randint(0, self.max_value))

        conv_node_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding,
                            'activation': activation, 'kernel_init': kernel_init, 'bias_init': bias_init}

        return conv_node_params

    def generate_random_pool2d_params(self, params):

        kernel_size = (
            self.random_state.randint(2, params['pool_kernel_dims'][0] + 1),
            self.random_state.randint(2, params['pool_kernel_dims'][1] + 1))

        stride = (self.random_state.randint(2, params['pool_stride'][0] + 1),
                  self.random_state.randint(2, params['pool_stride'][1] + 1))

        if params['pool_padding'] == 'random':

            padding_types = ['valid', 'same']

            padding = self.random_state.choice(padding_types)

        else:

            padding = params['pool_padding']

        pool_node_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}

        return pool_node_params

    def generate_random_layer_params(self, layer_type, n_nodes, parameters):

        layer_params = {'layer_type': layer_type, 'n_nodes': n_nodes}

        if self.only_2D:

            if layer_type == 'Conv2D':

                node_params = self.generate_random_conv2d_params(parameters)

            else:

                node_params = self.generate_random_pool2d_params(parameters)

        else:

            pass

        layer_params['node_params'] = node_params

        return layer_params

    def generate_random_structure(self, n_layers, mutation=False):

        if mutation:
            network_structure = [self.random_state.choice(
                self.layer_types, p=self.layer_probs) for _ in range(n_layers)]
        else:
            network_structure = ['Conv2D'] + [self.random_state.choice(
                self.layer_types, p=self.layer_probs) for _ in range(n_layers - 1)]

        return network_structure

    ####################################################################################################################
    #### Helper Functions
    ####################################################################################################################

    def check_input_shape(self, to_check, incoming_shape):

        smallest_dim = ()

        for idx, dim_ in enumerate(incoming_shape[1:-1]):
            min_dim_ = min(dim_, to_check[idx])

            smallest_dim = smallest_dim + (min_dim_,)

        return smallest_dim