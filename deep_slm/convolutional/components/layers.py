class Layer(object):

    def __init__(self, neurons, layer_type, layer_params):

        self.type = layer_type
        self.params = layer_params

        self.nodes = neurons
        self.mutation_nodes = []
        self.mutation_tensors = []
        self.tensor = None

class HiddenLayer(Layer):

    def __init__(self, neurons, type, layer_params):

        super().__init__(neurons, type, layer_params)

class OutputLayer(Layer):

    def __init__(self, neurons, tensors):

        super().__init__(neurons, 'Output', None)

        self.tensors = tensors

        del self.params
        del self.tensor
