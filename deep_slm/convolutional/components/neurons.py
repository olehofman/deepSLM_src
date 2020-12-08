class Neuron(object):

    def __init__(self, type):

        self.type = type


class InputNeuron(Neuron):

    def __init__(self, shape, tensor):

        super().__init__('Input')

        self.shape = shape
        self.tensor = tensor

class OutputNeuron(Neuron):

    def __init__(self, tensor):

        super().__init__('Output')

        self.tensor = tensor

class HiddenNeuron(Neuron):

    def __init__(self, keras_node, type, neuron_params):

        super().__init__(type)

        self.node = keras_node
        self.params = neuron_params

        self.semantics = []
        self.tensors = []


