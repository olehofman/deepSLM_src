

class Neuron:
    """Base class for different types of neural_network_components that compose a neural network.

    Warning: this class should not be used directly. Use derived classes instead.
    """
    __slots__ = 'neuron_id', 'bias', 'semantics'
    def __init__(self, neuron_id, bias, semantics):
        self.neuron_id = neuron_id
        self.bias = bias
        self.semantics = semantics

    def __repr__(self):
        return "Neuron"

    def clear_semantics(self):
        """Clears the semantics."""
        self.semantics = None

    def get_semantics(self):
        """Returns current semantics of the neuron."""
        return self.semantics
