from .neuron import Neuron


class InputNeuron(Neuron):
    """Class representing an neuron of the input layer in a neural network.

    Parameters
    ----------
    semantics : array of shape (num_samples,), optional

    Attributes
    ----------
    semantics : array of shape (num_samples,)
        Output vector of neuron. In this case, semantics is the vector with input data.

    bias : float
        Value for neuron's bias (inherited from Neuron).
    """
    def __init__(self, neuron_id, semantics):
        super().__init__(neuron_id=neuron_id, bias=None, semantics=semantics)

    def __repr__(self):
        return "InputNeuron"
