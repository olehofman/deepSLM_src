

class ConvolutionalPart(object):
    """
    Class representing the Convolutional Part (CP) of a Convolutional Neural Network (CNN).
    """

    def __init__(self, conv_network):
        self.conv_network = conv_network

    def __repr__(self):
        return "ConvolutionalPart"

    def predict(self, X):
        return self.conv_network.predict(X)
