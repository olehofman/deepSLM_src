

class NonConvolutionalPart:
    """
    Class representing the Non-Convolutional Part (NCP) of a Convolutional Neural Network (CNN).
    """
    
    def __init__(self, nn):
        self.nn = nn

    def __repr__(self):
        return "Non-ConvolutionalPart"
    
    def clear_semantics(self):
        self.nn.clear_semantics()
    
    def predict(self, X):
        return self.nn.predict(X)
