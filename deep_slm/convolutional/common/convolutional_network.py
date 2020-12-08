import numpy as np

from tensorflow.keras import Model

class Convolutional_Network(object):

    def __init__(self, input, hidden_layers, output_layer, reporter=True):

        self.input = input

        self.input_tensor = input.tensor

        self.hidden_layers = hidden_layers

        self.output_layer = output_layer

        self.reporter = reporter

        self.model = None

        self.mutation_model = None

    def calculate_output_semantics(self, X, X_test, mutation=False):
        if self.reporter:
            print('\t\tConvolutionalNetwork.calculate_output_semantics()')

        output_semantics = self.mutation_model.predict(X) if mutation else self.model.predict(X)
        output_test_semantics = self.mutation_model.predict(X_test) if mutation else self.model.predict(X_test)

        return (output_semantics, output_test_semantics) if type(output_semantics) == list else ([output_semantics], [output_test_semantics])

    def save_output_semantics(self, semantics, test_semantics, mutation=False, recompute=True):
        if self.reporter:
            print('\t\tConvolutionalNetwork.save_output_semantics()')

        output_nodes = self.output_layer.mutation_nodes if mutation else self.output_layer.nodes

        if mutation:
            self.output_layer.mutation_semantics = semantics
            self.output_layer.mutation_test_semantics = test_semantics
            # if not recompute:
            #
            #     self.output_layer.semantics.extend(semantics)
        else:
            self.output_layer.semantics = semantics
            self.output_layer.test_semantics = test_semantics

        for idx, node in enumerate(output_nodes):
            node.semantics = semantics[idx]

        return

    def initialize_network(self, recompute=True, mutation=False):

        if self.reporter:
            print('\t\tConvolutionalNetwork.initialize_network()')

        if mutation:

                self.mutation_model = Model(inputs=self.input_tensor, outputs=self.output_layer.mutation_tensors)

        else:

            self.model = Model(inputs=self.input_tensor, outputs=self.output_layer.tensors)

        return

    def evaluate_network(self, X, X_test, mutation=False, recompute=True):
        if self.reporter:
            print('\tCall to ConvolutionalNetwork.evaluate_network')

        self.initialize_network(mutation=mutation, recompute=recompute)

        semantics, test_semantics = self.calculate_output_semantics(X, X_test, mutation=mutation)

        self.save_output_semantics(semantics, test_semantics, mutation=mutation, recompute=recompute)

        return

    def predict(self, X):
        if self.reporter:
            print('\tCall to ConvolutionalNetwork.predict()')

        if self.mutation_model is not None:

            predictions = self.mutation_model.predict(X)

        else:

            predictions = self.model.predict(X)

        if type(predictions) == list:
            predictions = np.concatenate(predictions, axis=-1)

        return predictions

    # def predict(self, X):
    #     if self.reporter:
    #         print('\tCall to ConvolutionalNetwork.predict()')
    #
    #     predictions = self.model.predict(X)
    #
    #     if type(predictions) == list:
    #         predictions = np.concatenate(predictions, axis=-1)
    #
    #     return predictions

    # def initialize_network(self, recompute=True, mutation=False):
    #
    #     if self.reporter:
    #         print('\t\tConvolutionalNetwork.initialize_network()')
    #
    #     if mutation:
    #
    #         if recompute:
    #
    #             self.mutation_model = Model(inputs=self.input_tensor, outputs=self.output_layer.mutation_tensors)
    #
    #         else:
    #             # a different model is needed if Recompute=False
    #             pass
    #
    #     self.model = Model(inputs=self.input_tensor, outputs=self.output_layer.tensors)
    #
    #     return


    # def save_output_semantics(self, semantics, mutation=False, recompute=True):
    #     if self.reporter:
    #         print('\t\tConvolutionalNetwork.save_output_semantics()')
    #
    #     output_nodes = self.output_layer.mutation_nodes if mutation else self.output_layer.nodes
    #
    #     if mutation:
    #
    #         self.output_layer.mutation_semantics = semantics
    #
    #         if not recompute:
    #
    #             self.output_layer.semantics.extend(semantics)
    #
    #     else:
    #
    #         self.output_layer.semantics = semantics
    #
    #     for idx, node in enumerate(output_nodes):
    #
    #         node.semantics = semantics[idx]
    #
    #     return

