"""haakoas"""

import numpy as np
import functions as funcs

from layer import Layer


class LayerSoftmax(Layer):
    """
    Layer-class containing the nodes in a layer and all their input-weights.
    It also stores the output-value of the nodes during forward pass.
    """
    def __init__(self, input_dimensions: int, num_nodes: int):
        super().__init__(input_dimensions, num_nodes)

    def forward_pass(self, input_values, minibatch=True):
        """
        Computes the output of the layer's nodes given their inputs and
        the activation function.
        """
        if minibatch:
            pass
        else:
            pass

        # We store the temporary values
        self.input_vals = input_values
        self.output_vals = LayerSoftmax.softmax(input_values)
        # And return the output (this layer is assummed to always be the last layer)
        return self.output_vals

    @staticmethod
    def softmax(input_values):
        """
        Computes the softmax of the inputs and returns the result.
        """
        pass
