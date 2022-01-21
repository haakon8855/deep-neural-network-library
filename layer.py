"""Haakoas"""

import numpy as np
import activation_functions as af


class Layer():
    """
    Layer-class containing the nodes in a layer and all their input-weights.
    It also stores the output-value of the nodes during forward pass.
    """
    def __init__(self, input_dimensions: int, num_nodes: int, weight_range):
        self.input_dimension = input_dimensions + 1  # Add bias weights
        self.num_nodes = num_nodes  # Number of nodes
        self.weights = np.zeros((self.num_nodes, self.input_dimension),
                                dtype=np.float32)  # Create the weight matrix

        # input_vals is a vector containing the output of each node before
        # sending the results through the activation function.
        self.input_vals = np.zeros(num_nodes + 1, dtype=np.float32)
        # output_vals is a vector containing the output of each node after
        # sending the results through the activation function.
        # (i.e. output_vals = activation_func(input_vals))
        self.output_vals = np.zeros(num_nodes + 1, dtype=np.float32)
        # weight_range contains the minimum and maximum parameters for the
        # randomized initial weights.
        self.weight_range = weight_range
        # next_layer contains a reference to the next layer in the network,
        # it is None if the current layer is the last layer.
        self.next_layer = None

    def set_next_layer(self, next_layer: "Layer"):
        """
        Sets the reference to the next layer in order for the data to be sent
        directly to the next layer's forward_pass-method.
        """
        self.next_layer = next_layer

    def randomize_weights(self):
        """
        Randomizes the weights in the layer by creating a num_nodes x
        input_dim matrix and setting the values to random values in the
        specified range.
        """
        self.weights = np.random.uniform(
            self.weight_range[0], self.weight_range[1],
            (self.num_nodes, self.input_dimension)).astype(np.float32)

    def forward_pass(self, input_values, activation_func=af.relu):
        """
        Computes the output of the layer's nodes given their inputs and
        the activation function.
        """
        # The input values are copied to avoid editing the list reference, then
        # an activation of 1 is appended to represent the activation of the
        # bias node.
        values = np.append(input_values, [1]).astype(np.float32)
        # We calculate the matrix multiplication between the weights and
        # the input values. The @-operator is equivalent to
        # np.matmul(self.weights, values) (or np.dot(a, b) if both a and b
        # are 2d-arrays)
        raw_result = self.weights @ values
        # We store the temporary values
        self.input_vals = raw_result
        self.output_vals = activation_func(raw_result).astype(np.float32)
        # And return the output
        if self.next_layer is not None:
            return self.next_layer.forward_pass(self.output_vals)
        return self.output_vals
