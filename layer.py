"""haakoas"""

import numpy as np
import functions as funcs


class Layer():
    """
    Layer-class containing the nodes in a layer and all their input-weights.
    It also stores the output-value of the nodes during forward pass.
    """
    def __init__(self,
                 input_dimensions: int,
                 num_nodes: int,
                 weight_range=(-0.5, 0.5)):
        self.input_dimension = input_dimensions  # Add bias weights
        self.num_nodes = num_nodes  # Number of nodes

        # weight_range contains the minimum and maximum parameters for the
        # randomized initial weights.
        self.weight_range = weight_range

        # initialize weights randomly
        self.weights = np.random.uniform(
            self.weight_range[0], self.weight_range[1],
            (self.input_dimension, self.num_nodes)).astype(np.float32)
        # TODO: Bias range
        self.biases = np.random.uniform(self.weight_range[0],
                                        self.weight_range[1],
                                        (self.num_nodes, )).astype(np.float32)

        # input_vals is a vector containing the output of each node before
        # sending the results through the activation function.
        self.input_vals = np.zeros(num_nodes + 1, dtype=np.float32).T
        # output_vals is a vector containing the output of each node after
        # sending the results through the activation function.
        # (i.e. output_vals = activation_func(input_vals))
        self.output_vals = np.zeros(num_nodes + 1, dtype=np.float32).T
        # next_layer contains a reference to the next layer in the network,
        # it is None if the current layer is the last layer.
        self.next_layer = None
        # Initialize jacobians, defined later in forward_pass
        self.j_z_sum = None
        self.j_z_y = None
        self.j_z_w = None
        self.j_z_wb = None

    def set_next_layer(self, next_layer: "Layer"):
        """
        Sets the reference to the next layer in order for the data to be sent
        directly to the next layer's forward_pass-method.
        """
        self.next_layer = next_layer

    def backward_pass(self, j_l_z):
        j_l_w = j_l_z * self.j_z_w
        j_l_wb = j_l_z * self.j_z_wb
        j_l_y = j_l_z @ self.j_z_y
        return j_l_w, j_l_wb, j_l_y

    def forward_pass(self,
                     input_values,
                     activation_func=funcs.sigmoid,
                     activation_func_der=funcs.sigmoid_der):
        """
        Computes the output of the layer's nodes given their inputs and
        the activation function.
        """
        # The input values are copied to avoid editing the list reference, then
        # an activation of 1 is appended to represent the activation of the
        # bias node.
        # if minibatch:
        # biases = np.ones((input_values.shape[0], 1))
        # values = np.concatenate((input_values, biases), axis=1)
        # raw_result = np.einsum("ij,kj->ki", self.weights.T, values)

        # values = np.append(input_values, [1]).astype(np.float32)
        raw_result = self.weights.T @ input_values + self.biases
        # We calculate the matrix multiplication between the weights and
        # the input values. The @-operator is equivalent to
        # np.matmul(self.weights, values) (or np.dot(a, b) if both a and b
        # are 2d-arrays)

        # We store the temporary values
        self.input_vals = raw_result
        self.output_vals = activation_func(raw_result).astype(np.float32)

        # Compute Jacobian matrices before we return value
        self.j_z_sum = np.diag(activation_func_der(raw_result))
        self.j_z_y = self.j_z_sum @ self.weights.T
        self.j_z_w = np.outer(input_values, np.diag(self.j_z_sum))
        self.j_z_wb = np.outer([1], np.diag(self.j_z_sum))

        # And return the output
        if self.next_layer is not None:
            return self.next_layer.forward_pass(self.output_vals)
        return self.output_vals
