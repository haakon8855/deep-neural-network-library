"""haakon8855"""

import numpy as np
from utilities import Utilities as utils


class Layer:
    """
    Layer-class containing the nodes in a layer and all their input-weights.
    It also stores the output-value of the nodes during forward pass.
    """
    def __init__(self,
                 input_dimensions: int,
                 num_nodes: int,
                 lrate: float,
                 weight_range=(-0.5, 0.5),
                 bias_range=(-0.5, 0.5),
                 activation_func=utils.sigmoid,
                 activation_func_der=utils.sigmoid_der):
        self.input_dimensions = input_dimensions  # Add bias weights
        self.num_nodes = num_nodes  # Number of nodes
        self.lrate = lrate  # Layer-specific learning rate

        # weight_range contains the minimum and maximum parameters for the
        # randomized initial weights.
        self.weight_range = weight_range
        self.bias_range = bias_range

        # initialize weights randomly
        self.weights = np.random.uniform(
            self.weight_range[0], self.weight_range[1],
            (self.input_dimensions, self.num_nodes)).astype(np.float32)
        self.biases = np.random.uniform(self.bias_range[0], self.bias_range[1],
                                        (self.num_nodes, 1)).astype(np.float32)

        # Store activation function and its derivative function
        self.activation_func = activation_func
        self.activation_func_der = activation_func_der

        # input_values is a vector containing the output from the last layer
        # before multiplying the inputs with the current layer's weights
        # and adding biases.
        self.input_values = None
        # z_sum is a vector containing the output of each node before
        # sending the results through the activation function.
        self.z_sum = None
        # activations is a vector containing the output of each node after
        # sending the z_sum through the activation function.
        # (i.e. activations = activation_func(self.z_sum))
        self.activations = None
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
        """
        Computes the backward pass jacobian matrices and returns:
            J_l_w (delta for weights),
            J_l_wb (delta for bias weights),
            J_l_y (J_l_z for upstream layer)
        """
        # Compute diagonal vector for J_z_sum
        derivative = self.activation_func_der(self.z_sum)
        # J_z_sum is the matrix containing 'derivative' along its diagonal
        j_z_sum = np.eye(derivative.shape[1]) * derivative[:, np.newaxis, :]
        # J_z_y is matmul of J_z_sum and the transposed weight matrix
        j_z_y = j_z_sum @ self.weights.T

        # J_z_w is the outer product product of the inputs
        # and the diagonal of J_z_sum
        j_z_w = np.einsum('ij,ik->ijk', self.input_values, derivative)
        # Same goes for the bias weights
        j_z_wb = np.einsum('j,ik->ijk', [1], derivative)

        # J_l_w (weight gradients) is the multiplication between J_l_z and J_z_w
        j_l_w = j_l_z[:, np.newaxis, :] * j_z_w
        # Same goes for the bias weight gradient
        j_l_wb = j_l_z[:, np.newaxis, :] * j_z_wb
        # J_l_y is matmul of J_l_z and J_z_y
        # J_l_y is J_l_z for the upstream layer
        j_l_y = np.einsum('ij,ijk->ik', j_l_z, j_z_y)
        return j_l_w, j_l_wb, j_l_y

    def forward_pass(self, input_values):
        """
        Computes the output of the layer's nodes given their inputs and
        the activation function.
        """
        # Caching input_values, z_sum and activations allows for time saving
        # when performing the backward pass later.
        self.input_values = input_values
        self.z_sum = np.einsum("ij,kj->ki", self.weights.T,
                               self.input_values) + self.biases.reshape(1, -1)
        self.activations = self.activation_func(self.z_sum).astype(np.float32)

        # Return the downstream layer's return value from its forward pass
        if self.next_layer is not None:
            return self.next_layer.forward_pass(self.activations)
        # Return the activations from this layer if it is the last layer
        return self.activations

    def update_weights(self, delta_w, delta_b):
        """
        Updates weights and biases given by adding them and scaling them
        according to the layer's learning rate.
        """
        self.weights = self.weights - self.lrate * delta_w
        self.biases = self.biases - self.lrate * delta_b.reshape(-1, 1)

    def __str__(self):
        outstring = f"Input size: {self.input_dimensions}\n"
        outstring += f"Output size: {self.num_nodes}\n"
        outstring += f"Activation function: {self.activation_func}\n"
        outstring += f"Learning rate: {self.lrate}\n"
        return outstring
