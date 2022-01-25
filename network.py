"""haakoas"""

import numpy as np
from time import time

from layer import Layer
from layer_softmax import LayerSoftmax
from data_generator import DataGenerator
from configuration import Config
import functions as funcs


class Network():
    """
    Network class containing all layers, and conducting training of the network.
    """
    def __init__(self, config_file: str) -> None:
        self.layers = []

        self.config = Config.get_config(config_file)
        self.lrate = self.config["lrate"]

        generator = DataGenerator(10, 5, 10, 5, 10, 0.008)
        train, validation, test = generator.generate_images(10)

        self.train_x, self.train_y = train
        self.validation_x, self.validation_y = validation
        self.test_x, self.test_y = test

        # XOR example
        self.train_x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
                                 [1.0, 1.0]])
        self.train_y = np.array([0.0, 1.0, 1.0, 0.0])
        self.test_x = self.train_x
        self.test_y = self.train_y

        self.populate_layers()  # Create layer-objects

    def populate_layers(self) -> None:
        """
        Create the layer-class instances and add them to the list of layers.
        """
        last_layer_output_size = self.config["input"]
        for layer in self.config["layers"]:
            if "size" in layer:  # layer is a normal layer
                input_size = last_layer_output_size
                output_size = layer["size"]
                weight_range = layer["wr"]
                # TODO: activation function
                self.layers.append(
                    Layer(input_dimensions=input_size,
                          num_nodes=output_size,
                          weight_range=weight_range))
                last_layer_output_size = output_size
            else:
                if layer["type"] == "softmax":
                    input_size = last_layer_output_size
                    output_size = input_size
                    self.layers.append(
                        LayerSoftmax(input_dimensions=input_size,
                                     num_nodes=output_size))
        for i in range(0, len(self.layers) - 1):
            self.layers[i].set_next_layer(self.layers[i + 1])

    def backward_pass(self) -> None:
        """
        Runs backprop on the network to modify its weights and thus training it.
        """
        # TODO: Can be removed later:
        # Randomize order of examples in XOR
        permutation = np.random.permutation(len(self.train_x))
        self.train_x = self.train_x[permutation]
        self.train_y = self.train_y[permutation]

        # For each example in training set
        for i in range(len(self.train_x)):
            input_val = self.train_x[i]  # Extract example input values
            target_val = self.train_y[i]  # Extract example target values
            # Run forward pass and cache in-between computations
            prediction = self.forward_pass(input_val).reshape(-1, 1)
            # Compute initial jacobian from loss function to z-output
            j_l_z = funcs.mse_der(prediction, target_val)
            deltas = []

            # For j from n-1 to 0 (incl.) where n is number of layers
            for j in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[j]
                # Perform backward pass for layer j and get jacobians and
                # delta values for updating layer j's weights.
                delta_j, delta_jb, j_l_z = layer.backward_pass(j_l_z)
                # Cache deltas in order to update them later
                deltas.append((delta_j, delta_jb))

            # For j from 0 to n-1 (incl.) where n is length of 'deltas'
            for j in range(len(deltas)):
                # TODO: learning rate ref to layer here
                learning_rate = 0.75
                # Fetch the delta values corresponding to layer j.
                # Since 'deltas' is reversed in relation to layer order in
                # 'self.layers' we fetch with index = len(deltas) - j - 1
                delta_w, delta_b = deltas[len(deltas) - j - 1]
                # Update weights and biases by subtracting deltas multiplied
                # by the learning rate.
                # TODO: Do this in Layer-class?
                self.layers[j].weights = self.layers[
                    j].weights - learning_rate * delta_w
                self.layers[j].biases = (self.layers[j].biases -
                                         learning_rate * delta_b).reshape(
                                             self.layers[j].biases.shape)

    def forward_pass(self, test_x: np.ndarray, minibatch=False):
        """
        Given an example test_x (single-case or minibatch),
        we want to predict its class probability. Since the layers call each
        other's forward_pass-method recursively we simply return the first
        layer's forward_pass return value.
        """
        # if minibatch:
        #     return self.layers[0].forward_pass(test_x, minibatch=minibatch)

        # Call forward pass on first layer (after input layer). This layer
        # calls the next layers recursively and returns the result when
        # the last layer is reached.
        if not minibatch:
            test_x = test_x.reshape(-1, 1)
        return self.layers[0].forward_pass(test_x, minibatch=minibatch)


if __name__ == "__main__":
    NET = Network("config_file")

    for k in range(len(NET.train_x)):
        print("input: ", NET.train_x[k], "", end="")
        print("result: ", NET.forward_pass(NET.train_x[k]))
    # print("result: ", NET.forward_pass(NET.train_x[k], False))
    # print("input: ", NET.train_x, "", end="")
    # print("result: ", NET.forward_pass(NET.train_x, True))
    before = time()
    for i in range(10000):
        NET.backward_pass()
    after = time()
    print(f"time elapsed: {after-before}")
    print()
    for k in range(len(NET.train_x)):
        print("input: ", NET.train_x[k], "", end="")
        print("result: ", NET.forward_pass(NET.train_x[k]))
