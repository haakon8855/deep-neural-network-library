"""haakoas"""

import numpy as np

from layer import Layer
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
                bias_range = layer["br"]
                lrate = layer["lrate"]
                if layer["act"] == "sigmoid":
                    activation_func = funcs.sigmoid
                    activation_func_der = funcs.sigmoid_der
                elif layer["act"] == "relu":
                    activation_func = funcs.relu
                    activation_func_der = funcs.relu_der
                self.layers.append(
                    Layer(input_dimensions=input_size,
                          num_nodes=output_size,
                          weight_range=weight_range,
                          lrate=lrate,
                          bias_range=bias_range,
                          activation_func=activation_func,
                          activation_func_der=activation_func_der))
                last_layer_output_size = output_size
            else:
                if layer["type"] == "softmax":
                    pass
                    # TODO: softmax
        for i in range(0, len(self.layers) - 1):
            self.layers[i].set_next_layer(self.layers[i + 1])

    def backward_pass(self) -> None:
        """
        Runs backprop on the network to modify its weights and thus training it.
        """
        # For each example in training set
        target_vals = self.train_y  # Extract example target values
        # Run forward pass and cache in-between computations
        prediction = self.forward_pass(self.train_x, True).reshape(-1, 1)
        # Compute initial jacobian from loss function to z-output
        j_l_z = funcs.mse_der(prediction, target_vals.reshape(-1, 1))
        deltas = []

        # For j from n-1 to 0 (incl.) where n is number of layers
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            # Perform the backward pass for layer j and get the jacobian
            # matrices and delta values for updating layer j's weights.
            delta_j, delta_jb, j_l_z = layer.backward_pass(j_l_z)
            # Cache deltas in order to update them later
            deltas.append((delta_j, delta_jb))

        # For j from 0 to n-1 (incl.) where n is length of 'deltas'
        for i in range(len(deltas)):
            # Fetch the delta values corresponding to layer j.
            # Since 'deltas' is reversed in relation to layer order in
            # 'self.layers' we fetch with index = len(deltas) - j - 1
            delta_w, delta_b = deltas[len(deltas) - i - 1]
            delta_w = delta_w.mean(axis=0)
            delta_b = delta_b.mean(axis=0)
            self.layers[i].update_weights(delta_w, delta_b)

    def forward_pass(self, test_x: np.ndarray, minibatch=False):
        """
        Given an example test_x (single-case or minibatch),
        we want to predict its class probability. Since the layers call each
        other's forward_pass-method recursively we simply return the first
        layer's forward_pass return value.
        """
        # Call forward pass on first layer (after input layer). This layer
        # calls the next layers recursively and returns the result when
        # the last layer is reached.
        if not minibatch:
            test_x = test_x.reshape(1, -1)
        return self.layers[0].forward_pass(test_x)


if __name__ == "__main__":
    NET = Network("config_file")

    print("input: ", NET.train_x, "", end="")
    print("result: \n", NET.forward_pass(NET.train_x, True))

    for _ in range(10000):
        NET.backward_pass()

    print()
    for k in range(len(NET.train_x)):
        print("input: ", NET.train_x[k], "", end="")
        print("result: ", NET.forward_pass(NET.train_x[k]))
