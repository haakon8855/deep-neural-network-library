"""haakoas"""

import numpy as np
from time import time
from matplotlib import pyplot as plt

from layer import Layer
from data_generator import DataGenerator
from configuration import Config
from utilities import Utilities as utils


class Network():
    """
    Network class containing all layers, and conducting training of the network.
    """
    def __init__(self, config_file: str) -> None:
        self.layers = []

        self.config = Config.get_config(config_file)
        self.lrate = float(self.config['GLOBALS']["lrate"])
        self.use_softmax = False

        generator = DataGenerator(10, 5, 10, 5, 10, 0.008, seed=123)
        train, validation, test = generator.generate_images(10)

        self.train_x, self.train_y = train
        self.validation_x, self.validation_y = validation
        self.test_x, self.test_y = test

        # XOR example
        # self.train_x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
        #                          [1.0, 1.0]])
        # self.train_y = np.array([0.0, 1.0, 1.0, 0.0])
        # self.train_y = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0],
        #                          [0.0, 1.0]])
        self.test_x = self.train_x
        self.test_y = self.train_y

        # Store the historic loss values to track network training progress
        self.historic_loss = []

        self.populate_layers()  # Create layer-objects

    def populate_layers(self) -> None:
        """
        Create the layer-class instances and add them to the list of layers.
        """
        prev_layer_output_size = int(self.config['GLOBALS']["input"])
        if self.config['GLOBALS']["loss"] == "cross_entropy":
            self.loss_func = utils.cross_entropy
            self.loss_func_der = utils.cross_entropy_der
        else:
            self.loss_func = utils.mse
            self.loss_func_der = utils.mse_der
        self.use_softmax = self.config['GLOBALS']["softmax"]
        for layer_section in self.config.sections()[1:]:
            layer = self.config[layer_section]
            input_size = int(prev_layer_output_size)
            output_size = int(layer["size"])
            weight_range = (float(layer["wr_start"]), float(layer["wr_end"]))
            bias_range = (float(layer["br_start"]), float(layer["br_end"]))
            lrate = self.lrate
            if "lrate" in layer:
                lrate = float(layer["lrate"])
            if layer["activation"] == "sigmoid":
                activation_func = utils.sigmoid
                activation_func_der = utils.sigmoid_der
            elif layer["activation"] == "relu":
                activation_func = utils.relu
                activation_func_der = utils.relu_der
            self.layers.append(
                Layer(input_dimensions=input_size,
                      num_nodes=output_size,
                      weight_range=weight_range,
                      lrate=lrate,
                      bias_range=bias_range,
                      activation_func=activation_func,
                      activation_func_der=activation_func_der))
            prev_layer_output_size = output_size
        for i in range(0, len(self.layers) - 1):
            self.layers[i].set_next_layer(self.layers[i + 1])

    def backward_pass(self) -> None:
        """
        Runs backprop on the network to modify its weights and thus training it.
        """
        # For each example in training set
        target_vals = self.train_y  # Extract example target values
        # Run forward pass and cache in-between computations
        prediction = self.forward_pass(self.train_x,
                                       target=target_vals,
                                       minibatch=True)
        # Compute initial jacobian from loss function to softmax-output
        tval = target_vals.reshape(target_vals.shape[0], -1)
        if self.use_softmax:
            # j_l_s = utils.mse_der(prediction, tval)
            j_l_s = utils.cross_entropy_der(prediction, tval)
            # Compute j_s_z jacobian from softmax output to last layer output
            j_s_z = utils.j_soft(prediction)
            j_l_z = np.einsum('ij,ijk->ik', j_l_s, j_s_z)
        else:
            j_l_z = utils.mse_der(prediction, tval)
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

    def forward_pass(self,
                     test_x: np.ndarray,
                     target=None,
                     minibatch=True,
                     verbose=False):
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
        network_output = self.layers[0].forward_pass(test_x)
        if self.use_softmax:
            network_output = utils.softmax(network_output)

        # Calculate loss and store in loss log.
        current_loss = self.get_loss(network_output, target)
        self.historic_loss.append(current_loss)

        # Print information (inputs, outputs, target and loss)
        # if verbose flag is true.
        if verbose:
            print(f"\n\nNetwork Input: \n{test_x}")
            print(f"\nNetwork Output: \n{network_output}")
            if target is None:
                print("\nTarget provided was None,",
                      "please pass target to see loss.")
            else:
                print(f"\nTarget value(s): \n{target}")
                print(f"\nLoss: {current_loss}")

        return network_output

    def get_loss(self, network_output, target):
        """
        Returns the network's loss given its prediction and target value.
        """
        return self.loss_func(network_output, target)

    def __str__(self):
        outstring = "Network:\n"
        outstring += f"Input size: {self.layers[0].input_dimensions}\n"
        outstring += f"Output size: {self.layers[-1].num_nodes}\n"
        outstring += "\nLayers:\n"
        for i, layer in enumerate(self.layers):
            outstring += f"Layer {i}\n: {str(layer)}\n"
        return outstring


if __name__ == "__main__":
    NET = Network("config.ini")

    print(NET)

    NET.forward_pass(NET.train_x, NET.train_y, verbose=True)

    start_time = time()
    for _ in range(1000):
        NET.backward_pass()
    end_time = time()

    NET.forward_pass(NET.train_x, NET.train_y, verbose=True)

    print(f"Time to train: {round(end_time-start_time, 3)}")

    plt.xlabel("Minibatch")
    plt.ylabel("Loss")
    plt.plot(NET.historic_loss)
    plt.legend(["Train"])
    plt.show()
