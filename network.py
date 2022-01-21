"""Haakoas"""

import numpy as np

from layer import Layer
from data_generator import DataGenerator
from configuration import Config


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
                # TODO: Softmax-layer special case
                pass
        for i in range(0, len(self.layers) - 1):
            self.layers[i].set_next_layer(self.layers[i + 1])

    def backward_pass(self) -> None:
        """
        Runs backprop on the network to modify its weights and thus training it.
        """
        # TODO: implement backprop

    def forward_pass(self, test_x: np.ndarray, minibatch=True):
        """
        Given an example test_x (single-case or minibatch),
        we want to predict its class probability. Since the layers call each
        other's forward_pass-method recursively we simply return the first
        layer's forward_pass return value.
        """
        if minibatch:
            return self.layers[0].forward_pass(test_x, minibatch=minibatch)
        answers = []
        for case in test_x:
            answers.append(self.layers[0].forward_pass(case, minibatch))
        return answers


if __name__ == "__main__":
    NET = Network("config_file")
    print(NET.forward_pass(NET.train_x, True))
