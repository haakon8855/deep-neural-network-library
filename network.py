"""haakoas"""

import numpy as np

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
        p = np.random.permutation(len(self.train_x))
        self.train_x = self.train_x[p]
        self.train_y = self.train_y[p]
        for i in range(len(self.train_x)):
            case_x = self.train_x[i]
            case_y = self.train_y[i]
            target = case_y
            z = self.forward_pass(case_x)
            j_l_z = funcs.mse_der(z, target)
            deltas = []
            for j in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[j]
                delta_j, delta_jb, j_l_z = layer.backward_pass(j_l_z)
                deltas.append((delta_j, delta_jb))

            for j in range(len(deltas)):
                # TODO: learning rate ref to layer here
                lr = 0.75
                delta_w, delta_b = deltas[len(deltas) - j - 1]
                self.layers[j].weights = self.layers[j].weights - lr * delta_w
                self.layers[j].biases = (self.layers[j].biases -
                                         lr * delta_b).reshape(
                                             self.layers[j].biases.shape)

    def forward_pass(self, test_x: np.ndarray):
        """
        Given an example test_x (single-case or minibatch),
        we want to predict its class probability. Since the layers call each
        other's forward_pass-method recursively we simply return the first
        layer's forward_pass return value.
        """
        # if minibatch:
        #     return self.layers[0].forward_pass(test_x, minibatch=minibatch)
        return self.layers[0].forward_pass(test_x)


if __name__ == "__main__":
    NET = Network("config_file")
    # print(NET.forward_pass(NET.train_x[0]))
    # print(NET.backward_pass())

    for k in range(len(NET.train_x)):
        print("input: ", NET.train_x[k], "", end="")
        print("result: ", NET.forward_pass(NET.train_x[k]))
    for _ in range(5000):
        NET.backward_pass()
    print()
    for k in range(len(NET.train_x)):
        print("input: ", NET.train_x[k], "", end="")
        print("result: ", NET.forward_pass(NET.train_x[k]))
