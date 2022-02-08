"""haakon8855"""

from time import time
import numpy as np

from layer import Layer
from utilities import Utilities as utils


class Network():
    """
    Network class containing all layers, and conducting training of the network.
    """
    def __init__(self,
                 config,
                 train,
                 validation,
                 test,
                 wrt=None,
                 wreg=0.001) -> None:
        self.layers = []

        self.config = config
        self.lrate = float(self.config['GLOBALS']["lrate"])
        self.use_softmax = False
        self.wrt = wrt
        self.wreg = wreg

        self.train_x, self.train_y = train
        self.validation_x, self.validation_y = validation
        self.test_x, self.test_y = test

        # Store the historic loss values to track network training progress
        self.loss_index = -1
        self.train_loss = []
        self.train_loss_index = []
        self.validation_loss = []
        self.validation_loss_index = []

        self.populate_layers()  # Create layer-objects

    def populate_layers(self) -> None:
        """
        Create the layer-class instances and add them to the list of layers.
        """
        # Output size of last layer = number of input values in following layer
        prev_layer_output_size = int(self.config['GLOBALS']["input"])

        # Set correct loss funtion
        if self.config['GLOBALS']["loss"] == "cross_entropy":
            self.loss_func = utils.cross_entropy
            self.loss_func_der = utils.cross_entropy_der
        else:
            self.loss_func = utils.mse
            self.loss_func_der = utils.mse_der

        # Set whether to use softmax or not after last normal layer
        self.use_softmax = self.config['GLOBALS']["softmax"] == "true"

        # For each label in the config following the 'GLOBALS' label
        for layer_section in self.config.sections()[1:]:
            layer = self.config[layer_section]
            # Fetch config info specific to each layer
            input_size = int(prev_layer_output_size)
            output_size = int(layer["size"])
            weight_range = (float(layer["wr_start"]), float(layer["wr_end"]))
            bias_range = (float(layer["br_start"]), float(layer["br_end"]))

            lrate = self.lrate
            # if the layer has its own specified learning rate then use that
            if "lrate" in layer:
                lrate = float(layer["lrate"])

            # Set the layer's activation function
            if layer["activation"] == "sigmoid":
                activation_func = utils.sigmoid
                activation_func_der = utils.sigmoid_der
            elif layer["activation"] == "relu":
                activation_func = utils.relu
                activation_func_der = utils.relu_der
            elif layer["activation"] == "tanh":
                activation_func = utils.tanh
                activation_func_der = utils.tanh_der
            elif layer["activation"] == "linear":
                activation_func = utils.linear
                activation_func_der = utils.linear_der

            # Initialize the layer using the information from the config
            self.layers.append(
                Layer(input_dimensions=input_size,
                      num_nodes=output_size,
                      weight_range=weight_range,
                      lrate=lrate,
                      bias_range=bias_range,
                      activation_func=activation_func,
                      activation_func_der=activation_func_der))
            # Stores the output size for use as input size in the next layer
            prev_layer_output_size = output_size

        # Set reference to downstream layer for all layers, except the last one
        for i in range(0, len(self.layers) - 1):
            self.layers[i].set_next_layer(self.layers[i + 1])

    def fit(self, epochs=1, batch_size=5) -> float:
        """
        Runs backward_pass on minibatches to run through all training examples.
        Returns time to train.
        """
        start_time = time()
        # For each epoch
        for _ in range(epochs):
            # For each minibatch in epoch
            for i in range(0, len(self.train_x), batch_size):
                if i >= len(self.train_x) - batch_size:
                    minibatch_x = self.train_x[i:]
                    minibatch_y = self.train_y[i:]
                else:
                    minibatch_x = self.train_x[i:i + batch_size]
                    minibatch_y = self.train_y[i:i + batch_size]
                # Run backprop on the current minibatch
                self.backward_pass(minibatch_x, minibatch_y)
            # Run forward pass on the validation set at the end of each
            # epoch to be able to plot this after training is done.
            self.forward_pass(self.validation_x,
                              target=self.validation_y,
                              data_set=1)
        end_time = time()
        return end_time - start_time

    def backward_pass(self, train_x, train_y) -> None:
        """
        Runs backprop on the network to modify its weights and thus training it.
        """
        # For each example in training set
        target_vals = train_y  # Extract example target values
        # Run forward pass and cache in-between computations
        prediction = self.forward_pass(train_x,
                                       target=target_vals,
                                       minibatch=True)
        # Compute initial jacobian from loss function to softmax-output
        tval = target_vals.reshape(target_vals.shape[0], -1)
        if self.use_softmax:
            j_l_s = utils.cross_entropy_der(prediction, tval)
            # Compute j_s_z jacobian from softmax output to last layer output
            j_s_z = utils.j_soft(prediction)
            j_l_z = np.einsum('ij,ijk->ik', j_l_s, j_s_z)
        else:
            j_l_z = utils.mse_der(prediction, tval)

        deltas = []  # List of each layer's calculated gradients
        # For j from n-1 to 0 (incl.) where n is number of layers
        for j in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[j]
            # Perform the backward pass for layer j and get the jacobian
            # matrices and delta values for updating layer j's weights.
            delta_j, delta_jb, j_l_z = layer.backward_pass(j_l_z)
            # Apply weight regularization
            delta_j = self.add_regularization(delta_j, layer.weights)
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

    def add_regularization(self, delta_j, weights):
        """
        Add the weight regularization to the gradients depending on which
        regularization scheme is defined in the config.
        """
        if self.wrt == 'l1':
            delta_j = delta_j + self.wreg * np.sign(weights)
        elif self.wrt == 'l2':
            delta_j = delta_j + self.wreg * weights
        return delta_j

    def forward_pass(self,
                     test_x: np.ndarray,
                     target=None,
                     minibatch=True,
                     verbose=False,
                     data_set=0):
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
        if data_set == 0:
            self.loss_index += 1
            self.train_loss.append(current_loss)
            self.train_loss_index.append(self.loss_index)
        elif data_set == 1:
            self.validation_loss.append(current_loss)
            self.validation_loss_index.append(self.loss_index)
        else:
            print(f"Test data loss: {current_loss}")
            return network_output

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
