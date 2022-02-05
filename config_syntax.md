# Neural Network Config Files

The syntax of the neural network config files is identical to Windows'
.ini-files' syntax, where variables are defined on separate lines with an
equals-sign separating the variable and the value. JSON-like lists and
dictionaries are not allowed, thus each layer in the network is defined as its
own section.

## Sections

### Globals

A config file consists of several sections, the first one being the 'GLOBALS'-
section. Here, parameters for the data generator, and general parameters for
the neural network are defined such as the number of epochs, input values,
whether to use Softmax or not and the loss function to be used.

### Layers

Each section following the 'GLOBALS'-section is treated as a layer. The
section's actual name is not important, save for the fact that it must be
unique and different from all other section-labels. The layers are constructed
chronologically in the order they are encountered in the config file. This means
the first section following 'GLOBALS' is the first layer after the input layer.
In each layer, the number of inputs are implied from the number of input nodes
or from the number of outputs from the upstream layer. In each layer, one can
specify the number of nodes in that layer, its activation function, the range
for the initialization of the weights and biases and layer-specific learning
rate if this is desired.
