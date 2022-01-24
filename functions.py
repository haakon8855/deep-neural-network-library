"""haakoas"""

import numpy as np


def mse(value, target):
    """
    Mean squared error
    """
    ((value - target)**2).mean()


def mse_der(value, target):
    """
    The derivative of the mean squared error
    """
    return ((value - target) * (2 / len(value)))


def relu(inp):
    """
    ReLu activation function
    """
    return (inp > 0) * inp


def relu_der(inp):
    """
    The derivative of the ReLu activation function
    """
    return (inp > 0) * 1


def sigmoid(inp):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-inp))


def sigmoid_der(inp):
    """
    The derivative of the sigmoid activation function
    """
    return sigmoid(inp) * (1.0 - sigmoid(inp))
