"""Haakoas"""

import numpy as np


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
    sigmoid_x = sigmoid(
        inp)  # Calculate the activation of x only once to hopefully save time
    return sigmoid_x * (1 - sigmoid_x)
