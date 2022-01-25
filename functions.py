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
    return (value - target) * (2 / value.shape[1])


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


def softmax(inp):
    """
    Softmax of the 'inp' vector. If inp is list of vectors,
    softmax is computed individually for each vector.
    """
    # Raise e to all elements in inp
    e_inp = np.exp(inp)
    # Calculate the sum of e's for each example
    denominator = np.sum(e_inp, axis=1)
    # Divide each element in e_inp with the denominator for that row
    return e_inp / denominator[:, None]


def j_soft(inp):
    """
    The derivative of the softmax function in the form of a matrix.
    """
    j_soft = np.zeros((inp.shape[0], inp.shape[1], inp.shape[1]))
    for i in range(j_soft.shape[0]):
        for j in range(j_soft.shape[1]):
            for k in range(j_soft.shape[2]):
                if j == k:
                    num = inp[i, j] - inp[i, j]**2
                else:
                    num = -inp[i, j] * inp[i, k]
                j_soft[i, j, k] = num
    return j_soft
