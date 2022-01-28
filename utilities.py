"""haakoas"""

import numpy as np


class Utilities():
    @staticmethod
    def mse(value, target):
        """
        Mean squared error
        """
        loss = np.square(value - target)
        return loss.mean()

    @staticmethod
    def mse_der(value, target):
        """
        The derivative of the mean squared error
        """
        return (value - target) * (2 / value.shape[1])

    @staticmethod
    def cross_entropy(value, target):
        """
        Cross entropy loss function
        """
        loss = -np.log2(value) * target
        return loss.sum(axis=1).mean()

    @staticmethod
    def cross_entropy_der(value, target):
        """
        Derivative of cross entropy loss function
        """
        return -(target / (value * np.log(2)))

    @staticmethod
    def relu(inp):
        """
        ReLu activation function
        """
        return (inp > 0) * inp

    @staticmethod
    def relu_der(inp):
        """
        The derivative of the ReLu activation function
        """
        return (inp > 0) * 1

    @staticmethod
    def sigmoid(inp):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-inp))

    @staticmethod
    def sigmoid_der(inp):
        """
        The derivative of the sigmoid activation function
        """
        return Utilities.sigmoid(inp) * (1.0 - Utilities.sigmoid(inp))

    @staticmethod
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

    @staticmethod
    def j_soft(inp):
        """
        The derivative of the softmax function in the form of a matrix.
        """
        j_s_z = np.zeros((inp.shape[0], inp.shape[1], inp.shape[1]))
        for i in range(j_s_z.shape[0]):
            for j in range(j_s_z.shape[1]):
                for k in range(j_s_z.shape[2]):
                    if j == k:
                        num = inp[i, j] - inp[i, j]**2
                    else:
                        num = -inp[i, j] * inp[i, k]
                    j_s_z[i, j, k] = num
        return j_s_z
