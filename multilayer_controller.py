import sys

sys.path.insert(0, "evoman")
from controller import Controller
import numpy as np
import warnings


def sigmoid_activation(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 / (1 + np.exp(-x))


class PlayerController(Controller):
    def __init__(self, nodes_no):
        """
        Initializes the controller for evoman.

        :param nodes_no: List containing the no. of nodes for each layer (excluding output)
        """
        self.nodes_no = nodes_no

    # What is actually called controller in the demo are just weights
    def control(self, inputs, individual):
        weights_and_biases = individual["weights_and_biases"]
        # The biases are at the end of the array, the rest is weights
        biases = weights_and_biases[-(np.sum(self.nodes_no) - self.nodes_no[0]) :]
        weights = weights_and_biases[: -(np.sum(self.nodes_no) - self.nodes_no[0])]
        used_weights = 0
        used_biases = 0
        input = np.array(inputs)
        for layer in range(0, len(self.nodes_no) - 1):
            layer_weights = weights[
                used_weights : used_weights
                + (self.nodes_no[layer] * self.nodes_no[layer + 1])
            ].reshape(self.nodes_no[layer], self.nodes_no[layer + 1])
            layer_biases = biases[
                used_biases : used_biases + self.nodes_no[layer + 1]
            ].reshape((1, self.nodes_no[layer + 1]))
            input = sigmoid_activation(np.dot(input, layer_weights) + layer_biases)[0]
            used_weights += self.nodes_no[layer] * self.nodes_no[layer + 1]
            used_biases += self.nodes_no[layer + 1]
        actions = [1 if score > 0.5 else 0 for score in input]
        # [left, right, jump, shoot, release]
        return actions
