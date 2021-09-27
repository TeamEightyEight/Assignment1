import sys
sys.path.insert(0, 'evoman')
from controller import Controller
import numpy as np


def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))

# implements controller structure for player
class player_controller(Controller):
    def __init__(self):
        # Number of hidden neurons
        self.n_hidden = 10 #fixed in NEAT configuration file


    def control(self, inputs, net):

        output = net.activate(inputs)

        actions = [1 if score > 0.5 else 0 for score in output]
        # [left, right, jump, shoot, release]
        return actions



