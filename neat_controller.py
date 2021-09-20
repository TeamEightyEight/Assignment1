# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from controller import Controller
from neat.graphs import feed_forward_layers
import numpy as np


def sigmoid_activation(x):
    return 1.0 / (1.0 + np.exp(-x))

# implements controller structure for player
class player_controller(Controller):
    def __init__(self):
        # Number of hidden neurons
        self.nodes_no = None

    def get_node_vector(self, result):
        mid = [len(n) for n in result]
        return [20] + mid + [5]
    
        

        
    def control(self, inputs, controller):
        layers = feed_forward_layers(controller.input_keys,controller.output_keys,controller.connections)
        
        self.nodes_no = self.get_node_vector(layers)

        biases = np.array([controller.nodes[node].bias for node in controller.nodes])
        print(biases)

        weights = np.array([controller.connections[connection].weight for connection in controller.connections])

        print(weights)

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


