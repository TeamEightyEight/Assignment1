from controller import Controller
import numpy as np


# implements controller structure for player
class player_controller(Controller):
    def control(self, inputs, net):

        output = net.activate(inputs)

        actions = [1 if score > 0.5 else 0 for score in output]
        # [left, right, jump, shoot, release]
        return actions



