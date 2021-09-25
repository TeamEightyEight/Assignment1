# imports framework
import sys, os

sys.path.insert(0, "evoman")
from environment import Environment
from multilayer_controller import PlayerController

# imports other libs
import numpy as np

experiment_name = "logs/demo_run"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
LAYER_NODES = [20, 20, 15, 10, 5]

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(
    experiment_name=experiment_name,
    playermode="ai",
    player_controller=PlayerController(LAYER_NODES),
    speed="normal",
    enemymode="static",
    level=2,
    randomini="yes"
)


# tests saved demo solutions for each enemy
for en in range(3, 4):

    # Update the enemy
    env.update_parameter("enemies", [en])

    # Load specialist controller
    sol = {"weights_and_biases": np.loadtxt("best_individual.txt")}
    print("\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY " + str(en) + " \n")
    env.play(pcont=sol)
