"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import neat

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
from neat.graphs import feed_forward_layers

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        #genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        genome.input_keys = config.genome_config.input_keys
        genome.output_keys = config.genome_config.output_keys

        output = simulation(env,genome)
        #output = net.activate(simulation(env,genome))
        genome.fitness = output
        

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

#######################################
experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)



# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

##################



# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
print(config)
# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)
# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.

print('\nBest genome:\n{!s}'.format(winner))






