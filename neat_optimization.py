from __future__ import print_function
import neat

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
from custom_neat_classes import CoolPopulation88, CoolReporter88

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import pickle



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        [genome.fitness,genome.player_energy,genome.enemy_energy] = simulation(env,net)
        
    return [genome.fitness,genome.player_energy]

def best_individual_run(genome,config):
    net = neat.nn.FeedForwardNetwork.create(genome,config)
    [genome.fitness,genome.player_energy,genome.enemy_energy] = simulation(env,net)

    return [genome.fitness,genome.player_energy,genome.enemy_energy]
       
# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e

# evaluation -> not being use
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  randomini = 'yes',
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# Load configuration.
config = neat.Config(neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = CoolPopulation88(config)


# Add a stdout reporter to show progress in the terminal.
p.add_reporter(CoolReporter88(True))

# Run until a solution is found.

[best_ever,best_last_gen] = p.run(eval_genomes,13) #second parameter max num of generation

# Display the winning genome.
print("best_ever: {!s}".format(best_ever))
#print("best_last: {!s}".format(best_last_gen))
#print('\nBest genome:\n{!s}'.format([best_ever,best_last_gen]))


# Getting the winner list from the winners file
# If the file is empty then we make a new list with the mosth recent winner
with open('neat_winners.txt', 'rb') as pickle_in:
    try:
        winners = pickle.load(pickle_in)
        winners.append(winner)
    except Exception as e:
        winners = [winner]
#Now, for the best individual in the 13 generations, we run it 5 times and obtain his individual gain for each run
print("\n#######################################\n")
for i in range(0,5):
    results = best_individual_run(best_ever,config)
    print("Final Results: {!s}".format(results))

# Dumping the updated winners list to the file
with open('neat_winners.txt', 'wb') as pickle_out:
    pickle.dump(winners, pickle_out)





