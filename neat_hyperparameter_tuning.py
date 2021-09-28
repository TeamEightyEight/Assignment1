# imports framework
from __future__ import print_function
import sys

sys.path.insert(0, "evoman")
from environment import Environment
import os
from neat_optimization import *
from write_config import set_config

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
from hyperopt import hp, fmin, tpe
from hyperopt import SparkTrials, STATUS_OK


import neat
from neat_controller import player_controller
from custom_neat_classes import CoolPopulation88, CoolReporter88

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os
import pickle

ENEMY = 2
GENERATIONS = 4
LAMBDA = 7

space = hp.choice(
    "GA",
    [
        {
            "weight_coeff": hp.uniform("weight_coeff", 0, 1),
            "conn_add": hp.uniform("conn_add", 0, 1),
            "node_add": hp.uniform("node_add", 0, 1),
            "num_hidden": hp.quniform("num_hidden", 5, 30, 1),
        }
    ],
)


def test_hyperparameter_vector(args=None):
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            [genome.fitness, genome.player_energy, genome.enemy_energy] = simulation(
                env, net
            )

        return [genome.fitness, genome.player_energy]

    def best_individual_run(genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        [genome.fitness, genome.player_energy, genome.enemy_energy] = simulation(
            env, net
        )

        return [genome.fitness, genome.player_energy, genome.enemy_energy]

    # runs simulation
    def simulation(env, x):
        f, p, e, t = env.play(pcont=x)
        return f, p, e

    # evaluation -> not being use
    def evaluate(x):
        return np.array(list(map(lambda y: simulation(env, y), x)))

    # if args:
    #     set_config(args['weight_coeff'],
    #             args['conn_add'],
    #             args['node_add'],
    #             args['num_hidden'])
    # else:
    #     set_config()

    experiment_name = "neat_demo"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[2],
        playermode="ai",
        player_controller=player_controller(),
        enemymode="static",
        randomini="yes",
        level=2,
        speed="fastest",
    )

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state

    ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

    ini = time.time()  # sets time marker

    # genetic algorithm params

    run_mode = "train"  # train or test

    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-feedforward",
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = CoolPopulation88(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(CoolReporter88(True))

    # Run until a solution is found.

    [best_ever, best_last_gen] = p.run(
        eval_genomes, 3
    )  # second parameter max num of generation
    return -(best_ever.fitness)


spark_trials = SparkTrials()
best = fmin(
    test_hyperparameter_vector,
    space,
    trials=spark_trials,
    algo=tpe.suggest,
    max_evals=50,
)
print("The best combination of hyperparameters is:")
print(best)
