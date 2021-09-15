"""
This file contains the code for the game runner. 
What it does is creating an environment, running the game, and getting the final 
fitness score for the game.
"""
import datetime
import time
import sys

sys.path.insert(0, "evoman")
from environment import Environment
import numpy as np
from pathlib import Path


class GameRunner:
    def __init__(
        self, controller, experiment_name="", enemies=[2], level=2, speed="fastest"
    ):
        """
        This class instantiates an EVOMAN environment, runs the game and evaluates the fitness.
        """
        self.controller = controller
        self.enemies = enemies
        self.experiment_name = (
            experiment_name
            if experiment_name != ""
            else f"logs/Run {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        # Creates a directory for the experiment's logs
        Path(self.experiment_name).mkdir(parents=True, exist_ok=True)
        self.level = level
        self.speed = speed
        self.env = Environment(
            experiment_name=self.experiment_name,
            enemies=self.enemies,
            playermode="ai",
            player_controller=self.controller,
            enemymode="static",
            level=self.level,
            speed=self.speed,
        )
        self.env.state_to_log()

    def simulation(self, individual):
        """
        Method to actually run a play simulation. Returns the fitness only.
        :param individual: one individual from the population
        """
        f, p, e, t = self.env.play(pcont=individual)
        return f

    def evaluate(self, individual):
        """
        Basically, a wrapper for simulation(self, individual) to comply with DEAP's format.
        """
        return (self.simulation(individual),)
