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
import os
from pathlib import Path


class GameRunner:
    def __init__(
        self,
        controller,
        experiment_name="",
        enemies=[2],
        level=2,
        speed="fastest",
        headless=True,
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
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.env = Environment(
                experiment_name=self.experiment_name,
                enemies=self.enemies,
                playermode="ai",
                player_controller=self.controller,
                enemymode="static",
                level=self.level,
                speed=self.speed,
                logs="off",
                savelogs="no",
                sound="off",
                randomini="yes",
            )
        else:
            self.env = Environment(
                experiment_name=self.experiment_name,
                enemies=self.enemies,
                playermode="ai",
                player_controller=self.controller,
                enemymode="static",
                level=self.level,
                speed=self.speed,
                randomini="yes",
            )
        self.env.state_to_log()

    def evaluate(self, individual):
        """
        Method to actually run a play simulation.
        :param individual: one individual from the population
        """

        fitness, player_life, enemy_life, time = self.env.play(pcont=individual["weights_and_biases"])
        return fitness, player_life, enemy_life, time


