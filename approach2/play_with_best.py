import sys, os
import pandas as pd

sys.path.insert(0, "evoman")
from environment import Environment
from neat_controller import player_controller


# imports other libs
import glob
import re
import pickle
import neat


ENEMY = 8
RUNS_DIR = "runs"
N_GAMES = 5
BEST_INDIVIDUAL_PATTERN = "best_individual_run_"

def init_env(enemy):
    """
    Initializes environment for single objective mode (specialist)  with static enemy and ai player.
    """
    experiment_name = os.path.join(RUNS_DIR, 'enemy_' + str(enemy))

    return Environment(
        experiment_name=experiment_name,
        enemies=[ENEMY],
        playermode="ai",
        player_controller=player_controller(),
        speed="fastest",
        enemymode="static",
        level=2,
        logs="off",
        savelogs="no",
        sound="off",
        randomini="yes"
    )


def play_game(env, best_individual):
    """
    Play a game and return the individual gain of the execution.
    """

    # Load configuration.
    config = neat.Config(neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config-feedforward')
    
    net = neat.nn.FeedForwardNetwork.create(best_individual,config)
    
    f,p,e,t = env.play(pcont=net)
    return p-e



def main():
    # initialize the environment
    env = init_env(ENEMY)

    # dict containing all the results for each played games
    logbook = {}

    # iterate across the runs
    pattern = os.path.join(RUNS_DIR, 'enemy_' + str(ENEMY), BEST_INDIVIDUAL_PATTERN + "*[0-9]")
    for file_name in glob.glob(pattern):

        # extract the number of the run from the file name
        n_run = re.search("[0-9]+", os.path.basename(file_name)).group(0)
        print(f"RUN {n_run}:")

        # load the best individual
        with open(file_name, mode='rb') as file:
            best_individual = pickle.load(file)

        # play the game N_GAMES times
        individual_gains = [play_game(env, best_individual) for _ in range(N_GAMES)]
        [print(f"\tgame {game} - gain = {individual_gain}") for game, individual_gain in enumerate(individual_gains)]
        print()

        # save the results of the games of the current run
        logbook[n_run] = {"individual_gains":individual_gains}

    logbook_path = os.path.join(RUNS_DIR, "enemy_" + str(ENEMY), "games_played.csv")
    pd.DataFrame.from_dict(logbook, orient='index').to_csv(logbook_path, index=True, index_label='n_run', sep=";")
    print(
        f"Results of the games saved in {logbook_path} "
    )


if __name__ == "__main__":
    main()

