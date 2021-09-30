import matplotlib.pyplot as plt
import os
import glob

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from ast import literal_eval

import scipy.stats as stats

ENEMY = 8
RUNS_DIR = "ea1_runs"
FILE_NAME = "games_played.csv"


def read_files(dir_path):
    """
    Read the call the files in the folder and return a list of dataframes.
    """
    return [pd.read_csv(file) for file in glob.glob(os.path.join(dir_path, "logbook_run_*.csv"))]


def statistics_across_generations(data):
    """
    Aggregate statistics across a given list of dataframes and compute the mean and the standard deviation of the
    'avg_fitness' and 'max_fitness'. Return a dataframe.
    """
    aggregated_data = pd.concat(data)
    df_avg = aggregated_data.groupby(aggregated_data.index).agg(mean_avg_fitness=('avg_fitness', 'mean'), std_avg_fitness=('avg_fitness', 'std'))
    df_max = aggregated_data.groupby(aggregated_data.index).agg(mean_max_fitness=('max_fitness', 'mean'), std_max_fitness=('max_fitness', 'std'))
    return pd.concat([df_avg, df_max], axis=1)


def boxplot(data):
    """
    Plot the box plots of the played games.
    """
    x = data['n_run']

    fig, ax = plt.subplots(1)
    ax.boxplot(data['individual_gains'])
    ax.legend(loc='best')
    ax.set_xlabel('')
    ax.set_ylabel('individual gain')
    ax.grid()
    plt.show()


def main():
    # read the csv file containing the results of played games
    file_path = os.path.join(RUNS_DIR, "enemy_" + str(ENEMY), FILE_NAME)
    df_games = pd.read_csv(file_path, sep=";")

    # convert the element inside the column 'individual_gains' to an array
    df_games['individual_gains'] = df_games['individual_gains'].apply(literal_eval)

    # compute the mean of the individual_gains for each run
    df_games['individual_gains'] = df_games['individual_gains'].map(lambda x: np.array(x).mean())

    # show box plot
    boxplot(df_games)

    print(stats.ttest_ind(df_games['individual_gains'],
                df_games['individual_gains'])
    )







if __name__ == "__main__":
    main()
