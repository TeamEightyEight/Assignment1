import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from matplotlib.ticker import MaxNLocator

ENEMY = 2
EA_DIRS = ["approach1", "approach2"]
RUNS_DIR = "runs"
LOGBOOK_PATTERN = "logbook_run_"


def read_files(dir_path):
    """
    Read all the files in the folder and return a list of dataframes.
    """
    return [pd.read_csv(file, sep=";") for file in glob.glob(os.path.join(dir_path, LOGBOOK_PATTERN+"*.csv"))]


def statistics_across_generations(data):
    """
    Aggregate statistics across a given list of dataframes and compute the mean and the standard deviation of the
    'avg_fitness' and 'max_fitness'. Return a dataframe.
    """
    aggregated_data = pd.concat(data)
    df_avg = aggregated_data.groupby(aggregated_data.index).agg(mean_avg_fitness=('avg_fitness', 'mean'), std_avg_fitness=('avg_fitness', 'std'))
    df_max = aggregated_data.groupby(aggregated_data.index).agg(mean_max_fitness=('max_fitness', 'mean'), std_max_fitness=('max_fitness', 'std'))
    return pd.concat([df_avg, df_max], axis=1)


def line_plot(statistics):
    """
    Plot the lines of mean_avg_fitness and mean_max_fitness and their standard deviation across the generations.
    """
    x = statistics.index

    avg_fitness_lower_bound = statistics['mean_avg_fitness'] - statistics['std_avg_fitness']
    avg_fitness_upper_bound = statistics['mean_avg_fitness'] + statistics['std_avg_fitness']

    max_fitness_lower_bound = statistics['mean_max_fitness'] - statistics['std_max_fitness']
    max_fitness_upper_bound = statistics['mean_max_fitness'] + statistics['std_max_fitness']

    fig, ax = plt.subplots(1)
    ax.plot(x, statistics['mean_avg_fitness'], marker='o', linestyle='dashed', lw=2, label='mean avg fitness', color='blue')
    ax.plot(x, statistics['mean_max_fitness'], marker='o', linestyle='dashed', lw=2, label='mean max fitness', color='red')
    ax.fill_between(x, avg_fitness_lower_bound, avg_fitness_upper_bound, facecolor='blue', alpha=0.3,
                    label='std avg fitness')
    ax.fill_between(x, max_fitness_lower_bound, max_fitness_upper_bound, facecolor='red', alpha=0.3,
                    label='std max fitness')

    ax.legend(loc='best')
    ax.set_ylim(min(statistics['mean_avg_fitness'][1:]-1), max(statistics['mean_max_fitness'])+1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('generations')
    ax.set_ylabel('fitness')
    ax.grid()
    plt.show()


if __name__ == "__main__":
    dir_path = os.path.join(RUNS_DIR, "enemy_"+str(ENEMY))
    data = read_files(dir_path)
    stats = statistics_across_generations(data)
    line_plot(stats)
