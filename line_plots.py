import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter

ENEMY = 8
SPACE_LIM = 5
ARANGE_SPACE = 10.0
EA_DIR = "approach2"
APPROACH_NAME = "Approach 2"
COLOR_MEAN="red"
COLOR_MAX="blue"
RUNS_DIR = "runs"
LOGBOOK_PATTERN = "logbook_run_"
PLOTS_DIR = "plots"
PLOT_RESULT_NAME = "line_plot_enemy_"


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


def line_plot(ea_stats):
    """
    Plot the lines of mean_avg_fitness and mean_max_fitness and their standard deviation across the generations.
    """

    # read number of generations
    x = ea_stats.index

    avg_fitness_lower_bound = ea_stats['mean_avg_fitness'] - ea_stats['std_avg_fitness']
    avg_fitness_upper_bound = ea_stats['mean_avg_fitness'] + ea_stats['std_avg_fitness']

    max_fitness_lower_bound = ea_stats['mean_max_fitness'] - ea_stats['std_max_fitness']
    max_fitness_upper_bound = ea_stats['mean_max_fitness'] + ea_stats['std_max_fitness']

    fig, ax = plt.subplots(1, figsize=(5,2))

    # plot lines for ea1
    ax.plot(x, ea_stats['mean_avg_fitness'], marker='o', markersize=4, linestyle='dashed', lw=1, label='mean avg fitness', color=COLOR_MEAN)
    ax.plot(x, ea_stats['mean_max_fitness'], marker='o', markersize=4, linestyle='solid', lw=1, label='mean max fitness', color=COLOR_MAX)
    ax.fill_between(x, avg_fitness_lower_bound, avg_fitness_upper_bound, facecolor=COLOR_MEAN, alpha=0.3)
    ax.fill_between(x, max_fitness_lower_bound, max_fitness_upper_bound, facecolor=COLOR_MAX, alpha=0.3,)

    min_y_lim = min([
                        min(ea_stats['mean_avg_fitness'][1:])
                   ]) - SPACE_LIM
    max_y_lim = max([
                        max(ea_stats['mean_max_fitness'])
                    ]) + SPACE_LIM

    ax.set_ylim(min_y_lim, max_y_lim)
    ax.legend(loc='best')
    ax.set_title(f'Enemy {ENEMY} - {APPROACH_NAME}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('generations')
    ax.set_ylabel('fitness')
    ax.set_xticks(np.arange(0, 25, 2.0))
    ax.set_yticks(np.arange(np.floor(min_y_lim), np.ceil(max_y_lim), ARANGE_SPACE))
    ax.grid(linestyle="--")
    plt.show()
    return fig


if __name__ == "__main__":
    dir_path = os.path.join(EA_DIR, RUNS_DIR, "enemy_"+str(ENEMY))

    # read csv files as DataFrame
    ea_df = read_files(dir_path)

    # compute mean of mean fitnesses and mean of max fitnesses and their std
    ea_stats = statistics_across_generations(ea_df)

    # draw plot
    fig = line_plot(ea_stats)

    # save plot
    plot_file_name = os.path.join(PLOTS_DIR, PLOT_RESULT_NAME+str(ENEMY)+"_"+EA_DIR)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(plot_file_name, bbox_inches='tight')
    print(f"Plot saved in {plot_file_name}")
