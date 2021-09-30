import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from matplotlib.ticker import MaxNLocator

ENEMY = 2
EA_DIRS = ["approach1", "approach2"]
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


def line_plot(ea1_stats, ea2_stats):
    """
    Plot the lines of mean_avg_fitness and mean_max_fitness and their standard deviation across the generations.
    """

    # read number of generations
    x = ea1_stats.index

    ea1_avg_fitness_lower_bound = ea1_stats['mean_avg_fitness'] - ea1_stats['std_avg_fitness']
    ea1_avg_fitness_upper_bound = ea1_stats['mean_avg_fitness'] + ea1_stats['std_avg_fitness']

    ea1_max_fitness_lower_bound = ea1_stats['mean_max_fitness'] - ea1_stats['std_max_fitness']
    ea1_max_fitness_upper_bound = ea1_stats['mean_max_fitness'] + ea1_stats['std_max_fitness']

    ea2_avg_fitness_lower_bound = ea2_stats['mean_avg_fitness'] - ea2_stats['std_avg_fitness']
    ea2_avg_fitness_upper_bound = ea2_stats['mean_avg_fitness'] + ea2_stats['std_avg_fitness']

    ea2_max_fitness_lower_bound = ea2_stats['mean_max_fitness'] - ea2_stats['std_max_fitness']
    ea2_max_fitness_upper_bound = ea2_stats['mean_max_fitness'] + ea2_stats['std_max_fitness']

    fig, ax = plt.subplots(1)

    # plot lines for ea1
    ax.plot(x, ea1_stats['mean_avg_fitness'], marker='o', markersize=4, linestyle='dashed', lw=1, label='approach 1 - mean avg fitness', color='blue')
    ax.plot(x, ea1_stats['mean_max_fitness'], marker='o', markersize=4, linestyle='solid', lw=1, label='approach 1 - mean max fitness', color='blue')
    ax.fill_between(x, ea1_avg_fitness_lower_bound, ea1_avg_fitness_upper_bound, facecolor='blue', alpha=0.3)
    ax.fill_between(x, ea1_max_fitness_lower_bound, ea1_max_fitness_upper_bound, facecolor='blue', alpha=0.3,)

    # plot lines for ea2
    ax.plot(x, ea2_stats['mean_avg_fitness'], marker='o', markersize=4, linestyle='dashed', lw=1, label='approach 2 - mean avg fitness', color='red')
    ax.plot(x, ea2_stats['mean_max_fitness'], marker='o', markersize=4, linestyle='solid', lw=1, label='approach 2 - mean max fitness', color='red')
    ax.fill_between(x, ea2_avg_fitness_lower_bound, ea2_avg_fitness_upper_bound, facecolor='red', alpha=0.3)
    ax.fill_between(x, ea2_max_fitness_lower_bound, ea2_max_fitness_upper_bound, facecolor='red', alpha=0.3,)

    min_y_lim = min([
                        min(ea1_stats['mean_avg_fitness'][1:])
                      , min(ea2_stats['mean_avg_fitness'][1:])
                   ]) - 1
    max_y_lim = max([
                        max(ea1_stats['mean_max_fitness'])
                      , max(ea2_stats['mean_max_fitness'])
                    ]) + 1

    ax.set_ylim(min_y_lim, max_y_lim)
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('generations')
    ax.set_ylabel('fitness')
    ax.grid()
    plt.show()
    return fig


if __name__ == "__main__":
    ea1_dir_path = os.path.join(EA_DIRS[0], RUNS_DIR, "enemy_"+str(ENEMY))
    ea2_dir_path = os.path.join(EA_DIRS[1], RUNS_DIR, "enemy_" + str(ENEMY))

    # read csv files as DataFrame
    ea1_df = read_files(ea1_dir_path)
    ea2_df = read_files(ea2_dir_path)

    # compute mean of mean fitnesses and mean of max fitnesses and their std
    ea1_stats = statistics_across_generations(ea1_df)
    ea2_stats = statistics_across_generations(ea2_df)

    # draw plot
    fig = line_plot(ea1_stats, ea2_stats)

    # save plot
    plot_file_name = os.path.join(PLOTS_DIR, PLOT_RESULT_NAME+str(ENEMY))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(plot_file_name, bbox_inches='tight')
    print(f"Plot saved in {plot_file_name}")
