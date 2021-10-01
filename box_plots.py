import matplotlib.pyplot as plt
import os
import seaborn as sns

import numpy as np
import pandas as pd
from ast import literal_eval

import scipy.stats as stats

ENEMY = 5
EA_DIRS = ["approach1", "approach2"]
RUNS_DIR = "runs"
FILE_NAME = "games_played.csv"
PLOTS_DIR = "plots"
PLOT_RESULT_NAME = "box_plot_enemy_"


def statistics_across_generations(data):
    """
    Aggregate statistics across a given list of dataframes and compute the mean and the standard deviation of the
    'avg_fitness' and 'max_fitness'. Return a dataframe.
    """
    aggregated_data = pd.concat(data)
    df_avg = aggregated_data.groupby(aggregated_data.index).agg(mean_avg_fitness=('avg_fitness', 'mean'), std_avg_fitness=('avg_fitness', 'std'))
    df_max = aggregated_data.groupby(aggregated_data.index).agg(mean_max_fitness=('max_fitness', 'mean'), std_max_fitness=('max_fitness', 'std'))
    return pd.concat([df_avg, df_max], axis=1)


def main():
    # read the csv file containing the results of played games
    ea1_file_path = os.path.join(EA_DIRS[0], RUNS_DIR, "enemy_" + str(ENEMY), FILE_NAME)
    ea2_file_path = os.path.join(EA_DIRS[1], RUNS_DIR, "enemy_" + str(ENEMY), FILE_NAME)

    ea1_df_games = pd.read_csv(ea1_file_path, sep=";")
    ea2_df_games = pd.read_csv(ea2_file_path, sep=";")

    # convert the element inside the column 'individual_gains' to an array
    ea1_df_games['individual_gains'] = ea1_df_games['individual_gains'].apply(literal_eval)
    ea2_df_games['individual_gains'] = ea2_df_games['individual_gains'].apply(literal_eval)

    # compute the mean of the individual_gains for each run
    ea1_df_games['individual_gains'] = ea1_df_games['individual_gains'].map(lambda x: np.array(x).mean())
    ea2_df_games['individual_gains'] = ea2_df_games['individual_gains'].map(lambda x: np.array(x).mean())

    df = pd.concat([ea1_df_games['individual_gains'], ea2_df_games['individual_gains']]
                      , axis=1, keys=['Approach1', 'Approach2'])

    # compute statistic test
    t_statistic = stats.ttest_ind(df['Approach1'], df['Approach2'])

    # show box plot
    sns.set(rc={'figure.figsize': (3,5)})
    palette = {"Approach1": "blue", "Approach2": "red"}
    ax = sns.boxplot(data = df, palette=palette)
    ax.set(ylabel='Individual gain'
           , title=f'Enemy {ENEMY} (T-statistic = {t_statistic.statistic:.2f}, p-value = {t_statistic.pvalue:.2f})')
    plt.show()

    # save plot
    plot_file_name = os.path.join(PLOTS_DIR, PLOT_RESULT_NAME + str(ENEMY))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ax.get_figure().savefig(plot_file_name, bbox_inches='tight')
    print(f"Plot saved in {plot_file_name}")


if __name__ == "__main__":
    main()
