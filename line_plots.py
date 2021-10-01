import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter

ENEMY = 5
EA_DIRS = ["approach1", "approach2"]
RUNS_DIR = "runs"
LOGBOOK_PATTERN = "logbook_run_"
PLOTS_DIR = "plots"
PLOT_RESULT_NAME = "line_plot_enemy_"


class FitnessScale(mscale.ScaleBase):
    """
    Scales data in range -6 to 100 using

    The scale function:
      ln(tan(y) + sec(y))

    """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``gca().set_yscale("fitness")`` would
    # be used to select this scale.
    name = 'fitness'

    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        thresh: The degree above which to crop the data.
        """
        super().__init__(axis)

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The FitnessTransform class is defined below as a
        nested class of this one.
        """
        return self.FitnessTransform()

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.

        In our case, the Mercator example uses a fixed locator from -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        fmt = FuncFormatter(
            lambda x, pos=None: f"{np.degrees(x):.0f}\N{DEGREE SIGN}")
        axis.set(major_locator=FixedLocator(np.radians(range(-90, 90, 10))),
                 major_formatter=fmt, minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin), min(vmax)

    class FitnessTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = output_dims = 1

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, x):
            """
            This transform takes a numpy array and returns a transformed copy.
            Since the range of the Mercator scale is limited by the
            user-specified threshold, the input array must be masked to
            contain only valid values.  Matplotlib will handle masked arrays
            and remove the out-of-range data from the plot.  However, the
            returned array *must* have the same shape as the input array, since
            these values need to remain synchronized with values in the other
            dimension.
            """
            return np.log(x)

# Now that the Scale class has been defined, it must be registered so
# that Matplotlib can find it.
mscale.register_scale(FitnessScale)


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
    plt.yscale('log')
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
