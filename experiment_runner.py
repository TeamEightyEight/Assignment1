from multiprocessing import Pool
from genetic_optimization import run_optimization

START_RUN = 1
END_RUN = 10
ENEMY = 5


def run_experiment(run_number):
    """
    Runs an experiment with the default hyperparameters
        :param run_number: number of the run for the logs
    """
    return run_optimization(run_number, ENEMY)


if __name__ == "__main__":
    with Pool(12) as p:
        p.map(run_experiment, range(START_RUN, END_RUN + 1))