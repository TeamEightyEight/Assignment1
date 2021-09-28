from multiprocessing import Pool
from genetic_optimization import run_optimization


def run_experiment(experiment_no):
    """
    Runs an experiment with the default hyperparams
        :param experiment_no: number of the experiment for the logs
    """
    run_optimization(experiment_no)


if __name__ == "__main__":
    start = int(input("Enter the start run no.: "))
    end = int(input("Enter the end run no.: "))
    with Pool(12) as p:
        p.map(run_experiment, range(start, end + 1))
