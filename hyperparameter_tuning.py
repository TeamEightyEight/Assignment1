"""
Script to tune the hyperparameters.
Uses HyperOpt, a library for hyperparameter tuning.
The DeapOptimizer object will take care of the evolution and just return the best individual's fitness.
"""
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
from deap_optimization import DeapOptimizer
from game_runner import GameRunner
from simmys_multilayer_controller import PlayerController
from hyperopt import hp, fmin, tpe, space_eval
from hyperopt import SparkTrials, STATUS_OK


def test_hyperparameters_vector(args):
    """
    Tests an hyperparameter vector with DeapOptimizer
    """
    layer_nodes = [20] + [int(nodes) for nodes in list(args["layer_nodes"])] + [5]
    print(f"Now trying combination {args}")
    game_runner = GameRunner(PlayerController(layer_nodes), enemies=[3], headless=True)
    optimizer = DeapOptimizer(
        layer_nodes=layer_nodes,
        population_size=int(args["population_size"]),
        game_runner=game_runner,
        cx_probability=args["cx_probability"],
        mut_probability=args["mut_probability"],
        mutation_sigma=args["mutation_sigma"],
        mutation_indpb=args["mutation_indpb"],
        parallel=True,
    )
    max_fitness, best_individual = optimizer.evolve(
        generations=int(args["generations"])
    )
    return -max_fitness


space = hp.choice(
    "GA",
    [
        {
            "name": "1-layered NN",
            "population_size": hp.quniform("population_size_1", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_1", 0, 1),
            "mut_probability": hp.uniform("mut_probability_1", 0, 1),
            "generations": hp.quniform("generations_1", 5, 25, 1),
            "layer_nodes": [hp.quniform("layer_1_1", 10, 30, 1)],
            "mutation_sigma": hp.uniform("mutation_sigma_1", 0, 1),
            "mutation_indpb": hp.uniform("mutation_indpb_1", 0, 1),
        },
        {
            "name": "2-layered NN",
            "population_size": hp.quniform("population_size_2", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_2", 0, 1),
            "mut_probability": hp.uniform("mut_probability_2", 0, 1),
            "generations": hp.quniform("generations_2", 5, 25, 1),
            "layer_nodes": [
                hp.quniform("layer_1_2", 10, 30, 1),
                hp.quniform("layer_2_2", 10, 30, 1),
            ],
            "mutation_sigma": hp.uniform("mutation_sigma_2", 0, 1),
            "mutation_indpb": hp.uniform("mutation_indpb_2", 0, 1),
        },
        {
            "name": "3-layered NN",
            "population_size": hp.quniform("population_size_3", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_3", 0, 1),
            "mut_probability": hp.uniform("mut_probability_3", 0, 1),
            "generations": hp.quniform("generations_3", 5, 25, 1),
            "layer_nodes": [
                hp.quniform("layer_1_3", 10, 30, 1),
                hp.quniform("layer_2_3", 10, 30, 1),
                hp.quniform("layer_3_3", 10, 30, 1),
            ],
            "mutation_sigma": hp.uniform("mutation_sigma_3", 0, 1),
            "mutation_indpb": hp.uniform("mutation_indpb_3", 0, 1),
        },
        {
            "name": "4-layered NN",
            "population_size": hp.quniform("population_size_4", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_4", 0, 1),
            "mut_probability": hp.uniform("mut_probability_4", 0, 1),
            "generations": hp.quniform("generations_4", 5, 25, 1),
            "layer_nodes": [
                hp.quniform("layer_1_4", 10, 30, 1),
                hp.quniform("layer_2_4", 10, 30, 1),
                hp.quniform("layer_3_4", 10, 30, 1),
                hp.quniform("layer_4_4", 10, 30, 1),
            ],
            "mutation_sigma": hp.uniform("mutation_sigma_4", 0, 1),
            "mutation_indpb": hp.uniform("mutation_indpb_4", 0, 1),
        },
    ],
)
spark_trials = SparkTrials()
best = fmin(
    test_hyperparameters_vector,
    space,
    trials=spark_trials,
    algo=tpe.suggest,
    max_evals=30,
)
print(best)
