"""
Script to tune the hyperparameters.
Uses HyperOpt, a library for hyperparameter tuning.
The GeneticOptimizer object will take care of the evolution and just return the best individual's fitness.
"""
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
from approach1.genetic_optimization import GeneticOptimizer
from game_runner import GameRunner
from multilayer_controller import PlayerController
from hyperopt import hp, fmin, tpe
from hyperopt import SparkTrials

# fixed parameters
ENEMY = 2
GENERATIONS = 25
LAMBDA = 7
MUTATION_STEP_SIZE = 1.


def test_hyperparameters_vector(args):
    """
    Tests an hyperparameter vector with GeneticOptimizer
    """
    layer_nodes = [20] + [int(nodes) for nodes in list(args["layer_nodes"])] + [5]
    print(f"Now trying combination {args}")
    game_runner = GameRunner(PlayerController(layer_nodes), enemies=[ENEMY], headless=True)
    optimizer = GeneticOptimizer(
        layer_nodes=layer_nodes,
        population_size=int(args["population_size"]),
        game_runner=game_runner,
        generations=GENERATIONS,
        lambda_offspring = LAMBDA,
        cx_probability=round(args["cx_probability"], 2),
        cx_alpha=round(args["cx_alpha"], 2),
        mut_probability=round(args["mut_probability"], 2),
        mut_step_size=MUTATION_STEP_SIZE,
        mut_indpb=round(args["mutation_indpb"], 2),
        tournsize=int(args["tournsize"]),
        niche_size=int(args["niche_size"]),
        parallel=True,
    )
    best_individual = optimizer.evolve()
    return -best_individual["fitness"]


space = hp.choice(
    "GA",
    [
        {
            "name": "2-layered NN",
            "population_size": hp.quniform("population_size_2", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_2", 0.4, 1),
            "cx_alpha": hp.uniform("cx_alpha_2", 0, 1),
            "mut_probability": hp.uniform("mut_probability_2", 0, 0.6),
            "niche_size": hp.quniform("niche_size_2", 5, 10, 1),
            "tournsize": hp.quniform("tournsize_2", 5, 10, 1),
            "layer_nodes": [
                hp.quniform("layer_1_2", 10, 30, 1),
                hp.quniform("layer_2_2", 10, 30, 1),
            ],
            "mutation_indpb": hp.uniform("mutation_indpb_2", 0, 1),
        },
        {
            "name": "3-layered NN",
            "population_size": hp.quniform("population_size_3", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_3", 0.4, 1),
            "cx_alpha": hp.uniform("cx_alpha_3", 0, 1),
            "mut_probability": hp.uniform("mut_probability_3", 0, 0.6),
            "niche_size": hp.quniform("niche_size_3", 5, 10, 1),
            "tournsize": hp.quniform("tournsize_3", 5, 10, 1),
            "layer_nodes": [
                hp.quniform("layer_1_3", 10, 30, 1),
                hp.quniform("layer_2_3", 10, 30, 1),
                hp.quniform("layer_3_3", 10, 30, 1),
            ],
            "mutation_indpb": hp.uniform("mutation_indpb_3", 0, 1),
        },
        {
            "name": "4-layered NN",
            "population_size": hp.quniform("population_size_4", 50, 100, 1),
            "cx_probability": hp.uniform("cx_probability_4", 0.4, 1),
            "cx_alpha": hp.uniform("cx_alpha_4", 0, 1),
            "mut_probability": hp.uniform("mut_probability_4", 0, 0.6),
            "niche_size": hp.quniform("niche_size_4", 5, 10, 1),
            "tournsize": hp.quniform("tournsize_4", 5, 10, 1),
            "layer_nodes": [
                hp.quniform("layer_1_4", 10, 30, 1),
                hp.quniform("layer_2_4", 10, 30, 1),
                hp.quniform("layer_3_4", 10, 30, 1),
                hp.quniform("layer_4_4", 10, 30, 1),
            ],
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
    max_evals=100,
)
print("The best combination of hyperparameters is:")
print(best)
