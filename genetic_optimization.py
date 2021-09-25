import time
import numpy as np
import random
from game_runner import GameRunner
import pickle
import os
from tqdm import tqdm
from multilayer_controller import PlayerController
from scipy.spatial import distance_matrix
from tabulate import tabulate

# We can now fix the number of nodes to be used in our NN. The first HAS TO BE the number of inputs.
LAYER_NODES = [20, 20, 10, 5]
# Then, we can instantiate the Genetic Hyperparameters.
CX_PROBABILITY = 0.8
CX_ALPHA = 0.5
MUT_PROBABILITY = 0.3
MUTATION_MU = 0
MUTATION_STEP_SIZE = 1.0
MUTATION_INDPB = 0.76
POPULATION_SIZE = 20
GENERATIONS = 20
SAVING_FREQUENCY = 5
TOURNSIZE = 5
LAMBDA = 7  # literature advise to use LAMBDA=5-7
MIN_VALUE_INDIVIDUAL = -1
MAX_VALUE_INDIVIDUAL = 1
EPSILON_UNCORRELATED_MUTATION = 1.0e-6
ALPHA_FITNESS_SHARING = 1.0
# [K. Deb. Multi-objective Optimization using Evolutionary Algorithms. Wiley, Chichester, UK, 2001]
# suggests that a default value for the niche size should be in the range 5–10
# set it to 0.0 to disable the fitness sharing algorithm
NICHE_SIZE = 5.0


class GeneticOptimizer:
    def __init__(
        self,
        game_runner,
        layer_nodes=LAYER_NODES,
        cx_probability=CX_PROBABILITY,
        cx_alpha=CX_ALPHA,
        mut_probability=MUT_PROBABILITY,
        population_size=POPULATION_SIZE,
        lambda_offspring=LAMBDA,
        mutation_mu=MUTATION_MU,
        mutation_step_size=MUTATION_STEP_SIZE,
        mutation_indpb=MUTATION_INDPB,
        niche_size=NICHE_SIZE,
        checkpoint="checkpoint",
        parallel=False,
    ):
        """
        Initializes the Genetic Optimizer.
            :param layer_nodes: The number of nodes in each layer. (list)
            :param generations: The number of generations to run the GA for. (int)
            :param cx_probability: The probability of crossover. (float, 0<=x<=1)
            :param mut_probability: The probability of mutation. (float, 0<=x<=1)ù
            :param lambda_offspring: The scaling factor of the offspring size based on the population size
            :param mutation_mu: The mean of the normal distribution used for mutation. (float)
            :param mutation_step_size: The initial standard deviation of the normal distribution used for mutation. (float)
            :param mutation_indpb: The probability of an individual being mutated. (float, 0<=x<=1)
            :param population_size: The size of the population. (int)
            :param niche_size: The size of the niche considered to keep diversity with the fitness sharing.
                                If it is 0.0, the fitness sharing will be disabled. (float)
            :param checkpoint: The file name to save the checkpoint. (str)
            :param game_runner: The EVOMAN game runner. (GameRunner)
        """
        self.layer_nodes = layer_nodes
        self.checkpoint = checkpoint
        # The biases have to be the same amount of the nodes
        self.bias_no = np.sum(self.layer_nodes) - self.layer_nodes[0]
        self.cx_probability = cx_probability
        self.mut_probability = mut_probability
        self.population_size = population_size
        self.lambda_offspring = lambda_offspring
        self.game_runner = game_runner
        self.parallel = parallel
        self.niche_size = niche_size
        self.mutation_mu = mutation_mu
        self.mutation_step_size = mutation_step_size
        self.mutation_indpb = mutation_indpb
        self.cx_alpha = cx_alpha
        self.logbook = {}
        weights_no = 0
        for i in range(0, len(self.layer_nodes) - 1):
            weights_no += self.layer_nodes[i] * self.layer_nodes[i + 1]
        self.individual_size = weights_no + self.bias_no
        # compute the learning rate as suggested by the book
        # it is usually inversely proportional to the square root of the problem size
        self.learning_rate = 1 / (self.individual_size ** 0.5)
        self.verify_checkpoint()

    def create_population(self, n, wandb_size):
        """
        Creates a population of n individuals.
        Each individual consists in a np.array containing the weights of the NN, and an array containing the mutation step size.
            :param n: The size of the population. (int)
            :param wandb_size: The size of the array containing weights and biases. (int)
        """
        population = []
        for i in range(n):
            individual = {
                "weights_and_biases": np.random.uniform(
                    low=MIN_VALUE_INDIVIDUAL,
                    high=MAX_VALUE_INDIVIDUAL,
                    size=wandb_size,
                ),
                "step_size": self.mutation_step_size,
                "fitness": None,
            }
            population.append(individual)
        return population

    def blend_crossover(self, individual1, individual2, alpha):
        """
        Mates two individuals, randomly choosing a crossover point and performing a blend crossover as seen in book at page 67.
            :param individual1: The first parent. (np.array)
            :param individual2: The second parent. (np.array)
        """
        # For each weight/bias in the array, we decide a random shift quantity
        assert len(individual1["weights_and_biases"]) == len(
            individual2["weights_and_biases"]
        )
        for i in range(len(individual1["weights_and_biases"])):
            crossover = (1 - 2 * alpha) * random.random() - alpha
            individual1["weights_and_biases"][i] = (
                crossover * individual1["weights_and_biases"][i]
                + (1 - crossover) * individual2["weights_and_biases"][i]
            )
            # Then, we invert the two "crossover weights" for the second individual
            individual2["weights_and_biases"][i] = (
                crossover * individual2["weights_and_biases"][i]
                + (1 - crossover) * individual1["weights_and_biases"][i]
            )
        # We can then mutate the sigmas too!
        crossover = (1 - 2 * alpha) * random.random() - alpha
        individual1["step_size"] = (
            crossover * individual1["step_size"]
            + (1 - crossover) * individual2["step_size"]
        )
        individual2["step_size"] = (
            crossover * individual2["step_size"]
            + (1 - crossover) * individual1["step_size"]
        )
        return individual1, individual2

    def mutate_individual(self, individual):
        """
        Mutates an individual using a Gaussian distribution.
            :param individual: The individual to mutate. (np.array)
            :param indpb: The probability of a single weight of the individual being mutated. (float, 0<=x<=1)
        """
        # We mutate the weights and biases
        for i in range(len(individual["weights_and_biases"])):
            if random.random() < self.mutation_indpb:
                individual["weights_and_biases"][i] += (
                    random.gauss(0, 1) * individual["step_size"]
                )
        return individual

    def tournament_selection(self, population, k):
        """
        Selects the best individuals from a population.
            :param population: The population to select from. (list)
            :param k: The number of individuals to select. (int)
        """
        chosen_ones = []
        for i in range(0, k):
            tournament = random.choices(population, k=TOURNSIZE)
            chosen_ones.append(max(tournament, key=lambda x: x["fitness"]))
        return chosen_ones

    def best_selection(self, population, k):
        """
        Selects the best individuals from a population.
            :param population: The population to select from. (list)
            :param k: The number of individuals to select. (int)
        """
        return sorted(population, key=lambda x: x["fitness"])[-k:]

    def verify_checkpoint(self):
        """
        Tries to load the checkpoint if it exists, otherwise it creates the population.
        """
        # We have to define also an evaluation to compute the fitness sharing, if it is enabled
        if self.niche_size > 0:
            print(
                f"Evolutionary process started using the 'Fitness sharing' method with niche_size={self.niche_size}"
            )

        if (not self.parallel) and os.path.isfile(self.checkpoint):
            # If the checkpoint file exists, load it.
            with open(self.checkpoint, "rb") as cp_file:
                cp = pickle.load(cp_file)
                self.population = cp["population"]
                if self.population_size == len(self.population):
                    self.start_gen = cp["generation"]
                    self.logbook = cp["logbook"]
                    random.setstate(cp["rndstate"])
                    print(
                        f"We got a checkpoint and the sizes coincide! Starting from generation no. {self.start_gen}"
                    )
                else:
                    print(
                        "We got a checkpoint, but it was for a different population size. Gotta start from scratch."
                    )
                    self.initialize_population()
        else:
            self.initialize_population()

    def sharing(self, distance, niche_size, alpha):
        """
        Sharing function which count distant neighbourhoods less than close neighbourhoods
            :param distance: Distance between two individuals (float)
            :param niche_size: It is the share radius, the size of a niche in the genotype space; it decides both how many niches can be maintained and the granularity with which different niches can be discriminated (float)
            :param alpha: Determines the shape of the sharing function; for α=1 the function is linear, but for values greater than this the effect of similar individuals in reducing a solution’s fitness falls off more rapidly with distance (float)
        """
        if distance < niche_size:
            return 1 - (distance / niche_size) ** alpha
        else:
            return 0

    def fitness_sharing(self, individual, population, niche_size, alpha):
        """
        Compute the fitness of an individual and adjust it according to the number of individuals falling within some prespecified distance.
            :param individual: individual which you want compute to evaluate
            :param population: array of individual inside the actual population
            :param niche_size: It is the share radius, the size of a niche in the genotype space; it decides both how many niches can be maintained and the granularity with which different niches can be discriminated (float)
            :param alpha: Determines the shape of the sharing function; for α=1 the function is linear, but for values greater than this the effect of similar individuals in reducing a solution’s fitness falls off more rapidly with distance (float)
        """
        # compute the fitness of the individual
        fitness = self.game_runner.evaluate(individual)[0]

        # compute array of distances between the individual and all other individual in the population
        distances = distance_matrix(
            [individual["weights_and_biases"]],
            [individual["weights_and_biases"] for individual in population],
        )[0]
        return (fitness / sum([self.sharing(d, niche_size, alpha) for d in distances]),)

    def uncorrelated_mutation_one_step_size(
        self, mutation_step_size, mu, learning_rate, epsilon
    ):
        """
        Update of the mutation step size. It must be computed before of performing the mutation on the individual.
        """
        mutation_step_size *= np.exp(random.gauss(mu, learning_rate))

        # if the new mutation_step_size is too small, return epsilon
        return mutation_step_size if mutation_step_size > epsilon else epsilon

    def initialize_population(self):
        self.population = self.create_population(
            self.population_size, self.individual_size
        )
        self.start_gen = 0
        self.logbook = {}

    def evaluate_fitness_for_individuals(self, population, evaluate):
        """
        This loops over a given population of individuals,
        and saves the fitness to each Individual object (individual.fitness.values)
        :param population: The population of individuals to evaluate. (list)
        """
        fitnesses = map(evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind["fitness"] = fit

    def clone_individual(self, individual):
        """
        Clone an individual.
        """

        return {
            "weights_and_biases": individual["weights_and_biases"].copy(),
            "step_size": individual["step_size"],
            "fitness": individual["fitness"],
        }

    def evaluate_stats(self, population):
        """
        Compute the statistics of the population.
            :param population: The population of individuals to evaluate. (list)
        """
        return {
            "avg": np.average([ind["fitness"] for ind in population]),
            "min": np.min([ind["fitness"] for ind in population]),
            "max": np.max([ind["fitness"] for ind in population]),
            "std": np.std([ind["fitness"] for ind in population]),
            "best_individual": population[
                np.argmax([ind["fitness"] for ind in population])
            ],
        }

    def print_stats(self):
        """
        Pretty-prints the evolution stats
        """
        data = [
            [
                i,
                generation["avg"],
                generation["max"],
                generation["std"],
                generation["best_individual"]["step_size"],
            ]
            for i, generation in self.logbook.items()
        ]
        print(
            tabulate(
                data,
                headers=[
                    "Generation",
                    "Avg fitness",
                    "Max fitness",
                    "Fitness std_dev",
                    "step_size for the best",
                ],
            )
        )

    def evolve(self, generations):
        """
        This method is responsible of creating the population of individuals,
        """
        """
        Runs the GA for a given number of generations.
        """

        # First, evaluate the whole population's fitnesses.
        # This evaluation is done in different ways according to whether the fitness sharing is enabled or not.
        if self.niche_size > 0:
            self.evaluate_fitness_for_individuals(
                self.population,
                lambda individual: self.fitness_sharing(
                    individual,
                    population=self.population,
                    niche_size=self.niche_size,
                    alpha=ALPHA_FITNESS_SHARING,
                ),
            )
        else:
            self.evaluate_fitness_for_individuals(
                self.population, self.game_runner.evaluate
            )
        for g in tqdm(
            range(generations), desc=f"Run with nodes: {self.layer_nodes}", leave=False
        ):
            if not self.parallel:
                print(
                    f"\n👨‍👩‍👧 Generation {g} is about to give birth to children! 👨‍👩‍👧"
                )
            # We save every SAVING_FREQUENCY generations.
            if g % SAVING_FREQUENCY == 0 and not self.parallel:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(
                    population=self.population,
                    generation=g,
                    logbook=self.logbook,
                    rndstate=random.getstate(),
                )
                with open(self.checkpoint, "wb") as cp_file:
                    pickle.dump(cp, cp_file)

            # if the fitness sharing is enabled, you have to compute it for the new population
            if self.niche_size > 0:
                self.evaluate_fitness_for_individuals(
                    self.population,
                    lambda individual: self.fitness_sharing(
                        individual,
                        population=self.population,
                        niche_size=self.niche_size,
                        alpha=ALPHA_FITNESS_SHARING,
                    ),
                )

            # create a new offspring of size LAMBDA*len(population)
            offspring_size = self.lambda_offspring * len(self.population)
            offspring = []
            for i in range(1, offspring_size, 2):

                # selection of 2 parents with replacement
                parents = self.tournament_selection(self.population, k=2)

                # clone the 2 parents in the new offspring
                offspring.append(self.clone_individual(parents[0]))
                offspring.append(self.clone_individual(parents[1]))

                # apply mutation between the parents in a non-deterministic way
                if random.random() < self.cx_probability:
                    offspring[i - 1], offspring[i] = self.blend_crossover(
                        offspring[i - 1], offspring[i], self.cx_alpha
                    )
                    offspring[i - 1]["fitness"] = None
                    offspring[i]["fitness"] = None

                # apply mutation to the 2 new children
                if random.random() < self.mut_probability:
                    # mutate the step size
                    offspring[i - 1][
                        "step_size"
                    ] = self.uncorrelated_mutation_one_step_size(
                        offspring[i - 1]["step_size"],
                        mu=self.mutation_mu,
                        learning_rate=self.learning_rate,
                        epsilon=EPSILON_UNCORRELATED_MUTATION,
                    )
                    offspring[i][
                        "step_size"
                    ] = self.uncorrelated_mutation_one_step_size(
                        offspring[i]["step_size"],
                        mu=self.mutation_mu,
                        learning_rate=self.learning_rate,
                        epsilon=EPSILON_UNCORRELATED_MUTATION,
                    )
                    # mutate the individuals
                    offspring[i - 1] = self.mutate_individual(offspring[i - 1])
                    offspring[i - 1]["fitness"] = None

                    offspring[i] = self.mutate_individual(offspring[i])
                    offspring[i]["fitness"] = None

            start_time = time.time()

            if self.niche_size > 0:
                # Evaluate the fitness for the whole offspring
                self.evaluate_fitness_for_individuals(
                    offspring, self.game_runner.evaluate
                )
            else:
                # If the fitness sharing is disabled, is not needed to recalculate the fitness each individual

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if ind["fitness"] is None]

                # Then evaluate the fitness of individuals with an invalid fitness
                self.evaluate_fitness_for_individuals(
                    invalid_ind, self.game_runner.evaluate
                )

            print(
                f"Time to evaluate the fitness in the offspring: {round(time.time() - start_time, 3)} seconds"
            )

            # Select the survivors for next generation of individuals only between the new generation
            # (age-based selection)
            offspring = self.best_selection(offspring, len(self.population))

            # The population is entirely replaced by the offspring
            self.population = offspring

            # Compute the stats for the generation, and save them to the logbook.
            self.record = self.evaluate_stats(self.population)
            self.logbook[g] = self.record
            if not self.parallel:
                self.print_stats()

        # Return the best individual
        return self.record["max"], self.record["best_individual"]


if __name__ == "__main__":
    game_runner = GameRunner(PlayerController(LAYER_NODES), enemies=[3], headless=True)
    optimizer = GeneticOptimizer(
        population_size=POPULATION_SIZE, game_runner=game_runner
    )
    max_fitness, best_individual = optimizer.evolve(generations=GENERATIONS)
    if not optimizer.parallel:
        print(
            "Evolution is finished! I saved the best individual in best_individual.txt"
        )
        np.savetxt("best_individual.txt", best_individual["weights_and_biases"])
