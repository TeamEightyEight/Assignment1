from deap import base, creator, tools, algorithms
import numpy as np
import random
from game_runner import GameRunner
import pickle
import os
from tqdm import tqdm
from simmys_multilayer_controller import PlayerController


# We can now fix the number of nodes to be used in our NN. The first HAS TO BE the number of inputs.
LAYER_NODES = [20, 19, 28, 20, 5]
# Then, we can instantiate the Genetic Hyperparameters.
CX_PROBABILITY = 0.79
MUT_PROBABILITY = 0.54
MUTATION_MU = 0
MUTATION_SIGMA = 0.93
MUTATION_INDPB = 0.76
POPULATION_SIZE = 94
GENERATIONS = 13
SAVING_FREQUENCY = 5
TOURNSIZE = 5
LAMBDA = 3


class DeapOptimizer:
    def __init__(
        self,
        layer_nodes=LAYER_NODES,
        cx_probability=CX_PROBABILITY,
        mut_probability=MUT_PROBABILITY,
        population_size=POPULATION_SIZE,
        lambda_offspring=LAMBDA,
        mutation_mu=MUTATION_MU,
        mutation_sigma=MUTATION_SIGMA,
        mutation_indpb=MUTATION_INDPB,
        checkpoint="checkpoint",
        parallel=False,
        game_runner=GameRunner(PlayerController(LAYER_NODES), headless=False),
    ):
        """
        Initializes the Deap Optimizer.
            :param layer_nodes: The number of nodes in each layer. (list)
            :param generations: The number of generations to run the GA for. (int)
            :param cx_probability: The probability of crossover. (float, 0<=x<=1)
            :param mut_probability: The probability of mutation. (float, 0<=x<=1)ù
            :param lambda_offspring: The scaling factor of the offspring size based on the population size
            :param mutation_mu: The mean of the normal distribution used for mutation. (float)
            :param mutation_sigma: The standard deviation of the normal distribution used for mutation. (float)
            :param mutation_indpb: The probability of an individual being mutated. (float, 0<=x<=1)
            :param population_size: The size of the population. (int)
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
        self.mutation_mu = mutation_mu
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        self.register_toolbox()

    def register_toolbox(self):
        """
        Registers the aliases for the Deap Optimizer using Toolbox
        """
        self.toolbox = base.Toolbox()
        # Register the initialization function for an individual.
        self.toolbox.register("attr_float", random.random)
        """
        The following associates the individual alias to the initRepeat function, which creates WEIGHTS_NO 
        individuals using attr_float, the random float function we just created.
        """
        weights_no = 0
        for i in range(0, len(self.layer_nodes) - 1):
            weights_no += self.layer_nodes[i] * self.layer_nodes[i + 1]
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=weights_no + self.bias_no,
        )
        # Note that an individual is a flattened array of the weights.
        # We'll now create a population of individuals in the same way. We can now use a simple list.
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
            n=self.population_size,
        )

        self.population = self.toolbox.population()

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=self.mutation_mu,
            sigma=self.mutation_sigma,
            indpb=self.mutation_indpb,
        )
        self.toolbox.register(
            "select_parents", tools.selTournament, tournsize=TOURNSIZE
        )
        self.toolbox.register("select_survivors", tools.selBest)

        # Finally, we have to define the evaluation function. To do so, we gotta set up an evoman environment.
        self.toolbox.register("evaluate", self.game_runner.evaluate)
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

    def initialize_population(self):
        self.population = self.toolbox.population()
        self.start_gen = 0
        self.logbook = tools.Logbook()

    def evaluate_fitness_for_individuals(self, individuals):
        """
        This loops over a given population of individuals,
        and saves the fitness to each Individual object (individual.fitness.values)
        :param individuals: The population of individuals to evaluate. (list)
        """
        fitnesses = map(self.toolbox.evaluate, individuals)
        for ind, fit in zip(individuals, fitnesses):
            ind.fitness.values = fit

    def evolve(self, generations):
        """
        Runs the GA for a given number of generations.
        """

        # First, evaluate the whole population's fitnesses.
        self.evaluate_fitness_for_individuals(self.population)
        # Add the stats that we'll track to the logbook.
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("std", np.std)
        stats.register("best_individual", np.argmax)
        for g in tqdm(
            range(generations), desc=f"Run with nodes: {self.layer_nodes}", leave=False
        ):
            if not self.parallel:
                print(f"👨‍👩‍👧 Generation {g} is about to give birth to children! 👨‍👩‍👧")
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

            # create a new offspring of size LAMBDA*len(population)
            # literature advise to use LAMBDA=3
            offspring = []
            for i in range(1, self.lambda_offspring * len(self.population), 2):

                # selection of 2 parents with replacement
                parents = self.toolbox.select_parents(self.population, k=2)

                # clone the 2 parents in the new offspring
                offspring.append(self.toolbox.clone(parents[0]))
                offspring.append(self.toolbox.clone(parents[1]))

                # apply mutation between the parents in a non-deterministic way
                if random.random() < self.cx_probability:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

                # apply mutation to the 2 new children
                if random.random() < self.mut_probability:
                    (offspring[i - 1],) = self.toolbox.mutate(offspring[i - 1])
                    del offspring[i - 1].fitness.values

                    (offspring[i],) = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # Then evaluate their fitnesses.
            self.evaluate_fitness_for_individuals(invalid_ind)

            # Select the survivors for next generation of individuals between the old and the new generation
            # (fitness-based selection)
            offspring = self.toolbox.select_survivors(
                self.population + offspring, len(self.population)
            )

            # Compute the stats for the generation, and save them to the logbook.
            self.record = stats.compile(self.population)
            self.logbook.record(gen=g, evals=len(invalid_ind), **self.record)
            if not self.parallel:
                print(f"Right now, the average fitness is: {self.record['avg']}")
            # The population is entirely replaced by the offspring
            self.population = offspring
        # Return the best individual
        return self.record["max"], self.population[self.record["best_individual"]]


if __name__ == "__main__":
    game_runner = GameRunner(PlayerController(LAYER_NODES), enemies=[3])
    optimizer = DeapOptimizer(population_size=POPULATION_SIZE, game_runner=game_runner)
    max_fitness, best_individual = optimizer.evolve(generations=GENERATIONS)
    if not optimizer.parallel:
        print(
            "Evolution is finished! I saved the best individual in best_individual.txt"
        )
        np.savetxt("best_individual.txt", best_individual)
