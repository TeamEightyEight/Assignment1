import time

from deap import base, creator, tools, algorithms
import numpy as np
import random
from game_runner import GameRunner
import pickle
import os
from tqdm import tqdm
from simmys_multilayer_controller import PlayerController
from scipy.spatial import distance_matrix

# We can now fix the number of nodes to be used in our NN. The first HAS TO BE the number of inputs.
LAYER_NODES = [20, 5, 5]
# Then, we can instantiate the Genetic Hyperparameters.
CX_PROBABILITY = 0.8
CX_ALPHA = 0.5
MUT_PROBABILITY = 0.3
MUTATION_MU = 0
MUTATION_STEP_SIZE = 1.0
MUTATION_INDPB = 0.76
POPULATION_SIZE = 10
GENERATIONS = 25
SAVING_FREQUENCY = 5
TOURNSIZE = 5
LAMBDA = 5
MIN_VALUE_INDIVIDUAL = -1
MAX_VALUE_INDIVIDUAL = 1
EPSILON_UNCORRELATED_MUTATION = 1.e-6

# [K. Deb. Multi-objective Optimization using Evolutionary Algorithms. Wiley, Chichester, UK, 2001]
# suggests that a default value for the niche size should be in the range 5–10
NICHE_SIZE = 5.0
ALPHA_FITNESS_SHARING = 1.0

class DeapOptimizer:
    def __init__(
        self,
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
        game_runner=GameRunner(PlayerController(LAYER_NODES)),
    ):
        """
        Initializes the Deap Optimizer.
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
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # at each individual it is assigned an initial mutation step size
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, mutation_step=self.mutation_step_size)
        self.register_toolbox()

    def register_toolbox(self):
        """
        Registers the aliases for the Deap Optimizer using Toolbox
        """
        self.toolbox = base.Toolbox()

        # Register the initialization function for an individual.
        self.toolbox.register("attr_float", lambda: random.uniform(MIN_VALUE_INDIVIDUAL, MAX_VALUE_INDIVIDUAL))

        """
        The following associates the individual alias to the initRepeat function, which creates WEIGHTS_NO 
        individuals using attr_float, the random float function we just created.
        """
        weights_no = 0
        for i in range(0, len(self.layer_nodes) - 1):
            weights_no += self.layer_nodes[i] * self.layer_nodes[i + 1]
        individual_size = weights_no + self.bias_no

        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=individual_size,
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

        self.toolbox.register("mate", tools.cxBlend, alpha=self.cx_alpha)
        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=self.mutation_mu,
            indpb=self.mutation_indpb,
        )

        # compute the learning rate as suggested by the book
        # it is usually inversely proportional to the square root of the problem size
        learning_rate = 1/(individual_size**0.5)
        self.toolbox.register(
            "mutate_step_size",
            self.uncorrelated_mutation_one_step_size,
            mu=self.mutation_mu,
            learning_rate=learning_rate,
            epsilon=EPSILON_UNCORRELATED_MUTATION
        )
        self.toolbox.register(
            "select_parents", tools.selTournament, tournsize=TOURNSIZE
        )
        self.toolbox.register("select_survivors", tools.selBest)

        # Finally, we have to define the evaluation function. To do so, we gotta set up an evoman environment.
        self.toolbox.register("evaluate", self.game_runner.evaluate)

        # We have to define also an evaluation to compute the fitness sharing, if it is enabled
        if self.niche_size > 0:
            self.toolbox.register("evaluate_sharing", self.fitness_sharing, niche_size=self.niche_size, alpha=ALPHA_FITNESS_SHARING)

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
        distances = distance_matrix([individual], population)[0]
        return fitness / sum([self.sharing(d, niche_size, alpha) for d in distances]),

    def uncorrelated_mutation_one_step_size(self, mutation_step_size, mu, learning_rate, epsilon):
        """
        Update of the mutation step size. It must be computed before of performing the mutation on the individual.
        """
        mutation_step_size *= np.exp(random.gauss(mu, learning_rate))

        # if the new mutation_step_size is too small, return epsilon
        return mutation_step_size if mutation_step_size > epsilon else epsilon

    def initialize_population(self):
        self.population = self.toolbox.population()
        self.start_gen = 0
        self.logbook = tools.Logbook()


    def evaluate_fitness_for_individuals(self, population, evaluate):
        """
        This loops over a given population of individuals,
        and saves the fitness to each Individual object (individual.fitness.values)
        :param population: The population of individuals to evaluate. (list)
        """
        fitnesses = map(evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

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
                self.population
                , lambda individual: self.toolbox.evaluate_sharing(individual, population=self.population)
            )
        else:
            self.evaluate_fitness_for_individuals(
                self.population
                , self.toolbox.evaluate
            )

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
                print(f"\n👨‍👩‍👧 Generation {g} is about to give birth to children! 👨‍👩‍👧")
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
                    self.population
                    , lambda individual: self.toolbox.evaluate_sharing(individual, population=self.population)
                )

            print(f"Fitnesses in generation {g} of element 0: {self.population[0].fitness}")
            print(f"Mutation step sizes in generation {g}: {[ind.mutation_step for ind in self.population]}")

            # create a new offspring of size LAMBDA*len(population)
            # literature advise to use LAMBDA=5-7
            offspring_size = self.lambda_offspring * len(self.population)
            offspring = []
            for i in range(1, offspring_size, 2):

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

                # mutate the step size
                offspring[i - 1].mutation_step = self.toolbox.mutate_step_size(offspring[i - 1].mutation_step)
                offspring[i].mutation_step = self.toolbox.mutate_step_size(offspring[i].mutation_step)

                # apply mutation to the 2 new children
                if random.random() < self.mut_probability:

                    # mutate the individuals
                    (offspring[i - 1],) = self.toolbox.mutate(offspring[i - 1], sigma=offspring[i - 1].mutation_step)
                    del offspring[i - 1].fitness.values

                    (offspring[i],) = self.toolbox.mutate(offspring[i],  sigma=offspring[i].mutation_step)
                    del offspring[i].fitness.values

            start_time = time.time()

            if self.niche_size > 0:
                # Evaluate the fitness for the whole offspring
                self.evaluate_fitness_for_individuals(offspring, self.toolbox.evaluate)
            else:
                # If the fitness sharing is disabled, is not needed to recalculate the fitness each individual

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

                # Then evaluate the fitness of individuals with an invalid fitness
                self.evaluate_fitness_for_individuals(invalid_ind, self.toolbox.evaluate)

            print(f"Time to evaluate the fitness in the offspring: {time.time() - start_time} seconds")

            # Select the survivors for next generation of individuals only between the new generation
            # (age-based selection)
            offspring = self.toolbox.select_survivors(
               offspring, len(self.population)
            )

            # The population is entirely replaced by the offspring
            self.population = offspring

            print(f"Fitnesses sharing in generation {g} of element 0: {self.population[0].fitness}")

            # Compute the stats for the generation, and save them to the logbook.
            self.record = stats.compile(self.population)
            self.logbook.record(gen=g, evals=offspring_size, **self.record)
            if not self.parallel:
                print(f"Right now, the average fitness is: {self.record['avg']}\n")

        # Return the best individual
        return self.record["max"], self.population[self.record["best_individual"]]


if __name__ == "__main__":
    game_runner = GameRunner(PlayerController(LAYER_NODES), enemies=[3], headless=True)
    optimizer = DeapOptimizer(population_size=POPULATION_SIZE, game_runner=game_runner)
    max_fitness, best_individual = optimizer.evolve(generations=GENERATIONS)
    if not optimizer.parallel:
        print(
            "Evolution is finished! I saved the best individual in best_individual.txt"
        )
        np.savetxt("best_individual.txt", best_individual)
