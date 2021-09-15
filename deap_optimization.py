from deap import base, creator, tools
import numpy as np
import random
from game_runner import GameRunner
import pickle
import os
from simmys_multilayer_controller import PlayerController


# We can now fix the number of nodes to be used in our NN. The first HAS TO BE the number of inputs.
LAYER_NODES = [20, 30, 10, 5]
# Then, we can instantiate the Genetic Hyperparameters.
CX_PROBABILITY = 0.5
MUT_PROBABILITY = 0.2
POPULATION_SIZE = 100
SAVING_FREQUENCY = 10


class DeapOptimizer:
    def __init__(
        self,
        layer_nodes=LAYER_NODES,
        cx_probability=CX_PROBABILITY,
        mut_probability=MUT_PROBABILITY,
        population_size=POPULATION_SIZE,
        checkpoint="checkpoint",
        game_runner=GameRunner(PlayerController(LAYER_NODES)),
    ):
        """
        Initializes the Deap Optimizer.
        :param layer_nodes: The number of nodes in each layer. (list)
        :param generations: The number of generations to run the GA for. (int)
        :param cx_probability: The probability of crossover. (float, 0<=x<=1)
        :param mut_probability: The probability of mutation. (float, 0<=x<=1)
        :param population_size: The size of the population. (int)
        :param checkpoint: The file name to save the checkpoint. (str)
        """
        self.layer_nodes = layer_nodes
        self.checkpoint = checkpoint
        # The biases have to be the same amount of the nodes
        self.bias_no = np.sum(self.layer_nodes)
        self.cx_probability = cx_probability
        self.mut_probability = mut_probability
        self.population_size = population_size
        self.game_runner = game_runner
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
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=np.prod(self.layer_nodes) + self.bias_no,
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
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Finally, we have to define the evaluation function. To do so, we gotta set up an evoman environment.
        self.toolbox.register("evaluate", self.game_runner.evaluate)
        # If the checkpoint file exists, load it.
        if os.path.isfile(self.checkpoint):
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
        else:  # Otherwise start from scratch.
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
        stats.register("best_individual", np.argmax)
        for g in range(generations):
            print(f"👨‍👩‍👧 Generation {g} is about to give birth to children! 👨‍👩‍👧")
            # We save every SAVING_FREQUENCY generations.
            if g % SAVING_FREQUENCY == 0:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(
                    population=self.population,
                    generation=g,
                    logbook=self.logbook,
                    rndstate=random.getstate(),
                )
                with open(self.checkpoint, "wb") as cp_file:
                    pickle.dump(cp, cp_file)
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population))

            # Clone the selected individuals
            offspring = [self.toolbox.clone(individual) for individual in offspring]

            # Let's mate two random individuals of the offspring, repeating the process for the prob. of this happening*the individuals
            for i in range(0, int(len(offspring) * CX_PROBABILITY), 2):
                child1, child2 = tuple(random.choices(offspring, k=2))
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUT_PROBABILITY:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # Then evaluate their fitnesses.
            self.evaluate_fitness_for_individuals(invalid_ind)
            # Compute the stats for the generation, and save them to the logbook.
            self.record = stats.compile(self.population)
            self.logbook.record(gen=g, evals=len(invalid_ind), **self.record)
            print(f"Right now, the average fitness is: {self.record['avg']}")
            # The population is entirely replaced by the offspring
            self.population = offspring
        # Return the best individual
        return self.population[self.record["best_individual"]]


if __name__ == "__main__":
    game_runner = GameRunner(PlayerController(LAYER_NODES), enemies=[3])
    optimizer = DeapOptimizer(population_size=100, game_runner=game_runner)
    best_individual = optimizer.evolve(generations=20)
    print(f"The best individual is: {best_individual}")
    np.savetxt("best_individual.txt", best_individual)
