"""Implements the core evolution algorithm."""
from __future__ import division, print_function

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

import time
import csv
import os

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

# TODO: Add a curses-based reporter.
RUNS_DIR = 'ea2_runs'

class CompleteExtinctionException(Exception):
    pass


class CoolPopulation88(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                #if best is None or g.fitness > best.fitness:
                if best is None or g.individual_gain > best.individual_gain or ((g.individual_gain == best.individual_gain) and (g.fitness > best.fitness)): #first criteria is individual gain, then fitness
                    best = g

            self.reporters.post_evaluate(self.config, self.population, self.species, best)
            self.reporters.plot_reporter(self.config, self.generation, self.population, best, self.species)    #need to be done before reproduction

            # Track the best genome ever seen.
            #if self.best_genome is None or best.fitness > self.best_genome.fitness:
            if self.best_genome is None or best.individual_gain > self.best_genome.individual_gain or ((best.individual_gain == self.best_genome.individual_gain) and (best.fitness > self.best_genome.fitness)):
                self.best_genome = best
                

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            
            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        self.reporters.final_plot_report()
        return self.best_genome,best


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config, population, species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config, population, species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config, generation, best):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)

    def plot_reporter(self, config, generation, population, best_genome, species_set):
        for r in self.reporters:
            r.plot_reporter(config, generation, population, best_genome, species_set)

    def final_plot_report(self):
        for r in self.reporters:
            r.final_plot_report()

    def best_change(self,old_best, new_best):
        for r in self.reporters:
            r.best_change(old_best, new_best)

class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class CoolReporter88(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail,run_number,enemy):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        
        self.final_report_list = []
        self.generations = []
        self.max_fitnesses = []
        self.mean_fitnesses = []
        self.max_individual_gains = []

        self.run_number = run_number
        self.enemy      = enemy

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - best_species_id {2} - best_genome_key/id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)

    def plot_reporter(self, config, generation, population, best_genome, species_set):
        fitnesses = [c.fitness for c in itervalues(population)]
        ind_gains = [c.individual_gain for c in itervalues(population)]
        
        fit_mean = mean(fitnesses)
        
        print("  n_gen  max_fitness  mean_fitness  best_individual_gain")
        print("  =====  ===========  ============  ====================")
        print("   {0}   {1:3.5f}      {2:3.5f}      {3:3.5f}  ".format(generation,
                                                                      best_genome.fitness,
                                                                      fit_mean,
                                                                      best_genome.individual_gain))

        self.generations.append(generation)
        self.max_fitnesses.append(best_genome.fitness)
        self.mean_fitnesses.append(fit_mean)
        self.max_individual_gains.append(best_genome.individual_gain)

        generation_values = {
            'generation'    : generation,
            'max_fitness'   : best_genome.fitness,
            'mean_fitness'  : fit_mean,
            'max_ind_gain'  : best_genome.individual_gain
        }

        self.final_report_list.append(generation_values)

    def final_plot_report(self):

        base_path = os.path.join(RUNS_DIR, "enemy_" + str(self.enemy))
        os.makedirs(
            base_path,
            exist_ok=True,
        )
        
        csv_file_name = f"logbook_run_{self.run_number}.csv"

        with open(os.path.join(base_path,csv_file_name), mode='w', newline='') as line_plot_file:
            line_plot_writer = csv.writer(line_plot_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
            
            print("  n_gen  max_fitness  mean_fitness  best_individual_gain")
            print("  =====  ===========  ============  ====================")
            
            line_plot_writer.writerow(['n_gen','max_fitness','avg_fitness','best_individual_gain'])
            for gen_values in self.final_report_list:
                print("    {0}      {1:3.5f}     {2:3.5f}     {3}".format(gen_values['generation'],
                                                                   gen_values['max_fitness'],
                                                                   gen_values['mean_fitness'],
                                                                   gen_values['max_ind_gain']))
        
                line_plot_writer.writerow([gen_values['generation'],gen_values['max_fitness'],gen_values['mean_fitness'],gen_values['max_ind_gain']])
                