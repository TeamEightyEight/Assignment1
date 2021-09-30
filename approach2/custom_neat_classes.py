"""Implements the core evolution algorithm."""
from __future__ import division, print_function

from neat.reporting import ReporterSet, BaseReporter
from neat.population import Population, CompleteExtinctionException
from neat.six_util import iteritems, itervalues

import csv
import os

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

RUNS_DIR = 'runs'

class EvomanPopulation(Population):
    def __init__(self, config):
        super().__init__(config)

    def run(self, fitness_function, n=None):

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


            self.reporters.post_evaluate(self.config,self.generation, self.population, best)

            # Track the best genome ever seen.x
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

        # self.reporters.reporters[0].final_plot_report()
        return self.best_genome,best


class EvomanReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail,run_number,enemy):
        super().__init__()

        self.final_report_list = []
        self.generations = []
        self.max_fitnesses = []
        self.mean_fitnesses = []
        self.max_individual_gains = []
        self.run_number = run_number
        self.enemy      = enemy


    def post_evaluate(self, config, generation, population, best_genome):
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

    def plot_report(self):

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
                