from __future__ import print_function
import neat

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
from custom_neat_classes import CoolPopulation88, CoolReporter88

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import pickle
import csv


ENEMY = 8

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        [genome.fitness,genome.player_energy,genome.enemy_energy, genome.individual_gain] = simulation(env,net)
        
    return [genome.fitness,
            genome.player_energy,
            genome.enemy_energy,
            genome.individual_gain]

def best_individual_run(genome,config):
    net = neat.nn.FeedForwardNetwork.create(genome,config)
    [genome.fitness,genome.player_energy,genome.enemy_energy , genome.individual_gain] = simulation(env,net)
    
    return genome.individual_gain
       
# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    ind_gain = p-e
    return f,p,e,ind_gain

# evaluation -> not being use
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

if __name__ == "__main__":
    experiment_name = 'neat_results'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    for run_number in range(1,11):
        env = Environment(experiment_name=experiment_name,
                          enemies=[ENEMY],
                          playermode="ai",
                          player_controller=player_controller(),
                          enemymode="static",
                          randomini = 'yes',
                          level=2,
                          speed="fastest")

        env.state_to_log()
        ini = time.time()  # sets time marker
        run_mode = 'train' # train or test

        # Load configuration.
        config = neat.Config(neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'config-feedforward')

        # Create the population, which is the top-level object for a NEAT run.
        p = CoolPopulation88(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(CoolReporter88(True,run_number,ENEMY))

        # Run until a solution is found or max generation reached
        best_ever,best_last_gen = p.run(eval_genomes,13)

        # Display the winning genome.
        print("\nbest_ever: {!s}".format(best_ever.fitness))

        # Dumping the updated winners list to the file
        pickle_file_name = 'run%d_enemy%d_ea2_pickleBest'% (run_number,ENEMY)
        with open('neat_results/'+pickle_file_name, 'wb') as pickle_out:
            #pickle.dump(winners, pickle_out)
            pickle.dump(best_ever, pickle_out)

        with open('neat_results/'+pickle_file_name, mode='rb') as pickle_in:
            best_pickle = pickle.load(pickle_in)
    

        #Now, for the best individual in the all generations, we run it 5 times and obtain his individual gain for each run
        print("\n#######################################\n")
        box_plot_file_name = 'enemy%d_ea2.txt' % (ENEMY)

        with open('neat_results/'+box_plot_file_name, mode='a', newline='') as box_plot_file:
            box_plot_writer = csv.writer(box_plot_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
            
            individual_gains = []

            for i in range(0,5):
                individual_gain = best_individual_run(best_pickle,config)
                individual_gains.append(individual_gain)

            print(individual_gains)
            box_plot_writer.writerow(individual_gains)



