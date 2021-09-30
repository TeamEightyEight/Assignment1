from __future__ import print_function
import neat

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from neat_controller import player_controller
from custom_neat_classes import EvomanPopulation,EvomanReporter

# imports other libs
import os
import pickle

ENEMIES = [1]
GENERATIONS = 25
RUNS_DIR = 'runs'
NUM_RUNS = 10

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        [genome.fitness,genome.player_energy,genome.enemy_energy, genome.individual_gain] = simulation(env,net)
        
    return [genome.fitness,
            genome.player_energy,
            genome.enemy_energy,
            genome.individual_gain]

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    ind_gain = p-e
    return f,p,e,ind_gain

if __name__ == "__main__":
    for ENEMY in ENEMIES:
        experiment_name = 'neat_results2'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # choose this for not using visuals and thus making experiments faster
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"


        for run_number in range(1,NUM_RUNS+1):
            env = Environment(experiment_name=experiment_name,
                              enemies=[ENEMY],
                              playermode="ai",
                              player_controller=player_controller(),
                              enemymode="static",
                              randomini = 'yes',
                              level=2,
                              speed="fastest",
                              savelogs="no",
                              logs="off"
            )


            # Load configuration.
            config = neat.Config(neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                'config-feedforward')

            # Create the population, which is the top-level object for a NEAT run.
            p = EvomanPopulation(config)

            # Add a stdout reporter to show progress in the terminal.
            evoman_reporter = EvomanReporter(True,run_number,ENEMY)
            p.add_reporter(evoman_reporter)

            # Run until a solution is found or max generation reached
            best_ever,best_last_gen = p.run(eval_genomes,GENERATIONS)
            evoman_reporter.plot_report()

            # Display the winning genome.
            print("\nbest_ever: {!s}".format(best_ever.fitness))

            base_path = os.path.join(RUNS_DIR, "enemy_" + str(ENEMY))
            os.makedirs(
                base_path,
                exist_ok=True,
            )

            # Dumping the updated winners list to the file
            best_file_name = 'best_individual_run_%d'% (run_number)

            with open(os.path.join(base_path, best_file_name), 'wb') as file_out:
                pickle.dump(best_ever, file_out)