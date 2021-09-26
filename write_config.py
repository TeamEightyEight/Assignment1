def set_config(pop_size = 100,
    weight_coeff = 0.4,
    conn_add = 0.05,
    node_add = 0.03,
    num_hidden = 10):
    configs = f'''#--- parameters for the XOR-2 experiment ---#

[NEAT]
fitness_criterion     = max
fitness_threshold     = 100
pop_size              = {pop_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.5
activation_options      = sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.5
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
#The coefficient for the disjoint and excess gene counts� contribution to the genomic distance.
compatibility_disjoint_coefficient = 1w
#The coefficient for each weight, bias, or response multiplier difference�s contribution to the genomic distance
#original 0.5
compatibility_weight_coefficient   = {weight_coeff}

# connection add/remove rates
conn_add_prob           = {conn_add}
conn_delete_prob        = {conn_add}

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct


# node add/remove rates
#original 0.2
node_add_prob           = {node_add}
node_delete_prob        = {node_add}

# network parameters
num_hidden              = {num_hidden}
num_inputs              = 20
num_outputs             = 5

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
#original 0.8
weight_mutate_rate      = 0.9
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
#Individuals whose genomic distance is less than this threshold are considered to be in the same species.
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max 
max_stagnation       = 5
species_elitism      = 0

[DefaultReproduction]
# The number of most-fit individuals in each species that will be preserved as-is from one generation to the next
elitism            = 1
#The fraction for each species allowed to reproduce each generation.
survival_threshold = 1'''

    with open('config-feedforward', 'w') as configfile:
        configfile.write(configs)