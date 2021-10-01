# Assignment1

First assignment, due to 01/10/2021.

### Approach 1

For the first part of the experiment, meaning the evolution part, you need to:

    - Run the approach2/genetic\*optimization.py file. There you can set up the ENEMIES array to change trough different enenemies.
    - The results are stored in the approach1/runs/enemy\*# folder.
    - The `hyperparameter_tuning.py` file is used to tune the hyperparameters of the algorithm through hyperopt - `experiment_runner.py` runs optimization parallely
    Then, to run a good solution, you can run `approach1/play_with_best.py`, selecting the enemy you want to play with. Results are then stored in the `approach1/runs/enemy_#/best_solution.txt` file.

### Approach 2

For the first part of the experiment, meaning the evolution part, you need to: - Run the `approach2/neat*optmization.py` file. There you can set up the ENEMIES array to change trough different enenemies. - The results are stored in the `approach2/runs/enemy*#` folder.

For the second part, where the best individuals are confronted to different enemies, you need to: - Run the `approach2/play_with_best.py` file. There can change the ENEMY parameter to face different enemies (those enemies need to have a created folder and results from the first part of the experiment) - Results are stored in `approach2/runs/enemy*#/games_played.csv`
