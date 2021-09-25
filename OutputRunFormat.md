## Standard output format for each run

#### Line-plot across the generations

Create for each run a csv file containing:

```python
# eg. run1__enemy1_ea1.txt
n_gen; 	max_fitness; mean_fitness; best_individual_gain; best_individual
0
1
..
20
```

where:

- `best_individual_gain` is the `max([ (player_energy - enemy_energy) for ind in population])`
-  `best_individual` is the array of the weights+biases for which the individual_gain is the maximum between the population

At the end there should be 10 of these files.

Repeat this process for all 3 enemies.



#### Box-plot across the runs

For each run, consider the column `best_individual_gain` and take the`best_individual` for which the individual gain is maximum. Execute a game with this individual 5 times and create a file like this:

```python
# eg. enemy1_ea1.txt
n_run; 	individual_gains
1;		individual_gain_game1; .. ; individual_gain_game5
2;
..
10;
```

Compute the `mean(fitness_game1; .. ;fitness_game5)` for each run, so end up with 10 values and draw a box plot from them.

Repeat this process for all 3 enemies.

