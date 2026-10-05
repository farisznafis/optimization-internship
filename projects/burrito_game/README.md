# Burrito Optimization Game

Solutions for [Gurobi's Burrito Optimization Game](https://www.gurobi.com/burrito-optimization-game/):
decide where to park burrito trucks in Burritoville each day to maximise profit.

## Model

Sets: buildings with hungry customers $i$, available truck spots $j$.
Parameters: burrito price $r$, ingredient cost $k$, truck cost per day $f$, and scaled demand
$\alpha_{ij} d_i$ (demand of building $i$ that truck spot $j$ captures, lower for distant spots).

Variables: $x_j = 1$ if a truck is placed at spot $j$, $y_{ij} = 1$ if that truck serves building $i$.

$$
\begin{aligned}
\max\quad & \textstyle\sum_{i,j} (r - k)\,\alpha_{ij} d_i\, y_{ij} - \sum_j f\, x_j \\
\text{s.t.}\quad & \textstyle\sum_j y_{ij} \le 1 && \forall i \quad \text{(one truck per building)} \\
& y_{ij} \le x_j && \forall i, j \quad \text{(only placed trucks serve)} \\
& x_j, y_{ij} \in \{0, 1\}
\end{aligned}
$$

Only pairs with positive scaled demand are modelled. The model is solved with OR-Tools CP-SAT.

## Data

`data/<player>_<round>/` holds the four CSV files exported from the game for one round:
`*_problem_data.csv`, `*_truck_node_data.csv`, `*_demand_node_data.csv` and `*_demand_truck_data.csv`.
Bundled rounds: `fauzi_r1d1`, `muafi_r1d1`, `muafi_r1d1_2`, `zaid_r1d1`.

## Run

```bash
optim burrito-game                                  # default round: muafi_r1d1_2
optim burrito-game --round zaid_r1d1 --plot outputs/burrito_map.png
optim burrito-game --round-dir path/to/your/export  # any exported round
optim burrito-game --list-rounds
```

For `muafi_r1d1_2` the optimal plan places trucks at `truck15`, `truck27` and `truck55` for a profit
of 970. `--plot` saves a map of the buildings, truck spots and the chosen plan.

`burrito_game.ipynb` is the original Colab notebook, including the interactive map from Gurobi's
`show_map` helper.
