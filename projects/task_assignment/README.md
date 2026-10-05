# Project 1: Scrum task assignment

Assign sprint tasks from several client projects to a pool of data talents so that few people sit
idle, tasks go to the people whose skills fit them best, and workload stays balanced. The three goals
conflict, so they are combined with **goal programming**. The full problem statement is in
[`optimization_problem.pdf`](optimization_problem.pdf).

## Data

| File | Rows | Columns |
| --- | --- | --- |
| [`data/employees.csv`](data/employees.csv) | 109 talents | `employee_id`, `No`, `Role`, then one column per competency (level 0 to 5) |
| [`data/tasks.csv`](data/tasks.csv) | 300 tasks in 5 projects | `task_id`, `project_id`, `story_points`, then the same competency columns (required level) |
| [`data/mini/`](data/mini/) | 5 talents, 10 tasks, 3 projects | same format with 6 competencies, for quick runs |

Both files must have the same competency columns (order does not matter); blanks count as 0. The
loader checks this and reports the columns that differ.

## Step 1: skill-matching score

Every talent $j$ gets a score $s_{ji}$ for every task $i$. Choose the method with
`OVERQUALIFICATION` in the config.

- **Competency Assessment** (`true`, default). Each competency is weighted by its share of the task's
  total requirement. The weighted gap (talent level minus required level) is split into
  over-qualification and under-qualification, and their mean over all competencies is the
  **Mean Skill Gap (MSG)**. MSG >= 0 means qualified.
- **Weighted Euclidean Distance** (`false`). Distance between talent and task competency vectors,
  ignoring competencies the task does not need and down-weighting surplus skill; the score is
  $1 / (1 + \text{distance})$, so 1 is a perfect match.

Scores are written to `score.csv`.

## Step 2: optimisation model

Variables: $x_{ijk} = 1$ if task $i$ of project $k$ goes to talent $j$; $y_{jk} = 1$ if talent $j$
works on project $k$; $z_{ij} = 1$ if task $i$ goes to talent $j$; $W$ is the largest workload.

Constraints:

1. every task is assigned to exactly one talent;
2. $y_{jk} = 1$ exactly when talent $j$ has a task in project $k$, and each talent works on at most one project;
3. a talent's story points stay within `MAX_EMPLOYEE_WORKLOAD` (default 10);
4. $W$ is at least every talent's workload;
5. $z$ follows $x$: one talent per task, and only talents on the task's project.

Objectives:

| # | Goal | Expression |
| --- | --- | --- |
| 1 | minimise idle talents | $Z_1 = \sum_j (1 - \sum_k y_{jk})$ |
| 2 | maximise assessment score | $Z_2 = \sum_{i,j} s_{ji}\, z_{ij}$ |
| 3 | balance workload | $Z_3 = W$ (minimise) |

Each objective is solved on its own first, giving the best value $Z_k^*$. The multi-objective model
then adds $Z_k - d_k^+ + d_k^- = Z_k^*$ and minimises the weighted, normalised deviation

$$
D = \sum_{k=1}^{3} w_k \, \frac{d_k^+ + d_k^-}{Z_k^*}
$$

with weights `WEIGHT_OBJ1..3` (default 0.03, 0.9, 0.07). Only unwanted deviations exist: objective 1
has no $d^-$ and objective 3 has no $d^+$.

## Run

```bash
optim task-assignment --mini                      # 5 x 10 sample, runs in seconds
optim task-assignment                             # full dataset, needs a full Gurobi license
optim task-assignment --config my.yaml --employees e.csv --tasks t.csv --output outputs/run1
```

The full dataset is a large MIP. Set `TIME_LIMIT` in the config to stop each solve after that many
seconds and keep the best solution found; `MIPGAP` and `MIPGAP_MOO` control when a solve counts as
done. With `DISCORD: true` and `DISCORD_URL` set, progress and gap updates are posted to Discord.

## Output

Written to `--output` (default `outputs/task_assignment/`):

| File | Content |
| --- | --- |
| `score.csv` | skill-matching score of every talent for every task |
| `result_1.csv` ... `result_3.csv`, `result_MOO.csv` | per talent: projects, assigned tasks, story points used and unused, scores |
| `result_*.png` | box plot of the assessment scores of the assigned tasks |
| `score_comparison.png` | the four box plots side by side |

The console also prints, for each objective, the number of active and idle talents and of used and
wasted story points. [`results/`](results/) contains a sample run on the mini dataset, produced by the
team's earlier version of the solver (before constraint 5 applied to all projects, see the changelog),
so a fresh run may differ slightly.

![Score comparison on the mini dataset](results/score_comparison.png)

## Notebook

`task_assignment.ipynb` is the original exploration notebook (Colab). It imports the scoring classes
from an external `yippy` package; the same classes live in
`src/optimization_internship/task_assignment/scoring.py`. Earlier experiments (cosine similarity,
manpower planning, Weighted Euclidean runs) are in [`archive/task_assignment/`](../../archive/task_assignment/).
