# Challenge 1: Course selection

Choose which courses to take to finish a degree as cheaply as possible.

## Problem

Data: [`data/course_list.csv`](data/course_list.csv), 27 courses with `course_id`, `group` (`CS` or
`PROF`), `exam_type` (`EXAM` or `EOMA`, end-of-module assessment), `credit` and `cost`.

Let $x_i \in \{0, 1\}$ mean course $i$ is taken. Every variant requires

$$
\sum_i \text{credit}_i\, x_i = 180, \qquad \sum_{i \in CS} \text{credit}_i\, x_i \ge 120 .
$$

| Variant | Extra rule | Objective |
| --- | --- | --- |
| Single objective | no `EXAM` courses | minimise $\sum_i \text{cost}_i\, x_i$ |
| Multi objective | exams allowed | minimise $0.001 \cdot \text{cost} + 0.999 \cdot \#\text{exam courses}$ |

## Run

```bash
optim course-selection
optim course-selection --seed 7     # shuffle first to explore other optimal course sets
```

## Result

Both variants reach the same optimum: total cost **12,356** for M811, M813, S818 (CS) and DD870 (PROF),
no exams. Several course sets tie at this cost, so `--seed` may print a different but equally cheap set.

## Files

- `course_selection.ipynb`: original Colab notebook.
- `data/course_list.xlsx`: original spreadsheet.
