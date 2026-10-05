# Changelog

## Unreleased

- Gradio web demo (`app.py`) with a tab per project, deployable to Hugging Face Spaces.
- GitHub Actions: tests and lint, Docker build smoke test, automatic Space deployment from `main`.
- `course_selection` solvers take the credit rules as parameters; `burrito_game.draw_plan` and
  `reporting.comparison_figure` return matplotlib figures for reuse in the demo.

## 1.0.0 (2026-10-04)

The repository was restructured into an installable package with one CLI (`optim`) and a Docker image.

### Layout

- Code moved to `src/optimization_internship/`. Data, notebooks and per-project READMEs moved to
  `projects/<name>/`. The config moved to `configs/task_assignment.yaml`, and `.env.example` to the root.
- Deprecated notebooks, research scripts, old outputs and duplicate datasets moved to `archive/`.
- `requirements.txt` re-saved as UTF-8 (it was UTF-16, which pip cannot always read) and pinned to
  versions that were tested. `ortools` was added (it was missing); `scipy`, `seaborn` and `ace`
  were removed (not used by the code).

### Challenges

- The three scripts exported from Colab (`!pip install`, Google Drive paths, input files that
  were not in the repo) were rewritten as CLI commands that read the bundled data.
- Stock selection: added the link $x_i \le y_i$ so that only selected stocks receive money; before,
  the minimum-allocation rule could be skipped for unselected stocks. The budget is now
  $\sum x_i = 1$, matching the model description (it was $\le 1$).
- Burrito game: the map is drawn locally with matplotlib (`--plot`) instead of downloading
  `show_map.py` from GitHub at runtime.

### Task assignment

- **Bug fix:** constraint 5 ("each task is assigned to at most one employee" via $z$) only covered the
  tasks of the last project, because it reused a leaked loop variable. It now covers all tasks.
  Before the fix, objective 2 could count the score of a task for several employees.
- `y` variables were created inside the innermost loop, once per task, adding about 32,000 unused
  variables on the full dataset. Each variable is now created once, which also helps with the
  size-limited license.
- Errors are no longer swallowed: a failed step used to print a message and continue with empty data.
- Results from a time-limited or gap-limited solve are now reported (before, only `OPTIMAL`).
- Plots are saved without `plt.show()` and the background timer thread, so runs work on servers and in Docker.
- Paths no longer depend on running from the `solution/` folder; the circular imports between
  `helper`, `config` and `creds` are gone.
- New config key `TIME_LIMIT`. Unknown config keys are reported instead of ignored.
- The input loader checks that employee and task files have the same competency columns.
- Discord notifications use a timeout, and a failing webhook no longer stops the run.
- Output file `score_comaprison.png` renamed to `score_comparison.png`.
