"""End-to-end run: data -> model -> three single objectives -> goal programming."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..gurobi import make_env
from .callback import GapCallback
from .data import load_problem
from .model import OBJECTIVES, build_model, set_goal_programming
from .notify import Notifier
from .reporting import ObjectiveResult, collect_result, save_comparison, save_result, summarize
from .settings import Settings

log = logging.getLogger(__name__)

HEADER = """
==============================================

    TASK ASSIGNMENT OPTIMIZATION PROBLEM

==============================================
"""


def run(
    settings: Settings,
    employee_path: Path,
    task_path: Path,
    output_dir: Path,
    webhook_url: str | None = None,
    solver_log: bool = True,
) -> list[ObjectiveResult]:
    """Solve the three single objectives and the goal-programming model.

    Writes ``score.csv``, ``result_{1,2,3}.csv/png``, ``result_MOO.csv/png`` and
    ``score_comparison.png`` to ``output_dir`` and returns the four results.
    """
    if settings.discord and not webhook_url:
        log.warning("DISCORD is enabled in the config but DISCORD_URL is not set")
    notify = Notifier(webhook_url if settings.discord else None)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(HEADER)
    notify(f"Task Assignment Optimization Problem: START with {settings.metric}")

    with notify.stage("Defining data structure"):
        data = load_problem(employee_path, task_path, settings.overqualification)
        pd.DataFrame.from_dict(data.score, orient="index").to_csv(output_dir / "score.csv")
        log.info(
            "%d employees, %d tasks, %d projects",
            len(data.employees),
            len(data.tasks),
            len(data.company_tasks),
        )

    with make_env(output=solver_log) as env:
        with notify.stage("Model construction"):
            am = build_model(data, settings, env)

        try:
            results: list[ObjectiveResult] = []
            expressions, targets = {}, {}
            for objective in OBJECTIVES:
                with notify.stage(objective.title):
                    expressions[objective.index] = objective.expression(am, data)
                    am.model.setObjective(expressions[objective.index], objective.sense)
                    am.model.optimize()
                    result = collect_result(am, data, settings.max_employee_workload, objective.title)
                    # Z*_k: the single-objective optimum used as goal for the MOO.
                    targets[objective.index] = result.objective_value or 0.0
                    save_result(result, output_dir / f"result_{objective.index}")
                    print(summarize(result) + "\n")
                    results.append(result)

            with notify.stage("MOO with Goal Programming"):
                set_goal_programming(am, expressions, targets, settings.weights)
                am.model.Params.MIPGap = settings.mip_gap_moo
                am.model.optimize(GapCallback(notify))
                result = collect_result(am, data, settings.max_employee_workload, "Multi-Objective")
                save_result(result, output_dir / "result_MOO")
                print(summarize(result) + "\n")
                results.append(result)
        finally:
            am.model.dispose()

    save_comparison(results, output_dir / "score_comparison.png")
    notify(f"Task Assignment Optimization Problem: DONE, results in {output_dir}")
    return results
