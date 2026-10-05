"""Turn a solved model into CSV results, statistics and box plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from gurobipy import GRB
from matplotlib.figure import Figure  # no pyplot: works on headless servers and in Docker

from .data import ProblemData
from .model import AssignmentModel

STATUS_NAMES = {getattr(GRB.Status, name): name for name in dir(GRB.Status) if name.isupper()}
RESULT_COLUMNS = ["company", "assigned_task", "sum_sp", "wasted_sp", "assessment_score"]
COMPARISON_LABELS = [
    "Objective 1\nMin Idle Employee",
    "Objective 2\nMax Assessment Score",
    "Objective 3\nBalancing the Workload",
    "MOO with\nGoal Programming",
]


@dataclass
class ObjectiveResult:
    title: str
    status: str
    objective_value: float | None
    assignments: pd.DataFrame  # one row per active employee, columns RESULT_COLUMNS
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.objective_value is not None

    @property
    def scores(self) -> pd.Series:
        """Assessment score of every assigned task."""
        return self.assignments["assessment_score"].explode().dropna().astype(float).reset_index(drop=True)


def _statistics(assignments: pd.DataFrame, data: ProblemData) -> dict[str, int]:
    total_employee = len(data.employees)
    total_sp = sum(data.story_points.values())
    active_employee = len(assignments)
    active_sp = int(assignments["sum_sp"].sum()) if active_employee else 0
    return {
        "total_employee": total_employee,
        "active_employee": active_employee,
        "idle_employee": total_employee - active_employee,
        "total_sp": total_sp,
        "active_sp": active_sp,
        "wasted_sp": total_sp - active_sp,
    }


def collect_result(
    am: AssignmentModel, data: ProblemData, max_employee_workload: int, title: str
) -> ObjectiveResult:
    """Read the incumbent solution (optimal or not) from the model."""
    model = am.model
    status = STATUS_NAMES.get(model.Status, str(model.Status))
    if model.SolCount == 0:
        empty = pd.DataFrame(columns=RESULT_COLUMNS).rename_axis("employee")
        return ObjectiveResult(title, status, None, empty, _statistics(empty, data))

    rows = {}
    for j in data.employees:
        companies, tasks, scores, sp = [], [], [], 0
        for k, project_tasks in data.company_tasks.items():
            for i in project_tasks:
                if am.x[i, j, k].X > 0.5:
                    companies.append(k)
                    tasks.append(i)
                    scores.append(data.score[j][i])
                    sp += data.story_points[i]
        if tasks:
            rows[j] = [companies, tasks, sp, max_employee_workload - sp, scores]

    assignments = pd.DataFrame.from_dict(rows, orient="index", columns=RESULT_COLUMNS)
    assignments.index.name = "employee"
    return ObjectiveResult(title, status, model.ObjVal, assignments, _statistics(assignments, data))


def summarize(result: ObjectiveResult) -> str:
    s = result.stats

    def pct(part: int, whole: int) -> str:
        return f"{part / whole * 100:.2f}%" if whole else "-"

    value = f"{result.objective_value:.4f}" if result.found else "-"
    return "\n".join(
        [
            f"Statistics of {result.title}",
            f"Status\t\t\t\t: {result.status}",
            f"Objective value\t\t\t: {value}",
            f"Total Employee\t\t\t: {s['total_employee']}",
            f"Total Active Employee\t\t: {s['active_employee']}\t{pct(s['active_employee'], s['total_employee'])}",
            f"Total Idle Employee\t\t: {s['idle_employee']}\t{pct(s['idle_employee'], s['total_employee'])}",
            f"Total Story Points\t\t: {s['total_sp']}",
            f"Total Active Story Points\t: {s['active_sp']}\t{pct(s['active_sp'], s['total_sp'])}",
            f"Total Wasted Story Points\t: {s['wasted_sp']}\t{pct(s['wasted_sp'], s['total_sp'])}",
        ]
    )


def save_result(result: ObjectiveResult, output_base: Path) -> None:
    """Write ``<output_base>.csv`` and, if there is anything to plot, ``.png``."""
    result.assignments.to_csv(output_base.with_suffix(".csv"))
    scores = result.scores
    if scores.empty:
        return
    fig = Figure()
    ax = fig.subplots()
    ax.boxplot(scores)
    ax.set_title(f"Assessment Score Boxplot of {result.title}")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")


def comparison_figure(results: list[ObjectiveResult], labels: list[str] = COMPARISON_LABELS) -> Figure:
    """Box plots of the assessment scores of every objective, side by side."""
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()
    ax.boxplot([result.scores for result in results], tick_labels=labels)
    ax.set_title("Overall Assessment Score Boxplot")
    ax.tick_params(axis="x", labelrotation=15)
    return fig


def save_comparison(results: list[ObjectiveResult], output_file: Path) -> None:
    comparison_figure(results).savefig(output_file, bbox_inches="tight")
