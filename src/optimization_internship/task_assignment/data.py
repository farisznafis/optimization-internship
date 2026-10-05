"""Load employee and task data and compute the skill-matching scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .scoring import CompetencyAssessment, WeightedEuclideanDistance

DATA_DIR = Path("projects/task_assignment/data")
DEFAULT_EMPLOYEES = DATA_DIR / "employees.csv"
DEFAULT_TASKS = DATA_DIR / "tasks.csv"
MINI_EMPLOYEES = DATA_DIR / "mini" / "employees.csv"
MINI_TASKS = DATA_DIR / "mini" / "tasks.csv"

# Columns that are not competencies
EMPLOYEE_META = ["No", "Role"]
TASK_META = ["project_id", "story_points"]


@dataclass
class ProblemData:
    employees: list[str]
    tasks: list[str]
    story_points: dict[str, int]
    company_tasks: dict[str, list[str]]  # project id -> task ids, sorted by project id
    score: dict[str, dict[str, float]]  # score[employee][task]
    info: dict[str, Any] = field(default_factory=dict, repr=False)


def load_problem(employee_path: Path, task_path: Path, overqualification: bool) -> ProblemData:
    """Read both CSV files and score every employee against every task.

    Employee CSV: ``employee_id`` plus one column per competency (``No`` and
    ``Role`` are ignored). Task CSV: ``task_id``, ``project_id``,
    ``story_points`` plus the same competency columns. Competency levels are
    numbers (0-5); blanks count as 0.
    """
    employees_df = pd.read_csv(employee_path, index_col="employee_id")
    employees_df = employees_df.drop(columns=EMPLOYEE_META, errors="ignore")
    tasks_df = pd.read_csv(task_path, index_col="task_id")

    missing = [column for column in TASK_META if column not in tasks_df.columns]
    if missing:
        raise ValueError(f"{task_path} is missing columns: {missing}")
    for name, frame in (("employee_id", employees_df), ("task_id", tasks_df)):
        duplicated = frame.index[frame.index.duplicated()].unique().tolist()
        if duplicated:
            raise ValueError(f"Duplicate {name} values: {duplicated}")

    rcd_df = tasks_df.drop(columns=TASK_META).fillna(0)  # required competencies
    acd_df = employees_df.fillna(0)  # acquired competencies
    if set(rcd_df.columns) != set(acd_df.columns):
        only_tasks = sorted(set(rcd_df.columns) - set(acd_df.columns))
        only_employees = sorted(set(acd_df.columns) - set(rcd_df.columns))
        raise ValueError(
            "Employee and task competency columns differ. "
            f"Only in tasks: {only_tasks}. Only in employees: {only_employees}."
        )
    acd_df = acd_df[rcd_df.columns]

    scorer = CompetencyAssessment if overqualification else WeightedEuclideanDistance
    score, info = scorer(rcd_df, acd_df).fit()

    company_tasks = {
        project: tasks_df.index[tasks_df["project_id"] == project].tolist()
        for project in sorted(tasks_df["project_id"].unique())
    }

    return ProblemData(
        employees=employees_df.index.tolist(),
        tasks=tasks_df.index.tolist(),
        story_points={task: int(sp) for task, sp in tasks_df["story_points"].items()},
        company_tasks=company_tasks,
        score=score,
        info=info,
    )
