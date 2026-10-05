"""Challenge 1: course selection with CP-SAT.

Pick a set of courses worth exactly 180 credits, at least 120 of them from the
CS group, at minimum cost. Two variants are solved:

* single objective: exam-based courses are not allowed, minimise total cost;
* multi objective: exams are allowed but penalised, minimise
  ``w_cost * total_cost + w_exam * number_of_exam_courses``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

DEFAULT_DATA = Path("projects/course_selection/data/course_list.csv")
REQUIRED_COLUMNS = {"course_id", "group", "exam_type", "credit", "cost"}

TOTAL_CREDITS = 180
MIN_CS_CREDITS = 120


@dataclass
class Selection:
    status: str
    courses: pd.DataFrame

    @property
    def found(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")

    @property
    def total_cost(self) -> int:
        return int(self.courses["cost"].sum())

    @property
    def total_credits(self) -> int:
        return int(self.courses["credit"].sum())

    @property
    def exam_courses(self) -> int:
        return int((self.courses["exam_type"] == "EXAM").sum())


def load_courses(path: Path = DEFAULT_DATA, seed: int | None = None) -> pd.DataFrame:
    """Load the course list. With ``seed`` the rows are shuffled, which can surface
    a different course set among equally cheap optimal solutions."""
    courses = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(courses.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if seed is not None:
        courses = courses.sample(frac=1, random_state=seed).reset_index(drop=True)
    return courses


def _base_model(
    courses: pd.DataFrame, total_credits: int, min_cs_credits: int
) -> tuple[cp_model.CpModel, list[cp_model.IntVar]]:
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"course_{i}") for i in range(len(courses))]
    credits = courses["credit"].tolist()
    is_cs = (courses["group"] == "CS").tolist()

    model.add(sum(c * xi for c, xi in zip(credits, x)) == total_credits)
    model.add(sum(c * xi for c, xi, cs in zip(credits, x, is_cs) if cs) >= min_cs_credits)
    return model, x


def _solve(model: cp_model.CpModel, x: list, courses: pd.DataFrame) -> Selection:
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Selection(status_name, courses.iloc[0:0])

    chosen = [i for i, xi in enumerate(x) if solver.value(xi)]
    return Selection(status_name, courses.iloc[chosen].reset_index(drop=True))


def solve_single_objective(
    courses: pd.DataFrame, total_credits: int = TOTAL_CREDITS, min_cs_credits: int = MIN_CS_CREDITS
) -> Selection:
    """Minimise cost with exam-based courses forbidden."""
    model, x = _base_model(courses, total_credits, min_cs_credits)
    is_exam = (courses["exam_type"] == "EXAM").tolist()
    for xi, exam in zip(x, is_exam):
        if exam:
            model.add(xi == 0)

    model.minimize(sum(c * xi for c, xi in zip(courses["cost"].tolist(), x)))
    return _solve(model, x, courses)


def solve_multi_objective(
    courses: pd.DataFrame,
    weight_cost: float = 0.001,
    weight_exam: float = 0.999,
    total_credits: int = TOTAL_CREDITS,
    min_cs_credits: int = MIN_CS_CREDITS,
) -> Selection:
    """Minimise a weighted sum of total cost and number of exam-based courses."""
    model, x = _base_model(courses, total_credits, min_cs_credits)
    total_cost = sum(c * xi for c, xi in zip(courses["cost"].tolist(), x))
    is_exam = (courses["exam_type"] == "EXAM").tolist()
    total_exam = sum(xi for xi, exam in zip(x, is_exam) if exam)

    model.minimize(weight_cost * total_cost + weight_exam * total_exam)
    return _solve(model, x, courses)


def _report(title: str, selection: Selection) -> None:
    print(f"\n=== {title} ===")
    if not selection.found:
        print(f"No solution found (status: {selection.status})")
        return
    print(f"Status        : {selection.status}")
    print(f"Total cost    : {selection.total_cost}")
    print(f"Total credits : {selection.total_credits}")
    print(f"Exam courses  : {selection.exam_courses}")
    print(selection.courses.to_string(index=False))


def run(data: Path = DEFAULT_DATA, seed: int | None = None) -> int:
    courses = load_courses(data, seed)
    single = solve_single_objective(courses)
    multi = solve_multi_objective(courses)

    _report("Single objective: minimise cost, no exams", single)
    _report("Multi objective: weighted cost + number of exams", multi)
    return 0 if single.found or multi.found else 1
