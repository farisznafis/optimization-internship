"""MIP model for assigning Scrum tasks to employees.

Sets: tasks ``i``, employees ``j``, companies/projects ``k``.

Decision variables:
    x[i, j, k]    task i of project k is assigned to employee j
    y[j, k]       employee j works on project k
    z[i, j]       task i is assigned to employee j
    max_workload  largest story-point load of any employee

Objectives (solved one after another, then combined with goal programming):
    1. minimise idle employees         sum_j (1 - sum_k y[j, k])
    2. maximise assessment score       sum_ij score[j][i] * z[i, j]
    3. balance workload                minimise max_workload
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gurobipy as gp
from gurobipy import GRB, quicksum

from .data import ProblemData
from .settings import Settings


@dataclass
class AssignmentModel:
    model: gp.Model
    x: dict[tuple[str, str, str], gp.Var]
    y: dict[tuple[str, str], gp.Var]
    z: dict[tuple[str, str], gp.Var]
    max_workload: gp.Var


def build_model(data: ProblemData, settings: Settings, env: gp.Env | None = None) -> AssignmentModel:
    model = gp.Model("task_assignment", env=env)
    model.Params.Presolve = settings.presolve
    model.Params.MIPFocus = settings.mip_focus
    model.Params.MIPGap = settings.mip_gap
    model.Params.Heuristics = settings.heuristics
    model.Params.Threads = settings.threads
    if settings.time_limit is not None:
        model.Params.TimeLimit = settings.time_limit

    employees, company_tasks = data.employees, data.company_tasks
    x = {
        (i, j, k): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
        for k, tasks in company_tasks.items()
        for i in tasks
        for j in employees
    }
    y = {(j, k): model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}") for j in employees for k in company_tasks}
    z = {
        (i, j): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
        for tasks in company_tasks.values()
        for i in tasks
        for j in employees
    }
    max_workload = model.addVar(
        vtype=GRB.INTEGER, lb=0, ub=settings.max_employee_workload, name="max_workload"
    )

    am = AssignmentModel(model, x, y, z, max_workload)
    _add_constraints(am, data, settings.max_employee_workload)
    model.update()
    return am


def _add_constraints(am: AssignmentModel, data: ProblemData, max_employee_workload: int) -> None:
    model, x, y, z = am.model, am.x, am.y, am.z
    employees, company_tasks, sp = data.employees, data.company_tasks, data.story_points

    # 1. Every task is assigned to exactly one employee.
    for k, tasks in company_tasks.items():
        for i in tasks:
            model.addConstr(quicksum(x[i, j, k] for j in employees) == 1, name=f"assign_{i}")

    # 2. y[j, k] = 1 exactly when employee j has at least one task in project k,
    #    and every employee works on at most one project.
    for j in employees:
        for k, tasks in company_tasks.items():
            assigned = quicksum(x[i, j, k] for i in tasks)
            model.addGenConstrIndicator(y[j, k], True, assigned, GRB.GREATER_EQUAL, 1)
            model.addGenConstrIndicator(y[j, k], False, assigned, GRB.LESS_EQUAL, 0)
        model.addConstr(quicksum(y[j, k] for k in company_tasks) <= 1, name=f"one_project_{j}")

    # 3. An employee's workload in a project does not exceed the capacity.
    for j in employees:
        for k, tasks in company_tasks.items():
            model.addConstr(
                quicksum(sp[i] * x[i, j, k] for i in tasks) <= max_employee_workload,
                name=f"capacity_{j}_{k}",
            )

    # 4. max_workload bounds every employee's total workload.
    for j in employees:
        model.addConstr(
            am.max_workload
            >= quicksum(sp[i] * x[i, j, k] for k, tasks in company_tasks.items() for i in tasks),
            name=f"max_workload_{j}",
        )

    # 5. z links tasks to employees: at most one employee per task, z follows x,
    #    and a task can only go to an employee working on its project.
    for k, tasks in company_tasks.items():
        for i in tasks:
            model.addConstr(quicksum(z[i, j] for j in employees) <= 1, name=f"z_once_{i}")
            for j in employees:
                model.addGenConstrIndicator(x[i, j, k], True, z[i, j], GRB.EQUAL, 1)
                model.addGenConstrIndicator(z[i, j], True, y[j, k], GRB.EQUAL, 1)


def idle_employees(am: AssignmentModel, data: ProblemData) -> gp.LinExpr:
    return quicksum(1 - quicksum(am.y[j, k] for k in data.company_tasks) for j in data.employees)


def assessment_score(am: AssignmentModel, data: ProblemData) -> gp.LinExpr:
    return quicksum(
        data.score[j][i] * am.z[i, j]
        for tasks in data.company_tasks.values()
        for i in tasks
        for j in data.employees
    )


def workload_balance(am: AssignmentModel, data: ProblemData) -> gp.LinExpr:
    return gp.LinExpr(am.max_workload)


@dataclass(frozen=True)
class Objective:
    index: int
    title: str
    sense: int
    expression: Callable[[AssignmentModel, ProblemData], Any]


OBJECTIVES = (
    Objective(1, "Objective 1: Min Idle Employee", GRB.MINIMIZE, idle_employees),
    Objective(2, "Objective 2: Max Assessment Score", GRB.MAXIMIZE, assessment_score),
    Objective(3, "Objective 3: Balancing the Workload", GRB.MINIMIZE, workload_balance),
)


def set_goal_programming(
    am: AssignmentModel,
    expressions: dict[int, Any],
    targets: dict[int, float],
    weights: dict[int, float],
) -> None:
    """Replace the objective with a weighted goal-programming objective.

    For every objective k with a non-zero weight the constraint
    ``Z_k - d_plus_k + d_minus_k = Z*_k`` ties it to its single-objective
    optimum ``Z*_k``. The model then minimises
    ``sum_k w_k * (d_plus_k + d_minus_k) / Z*_k``. Only undesired deviations are
    modelled: objective 1 (minimise) has no d_minus, objective 3 (minimise) has
    no d_plus.
    """
    model = am.model
    d_plus: dict[int, Any] = {3: 0}
    d_minus: dict[int, Any] = {1: 0}
    for k in (1, 2):
        d_plus[k] = model.addVar(lb=0, name=f"d_plus_{k}")
    for k in (2, 3):
        d_minus[k] = model.addVar(lb=0, name=f"d_minus_{k}")

    for k, weight in weights.items():
        if weight != 0:
            model.addConstr(expressions[k] - d_plus[k] + d_minus[k] == targets[k], name=f"goal_{k}")

    normaliser = {k: (1 / value if value != 0 else 0) for k, value in targets.items()}
    model.setObjective(
        quicksum(weights[k] * (d_plus[k] + d_minus[k]) * normaliser[k] for k in (1, 2, 3)),
        GRB.MINIMIZE,
    )
