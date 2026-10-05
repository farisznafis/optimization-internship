# optimizer/create_objective.py

import pandas as pd
from .callback import GapCallback
from . import helper, config, utils
from gurobipy import GRB, Model, quicksum
from typing import Dict, List, Tuple, Any


def objective1(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    y: Dict[Tuple[str, str], Any],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    mu_Z_star: Dict[int, float],
) -> Tuple[Any, Dict[int, float], pd.Series]:
    """
    Sets the first objective to minimize the number of idle employees and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[str]]): Dictionary of company tasks.
        y (Dict[Tuple[str, str], Any]): Decision variable y.
        score (Dict[str, Dict[str, float]]): Dictionary of metric scores for each employee-task pair.
        story_points (Dict[str, int]): Dictionary of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.

    Returns:
        Tuple: Contains objective value, updated mu_Z_star, and assessment score.

    Example:
        mu_Z_1, mu_Z_star, assessment_score_1 = objective1(model, employees, company_tasks, y, score, story_points, max_employee_workload, mu_Z_star)
    """
    try:
        # single objective 1
        idle = []
        for j in employees:
            idle.append(1 - quicksum(y[j, k] for k in company_tasks.keys()))

        mu_Z_1 = quicksum(idle)
        model.setObjective(mu_Z_1, GRB.MINIMIZE)

        # solve the model
        model.optimize()

        # Process results using utility function
        mu_Z_star, assessment_score_1 = utils.model_results(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
            1,
            "./output/result_1",
            "Statistics of Objective 1",
            "Assessment Score Boxplot of Objective 1",
        )

        return mu_Z_1, mu_Z_star, assessment_score_1

    except Exception as e:
        msg = f"An error occurred in define objective1: {e}"
        helper.show(msg, helper.discord_status)

        return None, mu_Z_star, pd.Series()


def objective2(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    z: Dict[Tuple[str, str], Any],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    mu_Z_star: Dict[int, float],
) -> Tuple[Any, Dict[int, float], pd.Series]:
    """
    Sets the second objective to maximize the assessment score and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[str]]): Dictionary of company tasks.
        z (Dict[Tuple[str, str], Any]): Decision variable z.
        score (Dict[str, Dict[str, float]]): List of metric scores for each employee-task pair.
        story_points (Dict[str, int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.

    Returns:
        Tuple: Contains objective value, updated mu_Z_star, and assessment score.

    Example:
        mu_Z_2, mu_Z_star, assessment_score_2 = objective2(model, employees, company_tasks, z, score, story_points, max_employee_workload, mu_Z_star)
    """
    try:
        mu_Z_2 = quicksum(
            score[j][i] * z[i, j]
            for k, tasks in company_tasks.items()
            for i in tasks
            for j in employees
        )
        model.setObjective(mu_Z_2, GRB.MAXIMIZE)

        model.optimize()

        # Process results using utility function
        mu_Z_star, assessment_score_2 = utils.model_results(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
            2,
            "./output/result_2",
            "Statistics of Objective 2",
            "Assessment Score Boxplot of Objective 2",
        )

        return mu_Z_2, mu_Z_star, assessment_score_2

    except Exception as e:
        msg = f"An error occurred in define objective2: {e}"
        helper.show(msg, helper.discord_status)

        return None, mu_Z_star, pd.Series()


def objective3(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    max_workload: Any,
    mu_Z_star: Dict[int, float],
) -> Tuple[Any, Dict[int, float], pd.Series]:
    """
    Sets the third objective to balance the workload for each employee and solves the model.

    Args:
        model (Model): The optimization model.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[str]]): Dictionary of company tasks.
        score (Dict[str, Dict[str, float]]): List of metric scores for each employee-task pair.
        story_points (Dict[str, int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.
        max_workload (Any): Decision variable for maximum workload.
        mu_Z_star (Dict[int, float]): Dictionary to store objective values.

    Returns:
        Tuple: Contains objective value, updated mu_Z_star, and assessment score.

    Example:
        mu_Z_3, mu_Z_star, assessment_score_3 = objective3(model, employees, company_tasks, score, story_points, max_employee_workload, max_workload, mu_Z_star)
    """
    try:
        mu_Z_3 = max_workload
        model.setObjective(mu_Z_3, GRB.MINIMIZE)

        model.optimize()

        # Process results using utility function
        mu_Z_star, assessment_score_3 = utils.model_results(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
            3,
            "./output/result_3",
            "Statistics of Objective 3",
            "Assessment Score Boxplot of Objective 3",
        )

        return mu_Z_3, mu_Z_star, assessment_score_3

    except Exception as e:
        msg = f"An error occurred in define objective3: {e}"
        helper.show(msg, helper.discord_status)

        return None, mu_Z_star, pd.Series()


def MOO(
    model: Model,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
    mu_Z: Dict[int, float],
    mu_Z_star: Dict[int, float],
) -> Any:
    """
    Sets the multi-objective approach using the Goal Programming Optimization Method and solves the model.

    Args:
        model (Model): The optimization model.
        employees: List of employees.
        company_tasks: List of company tasks.
        score: List of metric scores for each employee-task pair.
        story_points: List of story points for each task.
        max_employee_workload: The maximum workload an employee can handle.
        mu_Z: Objective value for every objective.
        mu_Z_star: Dictionary to store objective values.

    Returns:
        Any: Assessment score for the multi-objective approach.

    Example:
        mu_Z = {1: mu_Z_1, 2: mu_Z_2, 3: mu_Z_3}
        assessment_score_4 = MOO(model, employees, company_tasks, score, story_points, max_employee_workload, mu_Z, mu_Z_star)
    """
    try:
        # define weight dictionary for each objective
        Weight = {
            1: config.weight_obj1,
            2: config.weight_obj2,
            3: config.weight_obj3,
        }

        # Define the deviation plus and minus variables
        d_plus = {}
        d_minus = {}

        # Add variables for d_plus and d_minus with specific conditions
        for i in range(1, 4):
            if i != 3:
                d_plus[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_plus_{i}")
            if i != 1:
                d_minus[i] = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"d_minus_{i}"
                )

        # Set specific variables to zero
        d_minus[1] = 0
        d_plus[3] = 0

        mu_Z_star_obj = mu_Z_star.copy()
        for i, value in mu_Z_star_obj.items():
            mu_Z_star_obj[i] = 1 / value if value != 0 else 0

        # This constraint is bound for each objective connected with each other
        for k, w in Weight.items():
            if w != 0:
                model.addConstr(mu_Z[k] - d_plus[k] + d_minus[k] == mu_Z_star[k])

        # Define D = sum k=1 to 3 ((W_plus_k * d_plus_k) + (W_minus_k * d_minus_k)) / mu_Z_star iterate
        D = quicksum(
            (
                (Weight[i] * (d_plus[i] + d_minus[i])) * mu_Z_star_obj[i]
                for i in range(1, 4)
            )
        )

        # Minimize D
        model.setObjective(D, GRB.MINIMIZE)

        # 8.2. Solve The Model
        gap_callback = GapCallback()
        model.setParam("MIPGap", config.MIPGap_moo)

        # Solve the model
        model.optimize(gap_callback)

        # 8.3. Print The Solver Results
        # Process results using utility function
        mu_Z_star, assessment_score_4 = utils.model_results(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
            4,
            "./output/result_MOO",
            "Statistics of Multi-Objective",
            "Assessment Score Boxplot of Multi-Objective",
        )

        return assessment_score_4

    except Exception as e:
        msg = f"An error occurred in define MOO: {e}"
        helper.show(msg, helper.discord_status)

        return pd.Series()
