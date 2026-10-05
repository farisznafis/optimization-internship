import gurobipy as gp
from . import config, helper
from gurobipy import GRB, Model, quicksum
from typing import Dict, List, Tuple, Any


def construct_model(license_params: Dict[str, Any]) -> Tuple[Model, Dict[int, float]]:
    """
    Constructs the optimization model.

    Args:
        license_params (Dict[str, Any]): The Gurobi license parameters.

    Returns:
        Model: The constructed optimization model.

    Example:
        model = construct_model(license_params)
    """

    try:
        # Create an environment with WLS license
        env = gp.Env(params=license_params) if license_params else None

        # Create the model within the Gurobi environment
        model = Model(name="task_assignment", env=env)

        # Set Gurobi parameters to improve performance
        model.setParam("Presolve", config.presolve)  # Aggressive presolve
        model.setParam("MIPFocus", config.MIPFocus)  # Focus on improving the best bound
        model.setParam("MIPGap", config.MIPGap)  # 1% optimality gap
        model.setParam("Heuristics", config.heuristics)  # Increase heuristics effort
        model.setParam(
            "Threads", config.threads
        )  # Use threads, adjust based on your CPU

        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        return model, mu_Z_star

    except Exception as e:
        msg = f"An error occurred in construct_model: {e}"
        helper.show(msg, helper.discord_status)

        return model


def decision_variables(
    model: Model,
    tasks: List[str],
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    max_employee_workload: int,
) -> Tuple[
    Dict[Tuple[str, str, str], Any],
    Dict[Tuple[str, str], Any],
    Dict[Tuple[str, str], Any],
    Any,
]:
    """
    Builds the decision variables for the optimization model.

    Args:
        model (Model): The optimization model.
        tasks (List[int]): List of task IDs.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.
        max_employee_workload (int): Maximum workload that can be assigned to an employee.

    Returns:
        Tuple: Contains decision variables x, y, z, and max_workload variable.

    Example:
        x, y, z, max_workload = decision_variables(model, tasks, employees, company_tasks, max_employee_workload)
    """

    try:
        x = (
            {}
        )  # Decision variable x to represent employee j is assigned to task i in project k
        y = {}  # Decision variable y to represent employee j is assigned to project k
        z = {}  # Decision variable z to represent task i is assigned to employee j

        for k, task in company_tasks.items():
            for i in task:
                for j in employees:
                    x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                    y[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")
                    z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # Decision variable for max workload that can be assigned
        max_workload = model.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            ub=max_employee_workload,
            name="max_workload",
        )

        # Integrate new variables
        model.update()

        return x, y, z, max_workload

    except Exception as e:
        msg = f"An error occurred in define decision_variables: {e}"
        helper.show(msg, helper.discord_status)

        return {}, {}, {}, None


def constraints(
    model: Model,
    x: Dict[Tuple[str, str, str], Any],
    y: Dict[Tuple[str, str], Any],
    z: Dict[Tuple[str, str], Any],
    max_workload: Any,
    employees: List[str],
    company_tasks: Dict[str, List[str]],
    story_points: Dict[str, int],
    max_employee_workload: int,
) -> None:
    """
    Adds constraints to the optimization model.

    Args:
        model (Model): The optimization model.
        x (Dict[Tuple[int, str, str], Any]): Decision variable x.
        y (Dict[Tuple[str, str], Any]): Decision variable y.
        z (Dict[Tuple[int, str], Any]): Decision variable z.
        max_workload (Any): Decision variable for maximum workload.
        employees (List[str]): List of employee IDs.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.
        story_points (Dict[int, int]): Dictionary of story points for each task.
        max_employee_workload (int): Maximum workload that can be assigned to an employee.

    Example:
        constraints(model, x, y, z, max_workload, employees, company_tasks, story_points, max_employee_workload)
    """

    try:
        # constraint 1: each task assigned to one talent
        for k, task in company_tasks.items():
            for i in task:
                model.addConstr(quicksum(x[(i, j, k)] for j in employees) == 1)

        # pre-processing constraint 2
        for j in employees:
            for k, task in company_tasks.items():
                # Use quicksum to sum up x[i][j][k] for all i
                temp_sum = quicksum(x[i, j, k] for i in task)

                # Add a constraint to the model: y[j][k] is 1 if the sum of x[i][j][k] for all i is > 0, and 0 otherwise
                model.addGenConstrIndicator(
                    y[j, k], True, temp_sum, GRB.GREATER_EQUAL, 1
                )
                model.addGenConstrIndicator(y[j, k], False, temp_sum, GRB.LESS_EQUAL, 0)

        # create constraint 2: each employee can only work on one task
        for j in employees:
            # The sum of y[j][k] for all companies (k) should be <= 1
            model.addConstr(quicksum(y[(j, k)] for k in company_tasks.keys()) <= 1)

        # Constraint 3: Employee workload doesn't exceed the capacity
        for j in employees:
            for k, tasks in company_tasks.items():
                model.addConstr(
                    quicksum(story_points[i] * x[(i, j, k)] for i in tasks)
                    <= max_employee_workload
                )

        # constraint 4: max_workload is greater than or equal to the workload of each employee
        for j in employees:
            model.addConstr(
                max_workload
                >= quicksum(
                    story_points[i] * x[i, j, k]
                    for k, tasks in company_tasks.items()
                    for i in tasks
                )
            )

        # # create constraint 5: each task can only assigned to one employee
        for i in tasks:
            model.addConstr(quicksum(z[(i, j)] for j in employees) <= 1)

        for k, tasks in company_tasks.items():
            for i in tasks:
                for j in employees:
                    model.addGenConstrIndicator(x[i, j, k], True, z[i, j], GRB.EQUAL, 1)
                    model.addGenConstrIndicator(z[i, j], True, y[j, k], GRB.EQUAL, 1)

    except Exception as e:
        msg = f"An error occurred in define constraints: {e}"
        helper.show(msg, helper.discord_status)
