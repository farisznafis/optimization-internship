from . import create_objective
from .helper import notify_and_time


@notify_and_time("Section 5: Objective 1")
def run_objective1(
    model,
    employees,
    company_tasks,
    y,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
):
    """
    Runs Objective 1: Minimize Idle Employee.

    Args:
        model: The optimization model.
        employees: List of employees.
        company_tasks: List of company tasks.
        y: Decision variable for tasks.
        score: Score data.
        story_points: Story points data.
        max_employee_workload: Maximum workload per employee.
        mu_Z_star: Dictionary for storing objective values.

    Returns:
        tuple: mu_Z_1, mu_Z_star, assessment_score_1

    Example:
        mu_Z_1, mu_Z_star, assessment_score_1 = run_objective1(
            model, employees, company_tasks, y, score, story_points, max_employee_workload, mu_Z_star
        )
    """
    return create_objective.objective1(
        model,
        employees,
        company_tasks,
        y,
        score,
        story_points,
        max_employee_workload,
        mu_Z_star,
    )


@notify_and_time("Section 6: Objective 2")
def run_objective2(
    model,
    employees,
    company_tasks,
    z,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
):
    """
    Runs Objective 2: Maximize Assessment Score.

    Args:
        model: The optimization model.
        employees: List of employees.
        company_tasks: List of company tasks.
        z: Decision variable for assessment.
        score: Score data.
        story_points: Story points data.
        max_employee_workload: Maximum workload per employee.
        mu_Z_star: Dictionary for storing objective values.

    Returns:
        tuple: mu_Z_2, mu_Z_star, assessment_score_2

    Example:
        mu_Z_2, mu_Z_star, assessment_score_2 = run_objective2(
            model, employees, company_tasks, z, score, story_points, max_employee_workload, mu_Z_star
        )
    """
    return create_objective.objective2(
        model,
        employees,
        company_tasks,
        z,
        score,
        story_points,
        max_employee_workload,
        mu_Z_star,
    )


@notify_and_time("Section 7: Objective 3")
def run_objective3(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    max_workload,
    mu_Z_star,
):
    """
    Runs Objective 3: Balancing the Workload.

    Args:
        model: The optimization model.
        employees: List of employees.
        company_tasks: List of company tasks.
        score: Score data.
        story_points: Story points data.
        max_employee_workload: Maximum workload per employee.
        max_workload: Maximum workload constraint.
        mu_Z_star: Dictionary for storing objective values.

    Returns:
        tuple: mu_Z_3, mu_Z_star, assessment_score_3

    Example:
        mu_Z_3, mu_Z_star, assessment_score_3 = run_objective3(
            model, employees, company_tasks, score, story_points, max_employee_workload, max_workload, mu_Z_star
        )
    """
    return create_objective.objective3(
        model,
        employees,
        company_tasks,
        score,
        story_points,
        max_employee_workload,
        max_workload,
        mu_Z_star,
    )


@notify_and_time("Section 8: MOO")
def run_MOO(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    mu_Z,
    mu_Z_star,
):
    """
    Runs Multi-Objective Optimization (MOO) with Goal Programming.

    Args:
        model: The optimization model.
        employees: List of employees.
        company_tasks: List of company tasks.
        score: Score data.
        story_points: Story points data.
        max_employee_workload: Maximum workload per employee.
        mu_Z: Dictionary for storing objective values.
        mu_Z_star: Dictionary for storing optimal objective values.

    Returns:
        assessment_score_4: Assessment score for the MOO.

    Example:
        assessment_score_4 = run_MOO(
            model, employees, company_tasks, score, story_points, max_employee_workload, mu_Z, mu_Z_star
        )
    """
    return create_objective.MOO(
        model,
        employees,
        company_tasks,
        score,
        story_points,
        max_employee_workload,
        mu_Z,
        mu_Z_star,
    )
