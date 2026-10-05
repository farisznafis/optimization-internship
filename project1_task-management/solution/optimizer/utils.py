import threading
import pandas as pd
import matplotlib.pyplot as plt
from . import helper
from gurobipy import Model, GRB
from typing import Dict, List, Tuple, Any
from .tools import CompetencyAssessment, WeightedEuclideanDistance


def get_employee_tasks(
    j: str,
    company_tasks: Dict[str, List[str]],
    model: Model,
    score: Dict[str, Dict[str, float]],
    story_points: Dict[str, int],
    max_employee_workload: int,
) -> Tuple[List[str], List[str], int, int, List[float]]:
    """
    Extracts and prints the tasks assigned to an employee and computes related metrics.

    Args:
        j (int): The employee ID.
        company_tasks (Dict[str, List[int]]): Dictionary of company tasks.
        model (Model): The optimization model.
        score (List[List[float]]): List of metric scores for each employee-task pair.
        story_points (List[int]): List of story points for each task.
        max_employee_workload (int): The maximum workload an employee can handle.

    Returns:
        Tuple[List[str], List[int], int, int, List[float]]: A tuple containing:
            - List of company names (comp)
            - List of task IDs (task)
            - Total story points (sp)
            - Wasted story points (wasted_sp)
            - List of metric scores (sim)

    Example:
        company_tasks = {'CompanyA': [1, 2], 'CompanyB': [3]}
        model = Model()
        score = [[0.5, 0.7, 0.6], [0.4, 0.8, 0.5]]
        story_points = [3, 2, 5]
        max_employee_workload = 10
        get_employee_tasks(1, company_tasks, model, score, story_points, max_employee_workload)
    """
    task = []
    sim = []
    comp = []
    sp = 0

    for k, tasks in company_tasks.items():
        for i in tasks:
            var = model.getVarByName(f"x_{i}_{j}_{k}")
            if var is not None and var.X == 1:
                print(f"Task {i} assigned to Employee {j}")
                print(f"Company\t\t\t: {k}")
                print(f"Story Points\t\t: {story_points[i]}")
                print(f"Metrics score\t: {score[j][i]:.10f}\n")

                task.append(i)
                sim.append(score[j][i])
                comp.append(k)
                sp += story_points[i]

    wasted_sp = max_employee_workload - sp if sp > 0 else 0
    return comp, task, sp, wasted_sp, sim


def define_data_structure(
    employee_path: str, task_path: str, overqualification: bool
) -> Tuple[
    List[str],  # employees
    List[str],  # tasks
    Dict[str, int],  # story_points
    Dict[str, List[str]],  # company_tasks
    Dict[str, Dict[str, float]],  # score
    Dict[str, Any],  # info
]:
    """
    Sets up the data structure by processing employee, task data, and calculate skills metric score.

    Args:
        employee_path (str): The path to the employee data CSV file.
        task_path (str): The path to the task data CSV file.
        overqualification (bool): Flag to choice skills metric methodhology.

    Returns:
        Tuple: Contains employees, tasks, story_points, company_tasks, score, and info.

    Dataset Structure:
    The following are examples of how the employee and task datasets must be created:

    Employee Data:
    >>> import pandas as pd
    >>> employee_skills_df = pd.DataFrame({
    ...     'math': [5, 3, 4, 4, 2],
    ...     'python': [5, 4, 4, 4, 3],
    ...     'sql': [3, 5, 4, 5, 2],
    ...     'cloud': [2, 4, 3, 5, 4],
    ...     'database': [2, 3, 4, 5, 4],
    ...     'optimization': [5, 5, 3, 4, 1]
    ... }, index=['Talent 1', 'Talent 2', 'Talent 3', 'Talent 4', 'Talent 5'])
    >>> employee_skills_df
            math  python  sql  cloud  database  optimization
    Talent 1     5       5    3      2         2             5
    Talent 2     3       4    5      4         3             5
    Talent 3     4       4    4      3         4             3
    Talent 4     4       4    5      5         5             4
    Talent 5     2       3    2      4         4             1

    Task Data:
    >>> task_df = pd.DataFrame({
    ...     'project_id': ['P2', 'P2', 'P3', 'P3', 'P2', 'P2', 'P3', 'P1', 'P1', 'P3'],
    ...     'story_points': [1, 2, 3, 4, 0, 0, 0, 5, 2, 5],
    ...     'math': [0, 3, 5, 4, 0, 4, 3, 3, 0, 5],
    ...     'python': [5, 3, 4, 3, 2, 1, 3, 4, 3, 5],
    ...     'sql': [3, 5, 4, 3, 1, 5, 4, 5, 2, 5],
    ...     'cloud': [4, 4, 5, 3, 0, 5, 4, 5, 0, 5],
    ...     'database': [4, 3, 5, 3, 1, 0, 3, 5, 2, 0],
    ...     'optimization': [0, 1, 5, 0, 5, 0, 4, 2, 2, 5]
    ... }, index=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
    >>> task_df
            project_id  story_points  math  python  sql  cloud  database  optimization
    T1              P2             1     0       5    3      4         4             0
    T2              P2             2     3       3    5      4         3             1
    T3              P3             3     5       4    4      5         5             5
    T4              P3             4     4       3    3      3         3             0
    T5              P2             0     0       2    1      0         1             5
    T6              P2             0     4       1    5      5         0             0
    T7              P3             0     3       3    4      4         3             4
    T8              P1             5     3       4    5      5         5             2
    T9              P1             2     0       3    2      0         2             2
    T10             P3             5     5       5    5      5         0             5

    Example:
    employees, tasks, story_points, company_tasks, score, info = define_data_structure('employees.csv', 'tasks.csv')
    """

    try:
        # 1.1. Pre-Processing: Employee Data
        # Read data
        employee_skills_df = pd.read_csv(employee_path, index_col="employee_id")
        employee_skills_df.drop(columns=["No", "Role"], inplace=True, errors="ignore")

        employees = employee_skills_df.index.tolist()
        skills_name = employee_skills_df.columns[1:].tolist()

        # 1.2. Pre-Processing: Task Data
        task_df = pd.read_csv(task_path, index_col="task_id")

        tasks = task_df.index.tolist()
        company_names = list(set(task_df["project_id"]))
        story_points = task_df["story_points"].to_dict()

        # 1.3. Group the task data by company/project
        # convert to dictionary each company and its task
        company_tasks = {}

        for company in company_names:
            company_tasks[company] = task_df[
                task_df["project_id"] == company
            ].index.tolist()

        # sort the company tasks from C1 to C5
        company_tasks = dict(sorted(company_tasks.items()))

        # 1.4. Pre-Processing: Skill Metric Score Calculation
        """
        # 1.4.1 Pre-Processing: Competency Assessment
        First, create RCD-ACD Dataframe that we get from Task Dataframe for RCD and from Employee Dataframe for ACD.
        """

        # 1.4.2 Required Competence Data
        rcd_df = task_df.drop(columns=["project_id", "story_points"])
        rcd_df = rcd_df.fillna(0)

        # 1.4.3 Acquired Competence Data
        # create a copy of the original DataFrame
        acd_df = employee_skills_df.copy()
        acd_df = acd_df.fillna(0)

        if overqualification:
            # Calculate with Competency Assessment
            ca = CompetencyAssessment(rcd_df, acd_df)
            score, info = ca.fit()
        else:
            # Calculate with Weighted Euclidean Distance
            wed = WeightedEuclideanDistance(rcd_df, acd_df)
            score, info = wed.fit()

        # Export the score dictionary to CSV
        score_df = pd.DataFrame.from_dict(score, orient="index")
        score_df.to_csv("./output/score.csv")

        return employees, tasks, story_points, company_tasks, score, info

    except Exception as e:
        msg = f"An error occurred in define data structure: {e}"
        helper.show(msg, helper.discord_status)

        return [], [], {}, {}, {}, {}


def model_results(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
    objective_index: int,
    result_file_path: str,
    statistics_title: str,
    boxplot_title: str,
) -> Tuple[Dict[int, float], pd.Series]:
    """
    Process the model results, update mu_Z_star and generate assessment score.

    Args:
        model: The optimization model.
        employees: List of employees.
        company_tasks: List of company tasks.
        score: Dictionary of metric scores for each employee-task pair.
        story_points: Dictionary of story points for each task.
        max_employee_workload: Maximum workload per employee.
        mu_Z_star: Dictionary to store objective values.
        objective_index: The index of the current objective.
        result_file_path: Path to save the result file.
        statistics_title: Title for the statistics section.
        boxplot_title: Title for the boxplot.

    Returns:
        Tuple: Updated mu_Z_star and assessment score.

    Example:
    >>> mu_Z_star, assessment_score_1 = model_results(
    ...     model,
    ...     employees,
    ...     company_tasks,
    ...     score,
    ...     story_points,
    ...     max_employee_workload,
    ...     mu_Z_star,
    ...     1
    ...     "./output/result_1",
    ...     "Statistics of Objective 1",
    ...     "Assessment Score Boxplot of Objective 1",
    ... )
    """
    if model.status == GRB.OPTIMAL:
        print("Solution Found!")
        print(f"Obj. Value {objective_index} i.e. Objective Value: {model.ObjVal}\n")
        mu_Z_star[objective_index] = model.ObjVal

        x_hat = {}
        for j in employees:
            result = get_employee_tasks(
                j, company_tasks, model, score, story_points, max_employee_workload
            )
            if len(result[1]) > 0:
                x_hat[j] = result
    else:
        print("No Solution Found!")
        x_hat = {}

    # Call the process_results function
    assessment_score = process_results(
        x_hat,
        employees,
        story_points,
        result_file_path,
        statistics_title,
        boxplot_title,
    )

    return mu_Z_star, assessment_score


def process_results(
    x_hat: Dict[str, Tuple[str, List[str], int, int, List[float]]],
    employees: List[str],
    story_points: Dict[str, int],
    output_file: str,
    title: str,
    boxplot_title: str,
) -> pd.Series:
    """
    Processes the results, saves the CSV file, shows statistics, and plots the box plot.

    Args:
        x_hat (Dict[str, Tuple[str, List[str], int, int, List[float]]]): Dictionary of results for each employee.
        employees (List[str]): List of employee IDs.
        story_points (Dict[str, int]): Dictionary of story points for each task.
        output_file (str): Path to the output CSV and plot PNG file.
        title (str): Title for the statistics.
        boxplot_title (str): Title for the box plot.

    Returns:
        pd.Series: Series of assessment scores.

    Example:
    >>> assessment_score = process_results(
    ...     x_hat,
    ...     employees,
    ...     story_points,
    ...     "./output/result_1",
    ...     "Statistics of Objective 1",
    ...     "Assessment Score Boxplot of Objective 1",
    ... )
    """
    # Set display options
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)

    # Convert dictionary to DataFrame and set 'employee' as index
    result = pd.DataFrame.from_dict(
        x_hat,
        orient="index",
        columns=[
            "company",
            "assigned_task",
            "sum_sp",
            "wasted_sp",
            "assessment_score",
        ],
    )
    result.index.name = "employee"
    result.to_csv(output_file + ".csv")

    # Statistics of The Objective
    total_employee = len(employees)
    total_sp = sum(story_points.values())
    total_active_employee = len(set(employee for employee in x_hat.keys()))
    total_active_sp = sum(value[2] for value in x_hat.values())
    total_idle_employee = total_employee - total_active_employee
    total_wasted_sp = total_sp - total_active_sp

    print(title)
    print(f"Total Employee\t\t\t: {total_employee}")
    print(
        f"Total Active Employee\t\t: {total_active_employee}\t{(total_active_employee/total_employee)*100:.2f}%"
    )
    print(
        f"Total Idle Employee\t\t: {total_idle_employee}\t{(total_idle_employee/total_employee)*100:.2f}%\n"
    )
    print(f"Total Story Points\t\t: {total_sp}")
    print(
        f"Total Active Story Points\t: {total_active_sp}\t{(total_active_sp/total_sp)*100:.2f}%"
    )
    print(
        f"Total Wasted Story Points\t: {total_wasted_sp}\t{(total_wasted_sp/total_sp)*100:.2f}%\n"
    )

    # Timer for auto close plot
    timer = threading.Timer(3, helper.close_plot)
    timer.start()

    # Make boxplot for the assessment score
    assessment_score = result["assessment_score"].explode().reset_index(drop=True)

    if len(assessment_score) != 0:
        assessment_score.plot(kind="box")
        plt.title(boxplot_title)
        plt.savefig(output_file + ".png")
        plt.show()
    else:
        print("No data to show")

    return assessment_score


def compare_scores(
    data: List[pd.Series], boxplot_title: List[str], output_file: str
) -> None:
    """
    Compare the score results for every objective with Box Plot.

    Args:
        data (List[pd.Series]): List of the input data
        boxplot_title (str): Title for the box plot.
        output_file (str): Path to the plot PNG file.

    Returns:
        None

    Example:
    >>> compare_scores(
    ...     data,
    ...     title,
    ...     "./output/score_comparison",
    ... )
    """
    # Timer for auto close plot
    timer = threading.Timer(3, helper.close_plot)
    timer.start()

    plt.figure(figsize=(10, 5))
    plt.boxplot(
        data,
        tick_labels=boxplot_title,
    )
    plt.title("Overall Assessment Score Boxplot")
    plt.xticks(rotation=15)
    plt.savefig(output_file + ".png")
    plt.show()
