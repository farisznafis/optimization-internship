"""
# Task Assignment Optimization Problem with Gurobi Framework

_by: TK-Bunga Matahari Team_

---

# 0. The Obligatory Part
"""

# Import library
import json
import requests
import datetime
import threading
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB, quicksum
from competency_assessment import CompetencyAssessment


# User Define Parameters

# Set the path of the license file
license_file_path = "./data/gurobi.lic"

"""
Data of Workforce and Task


On this notebook, if you want to use the full data, you can set the full variable to True
If you want to use the mini data, you can set the full variable to False
"""
full = False

# Run this if the data in Local/Repository
if full:
    employee_path = "./data/fixed_data_employee.csv"
    task_path = "./data/fixed_data_task.csv"

    # maximum workload for each employee
    max_employee_workload = 20
else:
    employee_path = "./mini_data/mini_data - employee.csv"
    task_path = "./mini_data/mini_data - task.csv"

    # maximum workload for each employee
    max_employee_workload = 8

# Methodology: Competency Assessment or Weighted Euclidean Distance
overqualification = True

# Weight for each objective
weight_obj1 = 0.03
weight_obj2 = 0.9
weight_obj3 = 0.07

# Define for Model Tune Parameters

# define the tuned parameters of the model
presolve = 2
MIPFocus = 1
MIPGap = 0.01
heuristics = 0.8
threads = 2
MIPGap_moo = 0.05


def s1_data_structure_CA(employee_path, task_path):
    """# 1. Define the Data Structure"""

    try:
        """## 1.1. Pre-Processing: Employee Data"""

        # Read data
        employee_skills_df = pd.read_csv(employee_path, index_col="employee_id")
        employee_skills_df.drop(columns=["No", "Role"], inplace=True, errors="ignore")

        employees = employee_skills_df.index.tolist()
        skills_name = employee_skills_df.columns[1:].tolist()

        """## 1.2. Pre-Processing: Task Data"""

        task_df = pd.read_csv(task_path, index_col="task_id")

        tasks = task_df.index.tolist()
        company_names = list(set(task_df["project_id"]))
        story_points = task_df["story_points"].to_dict()

        """## 1.3. Group the task data by company/project"""

        # convert to dictionary each company and its task
        company_tasks = {}

        for company in company_names:
            company_tasks[company] = task_df[
                task_df["project_id"] == company
            ].index.tolist()

        # sort the company tasks from C1 to C5
        company_tasks = dict(sorted(company_tasks.items()))

        """## 1.4. Pre-Processing: Competency Assessment

        First, create RCD-ACD Dataframe that we get from Task Dataframe for RCD and from Employee Dataframe for ACD.

        ### 1.4.1 Required Competence Data
        """

        rcd_df = task_df.drop(columns=["project_id", "story_points"])
        rcd_df = rcd_df.fillna(0)

        """### 1.4.2 Acquired Competence Data"""

        # create a copy of the original DataFrame
        acd_df = employee_skills_df.copy()
        acd_df = acd_df.fillna(0)

        """### 1.4.3 Fit the Data"""

        ca = CompetencyAssessment(rcd_df, acd_df)
        qs, info = ca.fit()

        """### 1.4.4 Qualification Space"""

        """### 1.4.5 Sorted MSG Score for All Tasks"""

        score = ca.rank_MSG(qs)
        score_df = pd.DataFrame.from_dict(score, orient="index")

        # Export the score dictionary to CSV
        score_df.to_csv("./output_CA/score.csv")

        return employees, tasks, story_points, company_tasks, score, info

    except Exception as e:
        send_discord_notification(f"An error occured in s1_data_structure_CA: {e}")
        print(f"An error occurred in s1_data_structure_CA: {e}")
        return [], [], [], {}, {}, {}


"""#### Generic Function"""


# Extracting and printing the results
def get_employee_tasks(
    j, company_tasks, model, score, story_points, max_employee_workload
):
    task = []
    sim = []
    comp = []
    sp = 0

    for k, tasks in company_tasks.items():
        for i in tasks:
            if model.getVarByName(f"x_{i}_{j}_{k}").X == 1:
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


# Get the License of Gurobi
def read_license_file(filepath):
    params = {}
    with open(filepath, "r") as file:
        for line in file:
            if line.startswith("WLSACCESSID"):
                params["WLSACCESSID"] = line.split("=")[1].strip()
            elif line.startswith("WLSSECRET"):
                params["WLSSECRET"] = line.split("=")[1].strip()
            elif line.startswith("LICENSEID"):
                params["LICENSEID"] = int(line.split("=")[1].strip())
    return params


def s2_construct_model(license_path):
    """# 2. Construct the Model"""

    try:
        license_params = read_license_file(license_path)

        # Create an environment with WLS license
        parameter = {
            "WLSACCESSID": license_params["WLSACCESSID"],
            "WLSSECRET": license_params["WLSSECRET"],
            "LICENSEID": license_params["LICENSEID"],
        }
        env = gp.Env(params=parameter)

        # Create the model within the Gurobi environment
        model = gp.Model(name="task_assignment", env=env)

        # Set Gurobi parameters to improve performance
        model.setParam("Presolve", presolve)  # Aggressive presolve
        model.setParam("MIPFocus", MIPFocus)  # Focus on improving the best bound
        model.setParam("MIPGap", MIPGap)  # 1% optimality gap
        model.setParam("Heuristics", heuristics)  # Increase heuristics effort
        # model.setParam("Threads", threads)  # Use 8 threads, adjust based on your CPU

        return model

    except Exception as e:
        send_discord_notification(f"An error occured in s2_construct_model: {e}")
        print(f"An error occurred in s2_construct_model: {e}")
        return None


def s3_decision_variable(model, tasks, employees, company_tasks):
    """# 3. Build the Decision Variable

    We have 3 sets:

    $$
    \text{sets} = \begin{cases}
    I &: \text{set of tasks} \\
    J &: \text{set of employees} \\
    K &: \text{set of projects}
    \\end{cases}
    $$

    Next, we define parameters, scalars, and data structures. Let:

    $$
    \begin{align*}
    i & = \text{task } i \\
    j & = \text{employee } j \\
    k & = \text{project } k \\
    s_i & = \text{story points of task } i \\
    e_{ij} & = \text{similarity skills of employee } j \text{ for task } i \\
    \\end{align*}
    $$

    **Decision Variables:**

    $$
    \begin{align*}
    x_{ijk} & = \text{Binary variable indicating whether employee } j \text{ is assigned to task } k \text{ for day } i \\
    y_{jk} & = \text{Binary variable indicating whether employee } j \text{ is assigned to any task from company } k \\
    \\end{align*}
    $$

    """

    try:
        # Create decision variables for x and y

        # Decision variable x to represent employee j is assigned to task i in project k
        x = {}
        for k, task in company_tasks.items():
            for i in task:
                for j in employees:
                    x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        # Decision variable y to represent employee j is assigned to project k
        y = {}
        for j in employees:
            for k in company_tasks.keys():
                y[(j, k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")

        # Decision variable z to represent task i is assigned to employee j
        z = {}
        for i in tasks:
            for j in employees:
                z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

        # Decision variable for max workload that can be assigned
        max_workload = model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=max_employee_workload, name="max_workload"
        )

        # Integrate new variables
        model.update()

        return x, y, z, max_employee_workload, max_workload

    except Exception as e:
        send_discord_notification(f"An error occured in s3_decision_variable: {e}")
        print(f"An error occurred in s3_decision_variable: {e}")
        return {}, {}, {}, 0, None


def s4_constraint(model, x, y, z, employees, company_tasks, story_points, max_workload):
    """# 4. Subject to the Constraint

    ## 4.1. Constraint 1: Each Task is Assigned to One Employee

    $$
    \\sum _{j\\in J}\\:x_{ijk}\\:=\\:1 \\quad \\forall i \\in k, \\: k \\in K
    $$

    """

    try:
        # constraint 1: each task assigned to one talent
        for k, task in company_tasks.items():
            for i in task:
                model.addConstr(quicksum(x[(i, j, k)] for j in employees) == 1)

        """## 4.2. Constraint 2: Each employee works for one company at a time

        Pre-Processing for Constraint 2:

        $$
        \\sum _{i\\in I_k}x_{ijk} > 0 \\: \\rightarrow \\: y_{jk}=1 \\quad \\forall j\\in J, \\: k\\in K\\:
        $$

        """

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

        """$$
        \\sum _{k\\in K}y_{jk}\\le 1 \\quad \\forall j\\in J
        $$

        """

        # create constraint 2: each employee can only work on one task
        for j in employees:
            # The sum of y[j][k] for all companies (k) should be <= 1
            model.addConstr(quicksum(y[(j, k)] for k in company_tasks.keys()) <= 1)

        """## 4.3. Constraint 3: Employee workload doesn't exceed the capacity

        $$
        \\sum _{i \\in I} s_i \\cdot x_{ijk} \\le max\\_workload \\quad \\forall j\\in J, \\: k \\in K
        $$

        """

        for j in employees:
            for k, tasks in company_tasks.items():
                model.addConstr(
                    quicksum(story_points[i] * x[(i, j, k)] for i in tasks)
                    <= max_workload
                )

        """## 4.4 Constraint 4: Maximum workload is greater than or equal to the workload of each employee For Objective 3

        $$
        max\\_workload \\ge \\sum_{i \\in I} \\sum_{k \\in K} s_i\\cdot x_{ijk}, \\quad \\forall j\\in J\\:\\:
        $$

        """

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
        send_discord_notification(f"An error occured in s4_constraint: {e}")
        print(f"An error occurred in s4_constraint: {e}")


def s5_objective1(
    model,
    employees,
    company_tasks,
    y,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
):
    """# 5. Single Objective Approach: 1) Minimize The Idle Employee
    ## 5.1. Set The Objective Model

    $$
    \\mu _{Z_1} = min.\\:I_j=\\sum _{j\\in \\:J}\\:\\left(1\\:-\\:\\sum _{k\\in \\:K}\\:y_{jk}\\right) \\quad \\tag{1}
    $$
    """

    try:
        # objective 1
        idle = []

        for j in employees:
            idle.append(1 - quicksum(y[j, k] for k in company_tasks.keys()))

        mu_Z_1 = quicksum(idle)

        # single objective 1
        model.setObjective(mu_Z_1, GRB.MINIMIZE)

        """## 5.2. Solve The Model of Objective $(1)$"""

        # solve the model
        model.optimize()

        """### 5.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 1 i.e. Total Idle Employees: {model.ObjVal}\n")
            mu_Z_star[1] = model.ObjVal

            x_hat_1 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_1[j] = result
        else:
            print("No Solution Found!")
            x_hat_1 = {}

        """## 5.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        result_1 = pd.DataFrame.from_dict(
            x_hat_1,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_1.index.name = "employee"
        result_1.to_csv("./output_CA/result_1.csv")

        """### 5.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_1.keys()))
        total_active_sp = sum(value[2] for value in x_hat_1.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

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

        """### 5.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_1 = (
            result_1["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_1) != 0:
            assessment_score_1.plot(kind="box")
            plt.title("Assessment Score Boxplot of Objective 1")
            plt.show()
        else:
            print("No data to show")

        return mu_Z_1, mu_Z_star, assessment_score_1

    except Exception as e:
        send_discord_notification(f"An error occured in s5_objective1: {e}")
        print(f"An error occurred in s5_objective1: {e}")
        return None, None, None


def s6_objective2(
    model,
    employees,
    company_tasks,
    z,
    score,
    story_points,
    max_employee_workload,
    mu_Z_star,
):
    """# 6. Single Objective Approach: 2) Maximize The Assessment Score
    ## 6.1. Set The Objective Model

    $$
    \\mu _{Z_2} = max.\\:A_{ij} = \\sum _{i\\in \\:I} \\sum _{j\\in \\:J} \\sum _{k\\in \\:K} \\: e_{ij} \\cdot x_{ijk} \\quad \\tag{2}
    $$
    """

    try:
        # objective 2
        # mu_Z_2 = quicksum(
        #     score[j][i] * x[i, j, k]
        #     for k, tasks in company_tasks.items()
        #     for i in tasks
        #     for j in employees
        # )

        # proposed objective 2
        mu_Z_2 = quicksum(
            score[j][i] * z[i, j]
            for k, tasks in company_tasks.items()
            for i in tasks
            for j in employees
        )

        # single objective 2
        model.setObjective(mu_Z_2, GRB.MAXIMIZE)

        """## 6.2. Solve The Model of Objective $(2)$"""

        # solve the model
        model.optimize()

        """### 6.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 2 i.e. Total Score: {model.ObjVal}\n")
            mu_Z_star[2] = model.ObjVal

            x_hat_2 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_2[j] = result
        else:
            print("No Solution Found!")
            x_hat_2 = {}

        """## 6.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        result_2 = pd.DataFrame.from_dict(
            x_hat_2,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_2.index.name = "employee"
        result_2.to_csv("./output_CA/result_2.csv")

        """### 6.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_2.keys()))
        total_active_sp = sum(value[2] for value in x_hat_2.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

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

        """### 6.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_2 = (
            result_2["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_2) != 0:
            assessment_score_2.plot(kind="box")
            plt.title("Assessment Score Boxplot of Objective 2")
            plt.show()
        else:
            print("No data to show")

        return mu_Z_2, mu_Z_star, assessment_score_2

    except Exception as e:
        send_discord_notification(f"An error occured in s6_objective2: {e}")
        print(f"An error occurred in s6_objective2: {e}")
        return None, None, None


def s7_objective3(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    max_workload,
    mu_Z_star,
):
    """# 7. Single Objective Approach: 3) Balancing Workload For Each Employee
    ## 7.1. Set The Objective Model

    $$
    \\mu_{Z_3} = min.\\:W_{j} \\quad \\tag{3}
    $$
    """

    try:
        # single objective 3
        mu_Z_3 = max_workload
        model.setObjective(mu_Z_3, GRB.MINIMIZE)

        """## 7.2. Solve The Model of Objective $(3)$"""

        # solve the model
        model.optimize()

        """### 7.2.1 Print The Solver Results"""

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(
                f"Obj. Value 3 i.e. Maximum Story Points Each Employee: {model.ObjVal}\n"
            )
            mu_Z_star[3] = model.ObjVal

            x_hat_3 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_3[j] = result
        else:
            print("No Solution Found!")
            x_hat_3 = {}

        """## 7.3. Show the Solver's Result"""

        # Set display options
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)

        # Convert dictionary to DataFrame and set 'employee' as index
        result_3 = pd.DataFrame.from_dict(
            x_hat_3,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )
        result_3.index.name = "employee"

        result_3.to_csv("./output_CA/result_3.csv")

        """### 7.3.1 Statistics of The Objective"""

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_3.keys()))
        total_active_sp = sum(value[2] for value in x_hat_3.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

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

        """### 7.3.2. Distribution With Respect to the Assessment Score"""

        # timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # make boxplot for objective 1 with respect to the assessment score
        assessment_score_3 = (
            result_3["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_3) != 0:
            assessment_score_3.plot(kind="box")
            plt.title("Assessment Score Boxplot of Objective 3")
            plt.show()
        else:
            print("No data to show")

        return mu_Z_3, mu_Z_star, assessment_score_3

    except Exception as e:
        send_discord_notification(f"An error occured in s7_objective3: {e}")
        print(f"An error occurred in s7_objective3: {e}")
        return None, None, None


def s8_MOO(
    model,
    employees,
    company_tasks,
    score,
    story_points,
    max_employee_workload,
    mu_Z_1,
    mu_Z_2,
    mu_Z_3,
    mu_Z_star,
    assessment_score_1,
    assessment_score_2,
    assessment_score_3,
):
    """# 8. Multi-Objective Approach: 2) Goal Programming Optimization Method
    ## 8.1. Set The Objective Model

    $$
    D = \\sum_{k=1}^{3} \\left( W_{plus_k} \\cdot d_{plus_k} + W_{minus_k} \\cdot d_{minus_k} \\right) / \\mu_{Z_star_k}
    $$
    """

    try:
        # define weight dictionary for each objective
        Weight = {
            1: weight_obj1,
            2: weight_obj2,
            3: weight_obj3,
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

        mu_Z = {1: mu_Z_1, 2: mu_Z_2, 3: mu_Z_3}

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

        ## 8.2. Solve The Model
        gap_callback = GapCallback()
        model.setParam("MIPGap", MIPGap_moo)

        # Solve the model
        model.optimize(gap_callback)

        ## 8.3. Print The Solver Results

        # Check and process the solution
        if model.status == GRB.OPTIMAL:
            print("Solution Found!")
            print(f"Obj. Value 5 i.e. Deviation: {model.ObjVal}\n")

            x_hat_4 = {}
            for j in employees:
                result = get_employee_tasks(
                    j, company_tasks, model, score, story_points, max_employee_workload
                )
                if len(result[1]) > 0:
                    x_hat_4[j] = result
        else:
            print("No Solution Found!")
            x_hat_4 = {}

        ## 8.4. Show the Solver's Result

        # Show data that has positive metrics score
        result_4 = pd.DataFrame.from_dict(
            x_hat_4,
            orient="index",
            columns=[
                "company",
                "assigned_task",
                "sum_sp",
                "wasted_sp",
                "assessment_score",
            ],
        )

        result_4.index.name = "employee"
        result_4.to_csv("./output_CA/result_4_MOO.csv")

        ## 8.5. Statistics of The Objective

        total_employee = len(employees)
        total_sp = sum(story_points.values())
        total_active_employee = len(set(employee for employee in x_hat_4.keys()))
        total_active_sp = sum(value[2] for value in x_hat_4.values())
        total_idle_employee = total_employee - total_active_employee
        total_wasted_sp = total_sp - total_active_sp

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

        ## 8.6. Distribution With Respect to the Assessment Score

        # Timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Make boxplot for x_hat_4
        assessment_score_4 = (
            result_4["assessment_score"].explode().reset_index(drop=True)
        )

        if len(assessment_score_4) != 0:
            assessment_score_4.plot(kind="box")
            plt.title("Assessment Score Boxplot of MOO Method 2")
            plt.show()
        else:
            print("No data to show")

        ## 8.7. Comparing MOO Method 2 to Single Objective and MOO Method 1

        # Timer for auto close plot
        timer = threading.Timer(3, close_plot)
        timer.start()

        # Merge all boxplot in one graph
        data = [
            assessment_score_1,
            assessment_score_2,
            assessment_score_3,
            assessment_score_4,
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot(
            data,
            label=[
                "Objective 1\nMin Idle Employee",
                "Objective 2\nMax Assessment Score",
                "Objective 3\nBalancing the Workload",
                "MOO with\nGoal Programming",
            ],
        )
        plt.title("Overall Assessment Score Boxplot")
        plt.xticks(rotation=15)
        plt.savefig("./output_CA/compare_SO_MOO.png")
        plt.show()

        return assessment_score_4

    except Exception as e:
        send_discord_notification(f"An error occured in s8_MOO: {e}")
        print(f"An error occurred in s8_MOO: {e}")


class GapCallback:
    def __init__(self):
        self.reported_gaps = set()

    def __call__(self, model, where):
        if where == GRB.Callback.MIP:
            nodecount = model.cbGet(GRB.Callback.MIP_NODCNT)
            if (
                nodecount % 100 == 0
            ):  # Adjust the frequency of the callback call if needed
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                if obj_best < GRB.INFINITY and obj_bound > -GRB.INFINITY:
                    gap = abs((obj_bound - obj_best) / obj_best) * 100
                    percentage_gap = gap

                    # Report gap for multiples of 5
                    if percentage_gap > 10 and int(percentage_gap) % 5 == 0:
                        if int(percentage_gap) not in self.reported_gaps:
                            print(f"Model reached {int(percentage_gap)}% gap.")
                            send_discord_notification(
                                f"Model reached {int(percentage_gap)}% gap."
                            )
                            self.reported_gaps.add(int(percentage_gap))

                    # Report gap for each integer when gap <= 10
                    elif percentage_gap <= 10:
                        if int(percentage_gap) not in self.reported_gaps:
                            print(f"Model reached {int(percentage_gap)}% gap.")
                            send_discord_notification(
                                f"Model reached {int(percentage_gap) - 1}% gap."
                            )
                            self.reported_gaps.add(int(percentage_gap))


def close_plot():
    plt.close()


def get_timestamp():
    return datetime.datetime.now().strftime("(%H.%M %d/%m/%Y)")


def send_discord_notification(message):
    url = "https://discord.com/api/webhooks/1245288786024206398/ZQEM6oSRWOYw0DV9_3WUNGYIk7yZQ-M1OdsZU6J3DhUKhZ-qmi8ecqJRAVBRqwpJt0q8"
    data = {"content": f"{get_timestamp()} {message}"}
    response = requests.post(
        url, data=json.dumps(data), headers={"Content-Type": "application/json"}
    )

    if response.status_code == 204:
        print("Notification sent successfully.")
    else:
        print("Failed to send notification.")


def main():
    header = """
    ==============================================

        TASK ASSIGNMENT OPTIMIZATION PROBLEM

    ==============================================
    """
    print(header)

    header_msg = (
        f"Task Assignment Optimization Problem: START with Competence Assessment"
    )
    send_discord_notification(header_msg)

    """
    =============================================
    
                Execute the Steps
    
    =============================================
    """

    try:
        # Section 1
        (
            employees,
            tasks,
            story_points,
            company_tasks,
            score,
            info,
        ) = s1_data_structure_CA(employee_path, task_path)

        section_1_msg_1 = f"Section 1: Data Structure Run Successfully"
        print(section_1_msg_1)
        send_discord_notification(section_1_msg_1)

        # Section 2
        model = s2_construct_model(license_file_path)
        if model:
            print(f"Section 2: Construct Model Run Successfully\n\n")
            send_discord_notification("Section 2: Construct Model Run Successfully")
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, z, max_employee_workload, max_workload = s3_decision_variable(
            model, tasks, employees, company_tasks
        )
        if x and y:
            print(f"Section 3: Build Decision Variable Run Successfully\n\n")
            send_discord_notification(
                "Section 3: Build Decision Variable Run Successfully"
            )
        else:
            raise Exception("Decision variable construction failed.")

        # Section 4
        s4_constraint(
            model, x, y, z, employees, company_tasks, story_points, max_workload
        )
        print(f"Section 4: Set Constraint Run Successfully\n\n")
        send_discord_notification("Section 4: Set Constraint Run Successfully")

        """
        =============================================
        
                    Solve The Objective
        
        =============================================
        """

        mu_Z_star = {i: 0.00 for i in range(1, 4)}

        # Section 5
        send_discord_notification("Section 5: Objective 1 START")
        start_time = datetime.datetime.now()
        mu_Z_1, mu_Z_star, assessment_score_1 = s5_objective1(
            model,
            employees,
            company_tasks,
            y,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_1 and assessment_score_1 is not None:
            print(f"Section 5: Objective 1 Run Successfully\n\n")
            send_discord_notification(
                f"Section 5: Objective 1 Run Successfully with {duration} seconds"
            )
        else:
            send_discord_notification("Objective 1 failed.")
            raise Exception("Objective 1 failed.")

        # Section 6
        send_discord_notification("Section 6: Objective 2 START")
        start_time = datetime.datetime.now()
        mu_Z_2, mu_Z_star, assessment_score_2 = s6_objective2(
            model,
            employees,
            company_tasks,
            z,
            score,
            story_points,
            max_employee_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_2 and assessment_score_2 is not None:
            send_discord_notification(
                f"Section 6: Objective 2 Run Successfully with {duration} seconds"
            )
            print(f"Section 6: Objective 2 Run Successfully\n\n")
        else:
            send_discord_notification("Objective 2 failed.")
            raise Exception("Objective 2 failed.")

        # Section 7
        send_discord_notification("Section 7: Objective 3 START")
        start_time = datetime.datetime.now()
        mu_Z_3, mu_Z_star, assessment_score_3 = s7_objective3(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            max_workload,
            mu_Z_star,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds

        if mu_Z_3 and assessment_score_3 is not None:
            send_discord_notification(
                f"Section 7: Objective 3 Run Successfully with {duration} seconds"
            )
            print(f"Section 7: Objective 3 Run Successfully\n\n")
        else:
            send_discord_notification("Objective 3 failed.")
            raise Exception("Objective 3 failed.")

        # Section 8
        send_discord_notification("Section 8: MOO START")
        start_time = datetime.datetime.now()
        assessment_score_4 = s8_MOO(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            max_employee_workload,
            mu_Z_1,
            mu_Z_2,
            mu_Z_3,
            mu_Z_star,
            assessment_score_1,
            assessment_score_2,
            assessment_score_3,
        )
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds
        if assessment_score_4 is not None:
            send_discord_notification(
                f"Section 8: MOO Run Successfully with {duration} seconds"
            )
            print(f"Section 8: MOO Run Successfully\n\n")
        else:
            send_discord_notification("MOO failed.")
            raise Exception("MOO failed.")

    except Exception as e:
        send_discord_notification(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
        send_discord_notification("Script ran successfully.")
    except Exception as e:
        error_message = str(e)
        send_discord_notification(f"Script failed with error: {error_message}")
