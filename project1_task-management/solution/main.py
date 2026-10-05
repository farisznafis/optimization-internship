"""
Module Name: main.py
Obective: Task Assignment Optimization Problem Solver

Description:
This module contains the solution for the Task Assignment Optimization Problem.
The solution is divided into several sections, each responsible for a specific task:
1. Data Input: Functions to read and validate input data for the task assignment problem.
2. Optimization Model: Implementation of the optimization model using appropriate algorithms.
3. Constraints and Objectives: Definition of constraints and objective functions used in the optimization.
4. Solution Execution: Functions to execute the optimization model and obtain results.
5. Output Processing: Functions to format and output the results in a user-friendly manner.

Functions:
- define_data_structure(employee_path, task_path): Pre-process and structure employee, task data, and score metric data.
- construct_model(license_path): Construct the optimization model with specified parameters.
- decision_variables(model, tasks, employees, company_tasks): Define decision variables for the model.
- constraints(model, x, y, z, employees, company_tasks, story_points, max_workload): Set constraints for the optimization model.
- objective1(model, employees, company_tasks, y, score, story_points, max_employee_workload, mu_Z_star): Minimize the idle employee.
- objective2(model, employees, company_tasks, z, score, story_points, max_employee_workload, mu_Z_star): Maximize the assessment score.
- objective3(model, employees, company_tasks, score, story_points, max_employee_workload, max_workload, mu_Z_star): Balance the workload for each employee.
- MOO(model, employees, company_tasks, score, story_points, max_employee_workload, mu_Z_1, mu_Z_2, mu_Z_3, mu_Z_star, assessment_score_1, assessment_score_2, assessment_score_3): Multi-Objective Optimization using Goal Programming.
- compare_scores(data, title, output_file): Compare the score results for every objective with Box Plot.

Classes:
- CompetencyAssessment: A class to assess the competencies of employees against required competencies for tasks using MSG scores.
- WeightedEuclideanDistance: A class to assess the competencies of employees against required competencies for tasks using Weighted Euclidean Distance.
- GapCallback: A callback class to report optimization progress and gap.

Usage:
Import the module in requirement file and optimizer package to process the assessment.
The solution can be executed by running the main() function, which orchestrates
the entire workflow from data input to output processing.

Example:
>>> if __name__ == "__main__":
...     main()

Authors:
TK Bunga Matahari Team
N. Muafi, I.G.P. Wisnu N., F. Zaid N., Fauzi I.S., Joseph C.L., S. Alisya

Last Modified:
June 2024

"""

# Import library
from optimizer import creds, config, helper, utils, create_model as m, run


def main():
    """
    The main function that executes the steps for the task assignment optimization problem.
    """

    helper.start()

    try:
        # Section 1
        employees, tasks, story_points, company_tasks, score, info = (
            utils.define_data_structure(
                creds.employee_path, creds.task_path, config.overqualification
            )
        )
        msg_1 = "Section 1: Defining Data Structure Run Successfully"
        helper.show(msg_1, helper.discord_status)

        # Section 2
        model, mu_Z_star = m.construct_model(creds.license_params)
        if model:
            msg_2 = "Section 2: Model construction Run Successfully"
            helper.show(msg_2, helper.discord_status)
        else:
            raise Exception("Model construction failed.")

        # Section 3
        x, y, z, max_workload = m.decision_variables(
            model, tasks, employees, company_tasks, config.max_employee_workload
        )
        if x and y and z and max_workload:
            msg_3 = "Section 3: Decision variables construction Run Successfully"
            helper.show(msg_3, helper.discord_status)
        else:
            raise Exception("Decision variables construction failed.")

        # Section 4
        m.constraints(
            model,
            x,
            y,
            z,
            max_workload,
            employees,
            company_tasks,
            story_points,
            config.max_employee_workload,
        )
        msg_4 = "Section 4: Set-up The Constraints Run Successfully"
        helper.show(msg_4, helper.discord_status)

        print("\nSolving The Objective...\n\n")

        # Section 5
        mu_Z_1, mu_Z_star, assessment_score_1 = run.run_objective1(
            model,
            employees,
            company_tasks,
            y,
            score,
            story_points,
            config.max_employee_workload,
            mu_Z_star,
        )

        # Section 6
        mu_Z_2, mu_Z_star, assessment_score_2 = run.run_objective2(
            model,
            employees,
            company_tasks,
            z,
            score,
            story_points,
            config.max_employee_workload,
            mu_Z_star,
        )

        # Section 7
        mu_Z_3, mu_Z_star, assessment_score_3 = run.run_objective3(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            config.max_employee_workload,
            max_workload,
            mu_Z_star,
        )

        # Section 8
        mu_Z = {1: mu_Z_1, 2: mu_Z_2, 3: mu_Z_3}

        assessment_score_4 = run.run_MOO(
            model,
            employees,
            company_tasks,
            score,
            story_points,
            config.max_employee_workload,
            mu_Z,
            mu_Z_star,
        )

        # compare score in every objective with Box Plot
        data = [
            assessment_score_1,
            assessment_score_2,
            assessment_score_3,
            assessment_score_4,
        ]

        title = [
            "Objective 1\nMin Idle Employee",
            "Objective 2\nMax Assessment Score",
            "Objective 3\nBalancing the Workload",
            "MOO with\nGoal Programming",
        ]

        utils.compare_scores(data, title, "output/score_comaprison")

    except Exception as e:
        msg = f"An error occurred: {e}"
        helper.show(msg, helper.discord_status)


if __name__ == "__main__":
    try:
        main()
        helper.show("Script ran successfully.", helper.discord_status)
    except Exception as e:
        error_message = str(e)
        helper.show(f"Script failed with error: {error_message}", helper.discord_status)
