import pandas as pd
import pytest

from optimization_internship.cli import main
from optimization_internship.task_assignment import data, pipeline
from optimization_internship.task_assignment.scoring import CompetencyAssessment, WeightedEuclideanDistance
from optimization_internship.task_assignment.settings import DEFAULT_CONFIG, Settings

RCD = pd.DataFrame({"math": [5, 0], "python": [5, 5]}, index=["T1", "T2"])
ACD = pd.DataFrame({"math": [5, 0], "python": [5, 5]}, index=["E1", "E2"])


def test_competency_assessment_prefers_matching_task():
    score, _ = CompetencyAssessment(RCD, ACD).fit()
    assert score["E2"]["T2"] > score["E2"]["T1"]
    assert score["E1"]["T1"] >= 0


def test_weighted_euclidean_perfect_match_scores_one():
    score, _ = WeightedEuclideanDistance(RCD, ACD).fit()
    assert score["E1"]["T1"] == pytest.approx(1.0)
    assert score["E2"]["T1"] < 1.0


def test_settings_reject_unknown_keys(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("MAX_EMPLOYEE_WORKLOAD: 5\nTYPO_KEY: 1\n")
    with pytest.raises(ValueError, match="TYPO_KEY"):
        Settings.from_yaml(config)


def test_default_config_loads():
    settings = Settings.from_yaml(DEFAULT_CONFIG)
    assert sum(settings.weights.values()) == pytest.approx(1.0)


def test_load_problem_rejects_mismatched_competencies(tmp_path):
    employees = tmp_path / "employees.csv"
    tasks = tmp_path / "tasks.csv"
    employees.write_text("employee_id,math\nE1,3\n")
    tasks.write_text("task_id,project_id,story_points,python\nT1,P1,1,3\n")
    with pytest.raises(ValueError, match="competency columns differ"):
        data.load_problem(employees, tasks, overqualification=True)


@pytest.mark.parametrize("overqualification", [True, False])
def test_pipeline_on_mini_data(tmp_path, overqualification):
    settings = Settings(overqualification=overqualification, threads=1)
    results = pipeline.run(settings, data.MINI_EMPLOYEES, data.MINI_TASKS, tmp_path, solver_log=False)

    assert len(results) == 4
    assert all(result.found for result in results)
    problem = data.load_problem(data.MINI_EMPLOYEES, data.MINI_TASKS, overqualification)
    for result in results:
        assigned = [task for tasks in result.assignments["assigned_task"] for task in tasks]
        assert sorted(assigned) == sorted(problem.tasks)  # every task exactly once
        assert all(len(set(companies)) == 1 for companies in result.assignments["company"])
        assert (result.assignments["sum_sp"] <= settings.max_employee_workload).all()

    for name in ("score.csv", "result_1.csv", "result_MOO.csv", "result_MOO.png", "score_comparison.png"):
        assert (tmp_path / name).exists(), name


def test_cli_mini_run(tmp_path):
    assert main(["task-assignment", "--mini", "--quiet", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "result_MOO.csv").exists()


def test_cli_missing_file_returns_error(tmp_path):
    assert main(["course-selection", "--data", str(tmp_path / "missing.csv")]) == 2
