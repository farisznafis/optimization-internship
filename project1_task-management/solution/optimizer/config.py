# import library
import yaml
from . import helper


try:
    # Load the .yaml file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Optimization Parameters from Environment Variables
    presolve = config["PRESOLVE"]
    MIPFocus = config["MIPFOCUS"]
    MIPGap = config["MIPGAP"]
    heuristics = config["HEURISTICS"]
    threads = config["THREADS"]
    MIPGap_moo = config["MIPGAP_MOO"]

    # Objective Weights from Environment Variables
    weight_obj1 = config["WEIGHT_OBJ1"]
    weight_obj2 = config["WEIGHT_OBJ2"]
    weight_obj3 = config["WEIGHT_OBJ3"]

    # Maximum Workload from Environment Variables
    max_employee_workload = config["MAX_EMPLOYEE_WORKLOAD"]

    # Methodology from Environment Variables
    overqualification = config["OVERQUALIFICATION"]

    metrics = (
        "CompetencyAssessment" if overqualification else "WeightedEuclideanDistance"
    )

    discord = config["DISCORD"]

except Exception as e:
    msg = f"Error in getting the Configuration Parameters: {e}"
    helper.show(msg)
