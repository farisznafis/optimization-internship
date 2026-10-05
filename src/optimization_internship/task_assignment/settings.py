"""Run settings for the task-assignment solver, loaded from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/task_assignment.yaml")

# YAML key -> Settings field
_KEYS = {
    "MAX_EMPLOYEE_WORKLOAD": "max_employee_workload",
    "PRESOLVE": "presolve",
    "MIPFOCUS": "mip_focus",
    "MIPGAP": "mip_gap",
    "HEURISTICS": "heuristics",
    "THREADS": "threads",
    "MIPGAP_MOO": "mip_gap_moo",
    "TIME_LIMIT": "time_limit",
    "WEIGHT_OBJ1": "weight_obj1",
    "WEIGHT_OBJ2": "weight_obj2",
    "WEIGHT_OBJ3": "weight_obj3",
    "OVERQUALIFICATION": "overqualification",
    "DISCORD": "discord",
}


@dataclass(frozen=True)
class Settings:
    max_employee_workload: int = 10

    # Gurobi parameters
    presolve: int = 2
    mip_focus: int = 1
    mip_gap: float = 0.01
    heuristics: float = 0.8
    threads: int = 2
    mip_gap_moo: float = 0.05
    time_limit: float | None = None  # seconds per solve, None = no limit

    # Goal-programming weights for objectives 1..3
    weight_obj1: float = 0.03
    weight_obj2: float = 0.9
    weight_obj3: float = 0.07

    # True: Competency Assessment (MSG), False: Weighted Euclidean Distance
    overqualification: bool = True

    # Send progress messages to the Discord webhook in DISCORD_URL
    discord: bool = False

    @property
    def weights(self) -> dict[int, float]:
        return {1: self.weight_obj1, 2: self.weight_obj2, 3: self.weight_obj3}

    @property
    def metric(self) -> str:
        return "CompetencyAssessment" if self.overqualification else "WeightedEuclideanDistance"

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        with open(path, encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        unknown = set(raw) - set(_KEYS)
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        return cls(**{_KEYS[key]: value for key, value in raw.items()})
