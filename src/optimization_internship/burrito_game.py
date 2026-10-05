"""Gurobi Burrito Optimization Game, solved with CP-SAT.

Decide where to park burrito trucks for a day to maximise profit:

    maximise  sum_ij (r - k) * scaled_demand_ij * y_ij  -  sum_j f * x_j
    s.t.      sum_j y_ij <= 1        every building is served by at most one truck
              y_ij <= x_j            only placed trucks can serve

Each round directory holds the four CSV files exported from the game
(``*_problem_data.csv``, ``*_truck_node_data.csv``, ``*_demand_node_data.csv``,
``*_demand_truck_data.csv``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from ortools.sat.python import cp_model

DATA_DIR = Path("projects/burrito_game/data")
DEFAULT_ROUND = "muafi_r1d1_2"


@dataclass
class BurritoInstance:
    burrito_price: float
    ingredient_cost: float
    truck_cost: float
    trucks: pd.DataFrame  # index: truck id; columns x, y
    buildings: pd.DataFrame  # index: building id; columns name, x, y, demand
    pairs: pd.DataFrame  # columns building, truck, distance, scaled_demand (> 0 only)


@dataclass
class BurritoPlan:
    status: str
    profit: float
    placed_trucks: list[str]
    assignments: dict[str, str]  # building -> truck

    @property
    def found(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")


def _find(round_dir: Path, suffix: str) -> Path:
    matches = sorted(round_dir.glob(f"*_{suffix}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *_{suffix}.csv in {round_dir}, found {len(matches)}")
    return matches[0]


def load_round(round_dir: Path) -> BurritoInstance:
    problem = pd.read_csv(_find(round_dir, "problem_data")).iloc[0]
    trucks = pd.read_csv(_find(round_dir, "truck_node_data"), index_col="index")
    buildings = pd.read_csv(_find(round_dir, "demand_node_data"), index_col="index")
    pairs = pd.read_csv(_find(round_dir, "demand_truck_data")).rename(
        columns={"demand_node_index": "building", "truck_node_index": "truck"}
    )
    pairs = pairs[pairs["scaled_demand"] > 0].reset_index(drop=True)

    return BurritoInstance(
        burrito_price=float(problem["burrito_price"]),
        ingredient_cost=float(problem["ingredient_cost"]),
        truck_cost=float(problem["truck_cost"]),
        trucks=trucks,
        buildings=buildings,
        pairs=pairs,
    )


def solve(instance: BurritoInstance) -> BurritoPlan:
    model = cp_model.CpModel()
    x = {j: model.new_bool_var(f"placed_{j}") for j in instance.trucks.index}
    y = {
        (row.building, row.truck): model.new_bool_var(f"served_{row.building}_{row.truck}")
        for row in instance.pairs.itertuples()
    }

    for i in instance.buildings.index:
        served_by = [var for (b, _), var in y.items() if b == i]
        if served_by:
            model.add(sum(served_by) <= 1)
    for (_, j), var in y.items():
        model.add(var <= x[j])

    margin = instance.burrito_price - instance.ingredient_cost
    scaled_demand = {(r.building, r.truck): r.scaled_demand for r in instance.pairs.itertuples()}
    revenue = sum(margin * scaled_demand[key] * var for key, var in y.items())
    truck_costs = sum(instance.truck_cost * var for var in x.values())
    model.maximize(revenue - truck_costs)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return BurritoPlan(status_name, 0.0, [], {})

    return BurritoPlan(
        status=status_name,
        profit=solver.objective_value,
        placed_trucks=[j for j, var in x.items() if solver.value(var)],
        assignments={b: j for (b, j), var in y.items() if solver.value(var)},
    )


def draw_plan(instance: BurritoInstance, plan: BurritoPlan) -> Figure:
    """Map of buildings, truck spots and the chosen plan."""
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    for building, truck in plan.assignments.items():
        b, t = instance.buildings.loc[building], instance.trucks.loc[truck]
        ax.plot([b.x, t.x], [b.y, t.y], color="tab:gray", linewidth=0.8, zorder=1)

    b = instance.buildings
    ax.scatter(
        b.x, b.y, s=b.demand * 4, color="tab:blue", alpha=0.6, label="Building (size = demand)", zorder=2
    )
    t = instance.trucks
    ax.scatter(t.x, t.y, marker="s", color="lightgray", edgecolor="gray", label="Truck spot", zorder=2)
    placed = t.loc[plan.placed_trucks]
    ax.scatter(placed.x, placed.y, marker="s", s=80, color="tab:red", label="Placed truck", zorder=3)

    ax.invert_yaxis()  # game coordinates grow downwards
    ax.set_title(f"Burritoville plan, profit {plan.profit:.0f}")
    ax.legend(loc="upper right")
    return fig


def plot_plan(instance: BurritoInstance, plan: BurritoPlan, output: Path) -> None:
    """Save the map from :func:`draw_plan` as an image file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_plan(instance, plan).savefig(output, bbox_inches="tight")


def run(round_dir: Path, plot: Path | None = None) -> int:
    instance = load_round(round_dir)
    print(
        f"Burritos cost {instance.ingredient_cost:g} to make and sell for "
        f"{instance.burrito_price:g}; each truck costs {instance.truck_cost:g} per day."
    )
    print(
        f"{len(instance.trucks)} truck spots, {len(instance.buildings)} buildings, "
        f"{len(instance.pairs)} building/truck pairs with demand."
    )

    plan = solve(instance)
    print("\n=== Burrito truck plan ===")
    if not plan.found:
        print(f"No solution found (status: {plan.status})")
        return 1

    print(f"Status : {plan.status}")
    print(f"Profit : {plan.profit:.2f}")
    print(f"Trucks : {', '.join(plan.placed_trucks) or '-'}")
    for building, truck in sorted(plan.assignments.items()):
        print(f"  {building:<10} -> {truck}")

    if plot is not None:
        plot_plan(instance, plan, plot)
        print(f"\nMap saved to {plot}")
    return 0
