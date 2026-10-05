"""Interactive demo of the optimization projects, built with Gradio.

Run locally:   pip install -e ".[demo]"  then  python app.py
Hosted demo:   Hugging Face Spaces, see deploy/huggingface/README.md
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # the default data paths are relative to the repository root
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
from gurobipy import GurobiError  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from optimization_internship import burrito_game, course_selection, stock_selection  # noqa: E402
from optimization_internship.gurobi import license_params_from_env  # noqa: E402
from optimization_internship.task_assignment import data as ta_data  # noqa: E402
from optimization_internship.task_assignment import pipeline  # noqa: E402
from optimization_internship.task_assignment.reporting import comparison_figure  # noqa: E402
from optimization_internship.task_assignment.settings import DEFAULT_CONFIG, Settings  # noqa: E402

REPO_URL = os.getenv("REPO_URL", "")  # set as a Space variable to show the source link

# Public demo limits. The license bundled with gurobipy allows 2,000 variables.
SIZE_LIMITED_MAX_VARIABLES = 2000
SOLVE_TIME_LIMIT = 30  # seconds per Gurobi solve
MAX_UPLOAD = "1mb"

MINI, UPLOAD = "Mini sample (5 talents, 10 tasks)", "Upload my own CSV files"
CA, WED = "Competency Assessment (MSG)", "Weighted Euclidean Distance"


# --------------------------------------------------------------------------- task assignment


def _estimate_variables(employee_path: Path, task_path: Path) -> int:
    """x and z (tasks x talents), y (talents x projects), max workload and 4 deviations."""
    n_employees = len(pd.read_csv(employee_path))
    tasks = pd.read_csv(task_path)
    n_projects = tasks["project_id"].nunique() if "project_id" in tasks else 0
    return 2 * len(tasks) * n_employees + n_employees * n_projects + 5


def solve_task_assignment(dataset, employees_file, tasks_file, method, max_workload, w1, w2, w3):
    if dataset == MINI:
        employee_path, task_path = ta_data.MINI_EMPLOYEES, ta_data.MINI_TASKS
    elif not employees_file or not tasks_file:
        raise gr.Error("Upload both the employees CSV and the tasks CSV.")
    else:
        employee_path, task_path = Path(employees_file), Path(tasks_file)

    try:
        n_variables = _estimate_variables(employee_path, task_path)
    except Exception as error:
        raise gr.Error(f"Could not read the CSV files: {error}") from error
    if n_variables > SIZE_LIMITED_MAX_VARIABLES and not license_params_from_env():
        raise gr.Error(
            f"This dataset needs about {n_variables:,} variables, but the free Gurobi license "
            f"used by this demo allows {SIZE_LIMITED_MAX_VARIABLES:,}. Try fewer talents or tasks "
            "(for example 10 talents x 40 tasks), or run it locally with your own license."
        )

    settings = dataclasses.replace(
        Settings.from_yaml(DEFAULT_CONFIG),
        overqualification=method == CA,
        max_employee_workload=int(max_workload),
        weight_obj1=w1,
        weight_obj2=w2,
        weight_obj3=w3,
        time_limit=SOLVE_TIME_LIMIT,
        threads=1,
        discord=False,
    )
    output_dir = Path(tempfile.mkdtemp(prefix="task_assignment_"))
    try:
        results = pipeline.run(settings, employee_path, task_path, output_dir, solver_log=False)
    except (ValueError, KeyError) as error:
        raise gr.Error(f"Invalid input: {error}") from error
    except GurobiError as error:
        raise gr.Error(f"Gurobi stopped: {error}") from error

    stats = pd.DataFrame(
        [
            {
                "Objective": result.title,
                "Status": result.status,
                "Objective value": round(result.objective_value, 4) if result.found else None,
                "Active talents": result.stats["active_employee"],
                "Idle talents": result.stats["idle_employee"],
                "Story points used": result.stats["active_sp"],
                "Spare capacity": int(result.assignments["wasted_sp"].sum()) if result.found else None,
                "Mean score": round(result.scores.mean(), 4) if not result.scores.empty else None,
            }
            for result in results
        ]
    )

    moo = results[-1].assignments.reset_index()
    moo_table = pd.DataFrame(
        {
            "Talent": moo["employee"],
            "Project": moo["company"].map(lambda projects: ", ".join(sorted(set(projects)))),
            "Tasks": moo["assigned_task"].map(", ".join),
            "Story points": moo["sum_sp"],
            "Scores": moo["assessment_score"].map(lambda scores: ", ".join(f"{s:.3f}" for s in scores)),
        }
    )

    files = sorted(str(path) for path in output_dir.iterdir() if path.suffix in (".csv", ".png"))
    return stats, comparison_figure(results), moo_table, files


def _toggle_upload(dataset):
    visible = dataset == UPLOAD
    return gr.File(visible=visible), gr.File(visible=visible), gr.File(visible=visible)


# --------------------------------------------------------------------------- burrito game

ROUNDS = sorted(path.name for path in burrito_game.DATA_DIR.iterdir() if path.is_dir())


def burrito_defaults(round_name):
    instance = burrito_game.load_round(burrito_game.DATA_DIR / round_name)
    return instance.burrito_price, instance.ingredient_cost, instance.truck_cost


def solve_burrito(round_name, burrito_price, ingredient_cost, truck_cost):
    instance = dataclasses.replace(
        burrito_game.load_round(burrito_game.DATA_DIR / round_name),
        burrito_price=burrito_price,
        ingredient_cost=ingredient_cost,
        truck_cost=truck_cost,
    )
    plan = burrito_game.solve(instance)
    if not plan.found:
        raise gr.Error(f"No solution found (status: {plan.status})")

    pairs = instance.pairs.set_index(["building", "truck"])["scaled_demand"]
    table = pd.DataFrame(
        [
            {
                "Building": building,
                "Name": instance.buildings.loc[building, "name"],
                "Truck": truck,
                "Burritos sold": int(pairs[building, truck]),
            }
            for building, truck in sorted(plan.assignments.items())
        ]
    )
    sold = int(table["Burritos sold"].sum()) if not table.empty else 0
    summary = (
        f"### Profit: ₲{plan.profit:,.0f}\n"
        f"**{len(plan.placed_trucks)}** trucks placed ({', '.join(plan.placed_trucks) or 'none'}), "
        f"serving **{len(plan.assignments)}** of {len(instance.buildings)} buildings and selling "
        f"**{sold}** burritos."
    )
    return summary, burrito_game.draw_plan(instance, plan), table


# --------------------------------------------------------------------------- stock selection

STOCKS = stock_selection.load_stocks()


def _stock_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Stock": STOCKS.index,
            "Annualised avg. return %": (STOCKS["aar"] * 100).round(2).values,
            "Risk %": (STOCKS["risk"] * 100).round(2).values,
        }
    )


def solve_stocks(max_risk, min_return, min_stocks, min_allocation, budget):
    rules = stock_selection.PortfolioRules(
        max_risk=max_risk / 100,
        min_return=min_return / 100,
        min_stocks=int(min_stocks),
        min_allocation=min_allocation / 100,
        budget=budget,
    )
    portfolio = stock_selection.optimize(STOCKS, rules)
    if not portfolio.found:
        raise gr.Error(
            "No portfolio satisfies these rules. Loosen the risk limit or lower the minimum return."
        )

    allocation = portfolio.allocation
    table = pd.DataFrame(
        {
            "Stock": allocation.index,
            "Allocation %": (allocation * 100).round(2).values,
            "Amount USD": (allocation * rules.budget).round(2).values,
            "Return contribution %": (allocation * STOCKS.loc[allocation.index, "aar"] * 100).round(2).values,
        }
    )
    summary = (
        f"### Expected annual return: {portfolio.expected_return * 100:.2f}%\n"
        f"Portfolio risk **{portfolio.risk * 100:.2f}%** (limit {max_risk:g}%), "
        f"{len(allocation)} stocks selected."
    )

    fig = Figure(figsize=(10, 4))
    left, right = fig.subplots(1, 2)
    left.bar(allocation.index, allocation * 100, color="tab:green")
    left.set_ylabel("Allocation %")
    left.set_title("Allocation")
    share = [float(allocation.get(stock, 0.0)) for stock in STOCKS.index]
    right.scatter(
        STOCKS["risk"] * 100,
        STOCKS["aar"] * 100,
        s=[40 + s * 1500 for s in share],
        color=["tab:green" if s > 0 else "lightgray" for s in share],
        edgecolor="gray",
    )
    for stock, row in STOCKS.iterrows():
        right.annotate(
            stock, (row["risk"] * 100, row["aar"] * 100), xytext=(6, 4), textcoords="offset points"
        )
    right.set_xlabel("Risk %")
    right.set_ylabel("Annualised avg. return %")
    right.set_title("Risk vs return (size = allocation)")
    fig.tight_layout()
    return summary, table, fig


# --------------------------------------------------------------------------- course selection


def solve_courses(total_credits, min_cs_credits, weight_exam):
    courses = course_selection.load_courses()
    rules = {"total_credits": int(total_credits), "min_cs_credits": int(min_cs_credits)}
    single = course_selection.solve_single_objective(courses, **rules)
    multi = course_selection.solve_multi_objective(
        courses, weight_cost=1 - weight_exam, weight_exam=weight_exam, **rules
    )

    def describe(name, selection):
        if not selection.found:
            return f"**{name}:** no feasible selection ({selection.status})."
        return (
            f"**{name}:** cost **{selection.total_cost:,}**, {selection.total_credits} credits, "
            f"{selection.exam_courses} exam course(s)."
        )

    summary = describe("No exams", single) + "\n\n" + describe("Weighted", multi)
    return summary, single.courses, multi.courses


# --------------------------------------------------------------------------- layout

INTRO = """
# Optimization Internship · TK Bunga Matahari

Four optimization problems from our internship, solved live with **Gurobi** and **OR-Tools**.
Change the inputs and press **Solve**: the models run on this server, not from cached results.
"""

TASK_INTRO = r"""
Assign Scrum tasks from several projects to data talents. Three goals pull in different directions:
**(1)** keep as few talents idle as possible, **(2)** give tasks to the people whose skills fit best,
**(3)** keep the busiest talent's workload low. Each goal is first optimised on its own; then
**goal programming** looks for one assignment that stays as close as possible to all three optima,
weighted by the sliders.
"""

ABOUT = """
### About

Built by the TK Bunga Matahari team during our internship:
N. Muafi, I.G.P. Wisnu N., F. Zaid N., Fauzi I.S., Joseph C.L., S. Alisya.
Supervisors: Yusuf F., Muhajir A.H., Alva A.S.

| Project | Technique | Solver |
| --- | --- | --- |
| Scrum task assignment | MIP, multi-objective goal programming | Gurobi |
| Burrito Optimization Game | facility location | OR-Tools CP-SAT |
| Stock selection | mixed-integer quadratically constrained program | Gurobi |
| Course selection | binary IP, weighted multi-objective | OR-Tools CP-SAT |

This demo uses the size-limited Gurobi license that ships with `gurobipy` (2,000 variables), so the
task-assignment project runs on small datasets here. The full dataset (109 talents, 300 tasks)
runs with the command-line tool and a full license.
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Optimization Internship") as demo:
        gr.Markdown(INTRO + (f"\nSource code: [{REPO_URL}]({REPO_URL})" if REPO_URL else ""))

        with gr.Tab("Task assignment"):
            gr.Markdown(TASK_INTRO)
            with gr.Row():
                with gr.Column(scale=1):
                    dataset = gr.Radio([MINI, UPLOAD], value=MINI, label="Dataset")
                    employees_file = gr.File(
                        label="Employees CSV", file_types=[".csv"], type="filepath", visible=False
                    )
                    tasks_file = gr.File(
                        label="Tasks CSV", file_types=[".csv"], type="filepath", visible=False
                    )
                    examples = gr.File(
                        value=[str(ta_data.MINI_EMPLOYEES), str(ta_data.MINI_TASKS)],
                        label="Example files (use them as a template)",
                        file_count="multiple",
                        interactive=False,
                        visible=False,
                    )
                    method = gr.Radio([CA, WED], value=CA, label="Skill-matching score")
                    max_workload = gr.Slider(1, 20, value=10, step=1, label="Max story points per talent")
                    gr.Markdown("**Goal-programming weights**")
                    w1 = gr.Slider(0, 1, value=0.03, step=0.01, label="1 · Min idle talents")
                    w2 = gr.Slider(0, 1, value=0.90, step=0.01, label="2 · Max skill score")
                    w3 = gr.Slider(0, 1, value=0.07, step=0.01, label="3 · Balance workload")
                    task_button = gr.Button("Solve", variant="primary")
                with gr.Column(scale=2):
                    task_stats = gr.Dataframe(label="Result per objective", interactive=False)
                    task_plot = gr.Plot(label="Skill score of the assigned tasks")
                    task_table = gr.Dataframe(
                        label="Assignment chosen by goal programming", interactive=False
                    )
                    task_files = gr.File(label="Download all results", file_count="multiple")
            dataset.change(
                _toggle_upload, dataset, [employees_file, tasks_file, examples], api_visibility="private"
            )
            task_button.click(
                solve_task_assignment,
                [dataset, employees_file, tasks_file, method, max_workload, w1, w2, w3],
                [task_stats, task_plot, task_table, task_files],
            )

        with gr.Tab("Burrito game"):
            gr.Markdown(
                "Place burrito trucks around Burritoville to maximise the day's profit. Each truck costs "
                "money; customers walk to a nearby truck, and fewer come the further they must walk. "
                "Based on [Gurobi's Burrito Optimization Game](https://www.gurobi.com/burrito-optimization-game/)."
            )
            price, cost, truck = burrito_defaults(burrito_game.DEFAULT_ROUND)
            with gr.Row():
                with gr.Column(scale=1):
                    round_name = gr.Dropdown(ROUNDS, value=burrito_game.DEFAULT_ROUND, label="Game round")
                    burrito_price = gr.Number(price, label="Burrito price (₲)", minimum=0)
                    ingredient_cost = gr.Number(cost, label="Ingredient cost per burrito (₲)", minimum=0)
                    truck_cost = gr.Number(truck, label="Truck cost per day (₲)", minimum=0)
                    burrito_button = gr.Button("Solve", variant="primary")
                with gr.Column(scale=2):
                    burrito_summary = gr.Markdown()
                    burrito_plot = gr.Plot(label="Plan")
                    burrito_table = gr.Dataframe(label="Who buys where", interactive=False)
            round_name.change(
                burrito_defaults,
                round_name,
                [burrito_price, ingredient_cost, truck_cost],
                api_visibility="private",
            )
            burrito_button.click(
                solve_burrito,
                [round_name, burrito_price, ingredient_cost, truck_cost],
                [burrito_summary, burrito_plot, burrito_table],
            )

        with gr.Tab("Stock selection"):
            gr.Markdown(
                "Split a portfolio over seven stocks to maximise the annualised average return, keeping "
                "the portfolio risk under a limit and spreading money over a minimum number of stocks."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    max_risk = gr.Slider(1, 15, value=15, step=0.5, label="Max portfolio risk %")
                    min_return = gr.Slider(0, 40, value=20, step=1, label="Min expected return %")
                    min_stocks = gr.Slider(1, 7, value=3, step=1, label="Min number of stocks")
                    min_allocation = gr.Slider(
                        0, 30, value=5.25, step=0.25, label="Min share per selected stock %"
                    )
                    budget = gr.Number(10_000, label="Budget (USD)", minimum=0)
                    stock_button = gr.Button("Solve", variant="primary")
                    gr.Dataframe(_stock_table(), label="Input data", interactive=False)
                with gr.Column(scale=2):
                    stock_summary = gr.Markdown()
                    stock_plot = gr.Plot(label="Portfolio")
                    stock_table = gr.Dataframe(label="Allocation", interactive=False)
            stock_button.click(
                solve_stocks,
                [max_risk, min_return, min_stocks, min_allocation, budget],
                [stock_summary, stock_table, stock_plot],
            )

        with gr.Tab("Course selection"):
            gr.Markdown(
                "Pick courses for a degree at minimum cost: an exact number of credits, enough of them "
                "from the CS group. Compare a plan without exams against a plan where exams are allowed "
                "but penalised."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    total_credits = gr.Slider(60, 300, value=180, step=15, label="Total credits")
                    min_cs = gr.Slider(0, 300, value=120, step=15, label="Min CS credits")
                    weight_exam = gr.Slider(
                        0, 1, value=0.999, step=0.001, label="Weight of exams vs cost (weighted plan)"
                    )
                    course_button = gr.Button("Solve", variant="primary")
                with gr.Column(scale=2):
                    course_summary = gr.Markdown()
                    single_table = gr.Dataframe(label="Plan without exams", interactive=False)
                    multi_table = gr.Dataframe(label="Weighted plan", interactive=False)
            course_button.click(
                solve_courses,
                [total_credits, min_cs, weight_exam],
                [course_summary, single_table, multi_table],
            )

        with gr.Tab("About"):
            gr.Markdown(ABOUT)

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20).launch(max_file_size=MAX_UPLOAD)
