"""Command-line interface: ``optim <command> [options]``.

Run from the repository root so the default data paths resolve, or pass your
own paths. Solver modules are imported lazily to keep ``--help`` fast.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from . import __version__

log = logging.getLogger("optim")


def _course_selection(args: argparse.Namespace) -> int:
    from . import course_selection

    return course_selection.run(args.data or course_selection.DEFAULT_DATA, args.seed)


def _stock_selection(args: argparse.Namespace) -> int:
    from . import stock_selection

    rules = stock_selection.PortfolioRules(
        max_risk=args.max_risk,
        min_return=args.min_return,
        min_stocks=args.min_stocks,
        min_allocation=args.min_allocation,
        budget=args.budget,
    )
    return stock_selection.run(args.data or stock_selection.DEFAULT_DATA, rules, verbose=args.verbose)


def _burrito_game(args: argparse.Namespace) -> int:
    from . import burrito_game

    if args.list_rounds:
        for round_dir in sorted(p for p in burrito_game.DATA_DIR.iterdir() if p.is_dir()):
            print(round_dir.name)
        return 0
    round_dir = args.round_dir or burrito_game.DATA_DIR / args.round
    return burrito_game.run(round_dir, args.plot)


def _task_assignment(args: argparse.Namespace) -> int:
    from .task_assignment import data, pipeline
    from .task_assignment.settings import Settings

    settings = Settings.from_yaml(args.config)
    if args.mini:
        employees, tasks = data.MINI_EMPLOYEES, data.MINI_TASKS
    else:
        employees = args.employees or Path(os.getenv("EMPLOYEE_PATH") or data.DEFAULT_EMPLOYEES)
        tasks = args.tasks or Path(os.getenv("TASK_PATH") or data.DEFAULT_TASKS)

    results = pipeline.run(
        settings,
        employees,
        tasks,
        args.output,
        webhook_url=os.getenv("DISCORD_URL") or None,
        solver_log=not args.quiet,
    )
    return 0 if results[-1].found else 1


def build_parser() -> argparse.ArgumentParser:
    from .task_assignment.settings import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(
        prog="optim",
        description="Optimization projects from the TK Bunga Matahari internship.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging and solver output")
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p = commands.add_parser("course-selection", help="Challenge 1: course selection (CP-SAT)")
    p.add_argument("--data", type=Path, help="default: projects/course_selection/data/course_list.csv")
    p.add_argument("--seed", type=int, help="shuffle courses first to explore tied optima")
    p.set_defaults(func=_course_selection)

    p = commands.add_parser("stock-selection", help="Challenge 2: portfolio optimisation (Gurobi)")
    p.add_argument("--data", type=Path, help="default: projects/stock_selection/data/returns.csv")
    p.add_argument("--max-risk", type=float, default=0.15, help="default: %(default)s")
    p.add_argument("--min-return", type=float, default=0.20, help="default: %(default)s")
    p.add_argument("--min-stocks", type=int, default=3, help="default: %(default)s")
    p.add_argument("--min-allocation", type=float, default=0.0525, help="default: %(default)s")
    p.add_argument("--budget", type=float, default=10_000, help="USD, default: %(default)s")
    p.set_defaults(func=_stock_selection)

    p = commands.add_parser("burrito-game", help="Burrito Optimization Game (CP-SAT)")
    p.add_argument(
        "--round",
        default="muafi_r1d1_2",
        help="round folder under projects/burrito_game/data (default: %(default)s)",
    )
    p.add_argument("--round-dir", type=Path, help="any folder with the four round CSV files")
    p.add_argument("--plot", type=Path, help="save a map of the plan to this PNG")
    p.add_argument("--list-rounds", action="store_true", help="list bundled rounds and exit")
    p.set_defaults(func=_burrito_game)

    p = commands.add_parser("task-assignment", help="Project 1: Scrum task assignment (Gurobi, MOO)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="default: %(default)s")
    p.add_argument("--employees", type=Path, help="employee CSV (env EMPLOYEE_PATH)")
    p.add_argument("--tasks", type=Path, help="task CSV (env TASK_PATH)")
    p.add_argument("--mini", action="store_true", help="use the bundled 5x10 mini dataset")
    p.add_argument(
        "--output", type=Path, default=Path("outputs/task_assignment"), help="default: %(default)s"
    )
    p.add_argument("--quiet", action="store_true", help="hide the Gurobi solver log")
    p.set_defaults(func=_task_assignment)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error, exc_info=args.verbose)
        return 2
