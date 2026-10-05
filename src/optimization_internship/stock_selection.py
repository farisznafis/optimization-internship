"""Challenge 2: stock selection for portfolio optimisation with Gurobi.

Find the allocation that maximises the annualised average return (AAR) of a
USD 10k portfolio subject to:

* portfolio risk ``sqrt(sum(r_i^2 * x_i^2))`` at most 15%;
* portfolio return at least 20%;
* at least 3 stocks selected, each selected stock receiving at least 5.25%;
* allocations sum to 100%.

The input is the yearly return table of the challenge (``returns.csv``). AAR is
taken from its ``aar`` column; risk is the population standard deviation of the
yearly returns, which matches ``summary.csv``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from .gurobi import make_env

DEFAULT_DATA = Path("projects/stock_selection/data/returns.csv")


@dataclass(frozen=True)
class PortfolioRules:
    max_risk: float = 0.15
    min_return: float = 0.20
    min_stocks: int = 3
    min_allocation: float = 0.0525
    budget: float = 10_000


@dataclass
class Portfolio:
    status: str
    allocation: pd.Series  # fraction of the budget per stock, only stocks > 0
    expected_return: float
    risk: float

    @property
    def found(self) -> bool:
        return not self.allocation.empty


def load_stocks(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    """Return one row per stock with decimal ``aar`` and ``risk`` columns."""
    raw = pd.read_csv(path)
    missing = {"stock", "return", "aar"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    risk = raw.groupby("stock")["return"].std(ddof=0) / 100
    aar = raw.dropna(subset=["aar"]).set_index("stock")["aar"]
    return pd.DataFrame({"aar": aar, "risk": risk}).dropna().sort_index()


def optimize(stocks: pd.DataFrame, rules: PortfolioRules | None = None, output: bool = False) -> Portfolio:
    rules = rules or PortfolioRules()
    names = stocks.index.tolist()
    aar = stocks["aar"].to_dict()
    risk = stocks["risk"].to_dict()

    with make_env(output) as env, gp.Model("portfolio", env=env) as model:
        x = model.addVars(names, lb=0.0, ub=1.0, name="allocation")
        y = model.addVars(names, vtype=GRB.BINARY, name="select")

        model.addQConstr(
            gp.quicksum(risk[s] ** 2 * x[s] * x[s] for s in names) <= rules.max_risk**2,
            "max_risk",
        )
        expected_return = gp.quicksum(aar[s] * x[s] for s in names)
        model.addConstr(expected_return >= rules.min_return, "min_return")
        model.addConstr(y.sum() >= rules.min_stocks, "min_stocks")
        for s in names:
            # A stock gets money only if selected, and then at least min_allocation.
            model.addConstr(x[s] <= y[s], f"link_{s}")
            model.addConstr(x[s] >= rules.min_allocation * y[s], f"min_alloc_{s}")
        model.addConstr(x.sum() == 1, "budget")

        model.setObjective(expected_return, GRB.MAXIMIZE)
        model.optimize()

        status = "OPTIMAL" if model.Status == GRB.OPTIMAL else f"STATUS_{model.Status}"
        if model.SolCount == 0:
            return Portfolio(status, pd.Series(dtype=float), math.nan, math.nan)

        allocation = pd.Series({s: x[s].X for s in names if x[s].X > 1e-6}, name="allocation")

    portfolio_return = float((allocation * stocks.loc[allocation.index, "aar"]).sum())
    portfolio_risk = math.sqrt(float(((allocation * stocks.loc[allocation.index, "risk"]) ** 2).sum()))
    return Portfolio(status, allocation, portfolio_return, portfolio_risk)


def run(data: Path = DEFAULT_DATA, rules: PortfolioRules | None = None, verbose: bool = False) -> int:
    rules = rules or PortfolioRules()
    stocks = load_stocks(data)
    portfolio = optimize(stocks, rules, output=verbose)

    print("\n=== Portfolio optimisation ===")
    if not portfolio.found:
        print(f"No feasible portfolio found (status: {portfolio.status})")
        return 1

    table = pd.DataFrame(
        {
            "allocation_%": portfolio.allocation * 100,
            "amount_usd": portfolio.allocation * rules.budget,
            "aar_%": stocks.loc[portfolio.allocation.index, "aar"] * 100,
            "return_contribution_%": portfolio.allocation
            * stocks.loc[portfolio.allocation.index, "aar"]
            * 100,
        }
    )
    print(f"Status                      : {portfolio.status}")
    print(table.round(2).to_string())
    print(f"\nExpected annual return      : {portfolio.expected_return * 100:.2f}%")
    print(f"Portfolio risk              : {portfolio.risk * 100:.2f}%")
    return 0
