import math

from optimization_internship import burrito_game, course_selection, stock_selection


def test_course_selection_single_objective_meets_rules():
    courses = course_selection.load_courses()
    result = course_selection.solve_single_objective(courses)

    assert result.status == "OPTIMAL"
    assert result.total_credits == course_selection.TOTAL_CREDITS
    assert result.courses.loc[result.courses["group"] == "CS", "credit"].sum() >= 120
    assert result.exam_courses == 0


def test_course_selection_multi_objective_is_not_more_expensive():
    courses = course_selection.load_courses()
    single = course_selection.solve_single_objective(courses)
    multi = course_selection.solve_multi_objective(courses)

    assert multi.found
    assert multi.total_credits == course_selection.TOTAL_CREDITS
    # Allowing (penalised) exams can only keep or lower the cost.
    assert multi.total_cost <= single.total_cost


def test_stock_risk_matches_summary_file():
    stocks = stock_selection.load_stocks()
    summary = stock_selection.pd.read_csv("projects/stock_selection/data/summary.csv", index_col="stock")

    for stock, risk in summary["risk"].items():
        assert math.isclose(stocks.loc[stock, "risk"] * 100, risk, rel_tol=1e-6)


def test_stock_selection_respects_rules():
    rules = stock_selection.PortfolioRules()
    portfolio = stock_selection.optimize(stock_selection.load_stocks(), rules)

    assert portfolio.found
    assert math.isclose(portfolio.allocation.sum(), 1.0, abs_tol=1e-6)
    assert len(portfolio.allocation) >= rules.min_stocks
    assert (portfolio.allocation >= rules.min_allocation - 1e-6).all()
    assert portfolio.risk <= rules.max_risk + 1e-6
    assert portfolio.expected_return >= rules.min_return - 1e-6


def test_burrito_every_bundled_round_solves(tmp_path):
    for round_dir in sorted(p for p in burrito_game.DATA_DIR.iterdir() if p.is_dir()):
        instance = burrito_game.load_round(round_dir)
        plan = burrito_game.solve(instance)

        assert plan.status == "OPTIMAL", round_dir.name
        assert set(plan.assignments.values()) <= set(plan.placed_trucks)
        assert len(plan.assignments) == len(set(plan.assignments))

    burrito_game.plot_plan(instance, plan, tmp_path / "map.png")
    assert (tmp_path / "map.png").stat().st_size > 0
