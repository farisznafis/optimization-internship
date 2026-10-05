import pandas as pd
import pytest

gr = pytest.importorskip("gradio")
app = pytest.importorskip("app")


def test_demo_has_all_tabs():
    tabs = [block.label for block in app.demo.blocks.values() if isinstance(block, gr.Tab)]
    assert tabs == ["Task assignment", "Burrito game", "Stock selection", "Course selection", "About"]


def test_burrito_tab_reacts_to_truck_cost():
    cheap, _, table = app.solve_burrito("muafi_r1d1_2", 10, 5, 250)
    expensive, _, _ = app.solve_burrito("muafi_r1d1_2", 10, 5, 600)
    assert "970" in cheap
    assert not table.empty
    assert cheap != expensive


def test_course_tab_reports_both_plans():
    summary, single, multi = app.solve_courses(180, 120, 0.999)
    assert "12,356" in summary
    assert single["credit"].sum() == 180
    assert multi["credit"].sum() == 180


def test_stock_tab():
    summary, table, _ = app.solve_stocks(15, 20, 3, 5.25, 10_000)
    assert "Expected annual return" in summary
    assert table["Allocation %"].sum() == pytest.approx(100, abs=0.05)


def test_task_assignment_tab_on_mini_data():
    stats, _, table, files = app.solve_task_assignment(app.MINI, None, None, app.CA, 10, 0.03, 0.9, 0.07)
    assert len(stats) == 4
    assert set(pd.Series(table["Talent"])) <= {f"Talent {i}" for i in range(1, 6)}
    assert any(path.endswith("score_comparison.png") for path in files)


def test_task_assignment_rejects_too_large_upload():
    with pytest.raises(gr.Error, match="free Gurobi license"):
        app.solve_task_assignment(
            app.UPLOAD,
            str(app.ta_data.DEFAULT_EMPLOYEES),
            str(app.ta_data.DEFAULT_TASKS),
            app.CA,
            10,
            0.03,
            0.9,
            0.07,
        )
