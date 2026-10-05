from __future__ import annotations

from collections.abc import Callable

from gurobipy import GRB, Model


class GapCallback:
    """
    Gurobi callback that reports MIP gap progress while the model is solving.

    Above 10% the gap is reported at every multiple of 5; at or below 10% it is
    reported at every integer percent. Each value is reported once.

    Example:
        model.optimize(GapCallback(print))
    """

    def __init__(self, notify: Callable[[str], None], every_n_nodes: int = 100) -> None:
        self.notify = notify
        self.every_n_nodes = every_n_nodes
        self.reported_gaps: set[int] = set()

    def __call__(self, model: Model, where: int) -> None:
        if where != GRB.Callback.MIP:
            return
        if model.cbGet(GRB.Callback.MIP_NODCNT) % self.every_n_nodes != 0:
            return

        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_best >= GRB.INFINITY or obj_bound <= -GRB.INFINITY or obj_best == 0:
            return

        gap = abs((obj_bound - obj_best) / obj_best) * 100
        bucket = int(gap)
        if bucket in self.reported_gaps:
            return
        if gap > 10 and bucket % 5 != 0:
            return

        self.reported_gaps.add(bucket)
        self.notify(f"Model reached {gap:.2f}% gap.")
