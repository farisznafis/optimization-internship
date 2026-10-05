"""Shared Gurobi environment setup.

Without license variables, gurobipy falls back to the size-limited license that
ships with the pip package (enough for the mini datasets and the small
challenges). For the full task-assignment dataset set the Web License Service
(WLS) variables ``WLSACCESSID``, ``WLSSECRET`` and ``LICENSEID``.
"""

from __future__ import annotations

import os
from typing import Any

import gurobipy as gp

LICENSE_VARS = ("WLSACCESSID", "WLSSECRET", "LICENSEID")


def license_params_from_env() -> dict[str, Any]:
    """Read WLS license parameters from the environment.

    Returns an empty dict when none are set. Raises ``ValueError`` when only some
    of them are set, because that is almost always a typo in ``.env``.
    """
    values = {name: os.getenv(name) for name in LICENSE_VARS}
    present = {name: value for name, value in values.items() if value}

    if not present:
        return {}
    if len(present) != len(LICENSE_VARS):
        missing = ", ".join(name for name in LICENSE_VARS if name not in present)
        raise ValueError(f"Incomplete Gurobi WLS license, missing: {missing}")

    return {
        "WLSACCESSID": present["WLSACCESSID"],
        "WLSSECRET": present["WLSSECRET"],
        "LICENSEID": int(present["LICENSEID"]),
    }


def make_env(output: bool = True) -> gp.Env:
    """Start a Gurobi environment using the WLS license if one is configured."""
    env = gp.Env(empty=True)
    for name, value in license_params_from_env().items():
        env.setParam(name, value)
    env.setParam("OutputFlag", int(output))
    env.start()
    return env
