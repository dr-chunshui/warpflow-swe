"""Lake-at-rest test: well-balancing of the Audusse scheme.

A still water surface zs = 0.5 over the TELEMAC bump bathymetry must stay
still (|u|, |v| < 1e-12) through 200 explicit steps with all-wall
boundaries.  This pins down the sign of the geometric source terms and the
HLLC wall flux.  Run directly: python tests/test_lake_at_rest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.config import (BathymetryConfig, BoundaryGroup, CaseConfig,
                           InitialConfig, MeshConfig, OutputConfig)
from solver.mesh import rect_mesh
from solver.solver import ShallowWaterSolver


def make_cfg(order: int, time_order: int) -> CaseConfig:
    return CaseConfig(
        name="lake_at_rest",
        analytic="",
        t_end=1.0e9,  # drive by steps in the test, not by t_end
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=20.0,
                        ymin=0.0, ymax=2.0, dx=0.25),
        initial=InitialConfig(type="lake_at_rest", params={"zs": 0.5}),
        bathymetry=BathymetryConfig(
            type="bump",
            params={"amplitude": 0.25, "x0": 10.0, "sigma": 1.0}),
        boundary=[
            BoundaryGroup(name="w_xmin", select={"xmin": [0.0, 2.0]}, type="wall"),
            BoundaryGroup(name="w_xmax", select={"xmax": [0.0, 2.0]}, type="wall"),
            BoundaryGroup(name="w_ymin", select={"ymin": [0.0, 20.0]}, type="wall"),
            BoundaryGroup(name="w_ymax", select={"ymax": [0.0, 20.0]}, type="wall"),
        ],
        cfl=0.8, order=order, limiter="barth", time_order=time_order,
        eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_test_lake_at_rest"),
    )


def run_case(order: int, time_order: int, n_steps: int) -> float:
    cfg = make_cfg(order, time_order)
    mesh = rect_mesh(0.0, 20.0, 0.0, 2.0, dx=0.25)
    s = ShallowWaterSolver(cfg, mesh)
    m0 = s.mass()
    for _ in range(n_steps):
        dt = s.compute_dt(1.0e9)
        s.step(dt)
    snap = s.snapshot()
    vel_max = float(max(np.abs(snap["u"]).max(), np.abs(snap["v"]).max()))
    h_err = float(np.abs(snap["zs"] - 0.5).max())
    m1 = s.mass()
    dm = abs(m1 - m0) / m0
    print(f"order {order} time_order {time_order}: {s.step_count} steps, "
          f"|u|max = {vel_max:.3e}, |zs-0.5|max = {h_err:.3e}, dmass = {dm:.3e}")
    assert vel_max < 1.0e-12, f"lake at rest violated: |u|max={vel_max}"
    assert h_err < 1.0e-13, f"surface moved: {h_err}"
    assert dm < 1.0e-12, f"mass drift: {dm}"
    return vel_max


def main() -> int:
    run_case(order=1, time_order=1, n_steps=200)
    run_case(order=2, time_order=2, n_steps=100)
    print("PASS: lake at rest (well-balanced) for order 1 and 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
