"""Closed-domain mass conservation with reflecting (wall) boundaries.

A Stoker dam break on a flat bed inside four walls must conserve mass to
machine precision: the mirror reflection makes the boundary mass flux
identically zero, and interior fluxes cancel pairwise in the FV update.
Run directly: python tests/test_mass.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.config import (BathymetryConfig, BoundaryGroup, CaseConfig,
                           InitialConfig, MeshConfig, OutputConfig)  # noqa: E402
from solver.mesh import rect_mesh  # noqa: E402
from solver.solver import ShallowWaterSolver  # noqa: E402


def make_cfg(order: int = 1, time_order: int = 1) -> CaseConfig:
    return CaseConfig(
        name="mass", analytic=None, t_end=0.5,
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=16.0,
                        ymin=0.0, ymax=0.45, dx=0.1),
        initial=InitialConfig(type="dam_snap",
                              params={"hl": 1.0, "hr": 0.2, "x_dam": 8.0}),
        bathymetry=BathymetryConfig(type="flat", params={"z": 0.0}),
        boundary=[
            BoundaryGroup(name="wmin", select={"xmin": [0.0, 0.45]},
                          type="wall"),
            BoundaryGroup(name="wmax", select={"xmax": [0.0, 0.45]},
                          type="wall"),
            BoundaryGroup(name="wmin_y", select={"ymin": [0.0, 16.0]},
                          type="wall"),
            BoundaryGroup(name="wmax_y", select={"ymax": [0.0, 16.0]},
                          type="wall"),
        ],
        cfl=0.8, order=order, limiter="barth", time_order=time_order,
        eps_dry=1.0e-6, output=OutputConfig(dir="output/_mass"))


def run_mass(order: int, time_order: int) -> tuple[float, float, float]:
    cfg = make_cfg(order, time_order)
    mesh = rect_mesh(cfg.mesh.xmin, cfg.mesh.xmax,
                     cfg.mesh.ymin, cfg.mesh.ymax,
                     cfg.mesh.dx, cfg.mesh.diagonal)
    s = ShallowWaterSolver(cfg, mesh)
    m0 = s.mass()
    while s.t < cfg.t_end - 1.0e-12:
        dt = s.compute_dt(cfg.t_end - s.t)
        s.step(dt)
    m1 = s.mass()
    hmin = float(s.w.numpy()[:, 0].min())
    return m0, m1, hmin


def main() -> int:
    for order, time_order in [(1, 1), (2, 2)]:
        m0, m1, hmin = run_mass(order, time_order)
        drift = abs(m1 - m0) / abs(m0)
        assert drift < 1.0e-12, \
            f"order {order}: mass drift {drift:.3e} ({m0:.12g} -> {m1:.12g})"
        assert hmin > -1.0e-12, f"order {order}: negative depth {hmin}"
        print(f"order {order}: mass {m0:.12g} -> {m1:.12g} "
              f"(drift {drift:.2e}), min h = {hmin:.3e}")
    print("PASS: closed-domain mass conservation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
