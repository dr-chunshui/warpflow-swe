"""Mesh-convergence study for the Ritter dam break (order 1).

Runs the case at dx = 0.1 / 0.05 / 0.025 m up to t = 1 s and reports the
band-averaged L1/L2 errors of h against the exact solution.  Run directly:
python tests/convergence_ritter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import analytic
from solver.analysis import band_cells, profile_errors
from solver.config import load_config
from solver.mesh import rect_mesh
from solver.solver import ShallowWaterSolver

T_END = 1.0


def run_dx(dx: float) -> dict:
    cfg = load_config(Path(__file__).resolve().parents[1]
                      / "test_cases" / "dambreak" / "ritter.yaml")
    cfg.t_end = T_END
    cfg.mesh.dx = dx
    cfg.output.band = 0.45 * dx  # keep the centerline row at every dx
    cfg.output.vtk_interval = 0.0
    cfg.output.csv_interval = 0.0

    mesh = rect_mesh(cfg.mesh.xmin, cfg.mesh.xmax,
                     cfg.mesh.ymin, cfg.mesh.ymax, dx, cfg.mesh.diagonal)
    s = ShallowWaterSolver(cfg, mesh)
    while s.t < T_END - 1.0e-12:
        s.step(s.compute_dt(T_END - s.t))

    snap = s.snapshot()
    idx = band_cells(mesh, cfg.output.centerline_y, cfg.output.band)
    c = mesh.cell_centroids
    h_ana, u_ana, _ = analytic.evaluate("ritter", cfg, s.t,
                                        c[:, 0], c[:, 1])
    eh = profile_errors(mesh, snap["h"], h_ana, idx)
    eu = profile_errors(mesh, snap["u"], u_ana, idx)
    print(f"dx = {dx:6.3f}  cells = {mesh.n_cells:6d}  steps = {s.step_count:5d}"
          f"  L1(h) = {eh['l1']:.4e}  L2(h) = {eh['l2']:.4e}"
          f"  L1(u) = {eu['l1']:.4e}")
    return eh


def main() -> int:
    errs = [run_dx(dx) for dx in (0.1, 0.05, 0.025)]
    print("refinement ratios L1(h):")
    ok = True
    for a, b in zip(errs[:-1], errs[1:]):
        r = a["l1"] / b["l1"]
        print(f"  {r:.3f}x")
        ok = ok and r > 1.3
    print("PASS: L1 decreases under refinement" if ok
          else "WARN: convergence rate below expectation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
