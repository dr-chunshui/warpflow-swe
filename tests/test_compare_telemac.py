"""Batch-1 capability tests: P1 final-field sampling + time-varying
inflow_q boundary.

Run directly:  python tests/test_compare_telemac.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.compare_telemac import p1_sample  # noqa: E402
from solver.config import load_config  # noqa: E402
from solver.mesh import rect_mesh  # noqa: E402


def main() -> int:
    # --- p1_sample: exact on a planar field, linear inside triangles ---
    ref_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                       [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    ref_tris = np.array([[0, 1, 2], [1, 4, 3], [3, 4, 5]])
    vals = 10.0 + 3.0 * ref_xy[:, 0] - 2.0 * ref_xy[:, 1]  # planar
    pts = np.array([[0.2, 0.2], [1.5, 0.5], [0.05, 0.9]])
    got = p1_sample(ref_xy, ref_tris, vals, pts)
    want = 10.0 + 3.0 * pts[:, 0] - 2.0 * pts[:, 1]
    assert np.allclose(got, want, atol=1e-12), (got, want)
    # outside the triangulation: nearest-node fallback (planar -> exact)
    out = p1_sample(ref_xy, ref_tris, vals, np.array([[-0.5, -0.5]]))
    assert np.allclose(out, vals[0], atol=1e-12), out
    print("p1_sample ok: planar exact, nearest fallback outside")

    # --- inflow_q time series: config parses, solver refreshes bparam ---
    mesh = rect_mesh(0.0, 4.0, 0.0, 1.0, 0.5, "alternating")
    seq = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    seq.write("# t  Q\n0.0  10.0\n100.0  30.0\n200.0  10.0\n")
    seq.close()
    y = """
case: {name: t_qseries, t_end: 1.0}
mesh: {type: rect, xmin: 0.0, xmax: 4.0, ymin: 0.0, ymax: 1.0, dx: 0.5}
physics: {g: 9.81}
initial: {type: lake_at_rest, params: {zs: 1.0}}
bathymetry: {type: flat, params: {zb: 0.0}}
boundary:
  groups:
    - {name: inlet, select: {xmin: [0.0, 1.0]}, type: inflow_q,
       value: 10.0, time: {series: "%s"}}
    - {name: outlet, select: {xmax: [0.0, 1.0]}, type: level, value: 1.0}
    - {name: walls, select: {ymin: [0.0, 4.0]}, type: wall}
    - {name: walls2, select: {ymax: [0.0, 4.0]}, type: wall}
numerics: {cfl: 0.8}
output: {dir: output/tmp_test}
""" % seq.name.replace("\\", "/")
    yf = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yf.write(y)
    yf.close()
    cfg = load_config(yf.name)
    from solver.solver import ShallowWaterSolver
    s = ShallowWaterSolver(cfg, mesh)
    # the time series lands in the per-group q_target slot; the kernel
    # sign convention keeps inflow negative (positive = out of domain)
    q1 = s._time_groups[0]["z"]
    assert np.all(q1[:1] < 0.0), q1[:1]
    s.update_time_bcs(0.0)
    assert s._q_target_host[s._time_groups[0]["gi"]] == -10.0, \
        s._q_target_host
    s.update_time_bcs(100.0)
    assert s._q_target_host[s._time_groups[0]["gi"]] == -30.0, \
        s._q_target_host
    s.update_time_bcs(50.0)
    assert s._q_target_host[s._time_groups[0]["gi"]] == -20.0, \
        s._q_target_host
    print("inflow_q series ok: Q(0)=10, Q(50)=20 (interp), Q(100)=30 m3/s")

    print("PASS: P1 sampling + time-varying inflow capability verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
