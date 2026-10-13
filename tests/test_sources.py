"""Unit tests for the source-term kernels (wind stress, Strickler friction)
and the harmonic time-dependent level boundary.  Run directly:
python tests/test_sources.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import kernels as K


def test_friction_riccati() -> None:
    """Semi-implicit Strickler friction is the exact flow map of the scalar
    ODE du/dt = -alpha u|u| (Riccati): u(T) = u0 / (1 + alpha u0 T)."""
    wp.init()
    h0, u0, v0 = 2.0, 0.7, -0.3
    kf = 30.0
    dt, n_steps = 0.1, 100
    w = wp.array(np.array([[h0, h0 * u0, h0 * v0]], dtype=np.float64),
                 dtype=wp.vec3d)
    ks = wp.array(np.array([kf], dtype=np.float64), dtype=wp.float64)
    for _ in range(n_steps):
        wp.launch(K.k_friction, dim=1, inputs=[w, ks, wp.float64(dt)])
    w_np = w.numpy()[0]
    alpha = 9.81 / (kf * kf * h0 ** (4.0 / 3.0))
    decay = 1.0 + alpha * np.hypot(u0, v0) * (dt * n_steps)
    expect_u = u0 / decay
    expect_v = v0 / decay
    assert abs(w_np[0] - h0) < 1.0e-14, f"friction moved h: {w_np[0]}"
    assert abs(w_np[1] / h0 - expect_u) < 1.0e-13, \
        f"u: {w_np[1] / h0} vs {expect_u}"
    assert abs(w_np[2] / h0 - expect_v) < 1.0e-13, \
        f"v: {w_np[2] / h0} vs {expect_v}"
    print(f"friction: u(T) = {w_np[1] / h0:.12f}, exact = {expect_u:.12f}")


def test_friction_dry_and_off() -> None:
    """Dry cells and kf = 0 cells are left untouched."""
    wp.init()
    w = wp.array(np.array([[0.0, 5.0, 5.0], [1.0, 1.0, 0.0]], dtype=np.float64),
                 dtype=wp.vec3d)
    ks = wp.array(np.array([0.0, 0.0], dtype=np.float64), dtype=wp.float64)
    wp.launch(K.k_friction, dim=2, inputs=[w, ks, wp.float64(1.0)])
    w_np = w.numpy()
    assert np.all(w_np == np.array([[0.0, 5.0, 5.0], [1.0, 1.0, 0.0]]))
    print("friction: dry / off cells untouched")


def test_wind_source() -> None:
    """k_wind adds area * c_w |W| W to the momentum rows (conservative
    d(hu)/dt rate -- NO h division; stress per unit area)."""
    wp.init()
    h = 2.0
    wu, wv, coeff = 5.0, 0.0, 1.2615e-3
    area = np.array([4.0, 9.0], dtype=np.float64)
    w = wp.array(np.array([[h, 0.0, 0.0], [1.0e-9, 0.0, 0.0]],
                          dtype=np.float64), dtype=wp.vec3d)
    rhs = wp.zeros(2, dtype=wp.vec3d)
    wp.launch(K.k_wind, dim=2, inputs=[
        w, wp.array(area, dtype=wp.float64), rhs,
        wp.float64(wu), wp.float64(wv), wp.float64(coeff)])
    r = rhs.numpy()
    sx = coeff * np.hypot(wu, wv) * wu  # tau/rho [m2/s2]
    assert r[0][0] == 0.0 and r[1][0] == 0.0, "wind touched mass"
    assert abs(r[0][1] - area[0] * sx) < 1.0e-15 * area[0]
    assert r[1][1] == 0.0, "wind forced a dry cell"
    print(f"wind: tau/rho = {sx:.6e} m2/s2, rhs[0] = {r[0][1]:.6e}")


def test_harmonic_bc() -> None:
    """Harmonic level boundary evaluates mean + amp sin(2 pi t/T + phase)."""
    from solver.config import (BoundaryGroup, CaseConfig, InitialConfig,
                               MeshConfig, OutputConfig)
    from solver.mesh import rect_mesh
    from solver.solver import ShallowWaterSolver

    cfg = CaseConfig(
        name="harmonic_bc", analytic="", t_end=1.0e9,
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=2.0,
                        ymin=0.0, ymax=0.4, dx=0.2),
        initial=InitialConfig(type="lake_at_rest", params={"zs": 10.0}),
        bathymetry=BathymetryFlat(),
        boundary=[
            BoundaryGroup(name="inlet", select={"xmin": [0.0, 0.4]},
                          type="level",
                          time={"mean": 10.0, "amp": 0.05, "period": 0.5,
                                "phase": 0.3}),
            BoundaryGroup(name="outlet", select={"xmax": [0.0, 0.4]},
                          type="level", value=10.0),
            BoundaryGroup(name="w_ymin", select={"ymin": [0.0, 2.0]},
                          type="wall"),
            BoundaryGroup(name="w_ymax", select={"ymax": [0.0, 2.0]},
                          type="wall"),
        ],
        cfl=0.8, order=1, limiter="barth", time_order=1, eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_test_harmonic"),
    )
    mesh = rect_mesh(0.0, 2.0, 0.0, 0.4, dx=0.2)
    s = ShallowWaterSolver(cfg, mesh)
    mean, amp, period, phase = 10.0, 0.05, 0.5, 0.3
    inlet_edges = s._time_groups[0][0]
    for t in (0.0, 0.125, 0.25, 0.437):
        s.update_time_bcs(t)
        got = s._bparam_host[inlet_edges]
        expect = mean + amp * np.sin(2.0 * np.pi * t / period + phase)
        assert np.allclose(got, expect, rtol=0.0, atol=1.0e-15), \
            f"t={t}: {got[:2]} vs {expect}"
    # the constant level group must be untouched by the refresh
    outlet_edges = s.mesh.boundary_edges[np.nonzero(s._group_of == 1)[0]]
    assert np.allclose(s._bparam_host[outlet_edges], 10.0)
    print("harmonic bc: zs(t) evaluated exactly at 4 sample times")


def BathymetryFlat():
    from solver.config import BathymetryConfig
    return BathymetryConfig(type="flat", params={"z": 0.0})


def main() -> int:
    test_friction_riccati()
    test_friction_dry_and_off()
    test_wind_source()
    test_harmonic_bc()
    print("PASS: source-term kernels + harmonic boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
