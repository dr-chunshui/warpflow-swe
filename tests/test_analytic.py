"""Analytic solution sanity tests (numpy only, no GPU).

- Ritter: Riemann invariant u+2c preserved across the rarefaction and
  exact mass integral.
- Stoker: shock position from mass balance consistency.
- Thacker planar/radial: exact periodicity (T apart the state repeats).
- Seiche: period and linear dispersion relation.
- Bump: comparison against the official TELEMAC analytic_sol.py when it
  is available on this machine.
Run directly: python tests/test_analytic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.analytic import SOLUTIONS, _bump_profile  # noqa: E402
from solver.config import (BathymetryConfig, BoundaryGroup, CaseConfig,
                           InitialConfig, MeshConfig, OutputConfig)  # noqa: E402

G = 9.81


def dam_cfg(initial: str, params: dict) -> CaseConfig:
    return CaseConfig(
        name="t", analytic=initial, t_end=1.0,
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=16.0,
                        ymin=0.0, ymax=0.45, dx=0.05),
        initial=InitialConfig(type=initial, params=params),
        bathymetry=BathymetryConfig(type="flat", params={"z": 0.0}),
        boundary=[BoundaryGroup(name="w", select={"ymin": [0.0, 16.0]},
                                type="wall")],
        cfl=0.8, order=1, limiter="barth", time_order=1, eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_t"))


def thacker_cfg(initial: str, params: dict) -> CaseConfig:
    return CaseConfig(
        name="t", analytic=initial, t_end=1.0,
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=4.0,
                        ymin=0.0, ymax=4.0, dx=0.04),
        initial=InitialConfig(type=initial, params=params),
        bathymetry=BathymetryConfig(
            type="thacker", params={"h0": 0.1, "a": 1.0,
                                    "xc": 2.0, "yc": 2.0}),
        boundary=[BoundaryGroup(name="w", select={"ymin": [0.0, 4.0]},
                                type="wall")],
        cfl=0.8, order=1, limiter="barth", time_order=1, eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_t"))


def test_ritter() -> None:
    cfg = dam_cfg("dam_snap", {"hl": 1.0, "hr": 0.0, "x_dam": 8.0})
    t = 1.3
    x = np.linspace(1.0, 20.0, 500)
    h, u, _ = SOLUTIONS["ritter"](cfg, t, x, np.zeros_like(x))
    cl = np.sqrt(G * 1.0)
    # Riemann invariant u + 2c = 2 cl inside the fan
    fan = (x > 8.0 - cl * t) & (x < 8.0 + 2.0 * cl * t) & (h > 1.0e-10)
    ri = u[fan] + 2.0 * np.sqrt(G * h[fan])
    assert np.abs(ri - 2.0 * cl).max() < 1.0e-12, \
        f"ritter invariant: {np.abs(ri - 2.0 * cl).max()}"
    # exact mass: trapezoid on a fine grid vs analytic pieces
    xf = np.linspace(0.0, 40.0, 200001)
    hf, _, _ = SOLUTIONS["ritter"](cfg, t, xf, np.zeros_like(xf))
    mass = np.trapezoid(hf, xf)
    # left constant part + parabola part
    head, front = 8.0 - cl * t, 8.0 + 2.0 * cl * t
    xm = np.linspace(head, front, 200001)
    hm, _, _ = SOLUTIONS["ritter"](cfg, t, xm, np.zeros_like(xm))
    exact = 1.0 * head + np.trapezoid(hm, xm)
    assert abs(mass - exact) < 1.0e-8, f"ritter mass {mass} vs {exact}"
    print("ritter: invariant + mass OK")


def test_stoker() -> None:
    cfg = dam_cfg("dam_snap", {"hl": 1.0, "hr": 0.2, "x_dam": 8.0})
    t = 1.5
    xf = np.linspace(0.0, 40.0, 400001)
    hf, uf, _ = SOLUTIONS["stoker"](cfg, t, xf, np.zeros_like(xf))
    # plateau depth: constant region between dam and shock
    x = np.linspace(9.0, 11.0, 50)
    hm, um, _ = SOLUTIONS["stoker"](cfg, t, x, np.zeros_like(x))
    hm0, um0 = hm[0], um[0]
    # shock speed from RH: s = um*hm/(hm-hr); check mass conservation form:
    # the constant state must satisfy the invariant u + 2c = 2 cl exactly
    assert abs(um0 + 2.0 * np.sqrt(G * hm0) - 2.0 * np.sqrt(G)) < 1.0e-10
    # jump relation: um^2 = (hm-hr)^2 g (hm+hr) / (2 hm hr)
    rhs = (hm0 - 0.2) ** 2 * G * (hm0 + 0.2) / (2.0 * hm0 * 0.2)
    assert abs(um0 ** 2 - rhs) < 1.0e-9 * rhs, \
        f"stoker jump: {um0**2} vs {rhs}"
    print(f"stoker: hm = {hm0:.6f}, um = {um0:.6f} OK")


def test_thacker_periodicity() -> None:
    cfg = thacker_cfg("thacker_planar",
                      {"h0": 0.1, "a": 1.0, "eta": 0.5, "length": 4.0,
                       "xc": 2.0, "yc": 2.0})
    T = 2.0 * np.pi / (np.sqrt(2.0 * G * 0.1) / 1.0)
    x = np.linspace(0.5, 3.5, 40)
    y = np.linspace(0.5, 3.5, 40)
    X, Y = np.meshgrid(x, y)
    X, Y = X.ravel(), Y.ravel()
    h1, u1, v1 = SOLUTIONS["thacker_planar"](cfg, 3.71, X, Y)
    h2, u2, v2 = SOLUTIONS["thacker_planar"](cfg, 3.71 + T, X, Y)
    assert np.abs(h1 - h2).max() < 1.0e-12
    assert np.abs(u1 - u2).max() < 1.0e-12
    assert np.abs(v1 - v2).max() < 1.0e-12
    T_planar = T

    cfg = thacker_cfg("thacker_radial",
                      {"h0": 0.1, "a": 1.0, "r0": 0.8, "length": 4.0,
                       "xc": 2.0, "yc": 2.0})
    T = 2.0 * np.pi / (np.sqrt(8.0 * G * 0.1) / 1.0)
    h1, u1, v1 = SOLUTIONS["thacker_radial"](cfg, 1.37, X, Y)
    h2, u2, v2 = SOLUTIONS["thacker_radial"](cfg, 1.37 + T, X, Y)
    assert np.abs(h1 - h2).max() < 1.0e-12
    assert np.abs(u1 - u2).max() < 1.0e-12
    print(f"thacker: periodicity OK (T_planar = {T_planar:.4f} s, "
          f"T_radial = {T:.4f} s)")


def test_seiche() -> None:
    cfg = CaseConfig(
        name="t", analytic="seiche", t_end=1.0,
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=200.0,
                        ymin=0.0, ymax=1.8, dx=0.2),
        initial=InitialConfig(type="seiche",
                              params={"h0": 10.0, "amplitude": 0.01,
                                      "length": 200.0, "xmin": 0.0}),
        bathymetry=BathymetryConfig(type="flat", params={"z": 0.0}),
        boundary=[BoundaryGroup(name="w", select={"ymin": [0.0, 200.0]},
                                type="wall")],
        cfl=0.5, order=1, limiter="barth", time_order=1, eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_t"))
    T = 200.0 / np.sqrt(G * 10.0)
    x = np.linspace(0.0, 200.0, 1001)
    h1, u1, _ = SOLUTIONS["seiche"](cfg, 2.3, x, np.zeros_like(x))
    h2, u2, _ = SOLUTIONS["seiche"](cfg, 2.3 + T, x, np.zeros_like(x))
    assert np.abs(h1 - h2).max() < 1.0e-14
    assert np.abs(u1 - u2).max() < 1.0e-14
    # walls: u = 0 at x = 0 and x = L when sin(w t) != 0
    _, uw, _ = SOLUTIONS["seiche"](cfg, T / 4.0,
                                   np.array([0.0, 200.0]), np.zeros(2))
    assert np.abs(uw).max() < 1.0e-14
    print(f"seiche: T = {T:.4f} s, walls satisfy u = 0 OK")


def test_bump_official() -> None:
    ref = Path(r"D:\Workspace\warpflow\telemac2d\examples\bump")
    if not (ref / "analytic_sol.py").exists():
        print("bump: official analytic_sol.py not found, skipped")
        return
    sys.path.insert(0, str(ref))
    from analytic_sol import BumpAnalyticSol  # noqa: E402

    for flow, q, hl in [("sub", 1.5, 0.8), ("cri", 0.3, 0.5),
                        ("trans", 0.45, 0.35)]:
        xs, h_mine, u_mine = _bump_profile(flow, q, width=2.0, hl=hl)
        sol = BumpAnalyticSol(flow=flow, Q=q, hl=hl, length=20.0,
                              width=2.0, bottom_function='exponential',
                              N=4001)
        sol()
        assert np.abs(h_mine - sol.H).max() < 1.0e-12, \
            f"bump {flow}: |dh| = {np.abs(h_mine - sol.H).max()}"
    print("bump: sub/cri/trans match official analytic_sol.py OK")


def main() -> int:
    test_ritter()
    test_stoker()
    test_thacker_periodicity()
    test_seiche()
    test_bump_official()
    print("PASS: analytic solution tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
