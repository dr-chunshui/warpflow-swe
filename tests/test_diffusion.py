"""Unit tests for the eddy-viscosity diffusion kernel (k_diffusion_velocity,
k_dt diffusion bound) and its solver wiring.  Run directly:
python tests/test_diffusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import kernels as K
from solver.mesh import build_topology

# Square split into two triangles: shared diagonal (1,2) is the only
# interior edge.
NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
TRIS = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


def _mesh():
    return build_topology(NODES.copy(), TRIS.copy())


def _w(h, ua, va, ub, vb, hb=None):
    hb = h if hb is None else hb
    return wp.array(np.array([[h, h * ua, h * va],
                              [hb, hb * ub, hb * vb]], dtype=np.float64),
                    dtype=wp.vec3d)


def test_two_cell_closed_form() -> None:
    """Face flux nu*(ha+hb)/2*L/d*(ub-ua) lands in rhs exactly; the pair
    contribution cancels (linear momentum conservation)."""
    wp.init()
    m = _mesh()
    e = int(np.nonzero(m.edge_cells[:, 1] >= 0)[0][0])
    a, b = (int(m.edge_cells[e, 0]), int(m.edge_cells[e, 1]))
    dist = float(np.hypot(*(m.cell_centroids[b] - m.cell_centroids[a])))
    nu, ua, va, ub, vb = 2.0, 0.0, 0.0, 1.0, 0.5
    w = _w(1.0, ua, va, ub, vb)
    rhs = wp.zeros(2, dtype=wp.vec3d)
    wp.launch(K.k_diffusion_velocity, dim=2, inputs=[
        w, wp.array(m.cell_edges, dtype=wp.vec3i),
        wp.array(m.edge_cells, dtype=wp.vec2i),
        wp.array(m.edge_len, dtype=wp.float64),
        wp.array(m.cell_centroids, dtype=wp.vec2d),
        rhs, wp.float64(nu)])
    r = rhs.numpy()
    c = nu * m.edge_len[e] * 0.5 * (1.0 + 1.0) / dist
    assert abs(r[a][1] - c * (ub - ua)) < 1.0e-13, f"hu: {r[a][1]} vs {c}"
    assert abs(r[a][2] - c * (vb - va)) < 1.0e-13, f"hv: {r[a][2]}"
    assert r[a][0] == 0.0 and r[b][0] == 0.0, "diffusion touched mass"
    assert abs(r[a][1] + r[b][1]) < 1.0e-13, "momentum not conserved"
    assert abs(r[a][2] + r[b][2]) < 1.0e-13, "momentum not conserved"
    print(f"diffusion: rhs[a] = ({r[a][1]:.12f}, {r[a][2]:.12f}), "
          f"exact = ({c:.12f}, {c * 0.5:.12f})")


def test_uniform_field_is_inert() -> None:
    """A spatially uniform velocity field produces zero diffusion."""
    wp.init()
    m = _mesh()
    w = _w(2.0, 0.7, -0.3, 0.7, -0.3)
    rhs = wp.zeros(2, dtype=wp.vec3d)
    wp.launch(K.k_diffusion_velocity, dim=2, inputs=[
        w, wp.array(m.cell_edges, dtype=wp.vec3i),
        wp.array(m.edge_cells, dtype=wp.vec2i),
        wp.array(m.edge_len, dtype=wp.float64),
        wp.array(m.cell_centroids, dtype=wp.vec2d),
        rhs, wp.float64(2.0)])
    assert np.all(rhs.numpy() == 0.0), "uniform field diffused"
    print("diffusion: uniform field inert")


def test_dry_neighbour_skipped() -> None:
    """DRYNEU: an edge with a dry side carries no flux."""
    wp.init()
    m = _mesh()
    w = _w(1.0, 0.0, 0.0, 1.0, 1.0, hb=0.0)
    rhs = wp.zeros(2, dtype=wp.vec3d)
    wp.launch(K.k_diffusion_velocity, dim=2, inputs=[
        w, wp.array(m.cell_edges, dtype=wp.vec3i),
        wp.array(m.edge_cells, dtype=wp.vec2i),
        wp.array(m.edge_len, dtype=wp.float64),
        wp.array(m.cell_centroids, dtype=wp.vec2d),
        rhs, wp.float64(2.0)])
    r = rhs.numpy()
    assert np.all(r == 0.0), f"dry edge carried flux: {r}"
    print("diffusion: dry neighbour skipped")


def test_boundary_edges_carry_no_flux() -> None:
    """A single triangle has only boundary edges -> zero rhs."""
    wp.init()
    m = build_topology(NODES[:3].copy(), np.array([[0, 1, 2]], dtype=np.int32))
    w = wp.array(np.array([[1.0, 0.3, 0.0]], dtype=np.float64),
                 dtype=wp.vec3d)
    rhs = wp.zeros(1, dtype=wp.vec3d)
    wp.launch(K.k_diffusion_velocity, dim=1, inputs=[
        w, wp.array(m.cell_edges, dtype=wp.vec3i),
        wp.array(m.edge_cells, dtype=wp.vec2i),
        wp.array(m.edge_len, dtype=wp.float64),
        wp.array(m.cell_centroids, dtype=wp.vec2d),
        rhs, wp.float64(2.0)])
    assert np.all(rhs.numpy() == 0.0), "boundary edge diffused"
    print("diffusion: boundary edges are Neumann-zero")


def test_k_dt_nu_zero_bitwise() -> None:
    """nu = 0 reproduces the pure caldt.f bound exactly."""
    wp.init()
    m = _mesh()
    h, u, v = 1.5, 0.4, -0.2
    w = _w(h, u, v, u, v)
    dt_out = wp.zeros(2, dtype=wp.float64)
    wp.launch(K.k_dt, dim=2, inputs=[
        w, wp.array(m.cell_area, dtype=wp.float64),
        wp.array(m.cell_edges, dtype=wp.vec3i),
        wp.array(m.edge_len, dtype=wp.float64),
        wp.array(m.cell_centroids, dtype=wp.vec2d),
        wp.array(m.edge_cells, dtype=wp.vec2i),
        wp.float64(0.0), dt_out])
    circum = m.edge_len[m.cell_edges].sum(axis=1)
    expect = (m.cell_area / circum) / max(0.01,
                                          np.hypot(u, v) + np.sqrt(9.81 * h))
    got = dt_out.numpy()
    assert np.allclose(got, expect, rtol=0, atol=1e-15), f"{got} vs {expect}"
    print(f"k_dt: nu = 0 exact ({got[0]:.12f})")


def test_k_dt_diffusion_bound() -> None:
    """nu > 0 clamps dt to 0.5*area/sum(nu*L/d) on every wet cell."""
    wp.init()
    m = _mesh()
    nu = 100.0  # large: the diffusion bound must win on both cells
    w = _w(1.0, 0.1, 0.0, 0.1, 0.0)
    dt_out = wp.zeros(2, dtype=wp.float64)
    wp.launch(K.k_dt, dim=2, inputs=[
        w, wp.array(m.cell_area, dtype=wp.float64),
        wp.array(m.cell_edges, dtype=wp.vec3i),
        wp.array(m.edge_len, dtype=wp.float64),
        wp.array(m.cell_centroids, dtype=wp.vec2d),
        wp.array(m.edge_cells, dtype=wp.vec2i),
        wp.float64(nu), dt_out])
    got = dt_out.numpy()
    e = int(np.nonzero(m.edge_cells[:, 1] >= 0)[0][0])
    a, b = (int(m.edge_cells[e, 0]), int(m.edge_cells[e, 1]))
    dist = float(np.hypot(*(m.cell_centroids[b] - m.cell_centroids[a])))
    for c, nb in ((a, b), (b, a)):
        acc = nu * m.edge_len[e] / dist  # only the interior edge is wet-wet
        assert abs(got[c] - 0.5 * m.cell_area[c] / acc) < 1.0e-13, \
            f"cell {c}: {got[c]} vs {0.5 * m.cell_area[c] / acc}"
    print(f"k_dt: diffusion bound active ({got[0]:.6e} s at nu = {nu})")


def test_solver_smoke() -> None:
    """Solver run with viscosity: mass conserved to machine precision."""
    from solver.config import (BathymetryConfig, BoundaryGroup, CaseConfig,
                               InitialConfig, MeshConfig, OutputConfig)
    from solver.mesh import rect_mesh
    from solver.solver import ShallowWaterSolver

    cfg = CaseConfig(
        name="diffusion_smoke", analytic="", t_end=1.0e9,
        mesh=MeshConfig(type="rect", xmin=0.0, xmax=2.0,
                        ymin=0.0, ymax=0.4, dx=0.2),
        initial=InitialConfig(type="dam_snap",
                              params={"hl": 1.0, "hr": 0.1, "x_dam": 1.0}),
        bathymetry=BathymetryConfig(type="flat", params={"z": 0.0}),
        boundary=[
            BoundaryGroup(name="w_xmin", select={"xmin": [0.0, 0.4]},
                          type="wall"),
            BoundaryGroup(name="w_xmax", select={"xmax": [0.0, 0.4]},
                          type="wall"),
            BoundaryGroup(name="w_ymin", select={"ymin": [0.0, 2.0]},
                          type="wall"),
            BoundaryGroup(name="w_ymax", select={"ymax": [0.0, 2.0]},
                          type="wall"),
        ],
        cfl=0.8, order=1, limiter="barth", time_order=1, eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_test_diffusion"),
        viscosity=0.5,
    )
    mesh = rect_mesh(0.0, 2.0, 0.0, 0.4, dx=0.2)
    s = ShallowWaterSolver(cfg, mesh)
    m0 = s.mass()
    dt_target = 0.05
    t_end = 2.5
    while s.t < t_end:
        dt = s.compute_dt(min(dt_target, t_end - s.t))
        s.step(dt)
    drift = abs(s.mass() - m0) / m0
    assert drift < 1.0e-12, f"mass drift {drift:.3e}"
    print(f"diffusion solver smoke: {s.step_count} steps, "
          f"mass drift {drift:.3e}")


def main() -> int:
    test_two_cell_closed_form()
    test_uniform_field_is_inert()
    test_dry_neighbour_skipped()
    test_boundary_edges_carry_no_flux()
    test_k_dt_nu_zero_bitwise()
    test_k_dt_diffusion_bound()
    test_solver_smoke()
    print("PASS: diffusion kernel + dt bound + solver wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
