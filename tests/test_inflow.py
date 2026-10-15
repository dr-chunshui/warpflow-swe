"""Unit tests for the DEBIMP inflow distribution (k_qw_sum +
k_inflow_alpha): a prescribed total discharge must be re-spread every
stage as the uniform normal velocity alpha = Q / sum(H_cell * L),
mirroring bord.f PROVEL=0 + cdl_fv.f:63 rescale.  Run directly:
python tests/test_inflow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import kernels as K
from solver.config import (BathymetryConfig, BoundaryGroup, CaseConfig,
                           InitialConfig, MeshConfig, OutputConfig)
from solver.mesh import rect_mesh
from solver.solver import ShallowWaterSolver


def make_solver(depth: float) -> ShallowWaterSolver:
    cfg = CaseConfig(
        name="test_inflow", analytic="", t_end=1.0,
        mesh=MeshConfig(),
        initial=InitialConfig(type="lake_at_rest", params={"zs": depth}),
        bathymetry=BathymetryConfig(type="flat", params={"z": 0.0}),
        boundary=[
            BoundaryGroup(name="inlet", select={"xmin": [0.0, 0.4]},
                          type="inflow_q", value=0.5),
            BoundaryGroup(name="outlet", select={"xmax": [0.0, 0.4]},
                          type="level", value=depth),
            BoundaryGroup(name="w_ymin", select={"ymin": [0.0, 2.0]},
                          type="wall"),
            BoundaryGroup(name="w_ymax", select={"ymax": [0.0, 2.0]},
                          type="wall"),
        ],
        cfl=0.8, order=1, limiter="barth", time_order=1, eps_dry=1.0e-6,
        output=OutputConfig(dir="output/_test_inflow"),
    )
    mesh = rect_mesh(0.0, 2.0, 0.0, 0.4, dx=0.2)
    return ShallowWaterSolver(cfg, mesh)


def test_alpha_uniform_depth() -> None:
    """Over a uniformly wet section alpha = Q / (H * L) exactly; with the
    kernel sign convention the inflow velocity is negative."""
    wp.init()
    depth, q_total = 2.0, 0.5
    s = make_solver(depth)
    # emulate the per-stage refresh exactly as _evaluate_rhs does
    wp.launch(K.k_zero_f64, dim=s.n_groups, inputs=[s.qw])
    wp.launch(K.k_qw_sum, dim=s.mesh.n_edges, inputs=[
        s.w, s.edge_cells, s.bedge_group, s.btype, s.edge_len, s.qw])
    wp.launch(K.k_inflow_alpha_ctrl, dim=s.mesh.n_edges, inputs=[
        s.bedge_group, s.btype, s.q_target, s.qw, s.q_deliv, s.bparam])
    inlet_edges = s.mesh.boundary_edges[np.nonzero(s._group_of == 0)[0]]
    alpha = s.bparam.numpy()[inlet_edges]
    wet_len = s.edge_len.numpy()[inlet_edges].sum()
    expect = -q_total / (depth * wet_len)
    assert np.allclose(alpha, expect, rtol=0.0, atol=1.0e-14), \
        f"alpha {alpha[:2]} vs {expect}"
    assert abs(float(s.qw.numpy()[0]) - depth * wet_len) < 1.0e-13
    # non-inflow edges keep bparam = 0
    wall_edges = s.mesh.boundary_edges[np.nonzero(s._group_of == 2)[0]]
    assert np.all(s.bparam.numpy()[wall_edges] == 0.0)
    print(f"debimp: alpha = {alpha[0]:.6f} m/s = Q/(H*L) exactly")


def test_alpha_concentrates_on_wet_part() -> None:
    """When part of the inlet section is dry, alpha grows so the wet part
    carries the full prescribed Q (TELEMAC's Q2 rescale behaviour) --
    the deep edges' velocity must exceed Q/(H_wet_all*L)."""
    wp.init()
    s = make_solver(2.0)
    # direct state surgery: dry the lower half of the inlet column
    # (cells whose centroid sits within one dx of the left boundary)
    w_np = s.w.numpy()
    cy = s.mesh.cell_centroids[:, 1]
    cx = s.mesh.cell_centroids[:, 0]
    dx = 0.2
    dry = (cx < dx) & (cy < 0.2)
    assert dry.any(), "test setup: no inlet cells selected"
    w_np[dry, 0] = 0.0
    w_np[dry, 1:] = 0.0
    s.w = wp.array(w_np, dtype=wp.vec3d)
    wp.launch(K.k_zero_f64, dim=s.n_groups, inputs=[s.qw])
    wp.launch(K.k_qw_sum, dim=s.mesh.n_edges, inputs=[
        s.w, s.edge_cells, s.bedge_group, s.btype, s.edge_len, s.qw])
    wp.launch(K.k_inflow_alpha_ctrl, dim=s.mesh.n_edges, inputs=[
        s.bedge_group, s.btype, s.q_target, s.qw, s.q_deliv, s.bparam])
    inlet_edges = s.mesh.boundary_edges[np.nonzero(s._group_of == 0)[0]]
    alpha = s.bparam.numpy()[inlet_edges]
    qw = float(s.qw.numpy()[0])
    expect = -0.5 / qw
    assert np.allclose(alpha, expect, rtol=0.0, atol=1.0e-12)
    # qw must only count the wet half of the section
    assert abs(qw - 2.0 * 0.2) < 1.0e-12, f"qw = {qw}"
    print(f"debimp: dry part of the section excluded, "
          f"alpha = {alpha[0]:.4f} (wet-only qw = {qw:.3f})")


def test_feedback_rescale() -> None:
    """Integral inflow control: with an 8% over-delivery history alpha
    shrinks by the damped factor sqrt(1/1.08); once delivery matches
    the target exactly (q_deliv = q_target) alpha is left unchanged --
    zero steady-state error."""
    wp.init()
    s = make_solver(2.0)
    wp.launch(K.k_zero_f64, dim=s.n_groups, inputs=[s.qw])
    wp.launch(K.k_qw_sum, dim=s.mesh.n_edges, inputs=[
        s.w, s.edge_cells, s.bedge_group, s.btype, s.edge_len, s.qw])
    wp.launch(K.k_inflow_alpha_ctrl, dim=s.mesh.n_edges, inputs=[
        s.bedge_group, s.btype, s.q_target, s.qw, s.q_deliv, s.bparam])
    inlet_edges = s.mesh.boundary_edges[np.nonzero(s._group_of == 0)[0]]
    base = s.bparam.numpy()[inlet_edges].copy()
    # previous stage over-delivered by 8%: q_deliv = 1.08 * |q_target|
    # (same sign convention: both negative for an inflow group)
    over = wp.array(np.array([1.08 * float(s._q_target_host[0]), 0.0, 0.0],
                             dtype=np.float64), dtype=wp.float64)
    wp.launch(K.k_inflow_alpha_ctrl, dim=s.mesh.n_edges, inputs=[
        s.bedge_group, s.btype, s.q_target, s.qw, over, s.bparam])
    got = s.bparam.numpy()[inlet_edges]
    assert np.allclose(got, base / np.sqrt(1.08), rtol=1.0e-14), \
        f"feedback: {got[:2]} vs {base[:2] / np.sqrt(1.08)}"
    # exact delivery -> no further change
    on_target = wp.array(np.array([float(s._q_target_host[0]), 0.0, 0.0],
                                  dtype=np.float64), dtype=wp.float64)
    wp.launch(K.k_inflow_alpha_ctrl, dim=s.mesh.n_edges, inputs=[
        s.bedge_group, s.btype, s.q_target, s.qw, on_target, s.bparam])
    assert np.allclose(s.bparam.numpy()[inlet_edges], got, rtol=0.0,
                       atol=0.0)
    print("feedback: 8% over-delivery damps alpha by sqrt(1/1.08); "
          "on-target delivery leaves alpha unchanged")


def main() -> int:
    test_alpha_uniform_depth()
    test_alpha_concentrates_on_wet_part()
    test_feedback_rescale()
    print("PASS: DEBIMP inflow distribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
