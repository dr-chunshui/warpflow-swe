"""ShallowWaterSolver: GPU time loop orchestrating the Warp kernels.

Scheme (mirrors TELEMAC-2D finite volume):
- HLLC flux + Audusse hydrostatic reconstruction + geometric sources
- explicit Euler (time_order=1) or SSP-RK2 (time_order=2)
- optional MUSCL reconstruction (order=2): Green-Gauss gradient + Barth
- dt = CFL * min(DTHAUT / max(0.01, c + |u|))  (caldt.f)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

from . import kernels as K
from .config import BC_TYPES, CaseConfig
from .io_csv import MassWriter, MonitorWriter
from .io_vtk import write_pvd, write_vtu
from .mesh import TriMesh, classify_boundaries
from .state import bathymetry_at, build_bathymetry, build_initial, nearest_cell


class ShallowWaterSolver:
    def __init__(self, cfg: CaseConfig, mesh: TriMesh):
        self.cfg = cfg
        self.mesh = mesh
        wp.init()
        self.dev = wp.get_device()

        n = mesh.n_cells
        self.n_cells = n

        # ---- geometry uploads -------------------------------------------
        self.w = wp.array(
            np.ascontiguousarray(self._initial_state(mesh), dtype=np.float64),
            dtype=wp.vec3d)
        self.wb = wp.zeros(n, dtype=wp.vec3d)  # stage / alternate buffer
        self.zb = wp.array(np.ascontiguousarray(build_bathymetry(cfg, mesh)),
                           dtype=wp.float64)
        self.rhs = wp.zeros(n, dtype=wp.vec3d)
        self.centroid = wp.array(np.ascontiguousarray(mesh.cell_centroids),
                                 dtype=wp.vec2d)
        self.area = wp.array(np.ascontiguousarray(mesh.cell_area),
                             dtype=wp.float64)
        self.cell_edges = wp.array(np.ascontiguousarray(mesh.cell_edges),
                                   dtype=wp.vec3i)
        self.edge_cells = wp.array(np.ascontiguousarray(mesh.edge_cells),
                                   dtype=wp.vec2i)
        self.edge_normal = wp.array(np.ascontiguousarray(mesh.edge_normal),
                                    dtype=wp.vec2d)
        self.edge_len = wp.array(np.ascontiguousarray(mesh.edge_len),
                                 dtype=wp.float64)
        self.edge_midpoint = wp.array(np.ascontiguousarray(mesh.edge_midpoint),
                                      dtype=wp.vec2d)
        self.dt_cell = wp.zeros(n, dtype=wp.float64)
        self.mass_arr = wp.zeros(1, dtype=wp.float64)

        ne = mesh.n_edges
        self.slope_w = wp.zeros(n, dtype=wp.vec2d)
        self.slope_u = wp.zeros(n, dtype=wp.vec2d)
        self.slope_v = wp.zeros(n, dtype=wp.vec2d)
        self.slope_z = wp.zeros(n, dtype=wp.vec2d)
        self.grad_w = wp.zeros(n, dtype=wp.vec2d)
        self.grad_u = wp.zeros(n, dtype=wp.vec2d)
        self.grad_v = wp.zeros(n, dtype=wp.vec2d)
        self.grad_z = wp.zeros(n, dtype=wp.vec2d)
        self.w2 = wp.zeros(n, dtype=wp.vec3d)  # RK2 output buffer

        # ---- boundary prep ----------------------------------------------
        groups = cfg.boundary
        self.n_groups = len(groups)
        group_of = classify_boundaries(mesh, [
            {"name": g.name, "select": g.select, "type": g.type} for g in groups
        ])
        bedge_group = np.full(ne, -1, dtype=np.int32)
        bparam = np.zeros(ne, dtype=np.float64)
        for gi, grp in enumerate(groups):
            sel_edges = mesh.boundary_edges[group_of == gi]
            bedge_group[sel_edges] = gi
            if grp.type == "level":
                bparam[sel_edges] = grp.value
            elif grp.type == "inflow_q":
                # Uniform unit-width discharge q = Q / total edge length:
                # sum(q * L_e) = Q exactly.  Kernel convention: bparam is
                # signed along the OUTWARD edge normal, so a positive cfg
                # value (inflow) is stored negative (value < 0 = outflow).
                sign = -1.0 if grp.value >= 0.0 else 1.0
                bparam[sel_edges] = sign * abs(grp.value) / \
                    np.sum(mesh.edge_len[sel_edges])
        self.bedge_group = wp.array(bedge_group, dtype=wp.int32)
        self.btype = wp.array(
            np.array([BC_TYPES[g.type] for g in groups], dtype=np.int32),
            dtype=wp.int32)
        self.bparam = wp.array(bparam, dtype=wp.float64)
        self.bflux = wp.zeros(self.n_groups, dtype=wp.float64)
        self._group_of = group_of

        # ---- monitors ----------------------------------------------------
        self.monitor_cells = [
            nearest_cell(mesh, float(m["x"]), float(m["y"]))
            for m in cfg.output.monitors
        ]

        self.t = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    def _initial_state(self, mesh: TriMesh) -> np.ndarray:
        zb = build_bathymetry(self.cfg, mesh)
        return build_initial(self.cfg, mesh, zb)

    # ------------------------------------------------------------------
    def compute_dt(self, t_remaining: float) -> float:
        wp.launch(K.k_dt, dim=self.n_cells, inputs=[
            self.w, self.area, self.cell_edges, self.edge_len, self.dt_cell])
        dt = float(self.dt_cell.numpy().min()) * self.cfg.cfl
        if dt < 1.0e-9:
            raise RuntimeError(
                f"timestep collapsed to {dt:.3e} s at t={self.t:.3f} "
                f"after {self.step_count} steps: solution blow-up")
        return min(dt, t_remaining)

    # ------------------------------------------------------------------
    def _evaluate_rhs(self, w_state: wp.array) -> None:
        """rhs = -div F(w_state) + S; resets rhs and bflux first."""
        wp.launch(K.k_zero_vec3, dim=self.n_cells, inputs=[self.rhs])
        wp.launch(K.k_zero_f64, dim=self.n_groups, inputs=[self.bflux])
        if self.cfg.order == 2:
            wp.launch(K.k_green_gauss, dim=self.n_cells, inputs=[
                w_state, self.zb, self.area, self.cell_edges, self.edge_cells,
                self.edge_normal, self.edge_len,
                self.grad_w, self.grad_u, self.grad_v, self.grad_z])
            if self.cfg.limiter == "barth":
                wp.launch(K.k_limit_barth, dim=self.n_cells, inputs=[
                    w_state, self.zb, self.grad_w, self.grad_u, self.grad_v,
                    self.grad_z, self.cell_edges, self.edge_cells,
                    self.slope_w, self.slope_u, self.slope_v, self.slope_z])
            else:
                wp.launch(K.k_slope_none, dim=self.n_cells, inputs=[
                    self.grad_w, self.grad_u, self.grad_v, self.grad_z,
                    self.slope_w, self.slope_u, self.slope_v, self.slope_z])
        wp.launch(K.k_flux, dim=self.mesh.n_edges, inputs=[
            w_state, self.zb,
            self.slope_w, self.slope_u, self.slope_v, self.slope_z,
            self.centroid, self.edge_midpoint, self.edge_cells,
            self.edge_normal, self.edge_len,
            self.bedge_group, self.btype, self.bparam,
            self.rhs, self.bflux,
            wp.int32(self.cfg.order),
        ])

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the state by one explicit step of size dt."""
        if self.cfg.time_order == 1:
            self._evaluate_rhs(self.w)
            wp.launch(K.k_update, dim=self.n_cells, inputs=[
                self.w, self.rhs, self.area, wp.float64(dt), self.wb])
            self.w, self.wb = self.wb, self.w
        else:
            # SSP-RK2: w1 = w + dt L(w); w_new = 0.5 (w + w1 + dt L(w1))
            self._evaluate_rhs(self.w)
            wp.launch(K.k_update, dim=self.n_cells, inputs=[
                self.w, self.rhs, self.area, wp.float64(dt), self.w2])
            self._evaluate_rhs(self.w2)
            wp.launch(K.k_update, dim=self.n_cells, inputs=[
                self.w2, self.rhs, self.area, wp.float64(dt), self.wb])
            wp.launch(K.k_comb, dim=self.n_cells, inputs=[
                self.w, self.wb, self.w2])
            self.w, self.w2 = self.w2, self.w
        self.t += dt
        self.step_count += 1

    # ------------------------------------------------------------------
    def mass(self) -> float:
        wp.launch(K.k_zero_f64, dim=1, inputs=[self.mass_arr])
        wp.launch(K.k_mass, dim=self.n_cells, inputs=[
            self.w, self.area, self.mass_arr])
        return float(self.mass_arr.numpy()[0])

    def snapshot(self) -> dict:
        """Host copies of the state for I/O (h, u, v, zb, zs per cell)."""
        w_np = self.w.numpy()
        zb_np = self.zb.numpy()
        h = np.maximum(w_np[:, 0], 0.0)
        wet = h > self.cfg.eps_dry
        u = np.where(wet, w_np[:, 1] / np.where(wet, h, 1.0), 0.0)
        v = np.where(wet, w_np[:, 2] / np.where(wet, h, 1.0), 0.0)
        return {"h": h, "u": u, "v": v, "zb": zb_np, "zs": zb_np + h}

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        vtk_on = cfg.output.vtk_interval > 0.0
        csv_on = cfg.output.csv_interval > 0.0
        next_vtk = 0.0
        next_csv = 0.0
        frames = []

        mon = MonitorWriter(out_dir / "monitors.csv",
                            [m.get("name", f"m{i}")
                             for i, m in enumerate(cfg.output.monitors)])
        mw = MassWriter(out_dir / "history.csv")

        def dump_frame() -> None:
            snap = self.snapshot()
            fname = f"frame_{len(frames):05d}.vtu"
            write_vtu(out_dir / fname, self.mesh.nodes, self.mesh.tris,
                      bathymetry_at(cfg,
                                    self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]),
                      {"h": snap["h"], "u": snap["u"], "v": snap["v"],
                       "zb": snap["zb"], "zs": snap["zs"]})
            frames.append((self.t, fname))

        def log_row() -> None:
            snap = self.snapshot()
            vals = [(snap["h"][c], snap["u"][c], snap["v"][c])
                    for c in self.monitor_cells]
            mon.write(self.t, vals)
            mw.write(self.t, self.mass(), self._last_dt if hasattr(self, "_last_dt") else 0.0,
                     float(snap["h"].min()), float(snap["h"].max()))

        if vtk_on:
            dump_frame()
        if csv_on:
            log_row()

        while self.t < cfg.t_end - 1.0e-12:
            next_event = cfg.t_end
            if vtk_on:
                next_event = min(next_event, next_vtk + cfg.output.vtk_interval)
            if csv_on:
                next_event = min(next_event, next_csv + cfg.output.csv_interval)
            dt = self.compute_dt(next_event - self.t)
            self._last_dt = dt
            self.step(dt)

            if vtk_on and self.t >= next_vtk + cfg.output.vtk_interval - 1.0e-9:
                next_vtk += cfg.output.vtk_interval
                dump_frame()
            if csv_on and self.t >= next_csv + cfg.output.csv_interval - 1.0e-9:
                next_csv += cfg.output.csv_interval
                log_row()

        if vtk_on:
            dump_frame()
        if csv_on:
            log_row()
        if vtk_on:
            write_pvd(out_dir / "animation.pvd", frames)

        mon.close()
        mw.close()
        print(f"done: {self.step_count} steps, t = {self.t:.4f} s, "
              f"mass = {self.mass():.12g} m^3 -> {out_dir}")
