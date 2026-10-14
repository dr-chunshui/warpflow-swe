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
from .mesh import TriMesh, classify_boundaries, cli_chains, read_cli
from .state import bathymetry_at, build_bathymetry, build_initial, edge_bed, nearest_cell


def _load_level_series(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Two-column time/elevation table for a level boundary.

    Whitespace- or comma-separated; lines that do not start with a number
    (headers, comments) are skipped -- TELEMAC liquid-boundary files and
    .prn gauge records both parse.
    """
    ts, zs = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for tok in line.replace(",", " ").split(";"):
                parts = tok.split()
                if len(parts) < 2:
                    continue
                try:
                    tv = float(parts[0])
                    zv = float(parts[1])
                except ValueError:
                    continue
                ts.append(tv)
                zs.append(zv)
    if len(ts) < 2:
        raise ValueError(f"{path}: need at least two time/elevation rows")
    return np.asarray(ts, dtype=np.float64), np.asarray(zs, dtype=np.float64)


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
        chain_map = self._load_chain_map(mesh, groups)
        group_of = classify_boundaries(mesh, [
            {"name": g.name, "select": g.select, "type": g.type} for g in groups
        ], chain_map[0] if chain_map else None)
        if chain_map:
            self._chains = chain_map[1]
        else:
            self._chains = None
        bedge_group = np.full(ne, -1, dtype=np.int32)
        bparam = np.zeros(ne, dtype=np.float64)
        # bord.f semantics: a prescribed level Z is stored as the ghost
        # DEPTH max(0, Z - ZF(boundary node)) -- over the node bed, not
        # the cell bed.  On steep valley banks (culm) the cell bed can sit
        # above the prescribed level where the node bed does not; using
        # the cell bed there dries the ghost and the boundary drains the
        # domain (culm lost 63% of its volume that way).
        bedge_zb = np.zeros(ne, dtype=np.float64)
        self._time_groups = []  # harmonic/series level boundary descriptors
        for gi, grp in enumerate(groups):
            sel_edges = mesh.boundary_edges[group_of == gi]
            bedge_group[sel_edges] = gi
            bedge_zb[sel_edges] = edge_bed(cfg, mesh, sel_edges)
            if grp.type == "level":
                if grp.time:
                    ts = grp.time
                    if "series" in ts:
                        # file-driven level: two-column table, np.interp
                        t_arr, z_arr = _load_level_series(ts["series"])
                        self._time_groups.append(
                            {"edges": sel_edges, "kind": "series",
                             "t": t_arr, "z": z_arr})
                        bparam[sel_edges] = np.interp(0.0, t_arr, z_arr)
                    elif "soliton" in ts:
                        # incident solitary wave (user_sl.f convention);
                        # base is the still-water level the crest rides on
                        # (= h0 unless the wave is a pure uplift, e.g.
                        # triangular_shelf prescribes eta only)
                        sp = ts["soliton"]
                        h0 = float(sp["h0"])
                        amp = float(sp["H"])
                        t0 = float(sp.get("t0", 0.0))
                        base = float(sp.get("base", h0))
                        self._time_groups.append(
                            {"edges": sel_edges, "kind": "soliton",
                             "h0": h0, "H": amp, "t0": t0, "base": base})
                        bparam[sel_edges] = base
                    else:
                        # harmonic level: zs(t) = mean + amp*sin(2 pi t/T + ph)
                        mean = float(ts["mean"])
                        amp = float(ts["amp"])
                        period = float(ts["period"])
                        phase = float(ts.get("phase", 0.0))
                        self._time_groups.append(
                            {"edges": sel_edges, "kind": "harm",
                             "mean": mean, "amp": amp, "period": period,
                             "phase": phase})
                        bparam[sel_edges] = mean + amp * np.sin(phase)
                else:
                    bparam[sel_edges] = grp.value
            elif grp.type == "inflow_q":
                # Uniform unit-width discharge q = Q / total edge length:
                # sum(q * L_e) = Q exactly.  Kernel convention: bparam is
                # signed along the OUTWARD edge normal, so a positive cfg
                # value (inflow) is stored negative (value < 0 = outflow).
                sign = -1.0 if grp.value >= 0.0 else 1.0
                bparam[sel_edges] = sign * abs(grp.value) / \
                    np.sum(mesh.edge_len[sel_edges])
                if grp.time and "series" in grp.time:
                    # time-varying total discharge (TELEMAC liquid
                    # boundary file); stored with the kernel sign
                    t_arr, q_arr = _load_level_series(grp.time["series"])
                    q_signed = sign * np.abs(q_arr) / \
                        np.sum(mesh.edge_len[sel_edges])
                    self._time_groups.append(
                        {"edges": sel_edges, "kind": "series",
                         "t": t_arr, "z": q_signed})
                    bparam[sel_edges] = q_signed[0]
        self.bedge_group = wp.array(bedge_group, dtype=wp.int32)
        self.btype = wp.array(
            np.array([BC_TYPES[g.type] for g in groups], dtype=np.int32),
            dtype=wp.int32)
        self.bparam = wp.array(bparam, dtype=wp.float64)
        self.bedge_zb = wp.array(bedge_zb, dtype=wp.float64)
        self.bflux = wp.zeros(self.n_groups, dtype=wp.float64)
        self._bcvol = wp.zeros(self.n_groups, dtype=wp.float64)
        self._group_of = group_of
        self._bparam_host = bparam

        # ---- source terms ------------------------------------------------
        if cfg.chezy > 0.0:
            self.strickler = wp.array(
                np.full(n, float(cfg.chezy), dtype=np.float64),
                dtype=wp.float64)
            self._chezy = 1
        elif cfg.strickler > 0.0:
            self.strickler = wp.array(
                np.full(n, float(cfg.strickler), dtype=np.float64),
                dtype=wp.float64)
            self._chezy = 0
        else:
            self.strickler = None
            self._chezy = 0
        self.wind_on = cfg.wind_coeff != 0.0

        # ---- monitors ----------------------------------------------------
        self.monitor_cells = [
            nearest_cell(mesh, float(m["x"]), float(m["y"]))
            for m in cfg.output.monitors
        ]

        self.t = 0.0
        self.step_count = 0
        self._mass0 = self.mass()

    # ------------------------------------------------------------------
    def _initial_state(self, mesh: TriMesh) -> np.ndarray:
        zb = build_bathymetry(self.cfg, mesh)
        return build_initial(self.cfg, mesh, zb)

    # ------------------------------------------------------------------
    def _load_chain_map(self, mesh: TriMesh,
                        groups: list) -> tuple[np.ndarray, list] | None:
        """Boundary-chain map for .cli-selected groups, else None.

        Follows the TELEMAC convention that the .cli sits next to the
        geometry file (geo.slf <-> geo.cli)."""
        if not any("chain" in g.select for g in groups):
            return None
        cli_path = Path(self.cfg.mesh.file).with_suffix(".cli")
        cli = read_cli(str(cli_path))
        return cli_chains(mesh, cli)

    # ------------------------------------------------------------------
    def update_time_bcs(self, t: float) -> None:
        """Refresh time-dependent level boundary values at time t (no-op
        if all boundaries are steady)."""
        if not self._time_groups:
            return
        for grp in self._time_groups:
            if grp["kind"] == "series":
                z = float(np.interp(t, grp["t"], grp["z"]))
            elif grp["kind"] == "soliton":
                h0, amp, t0 = grp["h0"], grp["H"], grp["t0"]
                k = np.sqrt(0.75 * amp / h0 ** 3)
                c = np.sqrt(9.81 * (h0 + amp))
                z = grp.get("base", h0) + amp / np.cosh(k * (c * t - t0)) ** 2
            else:
                z = grp["mean"] + grp["amp"] * np.sin(
                    2.0 * np.pi * t / grp["period"] + grp["phase"])
            self._bparam_host[grp["edges"]] = z
        self.bparam = wp.array(self._bparam_host, dtype=wp.float64)

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
    def boundary_inflow(self) -> float:
        """Cumulative net inflow VOLUME [m^3] through all open boundaries
        since t = 0 (exact accounting of the discrete update)."""
        if self.n_groups == 0:
            return 0.0
        return -float(self._bcvol.numpy().sum())

    # ------------------------------------------------------------------
    def _evaluate_rhs(self, w_state: wp.array, dtw: float = 0.0) -> None:
        """rhs = -div F(w_state) + S; resets rhs and bflux first.

        dtw > 0 additionally accumulates the boundary volume update
        (bflux * dtw) into self._bcvol; callers pass dt (Euler) or
        dt/2 per RK2 stage so boundary_inflow() integrates the exact
        flux that entered the discrete update."""
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
            self.bedge_group, self.btype, self.bparam, self.bedge_zb,
            self.rhs, self.bflux,
            wp.int32(self.cfg.order),
        ])
        if self.wind_on:
            wp.launch(K.k_wind, dim=self.n_cells, inputs=[
                w_state, self.area, self.rhs,
                wp.float64(self.cfg.wind_u), wp.float64(self.cfg.wind_v),
                wp.float64(self.cfg.wind_coeff)])
        if self.cfg.coriolis != 0.0:
            wp.launch(K.k_coriolis, dim=self.n_cells, inputs=[
                w_state, self.area, self.rhs, wp.float64(self.cfg.coriolis)])
        if dtw != 0.0:
            wp.launch(K.k_acc_f64, dim=self.n_groups, inputs=[
                self.bflux, self._bcvol, wp.float64(dtw)])

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance the state by one explicit step of size dt."""
        if self.cfg.time_order == 1:
            self._evaluate_rhs(self.w, dt)
            wp.launch(K.k_update, dim=self.n_cells, inputs=[
                self.w, self.rhs, self.area, wp.float64(dt), self.wb])
            if self.strickler is not None:
                wp.launch(K.k_friction, dim=self.n_cells, inputs=[
                    self.wb, self.strickler, wp.float64(dt), self._chezy])
            self.w, self.wb = self.wb, self.w
        else:
            # SSP-RK2: w1 = w + dt L(w); w_new = 0.5 (w + w1 + dt L(w1))
            self._evaluate_rhs(self.w, 0.5 * dt)
            wp.launch(K.k_update, dim=self.n_cells, inputs=[
                self.w, self.rhs, self.area, wp.float64(dt), self.w2])
            if self.strickler is not None:
                wp.launch(K.k_friction, dim=self.n_cells, inputs=[
                    self.w2, self.strickler, wp.float64(dt), self._chezy])
            self._evaluate_rhs(self.w2, 0.5 * dt)
            wp.launch(K.k_update, dim=self.n_cells, inputs=[
                self.w2, self.rhs, self.area, wp.float64(dt), self.wb])
            if self.strickler is not None:
                wp.launch(K.k_friction, dim=self.n_cells, inputs=[
                    self.wb, self.strickler, wp.float64(dt), self._chezy])
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
        avg = getattr(self, "_avg_w", None)
        w_np = self.w.numpy() if avg is None else avg
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
            if self.mesh.bottom is not None:
                node_zb = self.mesh.bottom
            else:
                node_zb = bathymetry_at(cfg, self.mesh.nodes[:, 0],
                                        self.mesh.nodes[:, 1])
            write_vtu(out_dir / fname, self.mesh.nodes, self.mesh.tris,
                      node_zb,
                      {"h": snap["h"], "u": snap["u"], "v": snap["v"],
                       "zb": snap["zb"], "zs": snap["zs"]})
            frames.append((self.t, fname))

        def log_row() -> None:
            snap = self.snapshot()
            vals = [(snap["h"][c], snap["u"][c], snap["v"][c])
                    for c in self.monitor_cells]
            mon.write(self.t, vals)
            mw.write(self.t, self.mass(), self.boundary_inflow(),
                     self._last_dt if hasattr(self, "_last_dt") else 0.0,
                     float(snap["h"].min()), float(snap["h"].max()))

        if vtk_on:
            dump_frame()
        if csv_on:
            log_row()

        # time-averaging window (wind set-up: the steady balance is extracted
        # as a time mean over the last `average_window` seconds, which
        # filters the residual basin-scale seiche)
        avg_on = cfg.output.average_window > 0.0
        acc = np.zeros((self.n_cells, 3), dtype=np.float64)
        acc_n = 0

        while self.t < cfg.t_end - 1.0e-12:
            self.update_time_bcs(self.t)
            next_event = cfg.t_end
            if vtk_on:
                next_event = min(next_event, next_vtk + cfg.output.vtk_interval)
            if csv_on:
                next_event = min(next_event, next_csv + cfg.output.csv_interval)
            dt = self.compute_dt(next_event - self.t)
            self._last_dt = dt
            self.step(dt)

            if avg_on and self.t >= cfg.t_end - cfg.output.average_window:
                acc += self.w.numpy()
                acc_n += 1

            if vtk_on and self.t >= next_vtk + cfg.output.vtk_interval - 1.0e-9:
                next_vtk += cfg.output.vtk_interval
                dump_frame()
            if csv_on and self.t >= next_csv + cfg.output.csv_interval - 1.0e-9:
                next_csv += cfg.output.csv_interval
                log_row()

        if avg_on and acc_n > 0:
            self._avg_w = acc / acc_n
            print(f"averaged state over [{cfg.t_end - cfg.output.average_window:g}, "
                  f"{cfg.t_end:g}] s ({acc_n} steps)")

        if vtk_on:
            dump_frame()
        if csv_on:
            log_row()
        if vtk_on:
            write_pvd(out_dir / "animation.pvd", frames)

        mon.close()
        mw.close()
        m_fin = self.mass()
        qbc = self.boundary_inflow()
        msg = (f"done: {self.step_count} steps, t = {self.t:.4f} s, "
               f"mass = {m_fin:.12g} m^3")
        if abs(qbc) > 0.0:
            msg += (f", q_bc = {qbc:.12g} m^3, mass-balance residual = "
                    f"{m_fin - self._mass0 - qbc:.3e} m^3")
        print(msg + f" -> {out_dir}")
