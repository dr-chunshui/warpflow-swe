"""Initial conditions and bathymetry on cell centroids (numpy, float64).

Cell-centred FV: conserved state w = (h, hu, hv) per cell; bottom zb per
cell.  All quantities are built on the cell centroids from analytic
formulas (TELEMAC geo files store flat bottoms -- terrain is applied at
runtime there via USER_CORFON, so we do the same here).
"""
from __future__ import annotations

import numpy as np

from .config import CaseConfig
from .mesh import TriMesh


def bathymetry_at(cfg: CaseConfig, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bottom elevation at arbitrary (x, y) points [m]."""
    p = cfg.bathymetry.params
    kind = cfg.bathymetry.type
    if kind == "flat":
        return np.full(len(x), float(p.get("z", 0.0)), dtype=np.float64)
    if kind == "bump":  # TELEMAC bump: zb = a*exp(-0.5*((x-x0)/s)^2)
        a = float(p.get("amplitude", 0.2))
        x0 = float(p.get("x0", 10.0))
        s = float(p.get("sigma", 1.0))
        return a * np.exp(-0.5 * ((x - x0) / s) ** 2)
    if kind == "thacker":  # parabolic bowl, zb = H0*(r^2/a^2 - 1) <= 0
        h0 = float(p["h0"])
        a = float(p["a"])
        xc = float(p.get("xc", 0.5 * (x.min() + x.max())))
        yc = float(p.get("yc", 0.5 * (y.min() + y.max())))
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        return h0 * (r2 / (a * a) - 1.0)
    if kind == "slope":  # plane bed, zb = z0 + slope*x (x absolute)
        return float(p["z0"]) + float(p["slope"]) * x
    if kind == "beach":
        # Wave-flume beach of the TELEMAC tests_channel case
        # (Synolakis solitary-wave run-up): flat to the toe, then a
        # plane 1:19.85 slope rising landward.
        toe = float(p.get("toe", 14.68))
        slope_inv = float(p.get("slope_inv", 19.85))
        return np.maximum(x - toe, 0.0) / slope_inv
    if kind == "vasque":
        # Plane 0.01 slope with a Gaussian hump (TELEMAC vasque
        # user_corfon.f; the manual prints (x-19)^2 but the Fortran
        # samples I = 1..47 over 46 m, i.e. hump centre x = 18).
        x0 = float(p.get("x0", 18.0))
        return (-0.6 + 0.01 * x
                + 0.1 * np.exp(-((x - x0) ** 2) / 20.0))
    if kind == "bolzano":
        # Tidal-flat channel of the TELEMAC balzano case
        # (user_fortran/user_corfon.f).  Piecewise linear in x; the two
        # middle branches overlap at x = 4800 exactly like the Fortran,
        # where the later assignment (the reservoir crest segment) wins.
        seg1 = -x / 2760.0
        seg2 = x / 2760.0 - 60.0 / 23.0
        seg3 = -x / 920.0 + 100.0 / 23.0
        z = seg1.copy()
        m2 = (x >= 3600.0) & (x <= 4800.0)
        z[m2] = seg2[m2]
        m3 = (x >= 4800.0) & (x <= 6000.0)
        z[m3] = seg3[m3]
        return z
    raise ValueError(f"unknown bathymetry type {kind!r}")


def build_bathymetry(cfg: CaseConfig, mesh: TriMesh) -> np.ndarray:
    """Bottom elevation zb at cell centroids [m]."""
    c = mesh.cell_centroids
    return bathymetry_at(cfg, c[:, 0], c[:, 1])


def build_initial(cfg: CaseConfig, mesh: TriMesh, zb: np.ndarray) -> np.ndarray:
    """Conserved state w (n_cells, 3) = (h, hu, hv) at t = 0."""
    p = cfg.initial.params
    xc = mesh.cell_centroids[:, 0]
    yc = mesh.cell_centroids[:, 1]
    kind = cfg.initial.type
    w = np.zeros((mesh.n_cells, 3), dtype=np.float64)

    if kind == "lake_at_rest":
        zs = float(p["zs"])
        w[:, 0] = np.maximum(zs - zb, 0.0)
    elif kind == "gaussian_hump":
        # TELEMAC gouttedo (user_condin_h.f): h = h0 + amp*exp(-r^2/s2),
        # at rest.  s2 = 4.0 (variance 2) in the manual case.
        h0 = float(p["h0"])
        amp = float(p["amp"])
        s2 = float(p.get("s2", 4.0))
        x0 = float(p.get("x0", 0.5 * (xc.max() + xc.min())))
        y0 = float(p.get("y0", 0.5 * (yc.max() + yc.min())))
        r2 = (xc - x0) ** 2 + (yc - y0) ** 2
        w[:, 0] = h0 + amp * np.exp(-r2 / s2)
    elif kind == "dam_snap":
        hl = float(p["hl"])
        hr = float(p["hr"])
        xd = float(p["x_dam"])
        w[:, 0] = np.where(xc < xd, hl, hr)
    elif kind == "seiche":
        # closed channel, h0 = H0 + A*cos(2*pi*x/L), u = 0
        h0 = float(p["h0"])
        amp = float(p["amplitude"])
        xmin = float(mesh.nodes[:, 0].min())
        length = float(mesh.nodes[:, 0].max() - xmin)
        w[:, 0] = np.maximum(h0 + amp * np.cos(2.0 * np.pi * (xc - xmin) / length), 0.0)
    elif kind == "bump_steady":
        # steady-state friendly start: constant free surface + given inflow
        # depth estimate; fine-tuned per case in the yaml params
        zs = float(p["zs"])
        w[:, 0] = np.maximum(zs - zb, 0.0)
    elif kind == "bump_exact":
        # steady bump profile (same construction as the TELEMAC examples:
        # initialise directly on the analytic solution); flow/q/width/hl
        # come from the yaml params
        from .analytic import _bump_profile
        xs, h_prof, u_prof = _bump_profile(
            str(p["flow"]), float(p["q"]),
            float(p.get("width", 2.0)), float(p.get("hl", 0.0)))
        w[:, 0] = np.interp(xc, xs, h_prof)
        w[:, 1] = w[:, 0] * np.interp(xc, xs, u_prof)
    elif kind == "thacker_planar":
        # exact solution at t=0 (examples/thacker user_fortran):
        # h = max( (eta*h0/a^2)*(2*(x-xc) - eta) - zb, 0 ), u = 0, v = eta*om
        h0 = float(p["h0"])
        a = float(p["a"])
        eta = float(p["eta"])
        L = float(p.get("length", 4.0))
        x0 = float(p.get("xc", 0.5 * L))
        y0 = float(p.get("yc", 0.5 * L))
        om = np.sqrt(2.0 * 9.81 * h0) / a
        zs = (eta * h0 / (a * a)) * (2.0 * (xc - x0) - eta)
        w[:, 0] = np.maximum(zs - zb, 0.0)
        # conserved momentum = h * v with the exact uniform velocity field
        # (u, v) = (0, eta*omega) at t = 0 (USER_CONDIN_H stores VELOCITY;
        # the FV state stores momentum)
        w[:, 2] = w[:, 0] * eta * om
    elif kind == "thacker_radial":
        # exact solution at t=0 (examples/thacker user_fortran_2):
        # h = max( h0*(sqrt(1-B^2)/(1-B) - 1 - (r^2/a^2)*((1-B^2)/(1-B)^2-1)) - zb, 0 )
        h0 = float(p["h0"])
        a = float(p["a"])
        r0 = float(p["r0"])
        L = float(p.get("length", 4.0))
        x0 = float(p.get("xc", 0.5 * L))
        y0 = float(p.get("yc", 0.5 * L))
        b = (a * a - r0 * r0) / (a * a + r0 * r0)
        r2 = (xc - x0) ** 2 + (yc - y0) ** 2
        zs = h0 * (np.sqrt(1.0 - b * b) / (1.0 - b) - 1.0
                   - (r2 / (a * a)) * ((1.0 - b * b) / (1.0 - b) ** 2 - 1.0))
        w[:, 0] = np.maximum(zs - zb, 0.0)
    else:
        raise ValueError(f"unknown initial type {kind!r}")

    return w


def nearest_cell(mesh: TriMesh, x: float, y: float) -> int:
    """Index of the cell whose centroid is closest to (x, y)."""
    d = (mesh.cell_centroids[:, 0] - x) ** 2 + (mesh.cell_centroids[:, 1] - y) ** 2
    return int(np.argmin(d))
