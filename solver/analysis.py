"""Comparison against analytic solutions: band-averaged profiles,
L1/L2 errors, PNG figure and errors.csv row.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import analytic
from .config import CaseConfig
from .mesh import TriMesh
from .solver import ShallowWaterSolver


def band_cells(mesh: TriMesh, y0: float, band: float) -> np.ndarray:
    """Indices of cells whose centroid lies within |y - y0| <= band."""
    cy = mesh.cell_centroids[:, 1]
    return np.where(np.abs(cy - y0) <= band)[0]


def profile_errors(mesh: TriMesh, num: np.ndarray, ana: np.ndarray,
                   idx: np.ndarray) -> dict:
    """Area-weighted L1/L2 errors over the selected cells."""
    a = mesh.cell_area[idx]
    e = num[idx] - ana[idx]
    norm = a.sum()
    return {
        "l1": float(np.abs(e * a).sum() / norm),
        "l2": float(np.sqrt((e * e * a).sum() / norm)),
    }


def compare_case(cfg: CaseConfig, mesh: TriMesh,
                 solver: ShallowWaterSolver) -> dict:
    """Numeric vs analytic at solver.t; writes compare.png + errors.csv."""
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = solver.snapshot()
    t = solver.t
    cx = mesh.cell_centroids[:, 0]
    cy = mesh.cell_centroids[:, 1]
    h_ana, u_ana, v_ana = analytic.evaluate(
        cfg.analytic, cfg, t, cx, cy)

    idx = band_cells(mesh, cfg.output.centerline_y, cfg.output.band)
    if len(idx) == 0:
        raise ValueError(
            f"{cfg.name}: band |y-{cfg.output.centerline_y}| <= "
            f"{cfg.output.band} contains no cell centroid; enlarge "
            f"output.band (mesh dy may be coarser than the band)")
    eh = profile_errors(mesh, snap["h"], h_ana, idx)
    # velocity errors are only meaningful where BOTH states are wet: the
    # analytic velocity is defined everywhere (e.g. the uniform Thacker
    # velocity field) while the numerical dry cells carry u = 0 by
    # construction, which would otherwise dominate the norm
    wet = (snap["h"] > cfg.eps_dry) & (h_ana > cfg.eps_dry)
    if np.any(wet[idx]):
        eu = profile_errors(mesh, snap["u"], u_ana, idx[wet[idx]])
    else:
        eu = {"l1": float("nan"), "l2": float("nan")}
    ef = profile_errors(mesh, snap["h"], h_ana,
                        np.arange(mesh.n_cells))
    errors = {"case": cfg.name, "t": t, "order": cfg.order,
              "time_order": cfg.time_order, "n_band": len(idx),
              "l1_h": eh["l1"], "l2_h": eh["l2"],
              "l1_u": eu["l1"], "l2_u": eu["l2"],
              "l1_h_full": ef["l1"], "l2_h_full": ef["l2"]}

    _write_errors_csv(out_dir / "errors.csv", errors)
    _plot(cfg, mesh, out_dir / "compare.png", t, snap, idx,
          {"h": h_ana, "u": u_ana}, errors)
    print("errors: " + "  ".join(f"{k}={errors[k]:.4e}"
                                 if isinstance(errors[k], float)
                                 else f"{k}={errors[k]}"
                                 for k in ("l1_h", "l2_h", "l1_u", "l2_u")))
    return errors


def _write_errors_csv(path: Path, errors: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(errors.keys())
        w.writerow([f"{v:.6e}" if isinstance(v, float) else v
                    for v in errors.values()])


def _plot(cfg: CaseConfig, mesh: TriMesh, path: Path, t: float,
          snap: dict, idx: np.ndarray, ana: dict, errors: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cx = mesh.cell_centroids[idx, 0]
    x_fine = np.linspace(cx.min(), cx.max(), 800)
    y_fine = np.full_like(x_fine, cfg.output.centerline_y)
    h_ana, u_ana, _ = analytic.evaluate(cfg.analytic, cfg, t,
                                        x_fine, y_fine)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    axes[0].plot(x_fine, h_ana, "k-", lw=1.2, label="analytic")
    axes[0].plot(cx, snap["h"][idx], "o", ms=2.5, mfc="none",
                 color="tab:red", label="numeric")
    axes[0].set_ylabel("h [m]")
    axes[0].legend(loc="upper right")
    axes[1].plot(x_fine, u_ana, "k-", lw=1.2, label="analytic")
    axes[1].plot(cx, snap["u"][idx], "o", ms=2.5, mfc="none",
                 color="tab:blue", label="numeric")
    axes[1].set_ylabel("u [m/s]")
    axes[1].set_xlabel("x [m]")

    fig.suptitle(
        f"{cfg.name}   t = {t:.3g} s   "
        f"order {cfg.order} / time {cfg.time_order}   "
        f"L1(h) = {errors['l1_h']:.3e}   L2(h) = {errors['l2_h']:.3e}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
