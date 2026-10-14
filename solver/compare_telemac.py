"""Comparison against a TELEMAC SELAFIN reference result
(``f2d_*.slf`` shipped with the validation examples).

Driven by the yaml block ``output.observed.telemac_ref`` with two modes:

- ``mode: gauges`` (default) -- for every named monitor the reference
  series is taken at the nearest reference mesh node, resampled onto the
  monitor sample times, and compared by RMSE / max |diff| / peak.
  Requires a multi-frame reference file; writes telemac_gauges.csv.
- ``mode: final_field`` -- cell-wise comparison of the final state
  against the last frame of the reference (works with single-frame
  files): the reference nodal field is P1-interpolated (barycentric,
  bucketed point-in-triangle lookup with nearest-node fallback) onto
  our cell centroids, so the metric is free of grid-misalignment
  sampling noise.  Writes telemac_final_field.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .mesh import read_selafin_series, nearest_index
from .compare_obs import load_monitors


_VAR_KEYS = {"S": "zs", "E": "zs", "H": "h", "U": "u", "V": "v",
             "B": "zb"}


def p1_sample(ref_xy: np.ndarray, ref_tris: np.ndarray, vals: np.ndarray,
              pts: np.ndarray, buckets: int = 96) -> np.ndarray:
    """P1-interpolate a reference nodal field at arbitrary points.

    Barycentric interpolation over the reference triangulation with a
    coarse bbox bucket index; points outside the reference domain fall
    back to the nearest reference node.
    """
    tri_xy = ref_xy[ref_tris]                      # (m, 3, 2)
    lo = tri_xy.min(axis=(0, 1))                   # domain bbox (2,)
    hi = tri_xy.max(axis=(0, 1))
    cell = np.maximum(hi - lo, 1e-9) / buckets

    grid: dict[tuple[int, int], list[int]] = {}
    bmin = np.floor((tri_xy.min(axis=1) - lo) / cell).astype(np.int64)
    bmax = np.ceil((tri_xy.max(axis=1) - lo) / cell).astype(np.int64)
    for it in range(len(ref_tris)):
        for ix in range(bmin[it, 0], bmax[it, 0] + 1):
            for iy in range(bmin[it, 1], bmax[it, 1] + 1):
                grid.setdefault((ix, iy), []).append(it)

    gx = (pts[:, 0] - lo[0]) / cell[0]
    gy = (pts[:, 1] - lo[1]) / cell[1]

    x1 = tri_xy[:, 0, 0]
    y1 = tri_xy[:, 0, 1]
    x2 = tri_xy[:, 1, 0]
    y2 = tri_xy[:, 1, 1]
    x3 = tri_xy[:, 2, 0]
    y3 = tri_xy[:, 2, 1]
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    ok = np.abs(det) > 1e-12

    out = np.empty(len(pts))
    for k in range(len(pts)):
        key = (int(gx[k]), int(gy[k]))
        found = False
        for cand in grid.get(key, ()):
            if not ok[cand]:
                continue
            a, b, c = ref_tris[cand]
            lam1 = ((y2[cand] - y3[cand]) * (pts[k, 0] - x3[cand])
                    + (x3[cand] - x2[cand]) * (pts[k, 1] - y3[cand])) / det[cand]
            lam2 = ((y3[cand] - y1[cand]) * (pts[k, 0] - x3[cand])
                    + (x1[cand] - x3[cand]) * (pts[k, 1] - y3[cand])) / det[cand]
            lam3 = 1.0 - lam1 - lam2
            if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
                out[k] = lam1 * vals[a] + lam2 * vals[b] + lam3 * vals[c]
                found = True
                break
        if not found:
            d2 = ((ref_xy[:, 0] - pts[k, 0]) ** 2
                  + (ref_xy[:, 1] - pts[k, 1]) ** 2)
            out[k] = vals[int(np.argmin(d2))]
    return out


def compare_final_field(cfg: CaseConfig, mesh, solver, ref_cfg: dict) -> dict:
    """Cell-wise comparison of the final state vs the reference frame.

    The reference nodal field is P1-interpolated onto our cell
    centroids (grid-misalignment-free metric); cells whose centroid
    falls outside the reference domain fall back to the nearest
    reference node.
    """
    ref_path = str(ref_cfg["file"])
    var = str(ref_cfg.get("var", "S")).upper()
    ref = read_selafin_series(ref_path, [var])
    vals = ref["series"][var][-1]  # last frame
    nodes = ref["nodes"]
    tris = ref["tris"]

    key = _VAR_KEYS[var]
    snap = solver.snapshot()
    ours = snap[key]
    ref_at_cells = p1_sample(nodes[:, :2], np.asarray(tris), vals,
                             mesh.cell_centroids[:, :2])
    e = ours - ref_at_cells
    metrics = {
        "l1": float(np.mean(np.abs(e))),
        "l2": float(np.sqrt(np.mean(e * e))),
        "linf": float(np.max(np.abs(e))),
        "ref_range": [float(vals.min()), float(vals.max())],
    }

    out = Path(cfg.output.dir) / "telemac_final_field.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["l1", f"{metrics['l1']:.6e}"])
        w.writerow(["l2", f"{metrics['l2']:.6e}"])
        w.writerow(["linf", f"{metrics['linf']:.6e}"])
        w.writerow(["t_ref", f"{ref['times'][-1]:.6g}"])
    print(f"telemac final-field compare ({var}, t_ref = "
          f"{ref['times'][-1]:g} s, P1 vs {len(mesh.cell_centroids)} cells): "
          f"L1 = {metrics['l1']:.4g}  L2 = {metrics['l2']:.4g}  "
          f"Linf = {metrics['linf']:.4g}")
    return metrics


def compare_telemac(cfg: CaseConfig, out_dir: str | Path) -> dict:
    """Compare each monitor against the TELEMAC reference series."""
    out_dir = Path(out_dir)
    ref = cfg.output.observed["telemac_ref"]
    var = str(ref.get("var", "S"))
    data = read_selafin_series(str(ref["file"]), [var])
    t_ref = data["times"]
    vals_ref = data["series"][var]
    nodes = data["nodes"]

    mon = load_monitors(out_dir / "monitors.csv")
    points = {m.get("name", f"m{i}"): (float(m["x"]), float(m["y"]))
              for i, m in enumerate(cfg.output.monitors)}

    rows = []
    for name, (mt, mh) in mon.items():
        px, py = points.get(name, (None, None))
        if px is None:
            continue
        j = int(np.argmin((nodes[:, 0] - px) ** 2 + (nodes[:, 1] - py) ** 2))
        ref_at = np.interp(mt, t_ref, vals_ref[:, j])
        d = mh - ref_at
        rmse = float(np.sqrt(np.mean(d * d)))
        dmax = float(np.max(np.abs(d)))
        rows.append([name, j, rmse, dmax, float(mh.max()),
                     float(ref_at.max())])

    with open(out_dir / "telemac_gauges.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gauge", "ref_node", "rmse", "max_abs_diff",
                    "peak_sim", "peak_ref"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.6g}", f"{r[3]:.6g}",
                        f"{r[4]:.6g}", f"{r[5]:.6g}"])

    print(f"telemac ref ({var}): {len(t_ref)} frames at "
          f"[{t_ref[0]:g}, {t_ref[-1]:g}] s")
    print(f"{'gauge':<8}{'rmse':>12}{'maxdiff':>12}{'peak_sim':>12}"
          f"{'peak_ref':>12}")
    for name, _, rmse, dmax, p_sim, p_ref in rows:
        print(f"{name:<8}{rmse:>12.4g}{dmax:>12.4g}{p_sim:>12.4g}"
              f"{p_ref:>12.4g}")
    return {r[0]: {"rmse": r[2], "max_abs_diff": r[3],
                   "peak_sim": r[4], "peak_ref": r[5]} for r in rows}
