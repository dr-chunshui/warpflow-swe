"""Comparison of monitored time series against measured observations.

Driven by the yaml block ``output.observed`` (points are auto-monitored
by config.load_config):

    observed:
      max_depth:     [{name: P10, x: ..., y: ..., measured: 11.8}]
      arrival:       [{name: A, x: ..., y: ..., threshold: 0.2}]
      arrival_diffs: [{name: AB, a: A, b: B, measured: 1140.0}]

- max_depth: peak simulated h at the point vs measured peak [m]
- arrival:   first time simulated h >= threshold at the point [s]
- arrival_diffs: simulated t_b - t_a vs measured travel time [s]
  (wave-front travel times, e.g. the malpasset transformers A->B/A->C)

Writes gauges.csv next to monitors.csv and returns a metrics dict.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import CaseConfig


def load_monitors(path: str | Path) -> dict:
    """Read monitors.csv -> {name: (t, h)} float64 arrays."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.asarray([[float(v) for v in r] for r in rows[1:] if r],
                      dtype=np.float64)
    out = {}
    for col in range(1, len(header), 3):
        name = header[col].removesuffix("_h")
        out[name] = (data[:, 0], data[:, col])
    return out


def _peak_h(t: np.ndarray, h: np.ndarray) -> float:
    return float(h.max())


def _arrival(t: np.ndarray, h: np.ndarray, threshold: float) -> float:
    hit = np.nonzero(h >= threshold)[0]
    if len(hit) == 0:
        return float("nan")
    return float(t[hit[0]])


def compare_observed(cfg: CaseConfig, out_dir: str | Path) -> dict:
    """Max-depth / arrival-time comparison; writes gauges.csv."""
    out_dir = Path(out_dir)
    mon = load_monitors(out_dir / "monitors.csv")
    obs = cfg.output.observed

    rows = []  # kind, name, ref, sim, meas, rel_err
    for entry in obs.get("max_depth", []):
        t, h = mon[entry["name"]]
        sim = _peak_h(t, h)
        meas = float(entry["measured"])
        rel = abs(sim - meas) / abs(meas)
        rows.append(["max_depth", entry["name"], "", sim, meas, rel])
    t_arr: dict = {}
    for entry in obs.get("arrival", []):
        t, h = mon[entry["name"]]
        t_arr[entry["name"]] = _arrival(t, h, float(entry["threshold"]))
        rows.append(["arrival", entry["name"], f"thr={entry['threshold']:g}",
                    t_arr[entry["name"]], float("nan"), float("nan")])
    for entry in obs.get("arrival_diffs", []):
        sim = t_arr[entry["b"]] - t_arr[entry["a"]]
        meas = float(entry["measured"])
        rel = abs(sim - meas) / abs(meas)
        rows.append(["arrival_diffs", entry["name"],
                     f"{entry['a']}->{entry['b']}", sim, meas, rel])

    with open(out_dir / "gauges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "name", "ref", "sim", "measured", "rel_err"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6g}", f"{r[4]:.6g}",
                        f"{r[5]:.4g}" if r[5] == r[5] else "nan"])

    print(f"{'kind':<14}{'name':<8}{'sim':>12}{'measured':>12}"
          f"{'rel_err':>10}")
    for kind, name, ref, sim, meas, rel in rows:
        mstr = f"{meas:>12.6g}" if meas == meas else f"{'-':>12}"
        print(f"{kind:<14}{name:<8}{sim:>12.6g}{mstr}"
              f"{rel:>10.4f}" if rel == rel else
              f"{kind:<14}{name:<8}{sim:>12.6g}{mstr}{'-':>10}")
    return {f"{kind}:{name}": {"sim": sim, "measured": meas, "rel_err": rel}
            for kind, name, _, sim, meas, rel in rows}
