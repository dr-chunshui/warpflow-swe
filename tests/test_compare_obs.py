"""Batch-1 capability tests: measured-observation comparison module.

Run directly:  python tests/test_compare_obs.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.compare_obs import compare_observed, load_monitors  # noqa: E402
from solver.config import load_config  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # synthetic monitors.csv: two points, one wave-like signal
        t = np.arange(0.0, 100.0, 1.0)
        hA = 2.0 * np.exp(-((t - 10.0) ** 2) / 8.0)   # peak 2.0 at t=10
        hB = 1.0 * np.exp(-((t - 50.0) ** 2) / 8.0)   # peak 1.0 at t=50
        with open(td / "monitors.csv", "w", encoding="ascii") as f:
            f.write("t,A_h,A_u,A_v,B_h,B_u,B_v\n")
            for k in range(len(t)):
                f.write(f"{t[k]:.3f},{hA[k]:.6g},0,0,{hB[k]:.6g},0,0\n")

        mon = load_monitors(td / "monitors.csv")
        assert set(mon) == {"A", "B"}
        assert abs(mon["A"][1].max() - 2.0) < 1e-12
        print("load_monitors ok")

        # yaml with observed block (points must merge into monitors)
        y = f"""
case: {{name: t_obs, t_end: 1.0}}
mesh: {{type: rect, xmin: 0, xmax: 1, ymin: 0, ymax: 1, dx: 0.1}}
initial: {{type: lake_at_rest, params: {{zs: 0.0}}}}
boundary:
  groups:
    - {{name: walls, select: {{xmin: [0.0, 1.0]}}, type: wall}}
    - {{name: w2, select: {{xmax: [0.0, 1.0]}}, type: wall}}
    - {{name: w3, select: {{ymin: [0.0, 1.0]}}, type: wall}}
    - {{name: w4, select: {{ymax: [0.0, 1.0]}}, type: wall}}
output:
  dir: {td.as_posix()}
  csv_interval: 1.0
  observed:
    max_depth:
      - {{name: A, x: 0.5, y: 0.5, measured: 2.05}}
      - {{name: B, x: 0.6, y: 0.5, measured: 0.95}}
    arrival:
      - {{name: A, x: 0.5, y: 0.5, threshold: 0.2}}
      - {{name: B, x: 0.6, y: 0.5, threshold: 0.2}}
    arrival_diffs:
      - {{name: AB, a: A, b: B, measured: 40.0}}
"""
        yp = td / "case.yaml"
        yp.write_text(y, encoding="utf-8")
        cfg = load_config(yp)
        # observed points appended to monitors with their names
        names = [m.get("name") for m in cfg.output.monitors]
        assert names == ["A", "B"], names

        metrics = compare_observed(cfg, td)
        md = metrics["max_depth:A"]
        assert abs(md["sim"] - 2.0) < 1e-12 and md["measured"] == 2.05
        ad = metrics["arrival_diffs:AB"]
        # first sampled crossing of h >= 0.2: fronts rise through the
        # threshold between t=5.7/6 and t=46.4/47 -> A at t=6, B at t=47
        assert abs(ad["sim"] - 41.0) < 1e-9, ad
        assert abs(ad["measured"] - 40.0) < 1e-12
        assert (td / "gauges.csv").is_file()
        print("compare_observed ok: max-depth/arrival/diffs all match")
        print("PASS: observation comparison capability verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
