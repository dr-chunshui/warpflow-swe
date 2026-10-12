"""Collect errors.csv from every case run under output/ into one table.

Usage: python summarize.py [output_dir]   (default: output)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    rows = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        err_csv = case_dir / "errors.csv"
        if not err_csv.exists():
            continue
        with err_csv.open() as fh:
            data = {}
            for k, v in csv.DictReader(fh).__next__().items():
                try:
                    data[k] = float(v)
                except (TypeError, ValueError):
                    pass
        hist = case_dir / "history.csv"
        t_end = mass_end = ""
        if hist.exists():
            with hist.open() as fh:
                lines = list(csv.DictReader(fh))
            if lines:
                last = lines[-1]
                t_end = last.get("t", "")
                mass_end = last.get("mass", "")
        rows.append({
            "case": case_dir.name,
            "l1_h": data.get("l1_h", ""),
            "l2_h": data.get("l2_h", ""),
            "l1_u": data.get("l1_u", ""),
            "l2_u": data.get("l2_u", ""),
            "l1_h_full": data.get("l1_h_full", ""),
            "t_end": t_end,
            "mass": mass_end,
        })

    if not rows:
        print(f"no errors.csv found under {root}")
        return 1

    cols = ["case", "l1_h", "l2_h", "l1_u", "l2_u", "l1_h_full", "t_end", "mass"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    out = root / "summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
