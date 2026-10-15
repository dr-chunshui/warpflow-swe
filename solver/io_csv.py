"""CSV loggers: monitor-point time series and global mass/dt history."""
from __future__ import annotations

import csv
from pathlib import Path


class MonitorWriter:
    """One row per time sample: t, then h/u/v per monitored point."""

    def __init__(self, path: str | Path, names: list):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="ascii")
        self._w = csv.writer(self._f)
        header = ["t"]
        for nm in names:
            header += [f"{nm}_h", f"{nm}_u", f"{nm}_v"]
        self._w.writerow(header)

    def write(self, t: float, values: list) -> None:
        row = [f"{t:.6f}"]
        for h, u, v in values:
            row += [f"{h:.10g}", f"{u:.10g}", f"{v:.10g}"]
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class MassWriter:
    """Global history: t, mass, q_bc (cumulative net boundary inflow
    volume), dt, min_h, max_h, plus one qbc_<group> column per boundary
    group with the same cumulative net volume split by group."""

    def __init__(self, path: str | Path, group_names: list | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="ascii")
        self._w = csv.writer(self._f)
        header = ["t", "mass", "q_bc", "dt", "h_min", "h_max"]
        for i, _ in enumerate(group_names or []):
            header.append(f"qbc_g{i}")
        self._w.writerow(header)

    def write(self, t: float, mass: float, qbc: float, dt: float,
              hmin: float, hmax: float, qbc_groups: list | None = None) -> None:
        row = [f"{t:.6f}", f"{mass:.12g}", f"{qbc:.12g}",
               f"{dt:.10g}", f"{hmin:.10g}", f"{hmax:.10g}"]
        for v in qbc_groups or []:
            row.append(f"{v:.12g}")
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()
