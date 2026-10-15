"""Boundary-excluded L1 metric (batch-1 follow-up, suggestion 1).

TELEMAC's output field pins the LIQUID boundary NODES to the prescribed
value (bord.f writes the FE array H), which pulls its P1 field down near
the level/inflow chains -- an artifact a cell-centred scheme cannot
reproduce.  This script reports the final-field comparison against the
reference with cells within R of any LIQUID boundary node excluded, so
the pinned-node artifact and the physical error are separately visible.

Usage: python scratch/boundary_excluded_metric.py <case.yaml> [frame.vtu]
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.compare_telemac import p1_sample  # noqa: E402
from solver.config import load_config  # noqa: E402
from solver.mesh import build_topology, cli_chains, read_cli, read_selafin, \
    read_selafin_series  # noqa: E402

RADII = [100.0, 300.0, 600.0, 1200.0]
LIQUID_TYPES = {"level", "inflow_q", "outflow"}


def read_vtu_scalar(path: str, names: list) -> dict:
    txt = open(path).read()
    out = {}
    for m in re.finditer(
            r'<DataArray type="[\w:]+" Name="(\w+)"[^>]*>\s*([+/\w=]*)\s*</DataArray>',
            txt):
        nm, b64 = m.group(1), m.group(2).strip()
        if nm in names:
            out[nm] = np.frombuffer(base64.b64decode(b64), dtype=np.float64)
    missing = [n for n in names if n not in out]
    if missing:
        raise KeyError(f"{path}: fields {missing} not found")
    return out


def liquid_chain_nodes(cfg, mesh) -> np.ndarray:
    """Nodes of every boundary chain a yaml group maps to a liquid type."""
    cli = read_cli(str(Path(cfg.mesh.file).with_suffix(".cli")))
    _, chains = cli_chains(mesh, cli)
    named = {}  # chain number (1-based) -> is_liquid
    rest_liquid = False
    for g in cfg.boundary:
        sel = g.select or {}
        ch = sel.get("chain", None)
        liquid = g.type in LIQUID_TYPES
        if ch == "rest":
            rest_liquid = liquid
        elif isinstance(ch, list):
            for c in ch:
                named[int(c)] = liquid
        elif ch is not None:
            named[int(ch)] = liquid
    pts = []
    for i, ch in enumerate(chains, start=1):
        if named.get(i, rest_liquid):  # unnamed chains default to 'rest'
            pts.append(mesh.nodes[np.asarray(ch["nodes"], dtype=np.int64)])
    return np.concatenate(pts, axis=0)


def main() -> int:
    yaml_path = sys.argv[1]
    cfg = load_config(yaml_path)
    nodes, tris, bottom = read_selafin(cfg.mesh.file)
    mesh = build_topology(nodes, tris, bottom)
    cc = mesh.cell_centroids
    area = mesh.cell_area

    frames = sorted(Path(cfg.output.dir).glob("frame_*.vtu"))
    fr = Path(sys.argv[2]) if len(sys.argv) > 2 else frames[-1]
    if not fr.exists():
        raise SystemExit(f"frame not found: {fr}")
    f = read_vtu_scalar(str(fr), ["h", "zs"])

    ref = read_selafin_series(cfg.output.observed["telemac_ref"]["file"], ["S"])
    s_ref = p1_sample(ref["nodes"], ref["tris"], ref["series"]["S"][-1], cc)

    zb = f["h"] * 0.0  # zb also stored; but h and zs must be consistent
    zb = f["zs"] - f["h"]
    e = f["zs"] - s_ref
    wet = f["h"] > cfg.eps_dry
    dry = ~wet

    bq = liquid_chain_nodes(cfg, mesh)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(bq).query(cc)

    print(f"case {cfg.name}  frame {fr.name}  cells {len(cc)}  "
          f"wet {int(wet.sum())}  dry {int(dry.sum())}")
    print(f"liquid boundary nodes: {len(bq)}")
    print(f"{'metric':34s} {'L1':>8s} {'L2':>8s} {'Linf':>8s} {'bias':>8s}")
    def row(tag, mask):
        if not mask.any():
            print(f"{tag:34s}   (empty)")
            return
        ee = e[mask]
        print(f"{tag:34s} {np.abs(ee).mean():8.4f} "
              f"{np.sqrt((ee*ee).mean()):8.4f} {np.abs(ee).max():8.4f} "
              f"{ee.mean():+8.4f}")
    row("all cells (headline basis)", np.ones_like(wet, dtype=bool))
    row("wet, all cells", wet)
    prev = 0.0
    for r in RADII:
        row(f"wet, exclude d < {r:g} m", wet & (d >= r))
    row("wet, NEAR chain d < 100 m", wet & (d < 100.0))
    row("wet, ring 100-300 m", wet & (d >= 100.0) & (d < 300.0))
    row("dry, all", dry)
    if dry.any():
        row("dry, exclude d < 300 m", dry & (d >= 300.0))
    # volume ledger over the same masks
    dv = (f["zs"] - s_ref) * area
    print(f"volume diff wet all: {dv[wet].sum():+,.0f} m3; "
          f"excl d<300: {dv[wet & (d >= 300.0)].sum():+,.0f} m3; "
          f"near d<300: {dv[wet & (d < 300.0)].sum():+,.0f} m3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
