"""M1 tests: mesh topology identities, classification, SELAFIN reader.

Run directly:  python tests/test_mesh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.mesh import build_topology, classify_boundaries, read_selafin, rect_mesh  # noqa: E402

SOLF_DIR = Path(r"D:\Workspace\warpflow\telemac2d\examples")


def divergence_identity(mesh) -> None:
    """For every cell: sum of outward L*n over its edges must vanish."""
    for c in range(mesh.n_cells):
        total = np.zeros(2)
        for k in range(3):
            e = mesh.cell_edges[c, k]
            sgn = 1.0 if mesh.edge_cells[e, 0] == c else -1.0
            total += sgn * mesh.edge_len[e] * mesh.edge_normal[e]
        assert np.max(np.abs(total)) < 1e-13, f"cell {c} divergence identity {total}"


def main() -> int:
    # --- rect mesh: counts, areas, identities ---
    m = rect_mesh(0.0, 16.0, 0.0, 0.45, dx=0.05)
    nx, ny = 320, 9
    assert m.n_cells == 2 * nx * ny, m.n_cells
    assert len(m.nodes) == (nx + 1) * (ny + 1)
    area_sum = float(np.sum(m.cell_area))
    assert abs(area_sum - 16.0 * 0.45) < 1e-12, area_sum
    divergence_identity(m)

    n_interior = np.count_nonzero(m.edge_cells[:, 1] != -1)
    assert n_interior + m.n_bedges == m.n_edges
    perim = 2.0 * (16.0 + 0.45)
    blen = float(np.sum(m.edge_len[m.boundary_edges]))
    assert abs(blen - perim) < 1e-12, blen
    print(f"rect mesh ok: {m.n_cells} cells, {m.n_edges} edges, {m.n_bedges} boundary edges")

    # alternating diagonals: symmetric mesh
    ms = rect_mesh(0.0, 4.0, 0.0, 4.0, dx=0.25)
    divergence_identity(ms)

    # --- boundary classification ---
    groups = [
        {"name": "w_lo", "select": {"xmin": [0.0, 0.45]}, "type": "wall"},
        {"name": "w_y", "select": {"ymin": [0.0, 16.0]}, "type": "wall"},
        {"name": "w_yp", "select": {"ymax": [0.0, 16.0]}, "type": "wall"},
        {"name": "outlet", "select": {"xmax": [0.0, 0.45]}, "type": "outflow"},
    ]
    g = classify_boundaries(m, groups)
    assert np.all(g >= 0)
    # xmin and xmax sides carry ny = 9 edges each
    assert np.count_nonzero(g == 0) == ny, np.count_nonzero(g == 0)
    assert np.count_nonzero(g == 3) == ny
    assert np.count_nonzero(g == 1) == nx
    assert np.count_nonzero(g == 2) == nx
    print("boundary classification ok")

    # --- SELAFIN reader on the real TELEMAC bump geometry ---
    # (TELEMAC geo files store flat bottoms; the bump is applied at runtime
    # by USER_CORFON, so our configs generate bathymetry analytically.)
    slf = SOLF_DIR / "bump" / "geo_bump.slf"
    nodes, tris, bottom = read_selafin(str(slf))
    assert len(tris) > 0 and len(nodes) > 0
    assert abs(nodes[:, 0].min()) < 1e-4 and abs(nodes[:, 0].max() - 20.0) < 1e-4
    assert abs(nodes[:, 1].min()) < 1e-4 and abs(nodes[:, 1].max() - 2.0) < 1e-4
    assert float(np.max(np.abs(bottom))) < 1e-6  # flat stored bathymetry
    msf = build_topology(nodes, tris)
    assert abs(float(np.sum(msf.cell_area)) - 40.0) < 1e-3
    divergence_identity(msf)
    print(f"selafin ok: {len(nodes)} nodes, {len(tris)} tris")

    print("PASS: mesh module verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
