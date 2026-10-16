"""Unit tests for the TELEMAC roughness-zone parsers (.bfr + roughness.tbl)
and the per-cell zone aggregation.  Run directly:
python tests/test_zones.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.mesh import (build_topology, read_bfr_zones, read_roughness_tbl,
                         zone_ks_for_cells)

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
TRIS = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


def _write(tmp: str, name: str, text: str) -> str:
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_bfr_and_tbl_parse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bfr = _write(tmp, "geo.bfr",
                     "1 100\n2 100\n3 200\n4 200\n")
        tbl = _write(tmp, "roughness.tbl",
                     "* comment row\n"
                     "100 NIKU 0.025 NULL\n"
                     "200 NIKU 0.150 NULL\n"
                     "END\n")
        zone = read_bfr_zones(bfr, 4)
        assert zone.tolist() == [100, 100, 200, 200]
        ks = read_roughness_tbl(tbl)
        assert ks == {100: 0.025, 200: 0.15}, ks
        m = build_topology(NODES.copy(), TRIS.copy())
        cell_ks, zone_of_cell = zone_ks_for_cells(m, zone, ks)
        # tri0 corners 100,100,200 -> majority 100; tri1 100,200,200 -> 200
        assert zone_of_cell.tolist() == [100, 200]
        assert cell_ks.tolist() == [0.025, 0.15]
    print("zones: .bfr/.tbl parse + majority vote exact")


def test_tie_nearest_centroid_node() -> None:
    """3 distinct zones -> the node nearest the cell centroid decides."""
    with tempfile.TemporaryDirectory() as tmp:
        bfr = _write(tmp, "geo.bfr",
                     "1 300\n2 100\n3 200\n4 100\n")
        tbl = _write(tmp, "roughness.tbl",
                     "100 NIKU 0.100\n200 NIKU 0.200\n300 NIKU 0.300\nEND\n")
        zone = read_bfr_zones(bfr, 4)
        ks = read_roughness_tbl(tbl)
        m = build_topology(NODES.copy(), TRIS.copy())
        cell_ks, zone_of_cell = zone_ks_for_cells(m, zone, ks)
        # tri0 corners zones [300,100,200]: node 0 at (0,0) is nearest the
        # centroid (1/3, 1/3) -> zone 300; tri1 [100,100,200] -> majority 100
        assert zone_of_cell.tolist() == [300, 100]
        assert cell_ks.tolist() == [0.3, 0.1]
    print("zones: tie broken by nearest-centroid node")


def test_uncovered_node_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bfr = _write(tmp, "geo.bfr", "1 100\n2 100\n3 200\n")
        try:
            read_bfr_zones(bfr, 4)
        except ValueError as e:
            assert "not covered" in str(e), e
        else:
            raise AssertionError("uncovered .bfr accepted")
        dup = _write(tmp, "dup.bfr", "1 100\n1 200\n2 100\n3 200\n4 100\n")
        try:
            read_bfr_zones(dup, 4)
        except ValueError as e:
            assert "twice" in str(e), e
        else:
            raise AssertionError("duplicate node accepted")
    print("zones: uncovered / duplicate nodes raise")


def test_non_niku_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tbl = _write(tmp, "roughness.tbl",
                     "100 NIKU 0.1\n200 STRI 30.0\nEND\n")
        try:
            read_roughness_tbl(tbl)
        except ValueError as e:
            assert "NIKU" in str(e), e
        else:
            raise AssertionError("non-NIKU law accepted")
    print("zones: non-Nikuradse law raises")


def main() -> int:
    test_bfr_and_tbl_parse()
    test_tie_nearest_centroid_node()
    test_uncovered_node_raises()
    test_non_niku_raises()
    print("PASS: roughness-zone parsers + per-cell aggregation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
