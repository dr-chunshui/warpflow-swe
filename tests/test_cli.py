"""Batch-1 capability tests: .cli parsing + boundary-chain grouping.

Run directly:  python tests/test_cli.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.mesh import build_topology, cli_chains, read_cli  # noqa: E402

SOLF_DIR = Path(r"D:\Workspace\warpflow\telemac2d\examples")
MALP_CLI = SOLF_DIR / "malpasset" / "geo_malpasset-small.cli"
MALP_SLF = SOLF_DIR / "malpasset" / "geo_malpasset-small.slf"


def write_cli(path: Path, lines: list) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write("/ synthetic boundary file\n")
        for node in lines:
            f.write(f"  2 2 2  0.000  0.000  0.000  0.0   2  "
                    f"0.000  0.000  0.000  {node + 1:>5d}     1\n")


def main() -> int:
    # --- reader on the real file ---
    cli = read_cli(str(MALP_CLI))
    assert len(cli["node"]) == 1080, len(cli["node"])
    assert set(np.unique(cli["lihbor"])) == {2}  # all wall
    assert cli["node"].min() >= 0
    print(f"read_cli ok: {len(cli['node'])} lines, LIHBOR values "
          f"{sorted(set(cli['lihbor'].tolist()))}")

    # --- chain grouping: single closed loop, all edges assigned ---
    from solver.mesh import read_selafin
    nodes, tris, _ = read_selafin(str(MALP_SLF))
    mesh = build_topology(nodes, tris)
    chain_of, chains = cli_chains(mesh, cli)
    assert np.all(chain_of >= 0)
    assert len(chains) == 1, len(chains)
    assert np.all(chain_of == 0)  # one closed loop covers everything
    print(f"malpasset chains ok: {len(chains)} chain, "
          f"{int((chain_of == 0).sum())} edges in chain 1")

    # --- synthetic: two disjoint triangles -> two chains ---
    # nodes 0,1,2 / 3,4,5; every node-pair inside a triangle is an edge
    quad_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                           [2.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    quad_tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    m2 = build_topology(quad_nodes, quad_tris)
    assert m2.n_bedges == 6
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "two.cli"
        write_cli(p, [0, 1, 2, 3, 4, 5])  # jump between 2 and 3
        cli2 = read_cli(str(p))
        chain2, chains2 = cli_chains(m2, cli2)
        assert np.all(chain2 >= 0)
        assert len(chains2) == 2, chains2
        assert sorted(np.unique(chain2).tolist()) == [0, 1]
        assert set(chains2[0]["nodes"]) == {0, 1, 2}
        assert set(chains2[1]["nodes"]) == {3, 4, 5}
        # loop closure: the wrap edges (2,0) and (5,3) have no consecutive
        # cli pair but must land in the right chains
        assert chain2[list(np.nonzero(chain2 == 0)[0])].size == 3
        assert int((chain2 == 0).sum()) == 3 and int((chain2 == 1).sum()) == 3
        print("two-chain synthetic ok (incl. loop closure)")

        # --- single walk without repeated first node -> one chain ---
        # square of two triangles: boundary edges (0,1),(1,2),(2,3),(3,0);
        # cli [0,1,2,3] covers three edges, (3,0) closes the loop
        sq_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        sq_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        msq = build_topology(sq_nodes, sq_tris)
        p1 = Path(td) / "one.cli"
        write_cli(p1, [0, 1, 2, 3])
        chain1, chains1 = cli_chains(msq, read_cli(str(p1)))
        assert len(chains1) == 1 and int((chain1 == 0).sum()) == 4
        print("single closed loop ok")

        # --- broken cli: node not on the boundary -> clear error ---
        pbad = Path(td) / "bad.cli"
        write_cli(pbad, [0, 1])
        try:
            cli_chains(m2, read_cli(str(pbad)))
            raise AssertionError("expected unassigned-edge error")
        except ValueError as e:
            assert "unassigned" in str(e)
        print("mismatched .cli rejected ok")

    print("PASS: .cli capability verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
