"""Boundary-flux audit of the TELEMAC culm reference run (f2d_culm.slf).

Integrates H * (U.n) * L along every liquid .cli chain of every frame
(P1 node means on the reference mesh, outward normal, positive = OUT of
the domain) and prints a per-chain / per-frame table.  Resolves the
"reference over-delivers 1.89x its inflow" attribution question: the
inflow chain's flux is compared against the prescribed Q at the same
instant (qsl_q.txt).

    python tools/culm_ref_inflow.py [path/to/f2d_culm.slf]

The geometry (.geo.slf / .geo.cli) and the prescribed-discharge series
(qsl_q.txt) are expected next to the results file, as in test_cases/culm.
Optional branch-B support (--scale-series OUT): writes a two-column
time/Q series with the prescribed hydrograph scaled by
Q_ref(t_last)/Q_prescribed(t_last), loadable as an inflow_q series
(positive = into the domain).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.mesh import (build_topology, cli_chains, read_cli, read_selafin,
                         read_selafin_series)

# chain type codes from cli_chains (lihbor of the walk)
WALL_TYPE = 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slf", nargs="?", default="test_cases/culm/f2d_culm.slf")
    ap.add_argument("--scale-series", metavar="OUT", default=None,
                    help="write the prescribed Q scaled by the measured "
                         "delivery ratio to OUT (two-column, into-domain "
                         "positive)")
    args = ap.parse_args()

    slf = Path(args.slf)
    case = slf.parent
    geo = case / slf.name.replace("f2d_", "geo_").replace(".slf", ".slf")
    if not geo.exists():
        # results file on the geo mesh under a different name convention
        geo = case / "geo_culm.slf"
    cli_path = geo.with_suffix(".cli")
    qsl = case / "qsl_q.txt"

    nodes, tris, _ = read_selafin(str(geo))
    mesh = build_topology(nodes, tris)
    cli = read_cli(str(cli_path))
    chain_of, chains = cli_chains(mesh, cli)

    data = read_selafin_series(str(slf), ["U", "V", "H"])
    U, V, H = data["series"]["U"], data["series"]["V"], data["series"]["H"]
    times = data["times"]
    print(f"{slf.name}: {len(times)} frame(s) at t = {times.tolist()}")

    # prescribed discharge at each frame time
    qsl_t, qsl_q = np.loadtxt(qsl, comments="#", unpack=True)
    q_pres = np.interp(times, qsl_t, qsl_q)

    liquid = [(ci, ch) for ci, ch in enumerate(chains)
              if ch["type"] != WALL_TYPE]
    print(f"\nper-chain boundary flux  H*(U.n)*L  [m^3/s, + = out of domain]")
    header = "frame  t [s]   " + "".join(
        f"chain{ci + 1:>2} (n={len(np.nonzero(chain_of == ci)[0]):2d}, "
        f"lih={ch['type']})  " for ci, ch in liquid)
    print(header)
    for k, t in enumerate(times):
        row = f"{k:5d} {t:8.1f}"
        for ci, _ in liquid:
            bes = mesh.boundary_edges[chain_of == ci]
            q = 0.0
            for be in bes:
                e = int(be)
                n1, n2 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
                ue = 0.5 * (U[k][n1] + U[k][n2])
                ve = 0.5 * (V[k][n1] + V[k][n2])
                he = 0.5 * (H[k][n1] + H[k][n2])
                nx, ny = mesh.edge_normal[e]
                q += (ue * nx + ve * ny) * he * mesh.edge_len[e]
            row += f"  {q:14.6f}"
        print(row)

    # attribution verdict: inflow chain vs prescribed Q
    inflow = None
    for ci, ch in liquid:
        bes = mesh.boundary_edges[chain_of == ci]
        e = int(bes[0])
        n1, n2 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        he0 = 0.5 * (H[-1][n1] + H[-1][n2])
        # the inflow carries water INTO the domain (negative outward flux);
        # identify it against the prescribed Q sign at the last frame
        q = 0.0
        for be in bes:
            ee = int(be)
            a, b = int(mesh.edges[ee, 0]), int(mesh.edges[ee, 1])
            ue = 0.5 * (U[-1][a] + U[-1][b])
            ve = 0.5 * (V[-1][a] + V[-1][b])
            he = 0.5 * (H[-1][a] + H[-1][b])
            nx, ny = mesh.edge_normal[ee]
            q += (ue * nx + ve * ny) * he * mesh.edge_len[ee]
        if q < 0.0 and (inflow is None or abs(q) > abs(inflow[1])):
            inflow = (ci, q)
    if inflow is None:
        print("\nno inflowing liquid chain found at the last frame")
        return 1
    ci, q_in = inflow
    ratio = abs(q_in) / q_pres[-1]
    print(f"\ninflow chain{ci + 1}: {abs(q_in):.4f} m^3/s at t = "
          f"{times[-1]:.1f} vs prescribed {q_pres[-1]:.4f} "
          f"-> ratio = {ratio:.5f}")
    if abs(ratio - 1.0) < 0.02:
        print("VERDICT: the reference INFLOW equals its prescription; the "
              "'1.89x over-delivery' attribution is REFUTED (that number "
              "was the OUTLET chain -- the reference is net DRAINING at "
              "t_end, i.e. a slower-receding flood, not an over-injecting "
              "boundary).")
        branch = "A"
    else:
        print(f"VERDICT: the reference over-delivers by {ratio:.2f}x -- "
              "branch B (drive our inflow with the measured delivery).")
        branch = "B"

    if args.scale_series:
        if branch != "B":
            print("--scale-series skipped: only meaningful on branch B")
            return 0
        scale = ratio
        out = Path(args.scale_series)
        with open(out, "w", encoding="utf-8") as f:
            f.write("# culm inflow scaled to the TELEMAC reference delivery "
                    f"(x{scale:.4f})\n")
            for t, q in zip(qsl_t, qsl_q):
                f.write(f"{t:.1f} {scale * q:.6f}\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
