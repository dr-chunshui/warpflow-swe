"""Run a case: python run_case.py test_cases/<case>/<config>.yaml"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver.config import load_config
from solver.mesh import rect_mesh, read_selafin, build_topology
from solver.solver import ShallowWaterSolver


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cfg = load_config(sys.argv[1])

    if cfg.mesh.type == "rect":
        mesh = rect_mesh(cfg.mesh.xmin, cfg.mesh.xmax,
                         cfg.mesh.ymin, cfg.mesh.ymax,
                         cfg.mesh.dx, cfg.mesh.diagonal)
    else:
        nodes, tris, _ = read_selafin(cfg.mesh.file)
        mesh = build_topology(nodes, tris)

    print(f"case {cfg.name}: {mesh.n_cells} cells, {mesh.n_edges} edges, "
          f"{mesh.n_bedges} boundary edges, t_end = {cfg.t_end} s")
    solver = ShallowWaterSolver(cfg, mesh)
    solver.run()

    if cfg.analytic:
        if not cfg.output.centerline_y:
            print("analytic set but output.centerline_y missing: skip compare")
        else:
            from solver.analysis import compare_case
            compare_case(cfg, mesh, solver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
