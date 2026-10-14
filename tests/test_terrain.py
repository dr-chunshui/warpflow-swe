"""Batch-1 capability tests: SELAFIN-carried bottom + line reservoir IC.

Run directly:  python tests/test_terrain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.config import load_config  # noqa: E402
from solver.mesh import build_topology, read_selafin, rect_mesh  # noqa: E402
from solver.state import build_bathymetry, build_initial  # noqa: E402

SOLF_DIR = Path(r"D:\Workspace\warpflow\telemac2d\examples")
MALP = SOLF_DIR / "malpasset" / "geo_malpasset-small.slf"


def main() -> int:
    # --- real malpasset geometry: FOND carried through build_topology ---
    nodes, tris, bottom = read_selafin(str(MALP))
    mesh = build_topology(nodes, tris, bottom)
    assert mesh.bottom is not None and len(mesh.bottom) == len(nodes)
    assert abs(float(mesh.bottom.min()) + 20.0) < 1e-4
    assert abs(float(mesh.bottom.max()) - 100.0) < 1e-4
    # centroid bottom is the P1 mean of the triangle's node values
    zb = build_bathymetry(_selafin_cfg(), mesh)
    zb_ref = mesh.bottom[mesh.tris].mean(axis=1)
    assert np.array_equal(zb, zb_ref)
    print(f"selafin bottom ok: {len(nodes)} nodes, "
          f"zb in [{zb.min():.2f}, {zb.max():.2f}] m")

    # --- reservoir_behind_line IC (distan.f convention + hole) ---
    cfg = _reservoir_cfg()
    w = build_initial(cfg, mesh, zb)
    # dam line of the TELEMAC case; positive side holds zs = 100 m
    h = w[:, 0]
    wet = h > 0.0
    assert wet.sum() > 0
    vol = float((h * mesh.cell_area).sum())
    assert 4.0e7 < vol < 6.0e7, vol  # real Malpasset capacity ~50e6 m^3
    # hole at (4500, 5350) r=200 forced dry even on the wet side
    from solver.state import nearest_cell
    c_in = nearest_cell(mesh, 4500.0, 5350.0)
    assert h[c_in] == 0.0, h[c_in]
    # dry side: far downstream transformer B (11900, 3250)
    c_b = nearest_cell(mesh, 11900.0, 3250.0)
    assert h[c_b] == 0.0, h[c_b]
    print(f"reservoir IC ok: {int(wet.sum())} wet cells, "
          f"V0 = {vol:.3e} m^3")

    # --- nearest_index: exact on a toy set, chunked path exercised ---
    from solver.mesh import nearest_index
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    xy = np.array([[0.1, 0.1], [4.9, 5.1], [0.8, 0.0]])
    j = nearest_index(xy, ref, chunk=1)
    assert list(j) == [0, 3, 1], j
    print("nearest_index ok (chunk size 1)")

    # --- ramp_cutoff IC (swash_solution user_condin_h.f) ---
    mesh_r = rect_mesh(0.0, 8.0, 0.0, 0.5, 0.5, "alternating")
    zb_r = np.zeros(mesh_r.n_cells)
    cfg_r = _cfg_with_initial(
        "ramp_cutoff", "slope: 0.05\n      intercept: 0.0500625\n"
        "      x_cut: 6.0")
    w_r = build_initial(cfg_r, mesh_r, zb_r)
    xh = mesh_r.cell_centroids[:, 0]
    h_r = w_r[:, 0]
    wet = h_r > 0.0
    assert xh[wet].max() <= 6.25 and xh[wet].min() >= 0.0
    c_in = nearest_cell(mesh_r, 3.0, 0.25)
    xr = mesh_r.cell_centroids[c_in, 0]
    assert abs(h_r[c_in] - (0.0500625 + 0.05 * xr)) < 1e-12
    c_dry = nearest_cell(mesh_r, 7.0, 0.25)
    assert h_r[c_dry] == 0.0
    print(f"ramp_cutoff ok: h(3) = {h_r[c_in]:.7f}, dry beyond x = 6")

    # --- level_profile IC: nearest surveyed section wins ---
    secfile = Path("output") / "tmp_sections.txt"
    secfile.parent.mkdir(exist_ok=True)
    secfile.write_text(
        "#  xli  yli  xre  yre  level\n"
        "0.0 0.0 4.0 0.0 10.0\n"
        "4.0 0.0 8.0 0.0 20.0\n", encoding="utf-8")
    cfg_s = _cfg_with_initial(
        "level_profile",
        f'file: "{secfile.resolve().as_posix()}"',
        mesh_file=None)
    w_s = build_initial(cfg_s, mesh_r, zb_r)
    c1 = nearest_cell(mesh_r, 1.0, 0.25)
    c2 = nearest_cell(mesh_r, 7.0, 0.25)
    assert abs(w_s[c1, 0] - 10.0) < 1e-12
    assert abs(w_s[c2, 0] - 20.0) < 1e-12
    print("level_profile ok: sections split at x = 4")

    # --- selafin_state hot start: S/U/V sampled at the nearest node ---
    f2d = (Path(__file__).resolve().parents[1]
           / "test_cases" / "conical_island" / "f2d_island.slf")
    if f2d.is_file():
        cfg_h = _cfg_with_initial(
            "selafin_state", f'file: "{f2d.as_posix()}"',
            mesh_file=MALP.as_posix())
        # island mesh for a hot start over the malpasset geometry: the
        # assertion is that every cell picked *some* valid node value
        from solver.mesh import read_selafin as _rs
        n2, t2, b2 = _rs(str(MALP))
        mesh_m = build_topology(n2, t2, b2)
        zb_m = build_bathymetry(_selafin_cfg(), mesh_m)
        w_h = build_initial(cfg_h, mesh_m, zb_m)
        assert np.all(np.isfinite(w_h))
        print("selafin_state ok: hot start field finite everywhere")

    print("PASS: terrain / line-reservoir capability verified")
    return 0


def _cfg_with_initial(kind: str, params_yaml: str,
                      mesh_file: str | None = None):
    import yaml
    import tempfile
    mf = mesh_file if mesh_file is not None else str(MALP).replace("\\", "/")
    y = """
case: {name: t_init, t_end: 1.0}
mesh: {type: selafin, file: "%s"}
initial:
  type: %s
  params:
      %s
bathymetry: {type: selafin}
boundary:
  groups:
    - {name: wall, select: {chain: 1}, type: wall}
numerics: {cfl: 0.8}
output: {dir: output/tmp_test}
""" % (mf, kind, params_yaml)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(y)
        p = f.name
    return load_config(p)


def _selafin_cfg():
    import yaml
    import tempfile
    y = """
case: {name: t_terrain, t_end: 1.0}
mesh: {type: selafin, file: "%s"}
initial: {type: lake_at_rest, params: {zs: 0.0}}
bathymetry: {type: selafin}
boundary:
  groups:
    - {name: wall, select: {chain: 1}, type: wall}
numerics: {cfl: 0.8}
output: {dir: output/tmp_test}
""" % str(MALP).replace("\\", "/")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(y)
        p = f.name
    return load_config(p)


def _reservoir_cfg():
    import yaml
    import tempfile
    y = """
case: {name: t_reservoir, t_end: 1.0}
mesh: {type: selafin, file: "%s"}
initial:
  type: reservoir_behind_line
  params:
    zs: 100.0
    dam: [4701.183, 4143.407, 4655.553, 4392.104]
    holes: [[4500.0, 5350.0, 200.0]]
bathymetry: {type: selafin}
boundary:
  groups:
    - {name: wall, select: {chain: 1}, type: wall}
numerics: {cfl: 0.8}
output: {dir: output/tmp_test}
""" % str(MALP).replace("\\", "/")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(y)
        p = f.name
    return load_config(p)


if __name__ == "__main__":
    raise SystemExit(main())
