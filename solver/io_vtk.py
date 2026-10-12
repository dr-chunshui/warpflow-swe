"""Minimal .vtu (XML, inline base64) + .pvd writer for ParaView."""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np


def _b64(data: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(data).tobytes()).decode("ascii")


def _cell_block(tris: np.ndarray) -> tuple[str, str]:
    n = len(tris)
    conn = np.concatenate([np.full((n, 1), 3, dtype=np.int32), tris], axis=1).ravel()
    offsets = np.arange(1, n + 1, dtype=np.int32) * 3
    return _b64(conn), _b64(offsets)


def write_vtu(
    path: str | Path,
    nodes: np.ndarray,  # (n, 2) xy
    tris: np.ndarray,  # (m, 3) int32
    point_z: np.ndarray,  # (n,) bottom elevation for the surface mesh
    cell_data: dict,  # name -> (m,) float64 arrays
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, m = len(nodes), len(tris)
    conn, offs = _cell_block(tris)
    node3 = np.column_stack([nodes, point_z])

    parts = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
        "  <UnstructuredGrid>",
        f'    <Piece NumberOfPoints="{n}" NumberOfCells="{m}">',
        "      <Points>",
        '        <DataArray type="Float64" NumberOfComponents="3" format="binary">'
        + _b64(node3.ravel()) + "</DataArray>",
        "      </Points>",
        "      <Cells>",
        f'        <DataArray type="Int32" Name="connectivity" format="binary">{conn}</DataArray>',
        f'        <DataArray type="Int32" Name="offsets" format="binary">{offs}</DataArray>',
        '        <DataArray type="UInt8" Name="types" format="binary">'
        + _b64(np.full(m, 7, dtype=np.uint8)) + "</DataArray>",
        "      </Cells>",
        "      <CellData>",
    ]
    for name, arr in cell_data.items():
        parts.append(
            f'        <DataArray type="Float64" Name="{name}" format="binary">'
            + _b64(np.ascontiguousarray(arr, dtype=np.float64)) + "</DataArray>"
        )
    parts += ["      </CellData>", "    </Piece>", "  </UnstructuredGrid>", "</VTKFile>", ""]
    path.write_text("\n".join(parts), encoding="ascii")


def write_pvd(path: str | Path, frames: list) -> None:
    """frames: list of (time, vtu_filename_relative_to_pvd_dir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="1.0">',
        "  <Collection>",
    ]
    for t, fname in frames:
        parts.append(f'    <DataSet timestep="{t!r}" part="0" file="{fname}"/>')
    parts += ["  </Collection>", "</VTKFile>", ""]
    path.write_text("\n".join(parts), encoding="ascii")
