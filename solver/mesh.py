"""Unstructured triangular mesh: generator, SELAFIN reader, edge topology.

Cell-centered finite volume on triangles.  Edges carry a unit normal
oriented from cell A (``edge_cells[e, 0]``) to cell B (``edge_cells[e, 1]``);
on boundary edges cell B is ``-1`` and the normal points outward.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class TriMesh:
    nodes: np.ndarray  # (n_nodes, 2) float64
    tris: np.ndarray  # (n_tris, 3) int32, node indices
    cell_centroids: np.ndarray  # (n_tris, 2) float64
    cell_area: np.ndarray  # (n_tris,) float64
    cell_edges: np.ndarray  # (n_tris, 3) int32, edge indices per cell
    edges: np.ndarray  # (n_edges, 2) int32, node indices (ascending)
    edge_cells: np.ndarray  # (n_edges, 2) int32, cell A/B (-1 at boundary)
    edge_normal: np.ndarray  # (n_edges, 2) float64, unit, cellA -> cellB
    edge_len: np.ndarray  # (n_edges,) float64
    edge_midpoint: np.ndarray  # (n_edges, 2) float64
    boundary_edges: np.ndarray  # (n_bedges,) int32 indices into edge arrays

    @property
    def n_cells(self) -> int:
        return len(self.tris)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_bedges(self) -> int:
        return len(self.boundary_edges)


def build_topology(nodes: np.ndarray, tris: np.ndarray) -> TriMesh:
    """Build edges, cell/edge adjacency and geometric quantities."""
    nodes = np.asarray(nodes, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int32)
    n_cells = len(tris)

    # Unique edges keyed by ascending node pair.
    edge_keys = {}
    cell_edges = np.empty((n_cells, 3), dtype=np.int32)
    pair_list = []
    for c, (n0, n1, n2) in enumerate(tris):
        for k, (a, b) in enumerate(((n0, n1), (n1, n2), (n2, n0))):
            key = (a, b) if a < b else (b, a)
            eid = edge_keys.get(key)
            if eid is None:
                eid = len(pair_list)
                edge_keys[key] = eid
                pair_list.append(key)
            cell_edges[c, k] = eid
    edges = np.array(pair_list, dtype=np.int32)
    n_edges = len(edges)

    edge_cells = np.full((n_edges, 2), -1, dtype=np.int32)
    for c in range(n_cells):
        for k in range(3):
            e = cell_edges[c, k]
            if edge_cells[e, 0] == -1:
                edge_cells[e, 0] = c
            else:
                edge_cells[e, 1] = c

    x = nodes[edges[:, 0]]
    y = nodes[edges[:, 1]]
    evec = y - x
    edge_len = np.sqrt(np.einsum("ij,ij->i", evec, evec))
    edge_midpoint = 0.5 * (x + y)

    # Perpendicular of the edge vector; orientation fixed below.
    nvec = np.stack([evec[:, 1], -evec[:, 0]], axis=1)

    cell_centroids = nodes[tris].mean(axis=1)
    tri_x = nodes[tris]
    cell_area = 0.5 * np.abs(
        (tri_x[:, 1, 0] - tri_x[:, 0, 0]) * (tri_x[:, 2, 1] - tri_x[:, 0, 1])
        - (tri_x[:, 2, 0] - tri_x[:, 0, 0]) * (tri_x[:, 1, 1] - tri_x[:, 0, 1])
    )

    # Orient normals from cell A to cell B (outward on boundary edges).
    for e in range(n_edges):
        a = edge_cells[e, 0]
        if a == -1:  # degenerate: should not happen for well-built meshes
            a = edge_cells[e, 1]
            nvec[e] = -nvec[e]
            edge_cells[e, 0], edge_cells[e, 1] = a, -1
            continue
        d = cell_centroids[a]
        b = edge_cells[e, 1]
        ref = (cell_centroids[b] - d) if b != -1 else (edge_midpoint[e] - d)
        if np.dot(ref, nvec[e]) < 0.0:
            nvec[e] = -nvec[e]
    edge_normal = nvec / edge_len[:, None]

    boundary_edges = np.nonzero(edge_cells[:, 1] == -1)[0].astype(np.int32)

    return TriMesh(
        nodes=nodes,
        tris=tris,
        cell_centroids=cell_centroids,
        cell_area=cell_area,
        cell_edges=cell_edges,
        edges=edges,
        edge_cells=edge_cells,
        edge_normal=edge_normal,
        edge_len=edge_len,
        edge_midpoint=edge_midpoint,
        boundary_edges=boundary_edges,
    )


def rect_mesh(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    dx: float,
    diagonal: str = "alternating",
) -> TriMesh:
    """Triangulate an axis-aligned rectangle.

    Quads are split into two triangles; ``diagonal='alternating'`` flips the
    diagonal in a checkerboard pattern (keeps symmetric cases symmetric),
    ``'fixed'`` always uses the same diagonal.
    """
    dx = float(dx)
    nx = max(1, int(round((xmax - xmin) / dx)))
    ny = max(1, int(round((ymax - ymin) / dx)))
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([gx.ravel(order="C"), gy.ravel(order="C")], axis=1)

    def vid(i: int, j: int) -> int:
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            n00, n10 = vid(i, j), vid(i + 1, j)
            n01, n11 = vid(i, j + 1), vid(i + 1, j + 1)
            flip = (i + j) % 2 == 1 if diagonal == "alternating" else False
            if flip:
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
            else:
                tris.append((n00, n10, n01))
                tris.append((n10, n11, n01))
    return build_topology(nodes, np.array(tris, dtype=np.int32))


def classify_boundaries(mesh: TriMesh, groups: list) -> np.ndarray:
    """Assign a group index to every boundary edge.

    ``groups`` is a list of dicts with ``select`` following the YAML schema,
    e.g. ``{"xmin": [0.0, 0.45]}`` selects boundary edges lying on the
    xmin side of the domain with midpoint y in [0, 0.45].  Returns
    ``(n_bedges,) int32`` group indices; raises on unmatched/ambiguous edges.
    """
    if mesh.n_bedges == 0:
        return np.zeros(0, dtype=np.int32)
    mids = mesh.edge_midpoint[mesh.boundary_edges]
    dom = {
        "xmin": float(mesh.nodes[:, 0].min()),
        "xmax": float(mesh.nodes[:, 0].max()),
        "ymin": float(mesh.nodes[:, 1].min()),
        "ymax": float(mesh.nodes[:, 1].max()),
    }
    scale = max(dom["xmax"] - dom["xmin"], dom["ymax"] - dom["ymin"])
    tol = 1e-6 * scale

    group_of = np.full(mesh.n_bedges, -1, dtype=np.int32)
    for gi, grp in enumerate(groups):
        sel = grp["select"]
        (side, span), = sel.items()
        for e in range(mesh.n_bedges):
            if group_of[e] != -1:
                continue
            mx, my = mids[e]
            if side in ("xmin", "xmax"):
                on_line = abs(mx - dom[side]) < tol
                in_span = span[0] - tol <= my <= span[1] + tol
            else:
                on_line = abs(my - dom[side]) < tol
                in_span = span[0] - tol <= mx <= span[1] + tol
            if on_line and in_span:
                group_of[e] = gi
    if np.any(group_of == -1):
        missing = mesh.boundary_edges[group_of == -1]
        pts = mesh.edge_midpoint[missing]
        raise ValueError(
            f"{len(missing)} boundary edge(s) not matched by any boundary "
            f"group, e.g. midpoint {pts[0].tolist()}"
        )
    return group_of


# ---------------------------------------------------------------------------
# SELAFIN (.slf) reader — enough of the format to load TELEMAC geometry files.
# All records are big-endian, wrapped in 4-byte length markers.
# ---------------------------------------------------------------------------


def _read_record(f):
    """Return (payload_bytes, declared_len) of one SELAFIN record."""
    header = f.read(4)
    if len(header) < 4:
        return None, 0
    (n,) = struct.unpack(">i", header)
    payload = f.read(n)
    (n2,) = struct.unpack(">i", f.read(4))
    if n != n2:
        raise ValueError(f"SELAFIN record marker mismatch: {n} != {n2}")
    return payload, n


def read_selafin(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a SELAFIN geometry file.

    Returns ``(nodes_xy float64 (n,2), tris int32 (m,3), bottom float64 (n,))``
    where ``bottom`` is the variable named FOND/BOTTOM if present, else zeros.

    Layout notes (validated against TELEMAC v8p4 example geometry files):
    the variable-count block may be a single ``[NBV, 0]`` int record or the
    classic pair (``NBV`` then ``NBV+2`` ints); variable names are 32-byte
    records holding 16 name chars + 16 unit chars.
    """
    with open(path, "rb") as f:
        payload, _ = _read_record(f)
        if len(payload) not in (72, 80):
            raise ValueError(f"{path}: unexpected SELAFIN title length {len(payload)}")

        def int_record() -> np.ndarray:
            buf, _ = _read_record(f)
            return np.frombuffer(buf, dtype=">i4")

        r = int_record()
        if len(r) == 1:
            nvar = int(r[0])
            head = int_record()  # classic: nvar+2 ints, first = space dim
            if len(head) != nvar + 2:
                raise ValueError(f"{path}: unexpected variable header {head[:5].tolist()}")
        else:
            nvar = int(r[0])  # combined layout: [NBV, 0]
        var_names = [
            _read_record(f)[0][:16].decode("ascii", errors="replace").strip()
            for _ in range(nvar)
        ]

        int_record()  # IPARAM block (10 ints)
        eh = int_record()
        nelem, npoin, ndp = int(eh[0]), int(eh[1]), int(eh[2])
        if ndp != 3:
            raise ValueError(f"{path}: only triangular SELAFIN supported (ndp={ndp})")

        ikle = np.frombuffer(_read_record(f)[0], dtype=">i4").reshape(nelem, ndp)
        np.frombuffer(_read_record(f)[0], dtype=">i4")  # IPOBO (boundary nodes)
        x = np.frombuffer(_read_record(f)[0], dtype=">f4")
        y = np.frombuffer(_read_record(f)[0], dtype=">f4")
        if len(x) != npoin or len(y) != npoin:
            raise ValueError(f"{path}: coordinate record size mismatch")

        # Dynamic part: skip time-marker records, then read the first frame
        # of every variable; take FOND/BOTTOM as bathymetry when present.
        bottom = np.zeros(npoin, dtype=np.float64)
        frame_bytes = npoin * 4
        for name in var_names:
            payload, n = _read_record(f)
            while payload is not None and n != frame_bytes:
                payload, n = _read_record(f)
            if payload is None:
                break
            if name.upper().startswith(("FOND", "BOTTOM")):
                bottom = np.frombuffer(payload, dtype=">f4").astype(np.float64)

    nodes = np.stack([x.astype(np.float64), y.astype(np.float64)], axis=1)
    tris = (ikle - 1).astype(np.int32)  # SELAFIN numbering is 1-based
    if tris.min() < 0 or tris.max() >= npoin:
        raise ValueError(f"{path}: connectivity out of range")
    return nodes, tris, bottom
