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
    bottom: np.ndarray | None = None  # (n_nodes,) float64, SELAFIN FOND

    @property
    def n_cells(self) -> int:
        return len(self.tris)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_bedges(self) -> int:
        return len(self.boundary_edges)


def build_topology(nodes: np.ndarray, tris: np.ndarray,
                   bottom: np.ndarray | None = None) -> TriMesh:
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
        bottom=(np.asarray(bottom, dtype=np.float64)
                if bottom is not None else None),
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


def classify_boundaries(mesh: TriMesh, groups: list,
                        chain_of_bedge: np.ndarray | None = None) -> np.ndarray:
    """Assign a group index to every boundary edge.

    ``groups`` is a list of dicts with ``select`` following the YAML schema.
    Two selection forms exist:

    - rectangle sides: ``{"xmin": [0.0, 0.45]}`` selects boundary edges on
      the xmin side of the domain with midpoint y in [0, 0.45];
    - .cli chains: ``{"chain": k}`` (1-based) selects the boundary edges of
      the k-th TELEMAC boundary chain parsed from the .cli file; requires
      ``chain_of_bedge`` (from :func:`cli_chains`).

    Returns ``(n_bedges,) int32`` group indices; raises on unmatched or
    ambiguous edges.
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
        if "chain" in sel:
            if chain_of_bedge is None:
                raise ValueError(
                    f"group {grp.get('name')!r}: chain select needs a "
                    f".cli chain map (mesh built without a .cli file)")
            sel_chains = (sel["chain"] if isinstance(sel["chain"], list)
                          else [sel["chain"]])
            n_chains = int(chain_of_bedge.max()) + 1
            if sel_chains == ["rest"]:
                # every chain not named by an earlier group
                taken = set()
                for prev in groups[:gi]:
                    psel = prev.get("select") or {}
                    if "chain" in psel:
                        pc = psel["chain"]
                        pc = pc if isinstance(pc, list) else [pc]
                        if pc != ["rest"]:
                            taken.update(int(k) for k in pc)
                sel_chains = [k for k in range(1, n_chains + 1)
                              if k not in taken]
            for k in sel_chains:
                k = int(k)
                _require_chain(k, n_chains, grp.get("name"))
                group_of[chain_of_bedge == k - 1] = gi
            continue
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


def _require_chain(k: int, n_chains: int, name: str | None) -> None:
    if not 1 <= k <= n_chains:
        raise ValueError(
            f"boundary group {name!r}: select chain {k} out of range "
            f"(.cli provides {n_chains} chain(s))")


# ---------------------------------------------------------------------------
# TELEMAC .cli boundary file: parsing and boundary-chain grouping.
# Each line holds 13 free-format fields (see get_bnd_value_srf in
# utils_serafin.F): LIHBOR LIUBOR LIVBOR HBOR UBOR VBOR CHBORD LITBOR TBOR
# ATBOR BTBOR NODE BORD; '/' lines are comments.  LIHBOR: 2 = wall,
# 4 = prescribed elevation (KENT), 5 = free (KSORT).
# ---------------------------------------------------------------------------


def read_cli(path: str) -> dict:
    """Parse a TELEMAC .cli boundary file.

    Returns a dict of 0-based arrays: ``lihbor/liubor/livbor`` (int32),
    ``hbor/ubor/vbor/chbord`` (float64) and ``node`` (int32, mesh node
    number, converted from 1-based).
    """
    cols = {"lihbor": [], "liubor": [], "livbor": [],
            "hbor": [], "ubor": [], "vbor": [], "chbord": [], "node": []}
    nline = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            nline += 1
            s = line.strip()
            if not s or s.startswith("/"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 13:
                raise ValueError(f"{path}: line {nline}: expected 13 "
                                 f"fields, got {len(parts)}")
            try:
                cols["lihbor"].append(int(parts[0]))
                cols["liubor"].append(int(parts[1]))
                cols["livbor"].append(int(parts[2]))
                cols["hbor"].append(float(parts[3]))
                cols["ubor"].append(float(parts[4]))
                cols["vbor"].append(float(parts[5]))
                cols["chbord"].append(float(parts[6]))
                cols["node"].append(int(parts[11]) - 1)
            except ValueError:
                raise ValueError(f"{path}: line {nline}: malformed: {s!r}")
    if not cols["node"]:
        raise ValueError(f"{path}: no boundary conditions found")
    return {k: np.asarray(v) for k, v in cols.items()}


def cli_chains(mesh: TriMesh, cli: dict) -> tuple[np.ndarray, list]:
    """Group boundary edges into the TELEMAC boundary chains of a .cli walk.

    Two-pass grouping: pass 1 pairs consecutive .cli nodes that share a
    boundary edge (walk order); pass 2 splits the paired edges into
    chains where the boundary type changes, so each wall / liquid segment
    becomes its own chain.  A shared edge at a type change belongs to the
    wall side (a liquid-boundary integral never swallows a wall edge);
    between two liquid types the incoming side wins.  Chains are numbered
    by first appearance (1-based in yaml select).  Returns
    ``(chain_of_bedge (n_bedges,) int32 0-based, chains)`` where each
    chain dict carries ``nodes`` and ``lihbor`` of its walk entries.
    """
    nodes = np.asarray(cli["node"], dtype=np.int64)
    lih = np.asarray(cli["lihbor"], dtype=np.int64)
    if nodes.min() < 0 or nodes.max() >= mesh.n_nodes:
        raise ValueError(
            f".cli references node(s) outside the mesh "
            f"[{nodes.min()}, {nodes.max()}] vs n_nodes {mesh.n_nodes}")

    bedge_key = {}
    for be in range(mesh.n_bedges):
        e = int(mesh.boundary_edges[be])
        a, b = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        bedge_key[(a, b)] = be

    # pass 1: first walk pairing of every boundary edge (the repeated
    # closing node of a closed loop pairs its edge twice; keep the first)
    pair = {}  # be -> (k, prev, node)
    for k in range(1, len(nodes)):
        prev, node = int(nodes[k - 1]), int(nodes[k])
        key = (prev, node) if prev < node else (node, prev)
        be = bedge_key.get(key)
        if be is not None and be not in pair:
            pair[be] = (k, prev, node)

    # pass 2: chains split where the walk jumps (non-edge step) or the
    # boundary type changes
    chain_of = np.full(mesh.n_bedges, -1, dtype=np.int32)
    chains: list = []
    cur_type = None
    prev_k = -10
    for be, (k, prev, node) in pair.items():  # walk order (insertion)
        ta, tb = int(lih[k - 1]), int(lih[k])
        t = ta if ta == tb else (2 if 2 in (ta, tb) else ta)
        if chain_of[be] != -1:
            continue
        if t != cur_type or k > prev_k + 1:  # type change / walk jump
            cur_type = t
            chains.append({"nodes": [], "lihbor": [], "type": int(t)})
        prev_k = k
        ci = len(chains) - 1
        chain_of[be] = ci
        chains[ci]["nodes"] += [prev, node]
        chains[ci]["lihbor"] += [ta, tb]

    # Edges still unpaired: the closing edge of a loop walked without
    # repeating its first node, or an edge touching a single-point .cli
    # entry (a corner point listed alone).  Assign by node membership;
    # when the ends sit in different chains the wall side wins, else the
    # first chain in walk order.
    if len(pair) != mesh.n_bedges:
        node_chain = {nd: ci
                      for ci, ch in enumerate(chains) for nd in ch["nodes"]}
        for be in np.nonzero(chain_of == -1)[0]:
            e = int(mesh.boundary_edges[be])
            a, b = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
            ca, cb = node_chain.get(a), node_chain.get(b)
            if ca is not None and (ca == cb or cb is None):
                chain_of[be] = ca
            elif cb is not None:
                if ca is None or 2 in chains[cb]["lihbor"]:
                    chain_of[be] = cb
                else:
                    chain_of[be] = min(ca, cb)
            else:
                mp = mesh.edge_midpoint[[
                    int(mesh.boundary_edges[be])]].tolist()
                raise ValueError(
                    f".cli walk leaves {int((chain_of == -1).sum())} boundary "
                    f"edge(s) unassigned (first midpoint {mp[0]}); .cli and "
                    f"mesh do not describe the same boundary")
    return chain_of, chains


# ---------------------------------------------------------------------------
# TELEMAC roughness zones: geo.bfr ("node_number zone_id" per row, 1-based,
# one row per mesh node) + roughness.tbl ("zone_id LAW value [NULL]" rows,
# '*' comments, END terminator).  Only the NIKURADSE law is supported
# (k_friction mode 2 carries ks); the zone-id digits carry no meaning
# (donau: 99999010 -> 10.0 m), the mapping must come from the table.
# ---------------------------------------------------------------------------


def read_bfr_zones(path: str, n_nodes: int) -> np.ndarray:
    """Parse a TELEMAC .bfr zone file -> zone id per node (0-based array).

    Every mesh node must be covered exactly by the "node zone" rows.
    """
    zone = np.full(n_nodes, -1, dtype=np.int64)
    nline = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            nline += 1
            s = line.strip()
            if not s or s.startswith(("*", "/", "#")):
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"{path}: line {nline}: expected "
                                 f"'node zone', got {s!r}")
            try:
                node = int(parts[0]) - 1
                zid = int(parts[1])
            except ValueError:
                raise ValueError(f"{path}: line {nline}: malformed: {s!r}")
            if not 0 <= node < n_nodes:
                raise ValueError(f"{path}: line {nline}: node {node + 1} "
                                 f"outside the mesh (1..{n_nodes})")
            if zone[node] != -1:
                raise ValueError(f"{path}: line {nline}: node {node + 1} "
                                 f"listed twice")
            zone[node] = zid
    missing = int(np.count_nonzero(zone == -1))
    if missing:
        raise ValueError(f"{path}: {missing} of {n_nodes} node(s) not "
                         f"covered by the zone file")
    return zone


def read_roughness_tbl(path: str) -> dict:
    """Parse a TELEMAC roughness table -> {zone_id: ks} (NIKURADSE rows)."""
    out: dict = {}
    nline = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            nline += 1
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            if s.upper().startswith("END"):
                break
            parts = s.split()
            if len(parts) < 3:
                raise ValueError(f"{path}: line {nline}: expected "
                                 f"'zone law value', got {s!r}")
            try:
                zid = int(parts[0])
            except ValueError:
                continue  # legend / comment row without '*' (donau header)
            if parts[1].upper() != "NIKU":
                raise ValueError(f"{path}: line {nline}: only the Nikuradse "
                                 f"law (NIKU) is supported, got {parts[1]!r}")
            try:
                out[zid] = float(parts[2])
            except ValueError:
                raise ValueError(f"{path}: line {nline}: malformed: {s!r}")
    if not out:
        raise ValueError(f"{path}: no NIKU roughness entries found")
    return out


def zone_ks_for_cells(mesh: TriMesh, zone_of_node: np.ndarray,
                      zone_ks: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell Nikuradse ks from node zones.

    Majority vote of the three corner zones; a tie (3 distinct zones) is
    resolved by the node nearest the cell centroid (deterministic; donau:
    57 of 93209 cells).  Returns ``(ks (n_cells,) float64,
    zone_of_cell (n_cells,) int64)``.
    """
    tri_zones = zone_of_node[mesh.tris]  # (n, 3)
    n_cells = len(tri_zones)
    zone_of_cell = np.empty(n_cells, dtype=np.int64)
    for c in range(n_cells):
        z = tri_zones[c]
        vals, counts = np.unique(z, return_counts=True)
        if len(vals) == 1 or counts.max() >= 2:
            zone_of_cell[c] = vals[np.argmax(counts)]
        else:
            nds = mesh.tris[c]
            cen = mesh.cell_centroids[c]
            k = np.argmin(((mesh.nodes[nds] - cen) ** 2).sum(axis=1))
            zone_of_cell[c] = z[k]
    unknown = sorted(set(zone_of_cell.tolist()) - set(zone_ks))
    if unknown:
        raise ValueError(
            f"zone(s) {unknown} present in the mesh but missing from the "
            f"roughness table {sorted(zone_ks)}")
    ks = np.array([zone_ks[int(z)] for z in zone_of_cell], dtype=np.float64)
    return ks, zone_of_cell


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


def _selafin_header(f, path: str):
    """Parse a SELAFIN header; returns (var_names, ikle, x, y)."""
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

    ip = int_record()  # IPARAM block (10 ints)
    if len(ip) >= 10 and int(ip[9]) != 0:
        int_record()  # 6-int starting-date block (results files)
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
    return var_names, ikle, x, y


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
        var_names, ikle, x, y = _selafin_header(f, path)
        npoin = len(x)

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


def read_selafin_series(path: str, wanted: list) -> dict:
    """Read every time frame of the named variables from a SELAFIN
    results file (TELEMAC reference runs).

    Returns ``{"times": (n_t,), "nodes": (n,2), "tris": (m,3),
    "series": {var: (n_t, n) float64}}`` -- ``wanted`` entries are matched
    against the file's variable names case-insensitively by prefix.
    """
    with open(path, "rb") as f:
        var_names, ikle, x, y = _selafin_header(f, path)
        npoin = len(x)
        # TELEMAC short variable codes -> full SELAFIN names
        alias = {"H": "WATER DEPTH", "S": "FREE SURFACE", "U": "VELOCITY U",
                 "V": "VELOCITY V", "B": "BOTTOM", "E": "FREE SURFACE"}
        idx = {}
        for want in wanted:
            key = alias.get(want.upper(), want.upper())
            hit = [i for i, nm in enumerate(var_names)
                   if nm.upper().startswith(key)]
            if not hit:
                hit = [i for i, nm in enumerate(var_names)
                       if nm.upper().startswith(want.upper())]
            if not hit:
                raise ValueError(
                    f"{path}: variable {want!r} not found in {var_names}")
            idx[want] = hit[0]

        times = []
        cols = {w: [] for w in wanted}
        frame_bytes = npoin * 4
        per_frame = [None] * len(var_names)
        while True:
            payload, n = _read_record(f)
            if payload is None:
                break
            if n not in (4, 8):  # not a time marker -> truncated
                break
            (t,) = struct.unpack(">d" if n == 8 else ">f", payload)
            missing = False
            for vi in range(len(var_names)):
                buf, n2 = _read_record(f)
                if n2 != frame_bytes:
                    missing = True
                    break
                per_frame[vi] = np.frombuffer(buf, dtype=">f4")
            if missing:
                break
            times.append(t)
            for w in wanted:
                cols[w].append(per_frame[idx[w]].astype(np.float64))

    if not times:
        raise ValueError(f"{path}: no complete time frames found")
    nodes = np.stack([x.astype(np.float64), y.astype(np.float64)], axis=1)
    tris = (ikle - 1).astype(np.int32)
    return {"times": np.asarray(times, dtype=np.float64),
            "nodes": nodes, "tris": tris,
            "series": {w: np.asarray(v) for w, v in cols.items()}}


def nearest_index(xy: np.ndarray, ref: np.ndarray,
                  chunk: int = 1024) -> np.ndarray:
    """Index of the nearest ``ref`` row for every ``xy`` row.

    Brute force in row blocks so the distance matrix stays bounded
    (donau: 93k cells x 47.5k reference nodes would need 33 GiB in one
    piece).
    """
    out = np.empty(len(xy), dtype=np.int64)
    rx = ref[:, 0]
    ry = ref[:, 1]
    for s in range(0, len(xy), chunk):
        e = min(s + chunk, len(xy))
        d2 = ((xy[s:e, 0][:, None] - rx[None, :]) ** 2
              + (xy[s:e, 1][:, None] - ry[None, :]) ** 2)
        out[s:e] = np.argmin(d2, axis=1)
    return out
