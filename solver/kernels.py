"""Warp kernels for the Godunov finite-volume shallow water solver.

Numerics mirror TELEMAC-2D v8p4 finite volume sources:

- HLLC flux:             flux_hllc.f (verbatim, including dry branches and
  the patched USTAR = 0.5(UL+UR) - (HR-HL)(AL+AR)/(HL+HR))
- Audusse hydrostatic reconstruction + geometric source terms:
                          hyd_fv.f   (DZIJ/HIJ, HGZI = 0.5*L*(h*+h)(h*-h))
- 2nd order MUSCL:       second_order.f (face depths corrected by the bottom
  increment, CORR_HL = DSH - DSZ; slope-source correction
  CORR_I = -0.5*L*(h_cell+h_face)*DSZ; per-edge fallback to 1st order when
  the bottom jump is large; boundary edges stay 1st order, like cdl_fv.f)
- CFL timestep:          caldt.f    (DTHAUT / max(0.01, c + |u|))
- Boundary ghost states: cdl_fv.f   (wall strong imposition, Riemann
  invariant ghost for level/discharge; bord.f elevation->depth conversion)

float64 discipline: every literal inside kernels is wp.float64(...);
gravity is the module constant G from numeric.py.
"""
from __future__ import annotations

import warp as wp

from .numeric import G, EPS_F64, U_MAX

# Boundary condition type codes (must match solver/config.py).
BC_WALL = 0
BC_OUTFLOW = 1
BC_INFLOW_Q = 2
BC_LEVEL = 3


# ---------------------------------------------------------------------------
# Primitive recovery: h<=eps -> zero velocity (hyd_fv.f EPS_FV guard);
# |velocity| clamped to U_MAX -- TELEMAC itself has no ceiling, but its
# cases never bottom a cell out mid-transient; ours did (see the
# transcritical bump) and the unbounded u = hu/h blew the HLLC flux up.
# Returns (h, u, v).
# ---------------------------------------------------------------------------
@wp.func
def primitive(hc: wp.float64, huc: wp.float64, hvc: wp.float64):
    h = wp.max(hc, wp.float64(0.0))
    if h <= EPS_F64:
        return h, wp.float64(0.0), wp.float64(0.0)
    u = huc / h
    v = hvc / h
    spd2 = u * u + v * v
    if spd2 > U_MAX * U_MAX:
        scale = U_MAX / wp.sqrt(spd2)
        u = u * scale
        v = v * scale
    return h, u, v


# ---------------------------------------------------------------------------
# HLLC Riemann flux in the edge-normal frame (mirrors flux_hllc.f).
# h1/h2 are the (possibly hydrostatically reconstructed) depths, u/v the
# physical-frame velocities; xnn/ynn the unit normal from side 1 to side 2.
# Returns the physical-frame flux (f_mass, f_mom_x, f_mom_y).
# ---------------------------------------------------------------------------
@wp.func
def hllc_flux(
    h1: wp.float64,
    h2: wp.float64,
    u1: wp.float64,
    v1: wp.float64,
    u2: wp.float64,
    v2: wp.float64,
    xnn: wp.float64,
    ynn: wp.float64,
):
    eps = EPS_F64
    gsur2 = G / wp.float64(2.0)

    # Rotation into the normal frame (flux_hllc.f).
    ul = xnn * u1 + ynn * v1
    vl = -ynn * u1 + xnn * v1
    ur = xnn * u2 + ynn * v2
    vr = -ynn * u2 + xnn * v2

    # Both sides dry -> zero flux (goto 20 in the Fortran).
    if h1 < eps and h2 < eps:
        return wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)

    hl = wp.max(h1, wp.float64(0.0))
    hr = wp.max(h2, wp.float64(0.0))
    al = wp.sqrt(G * hl)
    ar = wp.sqrt(G * hr)

    hstar = wp.float64(0.5) * (hl + hr) - wp.float64(0.25) * (ur - ul) * (hl + hr) / (al + ar)
    ustar = wp.float64(0.5) * (ul + ur) - (hr - hl) * (al + ar) / (hl + hr)

    # PQL/PQR: rarefaction -> 1, shock -> sqrt(0.5(hstar+h)hstar/h^2).
    pql = wp.float64(0.0)
    if hstar < hl:
        pql = wp.float64(1.0)
    else:
        if hl > eps:
            pql = wp.sqrt(wp.float64(0.5) * (hstar + hl) * hstar / (hl * hl))
    pqr = wp.float64(0.0)
    if hstar < hr:
        pqr = wp.float64(1.0)
    else:
        if hr > eps:
            pqr = wp.sqrt(wp.float64(0.5) * (hstar + hr) * hstar / (hr * hr))

    # Wave speeds SSL/SR with dry branches (flux_hllc.f lines 159-177).
    if hl > eps:
        ssl = ul - al * pql
        if hr > eps:
            sr = ur + ar * pqr
        else:
            ssl = ul - al
            sr = ul + wp.float64(2.0) * al
    else:
        ssl = ur - wp.float64(2.0) * ar
        sr = ur + ar

    denom = hr * (ur - sr) - hl * (ul - ssl)
    if wp.abs(denom) < eps:
        sstar = ustar
    else:
        sstar = (ssl * hr * (ur - sr) - sr * hl * (ul - ssl)) / denom

    # QSTARL / QSTARR middle-state conserved variables, with POND guard.
    pond = wp.float64(0.0)
    if wp.abs(ssl - sstar) > eps:
        pond = hl * (ssl - ul) / (ssl - sstar)
    qsl = pond
    qslm = pond * sstar
    qslt = pond * vl

    pond = wp.float64(0.0)
    if wp.abs(sr - sstar) > eps:
        pond = hr * (sr - ur) / (sr - sstar)
    qsr = pond
    qsrm = pond * sstar
    qsrt = pond * vr

    fl0 = hl * ul
    fl1 = hl * ul * ul + gsur2 * hl * hl
    fl2 = hl * ul * vl
    fr0 = hr * ur
    fr1 = hr * ur * ur + gsur2 * hr * hr
    fr2 = hr * ur * vr

    # Flux selection by wave speeds (FSTARL / FSTARR / FL / FR).
    if ssl > wp.float64(0.0):
        fx0 = fl0
        fx1 = fl1
        fx2 = fl2
    elif sstar > wp.float64(0.0) and ssl < wp.float64(0.0):
        fx0 = fl0 + ssl * (qsl - hl)
        fx1 = fl1 + ssl * (qslm - hl * ul)
        fx2 = fl2 + ssl * (qslt - hl * vl)
    elif sstar < wp.float64(0.0) and sr > wp.float64(0.0):
        fx0 = fr0 + sr * (qsr - hr)
        fx1 = fr1 + sr * (qsrm - hr * ur)
        fx2 = fr2 + sr * (qsrt - hr * vr)
    else:
        fx0 = fr0
        fx1 = fr1
        fx2 = fr2

    # Inverse rotation to the physical frame (flux_hllc.f).
    out0 = fx0
    out1 = xnn * fx1 - ynn * fx2
    out2 = ynn * fx1 + xnn * fx2
    return out0, out1, out2


# ---------------------------------------------------------------------------
# Boundary ghost state (mirrors cdl_fv.f).  Inputs: interior CELL-AVERAGE
# state (boundary edges stay first order, like the Fortran), cell bottom,
# bc type and per-edge parameter; outputs the ghost (h, u, v).
# ---------------------------------------------------------------------------
@wp.func
def ghost_state(
    bct: wp.int32,
    bparam: wp.float64,
    ha: wp.float64,
    ua: wp.float64,
    va: wp.float64,
    zba: wp.float64,
    nx: wp.float64,
    ny: wp.float64,
):
    eps = EPS_F64
    hg = ha
    ug = ua
    vg = va

    if bct == BC_WALL:
        # Reflecting wall: mirror the normal velocity.  The symmetric
        # Riemann problem gives exactly zero mass flux through the wall
        # and pressure f_n = 0.5 g h^2.  (Cell-centred FV: the boundary
        # flux enters the single adjacent cell, so the ghost MUST keep
        # mass balance; TELEMAC can afford strong un=0 imposition
        # because its FV control volumes are node-centred and boundary
        # values are reset afterwards in cdl_fv.f.)
        un = nx * ua + ny * va
        vt = -ny * ua + nx * va
        un = -un
        ug = nx * un - ny * vt
        vg = ny * un + nx * vt
        hg = ha
    elif bct == BC_OUTFLOW:
        # Zero gradient.
        pass
    elif bct == BC_LEVEL:
        # Prescribed free-surface elevation (bord.f: hg = max(0, zs-zb)).
        zs = bparam
        hg = wp.max(zs - zba, wp.float64(0.0))
        un = nx * ua + ny * va
        vt = -ny * ua + nx * va
        c1 = wp.sqrt(G * ha)
        cg = wp.sqrt(G * hg)
        regime = (un + c1) * (un - c1)
        if hg <= eps:
            ug = wp.float64(0.0)
            vg = wp.float64(0.0)
        elif regime < wp.float64(0.0):
            # subcritical: conserve the second Riemann invariant
            ugn = un + wp.float64(2.0) * (c1 - cg)
            ug = nx * ugn - ny * vt
            vg = ny * ugn + nx * vt
        elif un <= wp.float64(0.0):
            # supercritical inflow with missing data: lake at rest
            ug = wp.float64(0.0)
            vg = wp.float64(0.0)
        else:
            # supercritical outflow: no information (cdl_fv.f)
            hg = ha
            ug = wp.float64(0.0)
            vg = wp.float64(0.0)
    else:  # BC_INFLOW_Q
        # Prescribed discharge per unit width on this edge, signed with
        # the outward normal: q < 0 is inflow (mirrors cdl_fv.f where
        # UGN = U.n carries the direction and TELEMAC computes it as
        # Q/(h*L) from the current interior depth H1).
        q = bparam
        un = nx * ua + ny * va
        vt = -ny * ua + nx * va
        c1 = wp.sqrt(G * ha)
        regime = (un + c1) * (un - c1)
        if regime < wp.float64(0.0):
            # subcritical: ghost normal velocity from the current interior
            # depth; depth from conservation of the second Riemann invariant
            ugn = q / wp.max(ha, eps)
            hgt = un + wp.float64(2.0) * c1 - ugn
            hg = hgt * hgt / (wp.float64(4.0) * G)
            if hg <= eps:
                ug = wp.float64(0.0)
                vg = wp.float64(0.0)
            else:
                ug = nx * ugn - ny * vt
                vg = ny * ugn + nx * vt
        elif un <= wp.float64(0.0):
            # supercritical inflow without imposed depth: use interior depth
            hg = ha
            ugn = q / wp.max(ha, eps)
            ug = nx * ugn - ny * vt
            vg = ny * ugn + nx * vt
        else:
            # supercritical outflow: mirror cdl_fv.f (ghost at rest)
            hg = ha
            ug = wp.float64(0.0)
            vg = wp.float64(0.0)

    return hg, ug, vg


# ---------------------------------------------------------------------------
# Green-Gauss gradient of (w = h+zb free surface, u, v, zb) per cell (order 2).
# grad_c = 1/A sum_e 0.5(qA+qB) L n_out; boundary edges use the cell value.
# Reconstructing the free surface w (not the depth h) is what keeps the
# scheme exactly well-balanced at second order (Audusse 2004).
# ---------------------------------------------------------------------------
@wp.kernel
def k_green_gauss(
    w: wp.array(dtype=wp.vec3d),
    zb: wp.array(dtype=wp.float64),
    cell_area: wp.array(dtype=wp.float64),
    cell_edges: wp.array(dtype=wp.vec3i),
    edge_cells: wp.array(dtype=wp.vec2i),
    edge_normal: wp.array(dtype=wp.vec2d),
    edge_len: wp.array(dtype=wp.float64),
    grad_w: wp.array(dtype=wp.vec2d),
    grad_u: wp.array(dtype=wp.vec2d),
    grad_v: wp.array(dtype=wp.vec2d),
    grad_z: wp.array(dtype=wp.vec2d),
):
    c = wp.tid()
    wc = w[c]
    h0, uc, vc = primitive(wc[0], wc[1], wc[2])
    zc = zb[c]
    wsc = zc + wc[0]

    gxw = wp.float64(0.0)
    gyw = wp.float64(0.0)
    gxu = wp.float64(0.0)
    gyu = wp.float64(0.0)
    gxv = wp.float64(0.0)
    gyv = wp.float64(0.0)
    gxz = wp.float64(0.0)
    gyz = wp.float64(0.0)
    for k in range(3):
        e = cell_edges[c][k]
        cells = edge_cells[e]
        nb = cells[1]
        # normal is oriented A->B: outward for A, inward for B
        sgn = wp.float64(1.0)
        if cells[0] != c:
            sgn = wp.float64(-1.0)
        nx = sgn * edge_normal[e][0]
        ny = sgn * edge_normal[e][1]
        ln = edge_len[e]
        if nb < 0:
            we_w = wsc
            we_u = uc
            we_v = vc
            we_z = zc
        else:
            wb = w[nb]
            hb, ub, vb = primitive(wb[0], wb[1], wb[2])
            we_w = wp.float64(0.5) * (wsc + (zb[nb] + wb[0]))
            we_u = wp.float64(0.5) * (uc + ub)
            we_v = wp.float64(0.5) * (vc + vb)
            we_z = wp.float64(0.5) * (zc + zb[nb])
        gxw += we_w * ln * nx
        gyw += we_w * ln * ny
        gxu += we_u * ln * nx
        gyu += we_u * ln * ny
        gxv += we_v * ln * nx
        gyv += we_v * ln * ny
        gxz += we_z * ln * nx
        gyz += we_z * ln * ny
    inv_a = wp.float64(1.0) / cell_area[c]
    grad_w[c] = wp.vec2d(gxw * inv_a, gyw * inv_a)
    grad_u[c] = wp.vec2d(gxu * inv_a, gyu * inv_a)
    grad_v[c] = wp.vec2d(gxv * inv_a, gyv * inv_a)
    grad_z[c] = wp.vec2d(gxz * inv_a, gyz * inv_a)


# ---------------------------------------------------------------------------
# Barth-Jespersen limiter: per-field phi = min over interior edges of
# min(1, ratio) (ratio >= 0 by construction); slope = phi * grad.
# ---------------------------------------------------------------------------
@wp.kernel
def k_limit_barth(
    w: wp.array(dtype=wp.vec3d),
    zb: wp.array(dtype=wp.float64),
    grad_w: wp.array(dtype=wp.vec2d),
    grad_u: wp.array(dtype=wp.vec2d),
    grad_v: wp.array(dtype=wp.vec2d),
    grad_z: wp.array(dtype=wp.vec2d),
    cell_edges: wp.array(dtype=wp.vec3i),
    edge_cells: wp.array(dtype=wp.vec2i),
    slope_w: wp.array(dtype=wp.vec2d),
    slope_u: wp.array(dtype=wp.vec2d),
    slope_v: wp.array(dtype=wp.vec2d),
    slope_z: wp.array(dtype=wp.vec2d),
):
    c = wp.tid()
    wc = w[c]
    zc = zb[c]
    wsc = zc + wc[0]
    h0, uc, vc = primitive(wc[0], wc[1], wc[2])

    wm = wsc
    wx = wsc
    um = uc
    ux = uc
    vm = vc
    vx = vc
    zm = zc
    zx = zc
    for k in range(3):
        nb = edge_cells[cell_edges[c][k]][1]
        if nb >= 0:
            wb = w[nb]
            hb, ub, vb = primitive(wb[0], wb[1], wb[2])
            wnb = zb[nb] + wb[0]
            wm = wp.min(wm, wnb)
            wx = wp.max(wx, wnb)
            um = wp.min(um, ub)
            ux = wp.max(ux, ub)
            vm = wp.min(vm, vb)
            vx = wp.max(vx, vb)
            zm = wp.min(zm, zb[nb])
            zx = wp.max(zx, zb[nb])

    pw = wp.float64(1.0)
    pu = wp.float64(1.0)
    pv = wp.float64(1.0)
    pz = wp.float64(1.0)
    eps = EPS_F64
    for k in range(3):
        e = cell_edges[c][k]
        nb = edge_cells[e][1]
        if nb >= 0:
            wb = w[nb]
            hb, ub, vb = primitive(wb[0], wb[1], wb[2])
            wnb = zb[nb] + wb[0]
            dw = wp.float64(0.5) * (wsc + wnb) - wsc
            du = wp.float64(0.5) * (uc + ub) - uc
            dv = wp.float64(0.5) * (vc + vb) - vc
            dz = wp.float64(0.5) * (zc + zb[nb]) - zc
            if wp.abs(dw) > eps:
                if dw > wp.float64(0.0):
                    pw = wp.min(pw, (wx - wsc) / dw)
                else:
                    pw = wp.min(pw, (wm - wsc) / dw)
            if wp.abs(du) > eps:
                if du > wp.float64(0.0):
                    pu = wp.min(pu, (ux - uc) / du)
                else:
                    pu = wp.min(pu, (um - uc) / du)
            if wp.abs(dv) > eps:
                if dv > wp.float64(0.0):
                    pv = wp.min(pv, (vx - vc) / dv)
                else:
                    pv = wp.min(pv, (vm - vc) / dv)
            if wp.abs(dz) > eps:
                if dz > wp.float64(0.0):
                    pz = wp.min(pz, (zx - zc) / dz)
                else:
                    pz = wp.min(pz, (zm - zc) / dz)

    gw = grad_w[c]
    gu = grad_u[c]
    gv = grad_v[c]
    gz = grad_z[c]
    slope_w[c] = wp.vec2d(pw * gw[0], pw * gw[1])
    slope_u[c] = wp.vec2d(pu * gu[0], pu * gu[1])
    slope_v[c] = wp.vec2d(pv * gv[0], pv * gv[1])
    slope_z[c] = wp.vec2d(pz * gz[0], pz * gz[1])


# ---------------------------------------------------------------------------
# Unlimited slopes: copy gradients (limiter = none).
# ---------------------------------------------------------------------------
@wp.kernel
def k_slope_none(
    grad_w: wp.array(dtype=wp.vec2d),
    grad_u: wp.array(dtype=wp.vec2d),
    grad_v: wp.array(dtype=wp.vec2d),
    grad_z: wp.array(dtype=wp.vec2d),
    slope_w: wp.array(dtype=wp.vec2d),
    slope_u: wp.array(dtype=wp.vec2d),
    slope_v: wp.array(dtype=wp.vec2d),
    slope_z: wp.array(dtype=wp.vec2d),
):
    c = wp.tid()
    slope_w[c] = grad_w[c]
    slope_u[c] = grad_u[c]
    slope_v[c] = grad_v[c]
    slope_z[c] = grad_z[c]


# ---------------------------------------------------------------------------
# TELEMAC second_order.f per-edge fallback test: an edge stays first order
# when the bottom jump swallows one of the depths or the reconstructed
# bottom moves by more than half a depth (second_order.f lines 232-236).
# ---------------------------------------------------------------------------
@wp.func
def edge_falls_back(
    h0: wp.float64,
    h1: wp.float64,
    z0: wp.float64,
    z1: wp.float64,
    dsz0: wp.vec2d,
    dsz1: wp.vec2d,
):
    if z0 >= h1 + z1 or z1 >= h0 + z0:
        return wp.bool(1)
    if wp.float64(2.0) * wp.abs(dsz0[0]) >= h0 or wp.float64(2.0) * wp.abs(dsz0[0]) >= h1:
        return wp.bool(1)
    if wp.float64(2.0) * wp.abs(dsz0[1]) >= h0 or wp.float64(2.0) * wp.abs(dsz0[1]) >= h1:
        return wp.bool(1)
    if wp.float64(2.0) * wp.abs(dsz1[0]) >= h0 or wp.float64(2.0) * wp.abs(dsz1[0]) >= h1:
        return wp.bool(1)
    if wp.float64(2.0) * wp.abs(dsz1[1]) >= h0 or wp.float64(2.0) * wp.abs(dsz1[1]) >= h1:
        return wp.bool(1)
    return wp.bool(0)


# ---------------------------------------------------------------------------
# Main edge kernel: one thread per edge.  Interior edges: Audusse
# reconstruction + HLLC + geometric sources; boundary edges: ghost state
# from the cell average then the same HLLC (ghost bottom == cell bottom).
# Pairwise +-L*F accumulation keeps global mass to roundoff.
# order: 1 = first order, 2 = TELEMAC-style second order (second_order.f).
# ---------------------------------------------------------------------------
@wp.kernel
def k_flux(
    w: wp.array(dtype=wp.vec3d),
    zb: wp.array(dtype=wp.float64),
    slope_w: wp.array(dtype=wp.vec2d),
    slope_u: wp.array(dtype=wp.vec2d),
    slope_v: wp.array(dtype=wp.vec2d),
    slope_z: wp.array(dtype=wp.vec2d),
    centroid: wp.array(dtype=wp.vec2d),
    edge_midpoint: wp.array(dtype=wp.vec2d),
    edge_cells: wp.array(dtype=wp.vec2i),
    edge_normal: wp.array(dtype=wp.vec2d),
    edge_len: wp.array(dtype=wp.float64),
    bedge_group: wp.array(dtype=wp.int32),
    btype: wp.array(dtype=wp.int32),
    bparam: wp.array(dtype=wp.float64),  # per-edge: zs or q_edge [m^2/s]
    rhs: wp.array(dtype=wp.vec3d),
    bflux: wp.array(dtype=wp.float64),
    order: wp.int32,
):
    e = wp.tid()
    cells = edge_cells[e]
    a = cells[0]
    b = cells[1]
    nx = edge_normal[e][0]
    ny = edge_normal[e][1]
    le = edge_len[e]

    if b < 0:
        # -------- boundary edge: cell-average state + ghost (cdl_fv.f) --
        ha, ua, va = primitive(w[a][0], w[a][1], w[a][2])
        grp = bedge_group[e]
        hg, ug, vg = ghost_state(btype[grp], bparam[e], ha, ua, va, zb[a], nx, ny)
        f0, f1, f2 = hllc_flux(ha, hg, ua, va, ug, vg, nx, ny)
        wp.atomic_add(rhs, a, wp.vec3d(-le * f0, -le * f1, -le * f2))
        wp.atomic_add(bflux, grp, le * f0)
        return

    # -------- interior edge ------------------------------------------
    ha, ua, va = primitive(w[a][0], w[a][1], w[a][2])
    hb, ub, vb = primitive(w[b][0], w[b][1], w[b][2])
    zfa = zb[a]
    zfb = zb[b]
    h0a = w[a][0]  # raw cell depths (positive by construction)
    h0b = w[b][0]

    if order == 2:
        xm = edge_midpoint[e][0]
        ym = edge_midpoint[e][1]
        dxa = xm - centroid[a][0]
        dya = ym - centroid[a][1]
        dxb = xm - centroid[b][0]
        dyb = ym - centroid[b][1]
        sa = slope_z[a]
        sb = slope_z[b]
        dsza = wp.vec2d(sa[0] * dxa, sa[1] * dya)
        dszb = wp.vec2d(sb[0] * dxb, sb[1] * dyb)

        if edge_falls_back(h0a, h0b, zfa, zfb, dsza, dszb):
            # keep first order on this edge (second_order.f fallback)
            pass
        else:
            swa = slope_w[a]
            sua = slope_u[a]
            sva = slope_v[a]
            swb = slope_w[b]
            sub = slope_u[b]
            svb = slope_v[b]
            # Face reconstruction of the FREE SURFACE and the bottom
            # (Audusse 2nd order; TELEMAC CORR_HL/CORR_ZL with DSH the
            # surface increment): h_face = max(0, w_face - z_face).
            dswa = swa[0] * dxa + swa[1] * dya
            dswb = swb[0] * dxb + swb[1] * dyb
            wfa = (zfa + h0a) + dswa
            wfb = (zfb + h0b) + dswb
            zfa = zfa + dsza[0] + dsza[1]
            zfb = zfb + dszb[0] + dszb[1]
            hfa = wp.max(wfa - zfa, wp.float64(0.0))
            hfb = wp.max(wfb - zfb, wp.float64(0.0))
            ua = ua + sua[0] * dxa + sua[1] * dya
            va = va + sva[0] * dxa + sva[1] * dya
            ub = ub + sub[0] * dxb + sub[1] * dyb
            vb = vb + svb[0] * dxb + svb[1] * dyb
            if hfa <= wp.float64(0.0):
                hfa = wp.float64(0.0)
                ua = wp.float64(0.0)
                va = wp.float64(0.0)
            if hfb <= wp.float64(0.0):
                hfb = wp.float64(0.0)
                ub = wp.float64(0.0)
                vb = wp.float64(0.0)
            if hfa <= wp.float64(0.0):
                hfa = wp.float64(0.0)
                ua = wp.float64(0.0)
                va = wp.float64(0.0)
            if hfb <= wp.float64(0.0):
                hfb = wp.float64(0.0)
                ub = wp.float64(0.0)
                vb = wp.float64(0.0)

            # Audusse reconstruction on the face values (hyd_fv.f).
            dzi = wp.max(zfb - zfa, wp.float64(0.0))
            hij = wp.max(hfa - dzi, wp.float64(0.0))
            dzj = wp.max(zfa - zfb, wp.float64(0.0))
            hji = wp.max(hfb - dzj, wp.float64(0.0))

            f0, f1, f2 = hllc_flux(hij, hji, ua, va, ub, vb, nx, ny)

            # Geometric sources + slope corrections (hyd_fv.f HGZI with
            # CORR_I = -0.5*L*(h_cell+h_face)*DSZ, second_order.f:579).
            hgz_i = wp.float64(0.5) * le * (hij + hfa) * (hij - hfa) \
                - wp.float64(0.5) * le * (h0a + hfa) * (dsza[0] + dsza[1])
            hgz_j = wp.float64(0.5) * le * (hji + hfb) * (hji - hfb) \
                - wp.float64(0.5) * le * (h0b + hfb) * (dszb[0] + dszb[1])
            wp.atomic_add(
                rhs, a,
                wp.vec3d(-le * f0, -le * f1 + G * nx * hgz_i, -le * f2 + G * ny * hgz_i),
            )
            wp.atomic_add(
                rhs, b,
                wp.vec3d(le * f0, le * f1 - G * nx * hgz_j, le * f2 - G * ny * hgz_j),
            )
            return

    # -------- first-order path (also the order-2 fallback) ------------
    dzi = wp.max(zfb - zfa, wp.float64(0.0))
    hij = wp.max(ha - dzi, wp.float64(0.0))
    dzj = wp.max(zfa - zfb, wp.float64(0.0))
    hji = wp.max(hb - dzj, wp.float64(0.0))

    f0, f1, f2 = hllc_flux(hij, hji, ua, va, ub, vb, nx, ny)

    hgz_i = wp.float64(0.5) * le * (hij + ha) * (hij - ha)
    hgz_j = wp.float64(0.5) * le * (hji + hb) * (hji - hb)
    wp.atomic_add(
        rhs, a,
        wp.vec3d(-le * f0, -le * f1 + G * nx * hgz_i, -le * f2 + G * ny * hgz_i),
    )
    wp.atomic_add(
        rhs, b,
        wp.vec3d(le * f0, le * f1 - G * nx * hgz_j, le * f2 - G * ny * hgz_j),
    )


# ---------------------------------------------------------------------------
# Zeroing helpers.
# ---------------------------------------------------------------------------
@wp.kernel
def k_zero_vec3(dst: wp.array(dtype=wp.vec3d)):
    dst[wp.tid()] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.kernel
def k_zero_vec2(dst: wp.array(dtype=wp.vec2d)):
    dst[wp.tid()] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


@wp.kernel
def k_zero_f64(dst: wp.array(dtype=wp.float64)):
    dst[wp.tid()] = wp.float64(0.0)


# ---------------------------------------------------------------------------
# Per-cell CFL timestep (caldt.f): dt_c = DTHAUT / max(0.01, c + |u|),
# DTHAUT = area / sum(edge lengths).  Host takes min * CFL.
# ---------------------------------------------------------------------------
@wp.kernel
def k_dt(
    w: wp.array(dtype=wp.vec3d),
    cell_area: wp.array(dtype=wp.float64),
    cell_edges: wp.array(dtype=wp.vec3i),
    edge_len: wp.array(dtype=wp.float64),
    dt_out: wp.array(dtype=wp.float64),
):
    c = wp.tid()
    h, u, v = primitive(w[c][0], w[c][1], w[c][2])
    ce = cell_edges[c]
    circum = edge_len[ce[0]] + edge_len[ce[1]] + edge_len[ce[2]]
    dthaut = cell_area[c] / circum
    speed = wp.sqrt(u * u + v * v) + wp.sqrt(G * h)
    dt_out[c] = dthaut / wp.max(wp.float64(0.01), speed)


# ---------------------------------------------------------------------------
# Explicit Euler update: w_out = w_in + (dt/area) * rhs, then dry sanitize
# (h clipped at 0, velocity zeroed below eps -- hyd_fv.f discipline).
# ---------------------------------------------------------------------------
@wp.kernel
def k_update(
    w_in: wp.array(dtype=wp.vec3d),
    rhs: wp.array(dtype=wp.vec3d),
    cell_area: wp.array(dtype=wp.float64),
    dt: wp.float64,
    w_out: wp.array(dtype=wp.vec3d),
):
    c = wp.tid()
    da = dt / cell_area[c]
    h = w_in[c][0] + da * rhs[c][0]
    hu = w_in[c][1] + da * rhs[c][1]
    hv = w_in[c][2] + da * rhs[c][2]
    h, u, v = primitive(h, hu, hv)
    w_out[c] = wp.vec3d(h, h * u, h * v)


# ---------------------------------------------------------------------------
# SSP-RK2 combination: w_out = 0.5*(w_old + w_stage), same dry sanitize.
# ---------------------------------------------------------------------------
@wp.kernel
def k_comb(
    w_old: wp.array(dtype=wp.vec3d),
    w_stage: wp.array(dtype=wp.vec3d),
    w_out: wp.array(dtype=wp.vec3d),
):
    c = wp.tid()
    h = wp.float64(0.5) * (w_old[c][0] + w_stage[c][0])
    hu = wp.float64(0.5) * (w_old[c][1] + w_stage[c][1])
    hv = wp.float64(0.5) * (w_old[c][2] + w_stage[c][2])
    h, u, v = primitive(h, hu, hv)
    w_out[c] = wp.vec3d(h, h * u, h * v)


# ---------------------------------------------------------------------------
# Diagnostics: total water mass via atomic reduction (host zeroes mass[0]).
# ---------------------------------------------------------------------------
@wp.kernel
def k_mass(
    w: wp.array(dtype=wp.vec3d),
    cell_area: wp.array(dtype=wp.float64),
    mass: wp.array(dtype=wp.float64),
):
    c = wp.tid()
    wp.atomic_add(mass, 0, wp.max(w[c][0], wp.float64(0.0)) * cell_area[c])
