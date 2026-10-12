"""Analytic solutions for the validation cases (numpy, float64).

Each entry of SOLUTIONS is a function (cfg, t, x, y) -> (h, u, v)
evaluated at arbitrary points; parameters are read from cfg.initial.params
so the YAML file is the single source of truth.  All solutions assume
frictionless flow and g = 9.81 m/s^2.
"""
from __future__ import annotations

import numpy as np

from .config import CaseConfig

G = 9.81


SOLUTIONS: dict = {}


def _register(name: str):
    def deco(fn):
        SOLUTIONS[name] = fn
        return fn
    return deco


# ----------------------------------------------------------------------
# Dam break on a dry bed (Ritter 1892)
# ----------------------------------------------------------------------
@_register("ritter")
def ritter(cfg: CaseConfig, t: float, x: np.ndarray, y: np.ndarray):
    """Exact solution for hl > 0, hr = 0 on a flat bed.

    Rarefaction spanning x_d - sqrt(g hl) t  ...  x_d + 2 sqrt(g hl) t:
        u = 2/3 (sqrt(g hl) + (x - x_d)/t)
        h = 1/(9 g) (2 sqrt(g hl) - (x - x_d)/t)^2
    """
    p = cfg.initial.params
    hl = float(p["hl"])
    xd = float(p["x_dam"])
    cl = np.sqrt(G * hl)

    h = np.empty_like(x)
    u = np.empty_like(x)
    head = xd - cl * t
    front = xd + 2.0 * cl * t

    left = x < head
    mid = (x >= head) & (x <= front)
    right = x > front

    h[left] = hl
    u[left] = 0.0
    s = (x[mid] - xd) / t
    h[mid] = (2.0 * cl - s) ** 2 / (9.0 * G)
    u[mid] = 2.0 * (cl + s) / 3.0
    h[right] = 0.0
    u[right] = 0.0
    return h, u, np.zeros_like(x)


# ----------------------------------------------------------------------
# Dam break with a wet right state (Stoker 1957)
# ----------------------------------------------------------------------
@_register("stoker")
def stoker(cfg: CaseConfig, t: float, x: np.ndarray, y: np.ndarray):
    """Exact solution for hl > hr > 0 on a flat bed.

    Left rarefaction + constant middle state (hm, um) + right-moving shock.
    um = 2 (cl - cm)  and the shock Rankine-Hugoniot condition give

        2 (cl - cm) = (hm - hr) sqrt(g (hm + hr) / (2 hm hr))

    solved by bisection for hm in (hr, hl).
    """
    p = cfg.initial.params
    hl = float(p["hl"])
    hr = float(p["hr"])
    xd = float(p["x_dam"])
    cl = np.sqrt(G * hl)

    def imbalance(hm: float) -> float:
        cm = np.sqrt(G * hm)
        lhs = 2.0 * (cl - cm)
        rhs = (hm - hr) * np.sqrt(G * (hm + hr) / (2.0 * hm * hr))
        return lhs - rhs  # >0 at hm=hr, <0 at hm=hl

    lo, hi = hr + 1.0e-14, hl
    if hr <= 0.0:
        raise ValueError("stoker requires hr > 0 (use ritter)")
    if imbalance(lo) * imbalance(hi) > 0.0:
        raise ValueError("stoker: no middle state bracket")
    for _ in range(200):  # bisection to machine precision
        mid = 0.5 * (lo + hi)
        if imbalance(lo) * imbalance(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    hm = 0.5 * (lo + hi)
    cm = np.sqrt(G * hm)
    um = 2.0 * (cl - cm)
    shock_speed = um * hm / (hm - hr)

    h = np.empty_like(x)
    u = np.empty_like(x)
    head = xd - cl * t
    front = xd + shock_speed * t

    left = x < head
    fan = (x >= head) & (x < xd)
    mid_r = (x >= xd) & (x < front)
    right = x >= front

    h[left] = hl
    u[left] = 0.0
    s = (x[fan] - xd) / t
    h[fan] = (2.0 * cl - s) ** 2 / (9.0 * G)
    u[fan] = 2.0 * (cl + s) / 3.0
    h[mid_r] = hm
    u[mid_r] = um
    h[right] = hr
    u[right] = 0.0
    return h, u, np.zeros_like(x)


# ----------------------------------------------------------------------
# Steady flow over the TELEMAC bump (subcritical / critical / transcritical)
# ----------------------------------------------------------------------
# Mirrors examples/bump/analytic_sol.py of the TELEMAC distribution:
# specific energy  E = zb + h + q^2/(2 g h^2) = const, cubic solved per
# point; the transcritical variant locates the hydraulic jump by matching
# the momentum function 2 q^2/h + g h^2 across it.

BUMP_X0 = 10.0
BUMP_LENGTH = 20.0


def _bump_zb(x: np.ndarray) -> np.ndarray:
    # must match bathymetry_at(type="bump") defaults
    return 0.25 * np.exp(-0.5 * (x - BUMP_X0) ** 2)


def _cubic_branches(b: float, d: float):
    """Real roots of h^3 + b h^2 + d = 0 (linear term absent).

    Returns (h_small, h_large) -- the two positive roots when they exist
    (one root is negative for the flows considered here).
    """
    # h = y - b/3:  y^3 + p y + qq = 0
    p = -b * b / 3.0
    qq = 2.0 * b ** 3 / 27.0 + d
    disc = (qq / 2.0) ** 2 + (p / 3.0) ** 3
    if disc >= 0.0:  # single real root (should not happen on this profile)
        r = np.cbrt(-qq / 2.0 + np.sqrt(disc)) \
            + np.cbrt(-qq / 2.0 - np.sqrt(disc))
        h = r - b / 3.0
        return h, h
    m = 2.0 * np.sqrt(-p / 3.0)
    arg = np.clip(3.0 * qq / (2.0 * p) * np.sqrt(-3.0 / p), -1.0, 1.0)
    phi = np.arccos(arg) / 3.0
    ys = m * np.cos(phi - 2.0 * np.pi * np.arange(3) / 3.0)
    hs = np.sort(ys - b / 3.0)
    # root product = -d < 0: one root is negative, the two positive ones
    # (supercritical / subcritical branches) are hs[1] and hs[2]
    return hs[1], hs[2]


def _bump_profile(flow: str, q: float, width: float, hl: float,
                  n: int = 4001):
    """Steady-solution depth h(x) on a fine grid, following TELEMAC's
    BumpAnalyticSol.  Returns (x, h, u) arrays."""
    xs = np.linspace(0.0, BUMP_LENGTH, n)
    zb = _bump_zb(xs)
    zm = zb.max()
    ic = int(np.argmax(zb))
    q0 = q / width
    hc = (q0 * q0 / G) ** (1.0 / 3.0)
    h = np.empty(n)

    if flow == "sub":
        # energy line anchored by the downstream depth at zb = 0
        for i in range(n):
            b = zb[i] - q0 ** 2 / (2.0 * G * hl ** 2) - hl
            _, hsub = _cubic_branches(b, q0 ** 2 / (2.0 * G))
            h[i] = hsub
    elif flow == "cri":
        # energy line such that the flow is exactly critical at the crest:
        # E_cre = zm + hc + q0^2/(2 g hc^2); cubic coefficient b = zb - E
        b_cre = -q0 ** 2 / (2.0 * G * hc ** 2) - hc - zm
        for i in range(n):
            hsmall, hsub = _cubic_branches(zb[i] + b_cre, q0 ** 2 / (2.0 * G))
            h[i] = hsub if i < ic else (hc if i == ic else hsmall)
    elif flow == "trans":
        # Faithful port of TELEMAC BumpAnalyticSol.compute_transcritical_
        # solution.  b(x) is the cubic coefficient: h^3 + b h^2 + c = 0.
        b_down = -q0 ** 2 / (2.0 * G * hl ** 2) - hl          # hl-anchored
        b_cre = -q0 ** 2 / (2.0 * G * hc ** 2) - hc - zm      # crest-anch.
        c0 = q0 ** 2 / (2.0 * G)

        def h_down(i):
            _, hs = _cubic_branches(zb[i] + b_down, c0)
            return hs

        # phase 1: march upstream on the hl-anchored subcritical branch
        # until the depth drops below critical; like the official port,
        # the index is decremented once more after the trigger
        i = n - 1
        supercritical = False
        while i >= 0 and not supercritical:
            if i > ic and h_down(i) < hc:
                supercritical = True
            i -= 1
        if not supercritical:
            raise ValueError("trans: flow stays subcritical; check q/hl")

        # phase 2: shock location.  The official algorithm freezes the
        # crest-anchored supercritical depth at the transition point and
        # finds where its momentum function is first exceeded by the
        # hl-anchored branch momentum.
        h_star, _ = _cubic_branches(zb[i] + b_cre, c0)
        m_star = 2.0 * q0 ** 2 / h_star + G * h_star ** 2
        x_shock = None
        for j in range(ic, n - 1):
            hd = h_down(j + 1)
            if m_star < 2.0 * q0 ** 2 / hd + G * hd ** 2:
                x_shock = xs[j]
                break
        if x_shock is None:
            raise ValueError("trans: no jump position found")

        # phase 3: crest-anchored branches up to the shock, hl-anchored
        # subcritical branch downstream of it
        for k in range(n):
            if xs[k] > x_shock:
                h[k] = h_down(k)
            else:
                hsmall, hsub = _cubic_branches(zb[k] + b_cre, c0)
                h[k] = hsub if k < ic else (hc if k == ic else hsmall)
    else:
        raise ValueError(f"unknown bump flow {flow!r}")

    u = q0 / h
    return xs, h, u


def _bump_lookup(flow: str, cfg: CaseConfig, x: np.ndarray):
    p = cfg.initial.params
    q = float(p["q"])
    width = float(p.get("width", 2.0))
    hl = float(p.get("hl", 0.0))
    xs, h, u = _bump_profile(flow, q, width, hl)
    return np.interp(x, xs, h), np.interp(x, xs, u)


@_register("bump_sub")
def bump_sub(cfg, t, x, y):
    h, u = _bump_lookup("sub", cfg, x)
    return h, u, np.zeros_like(x)


@_register("bump_cri")
def bump_cri(cfg, t, x, y):
    h, u = _bump_lookup("cri", cfg, x)
    return h, u, np.zeros_like(x)


@_register("bump_trans")
def bump_trans(cfg, t, x, y):
    h, u = _bump_lookup("trans", cfg, x)
    return h, u, np.zeros_like(x)


# ----------------------------------------------------------------------
# Thacker solutions in a parabolic bowl (TELEMAC examples/thacker)
# ----------------------------------------------------------------------
# Domain [0,L]x[0,L], bowl  zb = H0 (r^2/a^2 - 1) centred at (xc, yc),
# H0 = 0.1, a = 1.  Parameters come from cfg.initial.params so the yaml
# matches the user_fortran files of the TELEMAC example.
def _bowl_zb(x, y, h0, a, xc, yc):
    r2 = (x - xc) ** 2 + (y - yc) ** 2
    return h0 * (r2 / (a * a) - 1.0)


@_register("thacker_planar")
def thacker_planar(cfg, t, x, y):
    """Planar free surface sloshing in a parabolic bowl
    (examples/thacker/user_fortran/user_condin_h.f)."""
    p = cfg.initial.params
    h0 = float(p["h0"])
    a = float(p["a"])
    eta = float(p["eta"])
    L = float(p.get("length", 4.0))
    xc = float(p.get("xc", 0.5 * L))
    yc = float(p.get("yc", 0.5 * L))
    om = np.sqrt(2.0 * G * h0) / a

    zs = (eta * h0 / (a * a)) * (
        2.0 * (x - xc) * np.cos(om * t)
        + 2.0 * (y - yc) * np.sin(om * t) - eta)
    h = np.maximum(zs - _bowl_zb(x, y, h0, a, xc, yc), 0.0)
    u = -eta * om * np.sin(om * t)
    v = eta * om * np.cos(om * t)
    return h, np.full_like(x, u), np.full_like(x, v)


@_register("thacker_radial")
def thacker_radial(cfg, t, x, y):
    """Radial oscillation in a parabolic bowl
    (examples/thacker/user_fortran_2/user_condin_h.f)."""
    p = cfg.initial.params
    h0 = float(p["h0"])
    a = float(p["a"])
    r0 = float(p["r0"])
    L = float(p.get("length", 4.0))
    xc = float(p.get("xc", 0.5 * L))
    yc = float(p.get("yc", 0.5 * L))
    om = np.sqrt(8.0 * G * h0) / a
    b = (a * a - r0 * r0) / (a * a + r0 * r0)

    denom = 1.0 - b * np.cos(om * t)
    c1 = np.sqrt(1.0 - b * b) / denom
    c2 = (1.0 - b * b) / (denom * denom) - 1.0
    r2 = (x - xc) ** 2 + (y - yc) ** 2
    zs = h0 * (c1 - 1.0 - (r2 / (a * a)) * c2)
    h = np.maximum(zs - _bowl_zb(x, y, h0, a, xc, yc), 0.0)
    u = 0.5 * om * (x - xc) * b * np.sin(om * t) / denom
    v = 0.5 * om * (y - yc) * b * np.sin(om * t) / denom
    return h, u, v


# ----------------------------------------------------------------------
# Linear seiche in a closed basin (flat bed, small-amplitude standing wave)
# ----------------------------------------------------------------------
@_register("seiche")
def seiche(cfg, t, x, y):
    """h = h0 + A cos(k x') cos(w t),  u = +(c/h0) A sin(k x') sin(w t),
    x' measured from the domain x-min; walls at both ends (wave crests).
    The u sign follows from continuity: dh/dt + h0 du/dx = 0."""
    p = cfg.initial.params
    h0 = float(p["h0"])
    amp = float(p["amplitude"])
    length = float(p["length"])
    x0 = float(p.get("xmin", 0.0))
    c = np.sqrt(G * h0)
    k = 2.0 * np.pi / length
    om = c * k
    xp = x - x0
    h = h0 + amp * np.cos(k * xp) * np.cos(om * t)
    u = (c / h0) * amp * np.sin(k * xp) * np.sin(om * t)
    return h, u, np.zeros_like(x)


def evaluate(name: str, cfg: CaseConfig, t: float,
             x: np.ndarray, y: np.ndarray):
    """Dispatch to the registered analytic solution."""
    if name not in SOLUTIONS:
        raise KeyError(f"unknown analytic solution {name!r}; "
                       f"available: {sorted(SOLUTIONS)}")
    return SOLUTIONS[name](cfg, t, x, y)
