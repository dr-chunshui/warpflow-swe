"""Analytic solutions for the validation cases (numpy, float64).

Each entry of SOLUTIONS is a function (cfg, t, x, y) -> (h, u, v)
evaluated at arbitrary points; parameters are read from cfg.initial.params
so the YAML file is the single source of truth.  All solutions assume
frictionless flow and g = 9.81 m/s^2.
"""
from __future__ import annotations

import numpy as np

from .config import BoundaryGroup, CaseConfig

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


# ----------------------------------------------------------------------
# Progressive wave in a channel (TELEMAC validation manual ch. 56, wave)
# ----------------------------------------------------------------------
def _wave_bc(cfg: CaseConfig) -> BoundaryGroup:
    """The harmonic level boundary driving the wave (yaml single source)."""
    for g in cfg.boundary:
        if g.type == "level" and g.time:
            return g
    raise ValueError("wave_prog requires a level boundary group with a "
                     "'time: {mean, amp, period}' spec")


@_register("wave_prog")
def wave_prog(cfg, t, x, y):
    """Linear progressive wave, H0 >> A (manual eq. 56.1-56.2):

        H = H0 + A sin(2 pi (t/T - x/(T c))),  c = sqrt(g H0)
        U = A sqrt(g/H0) sin(2 pi (t/T - x/(T c)))
    """
    g_bc = _wave_bc(cfg)
    h0 = float(g_bc.time["mean"])
    amp = float(g_bc.time["amp"])
    period = float(g_bc.time["period"])
    c = np.sqrt(G * h0)
    ph = 2.0 * np.pi * (t / period - x / (period * c))
    h = h0 + amp * np.sin(ph)
    u = amp * np.sqrt(G / h0) * np.sin(ph)
    return h, u, np.zeros_like(x)


# ----------------------------------------------------------------------
# Gradually varied flow with Strickler friction
# (TELEMAC validation manual ch. 27, friction; steady state, direct
# integration of the GVF equation like examples/friction exacte.f)
# ----------------------------------------------------------------------
@_register("friction_gvf")
def friction_gvf(cfg, t, x, y):
    """Steady channel flow on a mild slope with Strickler friction.

    Reference = the exact steady state of the 1D SWE with friction,
        dh/dx = (S0 - Sf) / (1 - Fr^2),
    marched upstream from the downstream (level) control with RK4.
    All parameters come from the yaml: discharge from the inflow_q group,
    outlet level from the xmax level group, slope from the bathymetry
    (exact for the linear beds the GVF reference applies to; zero for a
    flat bed, e.g. the canal case), Strickler K from physics.
    """
    from .state import bathymetry_at
    xa = np.array([cfg.mesh.xmin, cfg.mesh.xmax])
    zb_ends = bathymetry_at(cfg, xa, np.zeros(2))
    S0 = float((zb_ends[0] - zb_ends[1])
               / (cfg.mesh.xmax - cfg.mesh.xmin))
    width = cfg.mesh.ymax - cfg.mesh.ymin
    L = cfg.mesh.xmax - cfg.mesh.xmin
    q_in = 0.0
    zs_out = None
    for g in cfg.boundary:
        if g.type == "inflow_q":
            q_in += abs(float(g.value))
        elif g.type == "level" and g.select.get("xmax") is not None:
            zs_out = float(g.value)
    if q_in <= 0.0 or zs_out is None:
        raise ValueError("friction_gvf needs an inflow_q and an xmax "
                         "level boundary group")
    q = q_in / width
    zb_out = float(bathymetry_at(
        cfg, np.array([cfg.mesh.xmax]),
        np.array([0.5 * (cfg.mesh.ymin + cfg.mesh.ymax)]))[0])
    h_out = zs_out - zb_out
    xs, hs = _gvf_profile(q, cfg.strickler, S0, L, h_out)
    h = np.interp(x, xs, hs)
    u = q / h
    return h, u, np.zeros_like(x)


def _gvf_profile(q: float, k_str: float, s0: float, length: float,
                 h_out: float, n: int = 200001):
    """RK4 march of the GVF ODE from the outlet control (x = L) upstream.

    Cached: the reference only depends on (q, K, S0, L, h_out).
    """
    key = (q, k_str, s0, length, h_out)
    if key in _gvf_profile.cache:
        return _gvf_profile.cache[key]
    if k_str <= 0.0:
        raise ValueError("friction_gvf requires physics.strickler > 0")
    nm = 1.0 / k_str  # Manning n
    qn2 = (q * nm) ** 2

    def dhdx(h: float) -> float:
        fr2 = q * q / (G * h ** 3)
        sf = qn2 / h ** (10.0 / 3.0)
        return (s0 - sf) / (1.0 - fr2)

    xs = np.linspace(length, 0.0, n)  # marching upstream
    hs = np.empty(n)
    hs[0] = h_out
    dx = -length / (n - 1)
    for i in range(n - 1):
        h = hs[i]
        k1 = dhdx(h)
        k2 = dhdx(h + 0.5 * dx * k1)
        k3 = dhdx(h + 0.5 * dx * k2)
        k4 = dhdx(h + dx * k3)
        hs[i + 1] = h + dx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if np.any(hs <= 0.0) or np.any(~np.isfinite(hs)):
        raise ValueError("friction_gvf: GVF march left the physical range")
    xs_asc = np.ascontiguousarray(xs[::-1])
    hs_asc = np.ascontiguousarray(hs[::-1])
    _gvf_profile.cache[key] = (xs_asc, hs_asc)
    return xs_asc, hs_asc


_gvf_profile.cache: dict = {}


# ----------------------------------------------------------------------
# Wind set-up in a closed basin (TELEMAC validation manual ch. 59, wind)
# ----------------------------------------------------------------------
@_register("wind_setup")
def wind_setup(cfg, t, x, y):
    """Steady wind setup: surface slope balances the wind stress,

        H(x) = sqrt(H0^2 + 2 c_w |W|^2 x / g),
    with H0 fixed by mass conservation (examples/wind exacte.f).
    The depth-mean velocity vanishes at equilibrium (zero net discharge
    through every section: surface current + return flow cancel).
    """
    h_init = float(cfg.initial.params["zs"])
    L = cfg.mesh.xmax - cfg.mesh.xmin
    coeff = cfg.wind_coeff
    wu, wv = cfg.wind_u, cfg.wind_v
    wn2 = wu * wu + wv * wv
    if coeff <= 0.0 or wn2 <= 0.0:
        raise ValueError("wind_setup requires physics.wind u/v/coeff")
    a = 2.0 * coeff * wn2 / G
    target = h_init * L

    def volume(h0: float) -> float:
        # int_0^L sqrt(h0^2 + a x) dx
        return (2.0 / (3.0 * a)) * ((h0 * h0 + a * L) ** 1.5 - h0 ** 3)

    lo, hi = 1.0e-6, h_init
    if volume(hi) < target:
        raise ValueError("wind_setup: mass conservation unsolvable")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if volume(mid) < target:
            lo = mid
        else:
            hi = mid
    h0 = 0.5 * (lo + hi)
    h = np.sqrt(h0 * h0 + a * x)
    return h, np.zeros_like(x), np.zeros_like(x)


# ----------------------------------------------------------------------
# Tidal-flat draw-down equilibrium (TELEMAC validation manual ch. 3,
# balzano): at t -> inf both reaches are lakes at rest.  The right reach
# drains to the imposed boundary level zs = -2 m; the left reservoir is
# trapped behind the sill and settles at the crest elevation
# zs = zb(4800) = 100/23 - 4800/920 = -0.8695652174 m (manual section
# 3.2.5, DATA/SL_reservoir.txt).  Beaches above the local lake level
# are dry.
# ----------------------------------------------------------------------
@_register("balzano_drain")
def balzano_drain(cfg, t, x, y):
    from .state import bathymetry_at
    zb = bathymetry_at(cfg, x, y)
    crest = 100.0 / 23.0 - 4800.0 / 920.0
    zs = np.where(x < 4800.0, crest, -2.0)
    h = np.maximum(zs - zb, 0.0)
    return h, np.zeros_like(x), np.zeros_like(x)


# ----------------------------------------------------------------------
# Draw-down over a pool (TELEMAC validation manual ch. 54, vasque)
# ----------------------------------------------------------------------
@_register("vasque_dry")
def vasque_dry(cfg, t, x, y):
    """Final equilibrium of the draw-down over the vasque (pool).

    The offshore boundary falls to zs = -0.55 m over 300 s.  At the end
    the seaward beach (rising 0.01 per m) rests at the sea level while
    the depression landward of the Gaussian hump stays ponded at its
    spill level: the bed crest near x = 19,
        zb(19) = -0.6 + 0.01*19 + 0.1*exp(-1/20) = -0.3149 m,
    like the manual's final state (level horizontal in the pool, zero
    velocity on both sides).  h = max(zs - zb, 0), u = v = 0.
    """
    from .state import bathymetry_at
    zb = bathymetry_at(cfg, x, y)
    spill = -0.6 + 0.01 * 19.0 + 0.1 * np.exp(-1.0 / 20.0)
    zs = np.where(x < 19.0, -0.55, spill)
    h = np.maximum(zs - zb, 0.0)
    return h, np.zeros_like(x), np.zeros_like(x)


def evaluate(name: str, cfg: CaseConfig, t: float,
             x: np.ndarray, y: np.ndarray):
    """Dispatch to the registered analytic solution."""
    if name not in SOLUTIONS:
        raise KeyError(f"unknown analytic solution {name!r}; "
                       f"available: {sorted(SOLUTIONS)}")
    return SOLUTIONS[name](cfg, t, x, y)
