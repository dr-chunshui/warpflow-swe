"""YAML case configuration -> validated dataclasses.

Schema example (dambreak/ritter.yaml):

    case:     {name: dambreak_ritter, analytic: ritter, t_end: 2.5}
    mesh:     {type: rect, xmin: 0.0, xmax: 16.0, ymin: 0.0, ymax: 0.45,
               dx: 0.05, diagonal: alternating}
    physics:  {g: 9.81}
    initial:  {type: dam_snap, params: {hl: 1.0, hr: 0.0, x_dam: 8.0}}
    bathymetry: {type: flat, params: {z: 0.0}}
    boundary:
      groups:
        - {name: walls_xmin, select: {xmin: [0.0, 0.45]}, type: wall}
        - {name: outlet,     select: {xmax: [0.0, 0.45]}, type: outflow}
    numerics: {cfl: 0.8, order: 1, limiter: barth, time_order: 1,
               eps_dry: 1.0e-6}
    output:
      dir: output/dambreak_ritter
      vtk_interval: 0.1
      csv_interval: 0.01
      monitors: [{x: 13.5, y: 0.225}]
      centerline_y: 0.225
      band: 0.02

Extensions (TELEMAC validation cases wave / friction / wind):

    physics:  {strickler: 70.0,               # bottom friction K [m^(1/3)/s]
               # or chezy: C [m^(1/2)/s], or nikuradse: ks [m] (log law,
               # friction_calc.f CASE 5) -- mutually exclusive
               wind: {u: 5.0, v: 0.0, coeff: 1.2615e-3}}
    bathymetry: {type: slope, params: {z0: -4.0, slope: -1.85e-5}}
    boundary:
      groups:
        - {name: inlet, select: {xmin: [0.0, 0.3]}, type: level,
           time: {mean: 10.0, amp: 0.05, period: 0.5, phase: 0.0}}
    output:
      average_window: 1000.0   # time-average the state over the last N s
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Boundary type name -> kernel code (must match solver/kernels.py).
BC_TYPES = {"wall": 0, "outflow": 1, "inflow_q": 2, "level": 3}
# Boundary types that carry a scalar value in ``value``.
BC_WITH_VALUE = {"inflow_q", "level"}


@dataclass
class MeshConfig:
    type: str = "rect"  # rect | selafin
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    dx: float = 0.1
    diagonal: str = "alternating"
    file: str = ""  # selafin path for type == selafin


@dataclass
class InitialConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class BathymetryConfig:
    type: str = "flat"  # flat | bump | thacker | slope
    params: dict = field(default_factory=dict)


@dataclass
class BoundaryGroup:
    name: str
    select: dict  # {"xmin": [y0, y1]} / {"ymin": [x0, x1]} / {"chain": k}
    type: str  # wall | outflow | inflow_q | level
    value: float = 0.0  # zs for level [m], Q total for inflow_q [m^3/s]
    time: dict = field(default_factory=dict)
    # time-dependent level value, either
    #   harmonic: zs(t) = mean + amp*sin(2*pi*t/period+phase)
    #     keys: mean, amp, period; optional phase [rad] (default 0), or
    #   series: {file: <path>} -- two-column time/elevation table,
    #     linearly interpolated (TELEMAC liquid-boundary file), or
    #   soliton: {h0, H, t0} -- incident solitary wave
    #     zs(t) = h0 + H/cosh(K (c t - t0))^2 (conical_island user_sl.f).
    #     optional base (default h0): crest level the wave rides on --
    #     set 0 for a pure-uplift wave over zs = 0 water
    #     (triangular_shelf user_sl.f).
    # When ``time`` is non-empty it overrides ``value`` for ``level``
    # groups (a level series) and for ``inflow_q`` groups (a total-
    # discharge series; the series column is Q in m^3/s).
    # select chain k refers to the k-th (1-based) boundary chain of the
    # mesh's .cli file (unstructured meshes).


@dataclass
class OutputConfig:
    dir: str = "output"
    vtk_interval: float = 0.0  # <=0 disables frame dumps
    csv_interval: float = 0.0  # <=0 disables monitor/mass csv
    monitors: list = field(default_factory=list)  # [{x, y}]
    centerline_y: float = 0.0
    band: float = 0.05  # |y - centerline_y| half width for profile extraction
    average_window: float = 0.0  # time-average state over last N s (<=0 off)
    observed: dict = field(default_factory=dict)
    # Measured-data comparison (solver/compare_obs.py):
    #   max_depth: [{name, x, y, measured}]      peak h at the point [m]
    #   arrival:   [{name, x, y, threshold}]     first t with h >= threshold
    #   arrival_diffs: [{name, a, b, measured}]  t_b - t_a vs measured [s]
    # every x/y point is auto-appended to ``monitors``.


@dataclass
class CaseConfig:
    name: str
    analytic: str  # analytic solution key for analysis ("" = none)
    t_end: float
    mesh: MeshConfig
    initial: InitialConfig
    bathymetry: BathymetryConfig
    boundary: list
    cfl: float
    order: int  # 1 | 2
    limiter: str  # none | barth
    time_order: int  # 1 | 2
    eps_dry: float
    output: OutputConfig
    g: float = 9.81
    strickler: float = 0.0  # bottom friction K [m^(1/3)/s], 0 = off
    chezy: float = 0.0  # bottom friction C [m^(1/2)/s], 0 = off (wins over strickler)
    nikuradse: float = 0.0  # roughness ks [m] for the log-law friction
    # (friction_calc.f CASE 5), 0 = off (wins over strickler/chezy)
    wind_u: float = 0.0  # wind velocity [m/s]
    wind_v: float = 0.0
    wind_coeff: float = 0.0  # a_wind * rho_air / rho_water, 0 = off
    coriolis: float = 0.0  # Coriolis parameter f = 2*Omega*sin(lat) [1/s]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_config(path: str | Path) -> CaseConfig:
    """Parse and validate a case YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _require(isinstance(raw, dict), f"{path}: empty or invalid YAML")

    case = raw.get("case") or {}
    _require("name" in case, f"{path}: case.name missing")
    _require("t_end" in case, f"{path}: case.t_end missing")

    mesh = MeshConfig(**{k: v for k, v in (raw.get("mesh") or {}).items()})
    if mesh.type == "rect":
        _require(mesh.xmax > mesh.xmin and mesh.ymax > mesh.ymin,
                 f"{path}: mesh extent invalid")
        _require(mesh.dx > 0.0, f"{path}: mesh.dx must be positive")
        _require(mesh.diagonal in ("alternating", "fixed"),
                 f"{path}: mesh.diagonal must be alternating|fixed")
    elif mesh.type == "selafin":
        _require(mesh.file, f"{path}: mesh.file required for selafin")
        # resolve against the yaml's directory so cases stay relocatable
        fp = Path(mesh.file)
        if not fp.is_absolute():
            fp = path.parent / fp
        _require(fp.is_file(), f"{path}: mesh file missing: {fp}")
        mesh.file = str(fp)
    else:
        raise ValueError(f"{path}: unknown mesh.type {mesh.type!r}")

    initial = raw.get("initial") or {}
    _require("type" in initial, f"{path}: initial.type missing")
    ini = InitialConfig(type=initial["type"],
                        params=dict(initial.get("params") or {}))
    if ini.type == "selafin_state":
        _require("file" in ini.params,
                 f"{path}: initial selafin_state needs params.file")
        fp = Path(str(ini.params["file"]))
        if not fp.is_absolute():
            fp = path.parent / fp
        _require(fp.is_file(), f"{path}: initial state file missing: {fp}")
        ini.params["file"] = str(fp)
    if ini.type == "level_profile":
        _require("file" in ini.params,
                 f"{path}: initial level_profile needs params.file")
        fp = Path(str(ini.params["file"]))
        if not fp.is_absolute():
            fp = path.parent / fp
        _require(fp.is_file(), f"{path}: initial sections file missing: {fp}")
        ini.params["file"] = str(fp)

    bath = raw.get("bathymetry") or {"type": "flat", "params": {}}
    bathcfg = BathymetryConfig(type=bath.get("type", "flat"),
                               params=dict(bath.get("params") or {}))
    _require(bathcfg.type in ("flat", "bump", "thacker", "slope", "bolzano",
                              "beach", "vasque", "selafin", "selafin_ref"),
             f"{path}: unknown bathymetry type {bathcfg.type!r}")
    if bathcfg.type == "selafin":
        _require(mesh.type == "selafin",
                 f"{path}: bathymetry selafin requires mesh.type selafin")
    if bathcfg.type == "selafin_ref":
        _require("file" in bathcfg.params,
                 f"{path}: bathymetry selafin_ref needs params.file")
        fp = Path(str(bathcfg.params["file"]))
        if not fp.is_absolute():
            fp = path.parent / fp
        _require(fp.is_file(), f"{path}: bathymetry ref file missing: {fp}")
        bathcfg.params["file"] = str(fp)

    groups_raw = (raw.get("boundary") or {}).get("groups") or []
    _require(groups_raw, f"{path}: boundary.groups must not be empty")
    groups = []
    for g in groups_raw:
        _require("name" in g and "select" in g and "type" in g,
                 f"{path}: boundary group needs name/select/type: {g}")
        _require(g["type"] in BC_TYPES, f"{path}: unknown bc type {g['type']!r}")
        _require(isinstance(g["select"], dict) and len(g["select"]) == 1,
                 f"{path}: select must have exactly one key: {g.get('name')}")
        (side, span), = g["select"].items()
        if side == "chain":
            chain_sel = span if isinstance(span, list) else [span]
            _require(all((isinstance(c, int) and c >= 1) or c == "rest"
                         for c in chain_sel),
                     f"{path}: select chain must be 1-based int(s) or "
                     f"'rest': {g.get('name')}")
            if "rest" in chain_sel:
                _require(len(chain_sel) == 1,
                         f"{path}: chain rest cannot be combined: "
                         f"{g.get('name')}")
        else:
            _require(side in ("xmin", "xmax", "ymin", "ymax"),
                     f"{path}: select side {side!r} invalid")
            _require(isinstance(span, list) and len(span) == 2,
                     f"{path}: select span must be [a, b]: {g.get('name')}")
        tspec = dict(g.get("time") or {})
        if tspec:
            _require(g["type"] in ("level", "inflow_q"),
                     f"{path}: time-dependent value only for level or "
                     f"inflow_q: {g['name']}")
            if "series" in tspec:
                # file-driven level boundary: resolve the path against the
                # yaml's directory so cases stay relocatable
                fp = Path(str(tspec["series"]))
                if not fp.is_absolute():
                    fp = path.parent / fp
                _require(fp.is_file(),
                         f"{path}: bc series file missing: {fp}")
                tspec = {"series": str(fp)}
            elif "soliton" in tspec:
                # incident solitary wave (conical_island user_sl.f):
                # zs(t) = h0 + H/cosh(K (c t - t0))^2,
                # K = sqrt(0.75 H/h0^3), c = sqrt(g (h0 + H))
                sp = dict(tspec["soliton"])
                _require("h0" in sp and "H" in sp,
                         f"{path}: soliton needs h0/H: {g['name']}")
                tspec = {"soliton": sp}
            else:
                _require("mean" in tspec and "amp" in tspec and "period" in tspec,
                         f"{path}: bc time needs mean/amp/period or series: {g['name']}")
                _require(float(tspec["period"]) > 0.0,
                         f"{path}: bc time period must be positive: {g['name']}")
        elif g["type"] in BC_WITH_VALUE:
            _require("value" in g,
                     f"{path}: bc {g['type']} needs value: {g.get('name')}")
        groups.append(BoundaryGroup(
            name=g["name"], select=dict(g["select"]), type=g["type"],
            value=float(g.get("value", 0.0)), time=tspec,
        ))

    num = raw.get("numerics") or {}
    order = int(num.get("order", 1))
    time_order = int(num.get("time_order", 1))
    _require(order in (1, 2), f"{path}: numerics.order must be 1 or 2")
    _require(time_order in (1, 2), f"{path}: numerics.time_order must be 1 or 2")
    limiter = num.get("limiter", "barth")
    _require(limiter in ("none", "barth"), f"{path}: unknown limiter {limiter!r}")

    out = raw.get("output") or {}
    observed = dict(out.get("observed") or {})
    _require(all(k in ("max_depth", "arrival", "arrival_diffs",
                       "telemac_ref")
                 for k in observed),
             f"{path}: output.observed keys must be max_depth/arrival/"
             f"arrival_diffs/telemac_ref, got {sorted(observed)}")
    if "telemac_ref" in observed:
        ref = dict(observed["telemac_ref"])
        _require("file" in ref,
                 f"{path}: telemac_ref needs file: {ref}")
        fp = Path(str(ref["file"]))
        if not fp.is_absolute():
            fp = path.parent / fp
        _require(fp.is_file(), f"{path}: telemac_ref file missing: {fp}")
        ref["file"] = str(fp)
        observed["telemac_ref"] = ref
    output = OutputConfig(
        dir=out.get("dir", "output"),
        vtk_interval=float(out.get("vtk_interval", 0.0)),
        csv_interval=float(out.get("csv_interval", 0.0)),
        monitors=list(out.get("monitors") or []),
        centerline_y=float(out.get("centerline_y", 0.0)),
        band=float(out.get("band", 0.05)),
        average_window=float(out.get("average_window", 0.0)),
        observed=observed,
    )
    # observed points are monitored automatically (name carried through);
    # a point already monitored (e.g. listed under two observed kinds)
    # is kept once -- compare_obs keys entries by name, not by monitor row
    seen = {(round(float(m["x"]), 9), round(float(m["y"]), 9))
            for m in output.monitors}
    for entry in observed.get("max_depth", []) + observed.get("arrival", []):
        _require("name" in entry and "x" in entry and "y" in entry,
                 f"{path}: observed entries need name/x/y: {entry}")
        key = (round(float(entry["x"]), 9), round(float(entry["y"]), 9))
        if key not in seen:
            seen.add(key)
            output.monitors.append({"name": entry["name"],
                                    "x": float(entry["x"]),
                                    "y": float(entry["y"])})

    phys = raw.get("physics") or {}
    g = float(phys.get("g", 9.81))
    if abs(g - 9.81) > 1.0e-12:
        # kernels capture gravity as the compile-time constant G
        raise ValueError(f"{path}: physics.g={g} unsupported; kernels use 9.81")
    strickler = float(phys.get("strickler", 0.0))
    _require(strickler >= 0.0, f"{path}: physics.strickler must be >= 0")
    chezy = float(phys.get("chezy", 0.0))
    _require(chezy >= 0.0, f"{path}: physics.chezy must be >= 0")
    nikuradse = float(phys.get("nikuradse", 0.0))
    _require(nikuradse >= 0.0, f"{path}: physics.nikuradse must be >= 0")
    _require(sum(1.0 for v in (strickler, chezy, nikuradse) if v > 0.0) <= 1,
             f"{path}: physics friction law must be unique "
             f"(strickler/chezy/nikuradse are mutually exclusive)")
    wind = dict(phys.get("wind") or {})
    coriolis = float(phys.get("coriolis", 0.0))
    return CaseConfig(
        name=str(case["name"]),
        analytic=str(case.get("analytic", "")),
        t_end=float(case["t_end"]),
        mesh=mesh,
        initial=ini,
        bathymetry=bathcfg,
        boundary=groups,
        cfl=float(num.get("cfl", 0.8)),
        order=order,
        limiter=limiter,
        time_order=time_order,
        eps_dry=float(num.get("eps_dry", 1.0e-6)),
        output=output,
        g=g,
        strickler=strickler,
        chezy=chezy,
        nikuradse=nikuradse,
        wind_u=float(wind.get("u", 0.0)),
        wind_v=float(wind.get("v", 0.0)),
        wind_coeff=float(wind.get("coeff", 0.0)),
        coriolis=coriolis,
    )
