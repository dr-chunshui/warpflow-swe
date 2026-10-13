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
    select: dict  # {"xmin": [y0, y1]} or {"ymin": [x0, x1]}, single key
    type: str  # wall | outflow | inflow_q | level
    value: float = 0.0  # zs for level [m], Q total for inflow_q [m^3/s]
    time: dict = field(default_factory=dict)
    # time-dependent level value: zs(t) = mean + amp*sin(2*pi*t/period+phase)
    # keys: mean, amp, period; optional phase [rad] (default 0).  When
    # ``time`` is non-empty it overrides ``value`` for ``level`` groups.


@dataclass
class OutputConfig:
    dir: str = "output"
    vtk_interval: float = 0.0  # <=0 disables frame dumps
    csv_interval: float = 0.0  # <=0 disables monitor/mass csv
    monitors: list = field(default_factory=list)  # [{x, y}]
    centerline_y: float = 0.0
    band: float = 0.05  # |y - centerline_y| half width for profile extraction
    average_window: float = 0.0  # time-average state over last N s (<=0 off)


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
    wind_u: float = 0.0  # wind velocity [m/s]
    wind_v: float = 0.0
    wind_coeff: float = 0.0  # a_wind * rho_air / rho_water, 0 = off


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
    else:
        raise ValueError(f"{path}: unknown mesh.type {mesh.type!r}")

    initial = raw.get("initial") or {}
    _require("type" in initial, f"{path}: initial.type missing")
    ini = InitialConfig(type=initial["type"],
                        params=dict(initial.get("params") or {}))

    bath = raw.get("bathymetry") or {"type": "flat", "params": {}}
    bathcfg = BathymetryConfig(type=bath.get("type", "flat"),
                               params=dict(bath.get("params") or {}))
    _require(bathcfg.type in ("flat", "bump", "thacker", "slope"),
             f"{path}: unknown bathymetry type {bathcfg.type!r}")

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
        _require(side in ("xmin", "xmax", "ymin", "ymax"),
                 f"{path}: select side {side!r} invalid")
        _require(isinstance(span, list) and len(span) == 2,
                 f"{path}: select span must be [a, b]: {g.get('name')}")
        tspec = dict(g.get("time") or {})
        if tspec:
            _require(g["type"] == "level",
                     f"{path}: time-dependent value only for level: {g['name']}")
            _require("mean" in tspec and "amp" in tspec and "period" in tspec,
                     f"{path}: bc time needs mean/amp/period: {g['name']}")
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
    output = OutputConfig(
        dir=out.get("dir", "output"),
        vtk_interval=float(out.get("vtk_interval", 0.0)),
        csv_interval=float(out.get("csv_interval", 0.0)),
        monitors=list(out.get("monitors") or []),
        centerline_y=float(out.get("centerline_y", 0.0)),
        band=float(out.get("band", 0.05)),
        average_window=float(out.get("average_window", 0.0)),
    )

    phys = raw.get("physics") or {}
    g = float(phys.get("g", 9.81))
    if abs(g - 9.81) > 1.0e-12:
        # kernels capture gravity as the compile-time constant G
        raise ValueError(f"{path}: physics.g={g} unsupported; kernels use 9.81")
    strickler = float(phys.get("strickler", 0.0))
    _require(strickler >= 0.0, f"{path}: physics.strickler must be >= 0")
    wind = dict(phys.get("wind") or {})
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
        wind_u=float(wind.get("u", 0.0)),
        wind_v=float(wind.get("v", 0.0)),
        wind_coeff=float(wind.get("coeff", 0.0)),
    )
