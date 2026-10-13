# warpflow-swe

GPU Godunov finite-volume solver for the shallow-water equations,
written in Python + [NVIDIA Warp](https://github.com/NVIDIA/warp) and
mirroring the TELEMAC-2D finite-volume implementation. Verified against
the analytic solutions of the TELEMAC-2D validation manual — see
[VALIDATION.md](VALIDATION.md) for the full verification matrix,
acceptance numbers and reproduction steps.

## Features

- **float64 everywhere** — kernels, arrays and literals are typed 64-bit
  (Warp compiles bare Python floats as float32; `solver/numeric.py`
  centralises the discipline, `tests/test_float64.py` locks it).
- **Unstructured triangular FV mesh** — built-in rectangle mesher with
  alternating diagonals, plus a minimal SELAFIN (`.slf`) reader for
  TELEMAC meshes.
- **HLLC Riemann flux** with Audusse hydrostatic reconstruction and
  symmetric geometric source terms — exactly well-balanced
  (lake-at-rest to machine zero, `tests/test_lake_at_rest.py`).
- **Optional second order** — Green–Gauss gradients + Barth–Jespersen
  limiter + SSP-RK2, TELEMAC-style per-edge fallback to first order on
  steep-bed faces.
- **Wet/dry fronts** — EPS_FV-style drying guard plus a velocity ceiling
  on near-dry states (see VALIDATION.md note 7 for why).
- **Boundaries** — wall (mirror), outflow (zero-gradient), prescribed
  discharge (cdl_fv.f algorithm) and prescribed free-surface elevation
  (steady, harmonic in time, or a two-column time/level series file —
  e.g. a recorded wave-gauge signal).
- **Sources** — wind stress (TELEMAC option 2, conservative momentum
  form), Strickler bottom friction (semi-implicit Riccati splitting,
  stable for any dt) and Coriolis (conservative rotation form).
- **Outputs** — `.vtu` frames + `.pvd` for ParaView, monitor-point and
  mass-history CSV, analytic-comparison profiles with L1/L2 norms.
- **YAML-driven cases** — geometry, physics, numerics, boundaries and
  outputs are all data; adding a case needs no solver changes.

## Quick start

Requires Python 3.10+ and a CUDA GPU (any sm_60+; also runs on the Warp
CPU backend).

```
pip install -r requirements.txt
python test_warp_install.py                  # sanity-check the GPU toolchain
python run_case.py test_cases/dambreak/stoker.yaml
python summarize.py                          # error table over output/
python tests/test_riemann.py                 # unit tests: also
                                             #   test_float64, test_mesh,
                                             #   test_lake_at_rest, test_mass,
                                             #   test_analytic, test_sources
```

Each case writes to `output/<case>/`: `animation.pvd` (open in ParaView),
`monitors.csv`, `history.csv` (mass / dt / depth range) and, when an
analytic solution is configured, `compare.png` + `errors.csv`.

## Verification suite

| Case | Order | Result |
|---|---|---|
| dambreak (Ritter / Stoker) | 1 | L1 refinement ratio 1.65–1.70x (>= 1.6); shock 2.2 dx |
| Thacker planar / radial sloshing | 1 / 2 | front-dominated L1(h), 1.94x reduction on dx refinement |
| Seiche standing wave | 1 / 2 | amplitude decay 3.2% / 2.1% over 7.4 periods (<= 5%) |
| Bump subcritical / critical / transcritical | 1 / 2 | L1(h) 3.0e-3–5.2e-4; hydraulic jump 0.9–1.5 dx from analytic |
| Friction GVF, slope + Strickler | 1 | L1(h) 3.3e-2 m = 0.3% of depth vs the steady GVF profile |
| Harmonic progressive wave | 1 | L1(u) 6.9e-3, boundary-limited; interior-only 1.2e-3 (~1.5 orders) |
| Wind setup, closed basin | 1 | time-mean L1(h) 9.7e-4 m = 0.12% of the set-up dz |
| Balzano tidal-flat draw-down | 1 | L1(h) 5.3e-2 m at t = 70000 s; left reservoir locked at the sill crest; severe wet/dry |
| Canal stage-discharge flow | 1 | L1(h) 6.4e-4 m vs the steady GVF profile (surface 0.745 -> 0.53 m) |
| Vasque draw-down over a pool | 1 | L1(h) 4.8e-3 m; pool ponds at its spill level; partial drying |
| Tests-channel solitary-wave run-up | 1 | R/d 0.106 (0.099 with Cf = 0.001) vs experiment 0.076 / TELEMAC FV 0.084 |
| Gouttedo droplet, closed / open | 1 | mass 999.7833 vs 999.784 m3 (closed); outflow 31.2 vs 27.8 m3 (open) |

Closed-domain mass is conserved to machine precision (~1e-15 relative).

## Repository layout

```
run_case.py          # entry point: python run_case.py test_cases/<case>.yaml
summarize.py         # collect output/*/errors.csv into one table
solver/              # the library (mesh, kernels, solver, I/O, analytics)
test_cases/          # YAML case definitions (all solver input is data)
tests/               # unit tests (run individually, no framework needed)
```

## Provenance and license

The numerical methods (HLLC flux, hydrostatic reconstruction, second-order
fallback, CFL formula, boundary ghost states) mirror the TELEMAC-2D v8p4
finite-volume sources (`flux_hllc.f`, `hyd_fv.f`, `second_order.f`,
`caldt.f`, `cdl_fv.f`). This repository is an independent Python/Warp
implementation and is not part of, nor endorsed by, the TELEMAC-MASCARET
consortium. Licensed under GPL-3.0 (see [LICENSE](LICENSE)), the same
license family as TELEMAC-2D.
