"""HLLC flux properties: hydrostatic consistency, symmetry/conservativity,
dry-state handling and shock-capturing of a known dam-break wave.
Run directly: python tests/test_riemann.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver import kernels as K  # noqa: E402
from solver.numeric import G  # noqa: E402


@wp.kernel
def _probe(ql: wp.array(dtype=wp.vec3d),
           qr: wp.array(dtype=wp.vec3d),
           nn: wp.array(dtype=wp.vec2d),
           out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    # signature: hllc_flux(h1, h2, u1, v1, u2, v2, xnn, ynn) with primitive
    # (h, u, v) on each side.
    f = K.hllc_flux(ql[i][0], qr[i][0],
                    ql[i][1], ql[i][2],
                    qr[i][1], qr[i][2],
                    nn[i][0], nn[i][1])
    out[i] = wp.vec3d(f[0], f[1], f[2])


def flux(hl, ul, vl, hr, ur, vr, nx=1.0, ny=0.0) -> np.ndarray:
    wp.init()
    dev = "cpu"
    ql = wp.array(np.array([[hl, ul, vl]], dtype=np.float64),
                  dtype=wp.vec3d, device=dev)
    qr = wp.array(np.array([[hr, ur, vr]], dtype=np.float64),
                  dtype=wp.vec3d, device=dev)
    nn = wp.array(np.array([[nx, ny]], dtype=np.float64),
                  dtype=wp.vec2d, device=dev)
    out = wp.zeros(1, dtype=wp.vec3d, device=dev)
    wp.launch(_probe, dim=1, inputs=[ql, qr, nn, out], device=dev)
    return out.numpy()[0]


def main() -> int:
    # 1) hydrostatic: identical states at rest -> (0, 0.5 g h^2, 0)
    f = flux(0.7, 0.0, 0.0, 0.7, 0.0, 0.0)
    assert abs(f[0]) < 1e-14 and abs(f[2]) < 1e-14
    assert abs(f[1] - 0.5 * 9.81 * 0.7 ** 2) < 1e-13, f"hydrostatic {f}"
    print("hydrostatic flux OK:", f)

    # 2) conservativity / anti-symmetry: the same Riemann problem seen from
    #    the other side of the face -> f(UL,UR;n) = -f(UR,UL;-n)
    f1 = flux(1.0, 1.2, 0.3, 0.4, -0.5, 0.1)
    f2 = flux(0.4, -0.5, 0.1, 1.0, 1.2, 0.3, nx=-1.0)
    assert np.abs(f1 + f2).max() < 1e-13, f"not conservative {f1} vs {f2}"
    print("conservativity (swap + normal flip) OK:", f1[0])

    # 3) rotation to n = (0,1): hydrostatic -> (0, 0, 0.5 g h^2); the
    #    tangential velocity must not pollute the normal flux
    f3 = flux(0.7, 0.0, 0.0, 0.7, 0.0, 0.0, nx=0.0, ny=1.0)
    assert abs(f3[0]) < 1e-14 and abs(f3[1]) < 1e-14
    assert abs(f3[2] - 0.5 * 9.81 * 0.7 ** 2) < 1e-13, f"rot90 {f3}"
    print("rot90 hydrostatic flux OK:", f3)

    # 4) both dry -> zero flux
    f4 = flux(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.abs(f4).max() < 1e-14
    print("both-dry zero flux OK")

    # 5) strong dam break (hl=1, hr=0.1): intermediate velocity must be
    #    bracketed by the exact Sod-type solution values
    f5 = flux(1.0, 0.0, 0.0, 0.1, 0.0, 0.0)
    # HLLC star velocity for this state: u* = 0.5(ul+ur) - (hr-hl)(al+ar)/(hl+hr)
    al, ar = np.sqrt(9.81 * 1.0), np.sqrt(9.81 * 0.1)
    ustar = -(0.1 - 1.0) * (al + ar) / 1.1
    assert f5[0] > 0.0 and f5[0] < ustar * 1.0  # mass flux in +x, h* <= hl
    print(f"dam-break flux OK: f0 = {f5[0]:.6f} (ustar ~ {ustar:.4f})")

    print("PASS: HLLC flux tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
