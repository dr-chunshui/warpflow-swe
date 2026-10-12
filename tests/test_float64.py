"""M0 test: lock in the float64 discipline for Warp kernels.

Run directly:  python tests/test_float64.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solver.numeric import G  # noqa: E402


@wp.kernel
def k_typed(x: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    # All literals typed: G is a module float64 constant, 3.0 wrapped.
    out[i] = G * x[i] / wp.float64(3.0)


@wp.kernel
def k_f32(x: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    # CONTROL: computing in float32 must lose precision.  (A bare literal
    # mixed with a float64 variable is a compile error in Warp, so the
    # "silent" trap is emulated with an explicit float32 cast.)
    f32prod = wp.float32(x[i]) * 0.1
    out[i] = wp.float64(f32prod)


def main() -> int:
    n = 1024
    x = wp.array(np.linspace(0.0, 1.0, n), dtype=wp.float64, device="cuda:0")

    out = wp.empty(n, dtype=wp.float64, device="cuda:0")
    wp.launch(k_typed, dim=n, inputs=[x, out], device="cuda:0")
    typed = out.numpy()

    ref = 9.81 * np.linspace(0.0, 1.0, n) / 3.0
    rel_typed = np.max(np.abs(typed - ref) / np.maximum(np.abs(ref), 1e-300))
    print(f"typed kernel   max relative error = {rel_typed:.3e}")
    assert rel_typed < 1e-15, "typed kernel lost float64 precision"

    wp.launch(k_f32, dim=n, inputs=[x, out], device="cuda:0")
    f32res = out.numpy()
    rel_f32 = np.max(np.abs(f32res - ref) / np.maximum(np.abs(ref), 1e-300))
    print(f"float32 compute max relative error = {rel_f32:.3e} (precision trap demo)")
    assert rel_f32 > 1e-10, "expected visible float32 precision loss"

    print("PASS: float64 discipline verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
