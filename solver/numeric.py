"""Float64 discipline helpers for Warp kernels.

Warp compiles bare Python float literals inside ``@wp.kernel`` bodies as
float32 (the codegen maps Python ``float`` to ``wp.float32``).  Every
floating-point literal used inside a kernel must therefore be typed
explicitly as ``wp.float64(...)`` (typed constructor calls are emitted at
64-bit precision), or come from a module-level typed constant such as ``G``.

``wp.constant`` is NOT used: in Warp 1.18.0 it does not accept a ``dtype``
keyword and passes Python floats through as float32.

All arrays in this solver are created with ``dtype=wp.float64``; numpy
intermediates use ``np.float64``.
"""
from __future__ import annotations

import numpy as np
import warp as wp

# Warp scalar type used everywhere in this solver.
F64 = wp.float64

# Module-level typed constant (captured as a compile-time constant by Warp).
G = wp.float64(9.81)  # gravitational acceleration [m/s^2]

# Wet/dry tolerance, mirrors EPS_FV in TELEMAC-2D finite volume schemes.
EPS_F64 = wp.float64(1.0e-6)

# Velocity ceiling for the near-dry grey zone (EPS_F64 < h << flow depth).
# A "bottomed-out" cell (free surface collapsed onto the bed during a
# violent transient) keeps its momentum slot, so u = hu/h is unbounded
# and the HLLC flux feeds the momentum back in a positive loop until
# speed = inf and dt = 0.  50 m/s is an order of magnitude above every
# physical velocity of the validation suite (<= 5 m/s), so the clamp is
# inert on legitimate flow and only fires on non-physical states.
U_MAX = wp.float64(5.0e1)  # [m/s]


def as_np64(array) -> np.ndarray:
    """Return a float64 numpy copy of an array-like."""
    return np.asarray(array, dtype=np.float64)
