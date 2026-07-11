# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Host float64 reference for the deterministic HIL fixture (T-033).

This mirrors ``test/hil/fixtures/cartpole_hil.c`` byte-for-byte in its
recurrence — same constants, same explicit-Euler update order — but at
float64. It plays the role the zoo ``Source.simulate`` plays in the real
deployment path (T-034): the trusted host reference the on-target run is
compared against. The float32 (fixture) vs float64 (this) gap is exactly
what the ``cortex-a76`` × ``float32`` tolerance rows are sized for.

Keep this in lockstep with the C fixture. If one changes, the HIL
parity test will catch the drift — which is the point.
"""

from __future__ import annotations

import numpy as np

from jaxility.testing.equivalence import Trajectory

# Constants mirrored from cartpole_hil.c (the #define block).
_DT = 0.001
_G_OVER_L = 10.0
_DAMPING = 0.5
_GAIN_THETA = 8.0
_GAIN_RATE = 2.0
_THETA0 = 0.2
_SEED_STEP = 0.01


def cartpole_reference(n_steps: int, seed: int) -> Trajectory:
    """Return the float64 reference trajectory for ``(n_steps, seed)``.

    Quantities match :data:`jaxility.hil.trace.CARTPOLE_SCHEMA`:
    ``joint_position``, ``joint_velocity``, ``actuator_torque``, each of
    shape ``(n_steps, 1)``.
    """
    theta = _THETA0 + _SEED_STEP * float(seed)
    theta_dot = 0.0

    pos = np.empty((n_steps, 1), dtype=np.float64)
    vel = np.empty((n_steps, 1), dtype=np.float64)
    tau = np.empty((n_steps, 1), dtype=np.float64)

    for i in range(n_steps):
        u = -_GAIN_THETA * theta - _GAIN_RATE * theta_dot
        pos[i, 0] = theta
        vel[i, 0] = theta_dot
        tau[i, 0] = u

        accel = -_G_OVER_L * theta - _DAMPING * theta_dot + u
        theta_next = theta + _DT * theta_dot
        theta_dot_next = theta_dot + _DT * accel
        theta, theta_dot = theta_next, theta_dot_next

    return {
        "joint_position": pos,
        "joint_velocity": vel,
        "actuator_torque": tau,
    }
