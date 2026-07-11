# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Host closed-loop reference for the acados Cartpole controller (T-034).

The controller + dynamics live in :mod:`jaxility.bench.cartpole` (shared with the
benchmark CLI); this module adds the test-only *reference* closed loop: the
acados OCP solver (in Python) computes the control and a JAX ERK4 integrator
advances the plant. The generated on-target binary runs the same control with
the acados sim-solver plant; the HIL parity check compares the two. The
plant-model mismatch (JAX ERK4 vs acados sim) is the dominant, deliberate source
of divergence — measured ~1e-14 (ULP), so it is well inside the documented
`cortex-a76` x float64 bounds.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxility.bench.cartpole import (
    INITIAL_STATE,
    MODEL_NAME,
    PLANT_DT,
    build_cartpole_controller,
    cartpole_dynamics,
)
from jaxility.testing.equivalence import Trajectory

__all__ = [
    "INITIAL_STATE",
    "MODEL_NAME",
    "PLANT_DT",
    "build_cartpole_lqr",
    "closed_loop_reference",
]


def build_cartpole_lqr(work_dir):
    """Build the Cartpole LQR controller for the host; return the BuildBundle."""
    return build_cartpole_controller(work_dir)


def _rk4_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One ERK4 step of the JAX dynamics (matches acados' ERK integrator)."""
    x_j, u_j = jnp.asarray(x), jnp.asarray(u)
    k1 = cartpole_dynamics(x_j, u_j)
    k2 = cartpole_dynamics(x_j + 0.5 * dt * k1, u_j)
    k3 = cartpole_dynamics(x_j + 0.5 * dt * k2, u_j)
    k4 = cartpole_dynamics(x_j + dt * k3, u_j)
    return np.asarray(x_j + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def closed_loop_reference(solver, n_steps: int, seed: int) -> Trajectory:
    """Run the host closed loop: acados control + JAX ERK4 plant.

    Returns the trajectory in the nx=4 Cartpole quantity layout
    (`joint_position`/`joint_velocity` 2-vectors, `actuator_torque` 1-vector),
    matching jaxility.hil.CARTPOLE_LQR_SCHEMA and the generated binary's trace.
    """
    x = np.array(INITIAL_STATE, dtype=np.float64)
    x[0] += 0.01 * float(seed)  # same seed perturbation as the generated binary

    pos = np.empty((n_steps, 2), dtype=np.float64)
    vel = np.empty((n_steps, 2), dtype=np.float64)
    tau = np.empty((n_steps, 1), dtype=np.float64)

    for i in range(n_steps):
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        status = solver.solve()
        if status != 0:
            raise RuntimeError(f"reference acados solve failed at step {i}: {status}")
        u = np.asarray(solver.get(0, "u"), dtype=np.float64)
        pos[i] = x[:2]
        vel[i] = x[2:]
        tau[i] = u
        x = _rk4_step(x, u, PLANT_DT)

    return {"joint_position": pos, "joint_velocity": vel, "actuator_torque": tau}
