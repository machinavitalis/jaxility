# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""The Cartpole LQR controller — the launch / benchmark workload.

Factored into the library so both the benchmark CLI and the on-silicon tests
build the *same* controller: the analytical Cartpole dynamics, the LQR template,
and the host `build_for_target` invocation. nx=4, nu=1; 20-step horizon over 1 s
(a 50 ms shooting interval). This is the workload `jaxility bench cartpole`
measures and the T-034 HIL gate validates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp

from ..lowering import translate
from ..targets import current_host_target
from ..templates import lqr

if TYPE_CHECKING:
    from ..builder import BuildBundle

MODEL_NAME = "cartpole_lqr"
NX = 4
NU = 1
INITIAL_STATE: tuple[float, ...] = (0.3, 0.0, 0.0, 0.0)
HORIZON_STEPS = 20
TIME_HORIZON_S = 1.0
PLANT_DT = TIME_HORIZON_S / HORIZON_STEPS  # 0.05 s — the OCP shooting interval


def cartpole_dynamics(state, control):
    """Analytical Cartpole dynamics (JAX). State [x, theta, x_dot, theta_dot]."""
    g, mp, mc, length = 9.81, 0.1, 1.0, 0.5
    _, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    denom = mc + mp * sin_t * sin_t
    x_ddot = (
        control[0] + mp * sin_t * (length * theta_dot * theta_dot + g * cos_t)
    ) / denom
    theta_ddot = (
        -control[0] * cos_t
        - mp * length * theta_dot * theta_dot * cos_t * sin_t
        - (mc + mp) * g * sin_t
    ) / (length * denom)
    return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])


def build_cartpole_controller(
    work_dir: Path, *, model_name: str = MODEL_NAME
) -> BuildBundle:
    """Build the Cartpole LQR controller for the host; return the BuildBundle."""
    from ..builder import build_for_target

    cf = translate(cartpole_dynamics, in_shapes=((NX,), (NU,)), name=model_name)
    spec = lqr(
        cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=INITIAL_STATE,
        input_bounds=((-20.0,), (20.0,)),
        name=model_name,
        horizon_steps=HORIZON_STEPS,
        time_horizon_s=TIME_HORIZON_S,
    )
    return build_for_target(
        dynamics=cf,
        spec=spec,
        target=current_host_target(),
        # Self-contained benchmark/HIL workload — deliberately a fixed dummy
        # handle, not a robot's. This module measures controller solve time and
        # cross-checks parity (handle-independent), and must import cleanly
        # without Jaxterity. The *robot-rooted* deployment path — dynamics from
        # `reduced_params` and the manifest on the robot's real
        # `attestation_handle` (T-101/T-102) — lives in
        # `examples/cartpole_end_to_end.py`.
        source_attestation_handle=bytes.fromhex("34" * 32),
        work_dir=work_dir,
    )
