# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Equivalence-check on host artifacts (T-027).

Per invariant 1: every template's host
build passes the equivalence check against the source JAX simulation
within the documented tolerances. Cartpole is the priority; this
module covers it under both LQR (T-022) and TrackingMPC (T-023)
templates.

The comparison runs the JAX dynamics through an ERK4 forward
integration at the same per-step ``dt`` as acados, then compares
state-by-state against acados' own predicted state trajectory.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxility import build_for_target
from jaxility.lowering import translate
from jaxility.targets import current_host_target
from jaxility.templates import lqr, set_reference_trajectory, tracking_mpc
from jaxility.testing import compare

jax.config.update("jax_enable_x64", True)


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA = _tera_available()


def _cartpole(state, control):
    g, mp, mc, L = 9.81, 0.1, 1.0, 0.5
    _, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    denom = mc + mp * sin_t * sin_t
    x_ddot = (control[0] + mp * sin_t * (L * theta_dot * theta_dot + g * cos_t)) / denom
    theta_ddot = (
        -control[0] * cos_t
        - mp * L * theta_dot * theta_dot * cos_t * sin_t
        - (mc + mp) * g * sin_t
    ) / (L * denom)
    return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])


def _rk4_step(jax_fn, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One ERK4 step of the JAX dynamics. Matches acados' ERK integrator."""
    x_j = jnp.asarray(x)
    u_j = jnp.asarray(u)
    k1 = jax_fn(x_j, u_j)
    k2 = jax_fn(x_j + 0.5 * dt * k1, u_j)
    k3 = jax_fn(x_j + 0.5 * dt * k2, u_j)
    k4 = jax_fn(x_j + dt * k3, u_j)
    return np.asarray(x_j + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def _jax_forward_trajectory(
    jax_fn,
    x_0: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Forward-integrate JAX dynamics across ``u_seq`` with ERK4."""
    n = u_seq.shape[0]
    nx = x_0.shape[0]
    states = np.zeros((n + 1, nx), dtype=np.float64)
    states[0] = x_0
    for k in range(n):
        states[k + 1] = _rk4_step(jax_fn, states[k], u_seq[k], dt)
    return states


def _acados_trajectory(solver, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(state_traj, control_traj)`` from a solved AcadosOcpSolver."""
    nx = solver.get(0, "x").shape[0]
    nu = solver.get(0, "u").shape[0]
    states = np.zeros((n + 1, nx), dtype=np.float64)
    controls = np.zeros((n, nu), dtype=np.float64)
    for k in range(n):
        states[k] = solver.get(k, "x")
        controls[k] = solver.get(k, "u")
    states[n] = solver.get(n, "x")
    return states, controls


def _trajectory_to_quantities(
    states: np.ndarray, controls: np.ndarray
) -> dict[str, np.ndarray]:
    """Map a Cartpole (state, control) trajectory to the canonical quantities.

    Cartpole state is ``[x_cart, theta_pole, x_dot, theta_dot]``. The
    canonical quantity layout used by the tolerance table is:

    * ``joint_position`` = cart position + pole angle (shape: ``(N+1, 2)``)
    * ``joint_velocity`` = their derivatives (shape: ``(N+1, 2)``)
    * ``actuator_torque`` = control (shape: ``(N, 1)``)

    Controls are padded to length ``N+1`` so all three arrays share their
    first axis. The compare() shape-mismatch guard does not fire because
    we only compare quantities that exist in both source and candidate.
    """
    n_states = states.shape[0]
    n_controls = controls.shape[0]
    assert n_states == n_controls + 1
    padded_controls = np.vstack([controls, controls[-1:]])
    return {
        "joint_position": states[:, :2],
        "joint_velocity": states[:, 2:],
        "actuator_torque": padded_controls,
    }


# ---------------------------------------------------------------------------
# Acceptance: every template's host build passes equivalence.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_cartpole_lqr_host_passes_equivalence(tmp_path: Path) -> None:
    """LQR template + host build + Cartpole: acados traj matches JAX RK4."""
    cf = translate(_cartpole, in_shapes=((4,), (1,)), name="cartpole_lqr_t027")
    spec = lqr(
        cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.3, 0.0, 0.0, 0.0),
        input_bounds=((-20.0,), (20.0,)),
        name="cartpole_lqr_t027",
        horizon_steps=20,
        time_horizon_s=1.0,
    )
    bundle = build_for_target(
        dynamics=cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes.fromhex("e7" * 32),
        work_dir=tmp_path / "build",
    )
    status = bundle.solver.solve()
    assert status == 0

    acados_states, acados_controls = _acados_trajectory(
        bundle.solver, spec.horizon_steps
    )
    dt = spec.time_horizon_s / spec.horizon_steps
    jax_states = _jax_forward_trajectory(
        _cartpole, acados_states[0], acados_controls, dt
    )

    source = _trajectory_to_quantities(jax_states, acados_controls)
    candidate = _trajectory_to_quantities(acados_states, acados_controls)
    report = compare(
        source,
        candidate,
        target_family=current_host_target().family,
        dtype="float64",
    )
    assert report.overall_passed, str(report)


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_cartpole_tracking_mpc_host_passes_equivalence(tmp_path: Path) -> None:
    """TrackingMPC template + host build: acados traj matches JAX RK4."""
    cf = translate(_cartpole, in_shapes=((4,), (1,)), name="cartpole_track_t027")
    n = 20
    target_traj = [(0.5, 0.0, 0.0, 0.0)] * (n + 1)
    spec = tracking_mpc(
        cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        reference_trajectory=target_traj,
        input_bounds=((-20.0,), (20.0,)),
        name="cartpole_track_t027",
        horizon_steps=n,
        time_horizon_s=1.0,
    )
    bundle = build_for_target(
        dynamics=cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes.fromhex("aa" * 32),
        work_dir=tmp_path / "build",
    )
    set_reference_trajectory(bundle.solver, target_traj)
    status = bundle.solver.solve()
    assert status == 0

    acados_states, acados_controls = _acados_trajectory(bundle.solver, n)
    dt = spec.time_horizon_s / n
    jax_states = _jax_forward_trajectory(
        _cartpole, acados_states[0], acados_controls, dt
    )

    source = _trajectory_to_quantities(jax_states, acados_controls)
    candidate = _trajectory_to_quantities(acados_states, acados_controls)
    report = compare(
        source,
        candidate,
        target_family=current_host_target().family,
        dtype="float64",
    )
    assert report.overall_passed, str(report)


# ---------------------------------------------------------------------------
# Tolerance table parity (PATTERNS §7.4) — verify the new rows are in
# both TOLERANCE_TABLE and EQUIVALENCE.md.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_host_target_tolerances_documented() -> None:
    """The new host rows appear in test/EQUIVALENCE.md (cross-check guard)."""
    from jaxility.testing.tolerances import TOLERANCE_TABLE

    doc = (Path(__file__).resolve().parents[1] / "EQUIVALENCE.md").read_text()
    for target_family in ("host-darwin", "host-linux"):
        assert target_family in doc, f"missing {target_family} in EQUIVALENCE.md"
        for quantity in ("joint_position", "joint_velocity", "actuator_torque"):
            key = (target_family, "float64", quantity)
            assert key in TOLERANCE_TABLE
            assert quantity in doc
