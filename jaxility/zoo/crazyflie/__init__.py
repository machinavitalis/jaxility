# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Crazyflie — closed-form quadrotor dynamics on mock-cortex-m (zoo entry).

Source: Jaxterity zoo (real mesh-free Crazyflie 2.0 MJCF). Target:
mock-cortex-m. Controller template: TrackingMPC. See :doc:`README.md`
for source URL, license, and remaining work.

Like the Cartpole entry, the deployed plant is a **closed-form** rigid-body
model, not MJX: MJX exposes the dynamics through a constraint-solver
``while_loop`` that cannot be lowered to acados' fixed-size SQP graph
(ADR-016 — *"closed-form per-robot dynamics is the contract"*). A quadrotor is
a single free rigid body, so its closed form is the Newton-Euler equations of a
floating base, and it stays inside the smooth-op subset (arithmetic + products,
no branches) so it lowers to CasADi.

To keep it faithful to *this* robot, the scalars ``(m, I, g)`` are read from the
upstream Robot (:func:`_reduced_params`) rather than hardcoded — calibrating a
mass or inertia now propagates into the lowered binary, so the attestation
handle (:func:`_source`) and the deployed dynamics move together. The closed
form assumes the body COM sits at the free-joint frame origin (true for the
vendored ``cf2.xml``; ``links[0].com == [0, 0, 0]``); a calibration that offsets
the COM would need the parallel-axis term added here.
"""

from __future__ import annotations

from ...targets import MOCK_CORTEX_M
from ...testing import JaxteritySource, Source
from .. import ZooDeploymentConfig


def _load_robot():
    """Load the upstream Jaxterity Crazyflie Robot (floating base, ``nq == 7``).

    The *same* deterministic robot anchors both the attestation handle
    (:func:`_source`) and the lowered dynamics' scalars
    (:func:`_dynamics_factory`), so the manifest's provenance and the deployed
    plant describe one object (Jaxterity Invariant 1, "one model, one truth").
    """
    from jaxterity.zoo import load

    return load("crazyflie")


def _source() -> Source:
    # Zoo robots currently arrive UNCALIBRATED; SKILL.md advertises the
    # production compiler will refuse uncalibrated robots. A later change
    # tightens this to ``require_calibration_state="CALIBRATED"`` once the
    # sysid recipe lands upstream in Jaxterity.
    return JaxteritySource.from_robot(
        _load_robot(), dim=6, require_calibration_state=None
    )


def _reduced_params(robot) -> dict:
    """The scalars that parameterise the closed form: mass, principal inertia,
    gravity — read from the calibrated Robot (Crazyflie analog of Cartpole's
    ``jaxterity.zoo.cartpole.reduced_params``).

    The vendored ``cf2.xml`` is a single rigid body with a diagonal inertia
    tensor about a COM at the body origin, so the principal inertia is the
    diagonal ``(Ixx, Iyy, Izz)``.
    """
    import numpy as np

    link = robot.links[0]
    inertia = np.asarray(link.inertia)
    return {
        "m": float(link.mass),
        "I": (float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2])),
        "g": float(robot.gravity),
    }


def _dynamics_factory():
    """Closed-form quadrotor dynamics (smooth-op subset), parameterised by the Robot.

    Returns ``(f, state_shape, control_shape)`` where ``f(state, control) ->
    dstate`` over the 13-state floating base and the 4-vector body wrench.
    """
    return _quadrotor_ode(_reduced_params(_load_robot())), (13,), (4,)


def _quadrotor_ode(p):
    """Newton-Euler ``f(state, control) -> dstate`` of a free rigid body from the
    reduced scalars ``p = {m, I, g}``.

    State layout matches the upstream free joint
    (``jaxterity.zoo.crazyflie.thrust_dynamics``)::

        state   = [ pos(3), quat_wxyz(4), v_world(3), omega_body(3) ]   # 13
        control = [ thrust, Mx, My, Mz ]                    # 4 (body frame)
        dstate  = [ pos_dot(3), quat_dot(4), v_world_dot(3), omega_body_dot(3) ]

    Split out from :func:`_dynamics_factory` so the closed form can be exercised
    with parameters from *any* (e.g. recalibrated) robot — the faithfulness test
    feeds it ``_reduced_params`` of a heavier robot to show a mass change moves
    the deployed dynamics, not just the attestation handle.
    """
    import jax.numpy as jnp

    m, g = p["m"], p["g"]
    ix, iy, iz = p["I"]

    def quadrotor(state, control):
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        v_world = state[7:10]
        wx, wy, wz = state[10], state[11], state[12]
        thrust, mx, my, mz = control[0], control[1], control[2], control[3]

        # Body +z axis in the world frame (third column of R(quat)) — the same
        # expression the upstream ``control_to_generalized_force`` uses.
        z_body_world = jnp.array(
            [
                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ]
        )

        # Configuration derivative: position and unit-quaternion kinematics
        # q_dot = 0.5 * q ⊗ (0, omega_body).
        pos_dot = v_world
        quat_dot = 0.5 * jnp.array(
            [
                -(qx * wx + qy * wy + qz * wz),
                qw * wx + qy * wz - qz * wy,
                qw * wy - qx * wz + qz * wx,
                qw * wz + qx * wy - qy * wx,
            ]
        )

        # Translational: m * v_dot = thrust along body +z, in world, plus gravity.
        v_world_dot = thrust * z_body_world / m + jnp.array([0.0, 0.0, -g])

        # Rotational (Euler, diagonal inertia): I * omega_dot = M - omega x (I omega).
        omega_body_dot = jnp.array(
            [
                (mx - (wy * (iz * wz) - wz * (iy * wy))) / ix,
                (my - (wz * (ix * wx) - wx * (iz * wz))) / iy,
                (mz - (wx * (iy * wy) - wy * (ix * wx))) / iz,
            ]
        )

        return jnp.concatenate([pos_dot, quat_dot, v_world_dot, omega_body_dot])

    return quadrotor


def config() -> ZooDeploymentConfig:
    return ZooDeploymentConfig(
        name="crazyflie",
        source_factory=_source,
        target=MOCK_CORTEX_M,
        template="TrackingMPC",
        dtype="float32",
        n_steps=30,
        description=(
            "Crazyflie 2.0 nano-quadrotor, single free rigid body (13-state "
            "floating base). The deployed plant is the closed-form Newton-Euler "
            "dynamics sourced from the calibrated Robot; STM32H7 / Cortex-M7 is "
            "the launch-second target."
        ),
        license="MIT (Jaxterity zoo + Jaxility zoo entry).",
        upstream_status="real-robot",
        remaining_work=(
            "Wire the quaternion-aware tracking-MPC OCP template (follow-on to "
            "T-110; attitude in acados needs a unit-norm handling choice).",
            "Cortex-M7 cross-compilation lane + linker scripts (T-051/T-052).",
            "FVP-driven HIL parity (T-053).",
        ),
        jax_dynamics_factory=_dynamics_factory,
    )
