# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Cartpole — LQR on mock-cortex-a (zoo entry).

Source: Jaxterity zoo. Target: mock-cortex-a. Controller
template: LQR. See :doc:`README.md` for source URL, license, and
remaining work.
"""

from __future__ import annotations

from ...targets import MOCK_CORTEX_A
from ...testing import JaxteritySource, Source
from .. import ZooDeploymentConfig


def _load_robot():
    """Load the upstream Jaxterity Cartpole Robot.

    The *same* deterministic robot anchors both the attestation handle
    (:func:`_source`) and the lowered dynamics' scalars
    (:func:`_dynamics_factory`), so the manifest's provenance and the deployed
    plant describe one object (Jaxterity Invariant 1, "one model, one truth").
    """
    from jaxterity.zoo import load

    return load("cartpole")


def _source() -> Source:
    # Zoo robots currently arrive UNCALIBRATED; SKILL.md advertises the
    # production compiler will refuse uncalibrated robots. A later change
    # tightens this — once the sysid recipe lands upstream in Jaxterity,
    # bump to ``require_calibration_state="CALIBRATED"`` and add a CI
    # fixture that calibrates the zoo entry before deployment.
    return JaxteritySource.from_robot(
        _load_robot(), dim=2, require_calibration_state=None
    )


def _dynamics_factory():
    """Closed-form Cartpole dynamics (smooth-op subset), parameterised by the Robot.

    Jaxterity's ``Robot.build_system`` exposes the dynamics via MJX, whose
    constraint-solver ``while_loop`` cannot be lowered to acados' fixed-size SQP
    graph (ADR-016: *"closed-form per-robot dynamics is the contract"*). So the
    deployed plant is this four-state closed-form, not MJX. To keep it faithful
    to *this* robot, the four scalars (``g, mc, mp, L``) are read from the
    upstream Robot via :func:`jaxterity.zoo.cartpole.reduced_params` rather than
    hand-hardcoded here — calibrating a mass/length now propagates into the
    lowered binary, and the attestation handle (``_source``) and the deployed
    dynamics move together. Joint damping stays out of the closed-form by design
    (it is the frictionless host-equivalence reference; see ``reduced_params``).
    """
    from jaxterity.zoo.cartpole import reduced_params

    return _cartpole_ode(reduced_params(_load_robot())), (4,), (1,)


def _cartpole_ode(p):
    """Frictionless point-mass cartpole ``f(state, control) -> dstate`` from the
    reduced scalars ``p = {g, mc, mp, L}``.

    Split out from :func:`_dynamics_factory` so the closed-form can be exercised
    with parameters from *any* (e.g. calibrated) robot — the T-101 faithfulness
    test feeds it ``reduced_params`` of a recalibrated robot to show a parameter
    change moves the deployed dynamics, not just the attestation handle.
    """
    import jax.numpy as jnp

    g, mp, mc, L = p["g"], p["mp"], p["mc"], p["L"]

    def cartpole(state, control):
        theta, x_dot, theta_dot = state[1], state[2], state[3]
        sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
        denom = mc + mp * sin_t * sin_t
        x_ddot = (
            control[0] + mp * sin_t * (L * theta_dot * theta_dot + g * cos_t)
        ) / denom
        theta_ddot = (
            -control[0] * cos_t
            - mp * L * theta_dot * theta_dot * cos_t * sin_t
            - (mc + mp) * g * sin_t
        ) / (L * denom)
        return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])

    return cartpole


def config() -> ZooDeploymentConfig:
    return ZooDeploymentConfig(
        name="cartpole",
        source_factory=_source,
        target=MOCK_CORTEX_A,
        template="LQR",
        dtype="float64",
        n_steps=50,
        description=(
            "Cart + pole slider, 2 DoF, fixed base. Jaxility's launch demo "
            "is Cartpole-on-Pi-5 at 1 kHz; the zoo entry exercises "
            "the same pipeline on mock-cortex-a."
        ),
        license="MIT (Jaxterity zoo + Jaxility zoo entry).",
        upstream_status="real-robot",
        remaining_work=(
            "Replace the synthetic simulate with MJX-driven "
            "trajectory via robot.to_diagram() (T-026).",
            "Land the real LQR template (T-022).",
            "Wire to Pi 5 / Cortex-A76 toolchain (T-031).",
            "HIL parity tests against the deployed binary (T-033).",
        ),
        jax_dynamics_factory=_dynamics_factory,
        template_options={
            "Q": (10.0, 10.0, 1.0, 1.0),
            "R": (0.1,),
            "initial_state": (0.5, 0.0, 0.0, 0.0),
            "input_bounds": ((-20.0,), (20.0,)),
            "horizon_steps": 20,
            "time_horizon_s": 1.0,
        },
    )
