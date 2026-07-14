# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""SO-100 — task-space WBC on mock-cortex-a (zoo entry).

Source: Jaxterity zoo (real SO-101 URDF). Target: mock-cortex-a.
Controller template: WBC. See :doc:`README.md` for source URL,
license, and remaining work.
"""

from __future__ import annotations

from ...targets import MOCK_CORTEX_A
from ...testing import JaxteritySource, Source
from .. import ZooDeploymentConfig


def _load_robot():
    """Load the upstream Jaxterity SO-100 / SO-101 Robot (serial 6-DoF arm).

    The *same* deterministic robot anchors both the attestation handle
    (:func:`_source`) and the lowered dynamics' spatial tree
    (:func:`_dynamics_factory`) — one model, one truth.
    """
    from jaxterity.zoo import load

    return load("so100")


def _source() -> Source:
    # Zoo robots currently arrive UNCALIBRATED; SKILL.md advertises the
    # production compiler will refuse uncalibrated robots. A later change
    # tightens this — bump to ``require_calibration_state="CALIBRATED"``
    # once the sysid recipe lands.
    return JaxteritySource.from_robot(
        _load_robot(), dim=6, require_calibration_state=None
    )


def _dynamics_factory():
    """Closed-form manipulator dynamics (Featherstone ABA) for SO-100.

    Unlike the flyers' explicit closed form, a manipulator needs ``M(q)⁻¹`` —
    absent from the smooth-op subset — so the deployed plant is the O(n)
    articulated-body recursion (:mod:`._dynamics`), which lowers to CasADi
    using only spatial matmuls and scalar reciprocals. The spatial tree
    (transforms, axes, per-link mass/com/inertia) is read from the calibrated
    Robot, so a recalibration propagates into the lowered binary.

    Returns ``(f, state_shape, control_shape)`` with ``state = [q(6), q̇(6)]``
    and ``control = τ(6)``.
    """
    from ._dynamics import manipulator_ode, spatial_tree

    robot = _load_robot()
    f, n = manipulator_ode(spatial_tree(robot), float(robot.gravity))
    return f, (2 * n,), (n,)


def config() -> ZooDeploymentConfig:
    return ZooDeploymentConfig(
        name="so100",
        source_factory=_source,
        target=MOCK_CORTEX_A,
        template="WBC",
        dtype="float64",
        n_steps=50,
        description=(
            "SO-100 / SO-101 desktop manipulator, 6 DoF, fixed base. The "
            "zoo entry exercises the mock pipeline on the real "
            "Jaxterity SO-101 URDF."
        ),
        license="MIT (Jaxterity zoo + Jaxility zoo entry).",
        upstream_status="real-robot",
        remaining_work=(
            "Consume the Jaxterity Task DSL for richer WBC tasks (T-024); the "
            "current entry uses a single joint-space regulation task.",
            "Bring up a real-target lane (post-launch — T-070).",
        ),
        jax_dynamics_factory=_dynamics_factory,
        # WBC as a single joint-space regulation task: hold a bent pose (T-024).
        # State = [q(6), q̇(6)]; the CLI auto-builds one WBCTask from Q/R.
        template_options={
            "Q": (12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "R": (0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            "initial_state": (
                0.3,
                -0.8,
                1.2,
                -0.6,
                0.4,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            "input_bounds": ((-5.0,) * 6, (5.0,) * 6),  # joint torque limits (N·m)
            "horizon_steps": 20,
            "time_horizon_s": 1.0,
        },
    )
