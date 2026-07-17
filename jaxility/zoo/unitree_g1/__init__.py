# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Unitree G1 — the zoo's first *branched* robot, dynamics from the generator.

Source: Jaxterity zoo (real Unitree G1, fixed-base, 29-DoF). Target:
mock-cortex-a. Controller template: WBC (single joint-space regulation task).

Unlike the hand-written zoo entries (Cartpole / Crazyflie / SO-100), G1's
deployable dynamics are **generated** from the robot's MJCF by
:func:`jaxility.lowering.generate_dynamics` with ``tree_source="mujoco"`` — a
branched humanoid the hand-written serial-chain ABA could never express (T-126).
The MuJoCo tree source reads ``dof_armature`` directly (which dominates the
effective inertia of the light distal joints), so the deployed dynamics tracks
MJX. See :doc:`README.md` for fidelity and remaining work.
"""

from __future__ import annotations

from typing import Any

from ...targets import MOCK_CORTEX_A
from ...testing import JaxteritySource, Source
from .. import ZooDeploymentConfig

_NV = 29  # 29 revolute DoF, fixed base -> nx = 58, nu = 29
_NX = 2 * _NV


def _robot() -> Any:
    from jaxterity.zoo import load

    return load("unitree_g1")


def _source() -> Source:
    # Zoo robots arrive UNCALIBRATED (see SO-100); opt out of the gate for now.
    return JaxteritySource.from_robot(_robot(), dim=_NV, require_calibration_state=None)


def _casadi_dynamics() -> Any:
    """Generate G1's branched rigid-body dynamics from its MJCF (T-126).

    ``tree_source="mujoco"`` builds the tree from the compiled MuJoCo model and
    auto-reads ``dof_armature`` — validated against unconstrained RBD to ~1e-7.
    """
    from ...lowering import generate_dynamics

    return generate_dynamics(
        _robot().to_mjcf(),
        source_format="mjcf",
        tree_source="mujoco",
        name="unitree_g1",
    )


def config() -> ZooDeploymentConfig:
    return ZooDeploymentConfig(
        name="unitree_g1",
        source_factory=_source,
        target=MOCK_CORTEX_A,
        template="WBC",
        dtype="float64",
        n_steps=50,
        description=(
            "Unitree G1 humanoid, 29 DoF, fixed base (welded pelvis). The zoo's "
            "first branched robot; its deployable dynamics are generated from the "
            "MJCF via the MuJoCo tree source, not hand-written."
        ),
        license="MIT (Jaxterity zoo + Jaxility zoo entry).",
        upstream_status="real-robot",
        remaining_work=(
            "Richer multi-task whole-body control via the Jaxterity Task DSL (T-024).",
            "Floating base + contact for locomotion (parked — T-122).",
            "Bring up a real-target lane (post-launch — T-070).",
        ),
        casadi_dynamics_factory=_casadi_dynamics,
        template_options={
            # Single joint-space regulation to the neutral pose. WBC builds one
            # task from Q / R (build_cmd) until the Jaxterity Task DSL lands.
            "Q": tuple([1.0] * _NX),
            "R": tuple([0.1] * _NV),
            "initial_state": tuple([0.0] * _NX),
        },
    )
