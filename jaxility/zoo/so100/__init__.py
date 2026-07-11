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


def _source() -> Source:
    from jaxterity.zoo import load

    robot = load("so100")
    # Zoo robots currently arrive UNCALIBRATED; SKILL.md advertises the
    # production compiler will refuse uncalibrated robots. A later change
    # tightens this — bump to ``require_calibration_state="CALIBRATED"``
    # once the sysid recipe lands.
    return JaxteritySource.from_robot(robot, dim=6, require_calibration_state=None)


# Note: no ``_dynamics_factory`` for SO-100 yet. The
# manipulator dynamics live inside Jaxterity / MJX and use scatter /
# gather primitives outside the T-020 smooth-op subset; no analytical
# 6-DOF replacement is bundled. The CLI surfaces this with a clear
# "no jax_dynamics_factory" message. A later change either lands
# the MJX → CasADi pass (smoothed scatter handlers) or bundles an
# analytical SO-100 dynamics shim.


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
            "Replace the synthetic simulate with MJX-driven "
            "trajectory via robot.to_diagram() (T-026).",
            "Land the WBC template + Jaxterity Task DSL consumer (T-024).",
            "Bring up a real-target lane (post-launch — T-070).",
        ),
        # ``jax_dynamics_factory=None`` (the default). MJX gap surfaces
        # in the CLI as a "no dynamics factory" structured error; the
        # Python API (``build_for_target``) still works for any caller
        # who hand-rolls SO-100 dynamics in the smooth-op subset.
    )
