# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Crazyflie — TrackingMPC on mock-cortex-m (zoo entry, stub source).

Crazyflie ships in the Jaxility documentation but is **not yet in the
Jaxterity zoo**. The current entry uses a
:class:`~jaxility.testing.MockSource` stub so the Jaxility pipeline
can be exercised end-to-end now; Jaxterity is expected to land a real
Crazyflie zoo robot (rotor + contact modelling) before the
STM32H7 / Cortex-M7 bring-up. The mock contract is documented in
:doc:`README.md`.
"""

from __future__ import annotations

from ...targets import MOCK_CORTEX_M
from ...testing import MockSource, Source
from .. import ZooDeploymentConfig

_CRAZYFLIE_HANDLE_SALT = b"jaxility-zoo-crazyflie-stub"
"""Salt that pins the stub's attestation handle. When the real Jaxterity
zoo Crazyflie ships, swap this for a :class:`JaxteritySource.from_robot`
call and delete the salt."""


def _source() -> Source:
    return MockSource(
        name="crazyflie",
        initial_state=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0),
        dim=4,
        handle_salt=_CRAZYFLIE_HANDLE_SALT,
    )


def config() -> ZooDeploymentConfig:
    return ZooDeploymentConfig(
        name="crazyflie",
        source_factory=_source,
        target=MOCK_CORTEX_M,
        template="TrackingMPC",
        dtype="float32",
        n_steps=30,
        description=(
            "Crazyflie 2.X quadrotor, 4-rotor floating base. STM32H7 / "
            "Cortex-M7 is the launch-second target (FVP at v0.1). "
            "Stub source for now — Jaxterity has not yet landed "
            "Crazyflie in its zoo (needs rotor + contact modelling)."
        ),
        license="MIT (Jaxility zoo entry).",
        upstream_status="stub-pending-jaxterity",
        remaining_work=(
            "Promote to real-robot when Jaxterity ships a Crazyflie zoo "
            "entry (rotor + contact modelling required).",
            "Land the trajectory-tracking MPC template (T-023).",
            "Cortex-M7 cross-compilation lane + linker scripts (T-051).",
            "FVP-driven HIL parity (T-053).",
        ),
    )
