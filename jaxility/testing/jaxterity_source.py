# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Adapter from a real Jaxterity :class:`Robot` to the :class:`Source`.

T-016 is the end-to-end attestation-chain integration test. Jaxterity
exposes the (currently-named) :class:`jaxterity.robot.Robot` with an
``attestation_handle`` hex string; this module bridges it onto the
:class:`~jaxility.testing.sources.Source` Protocol the mock
pipeline consumes, so the chain runs unmodified from a Jaxterity zoo
robot down through a Jaxility manifest and artifact.

Naming note: the Jaxility documentation set was drafted while the
upstream type was still called ``CalibratedRobot``. The real Jaxterity
ships a single :class:`Robot` whose ``calibration_state`` enum
(``UNCALIBRATED`` / ``PARTIAL`` / ``CALIBRATED``) marks where the robot
is in the sysid pipeline. :func:`from_robot` accepts the real Robot
and verifies its calibration state matches the caller's expectation —
the deployment compiler should refuse to consume an uncalibrated robot
in production (CONTEXT.md ADR-005 chain) but tests opt in to
``UNCALIBRATED`` so the bare zoo robots can be exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..errors import SourceError
from .equivalence import Trajectory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jaxterity.robot import Robot

_CalibrationState = Literal["UNCALIBRATED", "PARTIAL", "CALIBRATED"]


@dataclass(frozen=True)
class JaxteritySource:
    """Wraps a :class:`jaxterity.robot.Robot` as a Jaxility ``Source``.

    Carries:

    * :attr:`name` — the robot's name (``"physics"`` for Cartpole's
      MJCF, ``"so100"`` for the SO-101 zoo entry — Jaxterity's naming
      derives from the URDF / MJCF model name).
    * :attr:`attestation_handle` — the real Robot's BLAKE3 handle,
      decoded from hex to bytes for the manifest hash chain.

    :meth:`simulate` returns a deterministic synthetic trajectory
    seeded from the handle bytes. The mock pipeline's equivalence
    check then trivially passes (the bundle's ``simulate`` re-runs
    the same synthetic). A later change will replace this with a
    real Jaxonomy-driven simulation through ``robot.to_diagram()``.

    The synthetic-trajectory choice is deliberate for now: the
    test surface for T-016 is the attestation chain, not the
    dynamics. Stubbing the dynamics keeps the test fast and avoids a
    transitive dependency on MJX during ``pytest``.
    """

    name: str
    attestation_handle: bytes
    robot: object  # the underlying jaxterity.robot.Robot
    dim: int = 2

    @classmethod
    def from_robot(
        cls,
        robot: Robot,
        *,
        dim: int = 2,
        require_calibration_state: _CalibrationState | None = None,
    ) -> JaxteritySource:
        """Construct a source from a Jaxterity Robot.

        Args
        ----
        robot : jaxterity.robot.Robot
            The Jaxterity robot. (Documentation references
            ``CalibratedRobot`` — see module docstring for why this
            wrapper accepts the real type.)
        dim : int
            Trajectory output dimension. Defaults to ``2`` (cartpole).
        require_calibration_state : "UNCALIBRATED" | "PARTIAL" | "CALIBRATED" | None
            If supplied, raise when the robot's calibration state
            does not match. ``None`` accepts any state — tests
            opt into ``UNCALIBRATED`` so zoo robots can be
            exercised without running sysid first.
        """
        if require_calibration_state is not None:
            actual = robot.calibration_state.name  # type: ignore[attr-defined]
            if actual != require_calibration_state:
                raise SourceError(
                    f"expected robot.calibration_state == "
                    f"{require_calibration_state!r}, got {actual!r}. "
                    "Run the upstream sysid recipe before deployment, "
                    "or pass require_calibration_state=None to override."
                )
        handle_hex = robot.attestation_handle  # type: ignore[attr-defined]
        return cls(
            name=robot.name,  # type: ignore[attr-defined]
            attestation_handle=bytes.fromhex(handle_hex),
            robot=robot,
            dim=dim,
        )

    def simulate(self, n_steps: int) -> Trajectory:
        if n_steps < 1:
            raise SourceError(f"n_steps must be >= 1, got {n_steps}")
        # Seed the trajectory phase from the first 8 handle bytes so any
        # chain mutation perturbs the trajectory deterministically. This
        # keeps the equivalence test trivially passing (source vs. bundle
        # are byte-identical inside one mock_lower call) while making
        # cross-test trajectories visibly different.
        seed = int.from_bytes(self.attestation_handle[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        t = np.linspace(0.0, 1.0, n_steps)
        pos = np.stack(
            [np.sin(2.0 * np.pi * (t + 0.1 * i) + phase) for i in range(self.dim)],
            axis=1,
        )
        vel = np.stack(
            [
                2.0 * np.pi * np.cos(2.0 * np.pi * (t + 0.1 * i) + phase)
                for i in range(self.dim)
            ],
            axis=1,
        )
        torque = 0.1 * np.cos(np.outer(t, np.arange(self.dim) + 1))
        return {
            "joint_position": pos,
            "joint_velocity": vel,
            "actuator_torque": torque,
        }
