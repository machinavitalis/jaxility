# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""``Source`` protocol + a mock implementation.

A *source* is the thing the lowering pipeline consumes. Real usage:
the source is a :class:`jaxterity.robot.Robot` (or, after Jaxterity
tags 0.2+, a ``CalibratedRobot`` if that rename happens). For
testing, the source is a small in-memory mock satisfying the same
Protocol so the pipeline can be exercised end-to-end without
importing Jaxterity (T-016 wires the real bridge).

A source carries an :attr:`attestation_handle` (the upstream hash
that the manifest's ``source_attestation_handle`` becomes) and can
produce a :data:`~jaxility.testing.equivalence.Trajectory` for the
agreed quantities at a chosen number of steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..errors import SourceError
from .equivalence import Trajectory


@runtime_checkable
class Source(Protocol):
    """Duck-typed interface every lowering source must satisfy.

    Jaxterity's :class:`Robot` will satisfy this via a small adapter
    that lands in T-016. Tests use :class:`MockSource` below.
    """

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``"cartpole"``."""

    @property
    def attestation_handle(self) -> bytes:
        """BLAKE3 digest from the upstream calibration / sysid pipeline.

        Jaxterity exposes this as a hex string; the source adapter
        must decode it to bytes so the canonical hash chain works
        end-to-end.
        """

    def simulate(self, n_steps: int) -> Trajectory:
        """Return a trajectory keyed by quantity name.

        The quantity keys must match the tolerance table for the
        chosen ``(target_family, dtype)``; missing keys raise on the
        equivalence-check side (PATTERNS §7.4 — no silent default).
        """


@dataclass(frozen=True)
class MockSource:
    """An in-memory source for the mock lowering pipeline.

    Carries a stable :attr:`attestation_handle` derived from
    ``identifier`` + ``initial_state``, plus a deterministic
    :meth:`simulate` that returns sinusoidal-style trajectories
    keyed by ``joint_position`` / ``joint_velocity`` /
    ``actuator_torque`` — the same keys the tolerance table
    covers.
    """

    name: str
    initial_state: tuple[float, ...]
    """Initial-condition tuple. Different states produce different handles."""

    dim: int = 2
    """Number of joints. Default ``2`` mirrors a cartpole."""

    handle_salt: bytes = b"jaxility-mock-source"
    """Salt mixed into the BLAKE3 input so distinct sources of the same shape
    still produce distinct handles. Tests override this to model an upstream
    sysid recipe change."""

    extra: tuple[bytes, ...] = field(default_factory=tuple)
    """Optional additional inputs that participate in the handle
    (provenance, recipe identifiers, …)."""

    @property
    def attestation_handle(self) -> bytes:
        import blake3

        h = blake3.blake3(self.handle_salt)
        h.update(self.name.encode("utf-8"))
        h.update(str(self.initial_state).encode("utf-8"))
        h.update(str(self.dim).encode("utf-8"))
        for chunk in self.extra:
            h.update(chunk)
        return h.digest()

    def simulate(self, n_steps: int) -> Trajectory:
        if n_steps < 1:
            raise SourceError(f"n_steps must be >= 1, got {n_steps}")
        t = np.linspace(0.0, 1.0, n_steps)
        # Seed the phase from the initial state so different initial
        # conditions produce different trajectories — useful for the
        # "changing source input changes artifact hash" test (T-015).
        phase = float(sum(self.initial_state))
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
