# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""The HIL trace contract: target stdout -> a comparable Trajectory.

A deployed artifact run in HIL mode emits one JSON object per control
cycle on stdout (JSON Lines), in step order — each line names its
quantities::

    {"step": 0, "joint_position": [0.2],
     "joint_velocity": [0.0], "actuator_torque": [-1.6]}
    {"step": 1, "joint_position": [0.2], ...}
    ...

This module parses that stream into a
:data:`jaxility.testing.equivalence.Trajectory` (``quantity -> array of
shape (n_steps, *quantity_shape)``) so it can be fed straight into
:func:`jaxility.testing.equivalence.compare`. The format is deliberately
self-describing — each line names its quantities — so the parser never
has to guess a column order, and a schema mismatch fails loudly
(invariant 7) rather than silently misaligning a vector.

The contract is shared with the on-target side (the C fixture in
``test/hil/fixtures/cartpole_hil.c`` today; the generated acados shim in
T-034). Keep this parser and that emitter in lockstep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..errors import HILError
from ..testing.equivalence import Trajectory

HIL_TRACE_SCHEMA_V0 = 0
"""Schema version of the JSONL HIL trace line format."""


@dataclass(frozen=True)
class StateSchema:
    """The quantities a HIL trace must carry, with their per-step shapes.

    ``quantities`` maps each quantity name (which must also be a key of
    the tolerance table for the run's ``(target_family, dtype)``) to its
    per-cycle shape — ``(1,)`` for a scalar carried as a 1-vector,
    ``(3,)`` for a 3-vector, and so on. The parser validates every line
    against this schema and raises :class:`~jaxility.errors.HILError` on
    any mismatch.
    """

    quantities: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.quantities)


# The 1-DOF fixture schema (T-033): angle, angular rate, and the scalar
# actuator command, each a 1-vector. Matches the deterministic
# `cartpole_hil.c` fixture and the `cortex-a76` x float32 tolerance rows.
CARTPOLE_SCHEMA = StateSchema(
    quantities=(
        ("joint_position", (1,)),
        ("joint_velocity", (1,)),
        ("actuator_torque", (1,)),
    )
)

# The full Cartpole LQR schema (T-034): state is
# [x_cart, theta, x_dot, theta_dot] (nx=4), control is [force] (nu=1).
# joint_position = (cart, pole) positions; joint_velocity = their rates;
# actuator_torque = the scalar force. Matches the generated acados
# controller HIL binary and the `cortex-a76` x float64 tolerance rows.
CARTPOLE_LQR_SCHEMA = StateSchema(
    quantities=(
        ("joint_position", (2,)),
        ("joint_velocity", (2,)),
        ("actuator_torque", (1,)),
    )
)


def parse_trace(stdout: str, schema: StateSchema, *, n_steps: int) -> Trajectory:
    """Parse a JSONL HIL trace into a step-locked :data:`Trajectory`.

    Args
    ----
    stdout : str
        The raw standard output captured from the target run.
    schema : StateSchema
        The quantities (and per-step shapes) every line must carry.
    n_steps : int
        The exact number of cycles expected. The trace must carry steps
        ``0 .. n_steps - 1`` in order; a short, long, or out-of-order
        trace is a loud failure.

    Returns
    -------
    Trajectory
        ``quantity -> np.ndarray`` of shape ``(n_steps, *quantity_shape)``,
        dtype ``float64`` (the comparison upcasts anyway).

    Raises
    ------
    HILError
        On any malformed line, missing/extra quantity, wrong per-step
        shape, non-contiguous step index, or wrong step count. The HIL
        transport is allowed to fail; it is never allowed to silently
        return a misaligned trajectory.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if len(lines) != n_steps:
        raise HILError(
            f"HIL trace carried {len(lines)} cycle line(s); expected "
            f"{n_steps}. The target run was truncated, over-ran, or emitted "
            f"non-trace output on stdout."
        )

    buffers: dict[str, np.ndarray] = {
        name: np.empty((n_steps, *shape), dtype=np.float64)
        for name, shape in schema.quantities
    }

    for expected_step, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HILError(
                f"HIL trace line {expected_step} is not valid JSON: {line!r} ({exc})."
            ) from exc
        if not isinstance(record, dict):
            raise HILError(
                f"HIL trace line {expected_step} is not a JSON object: {line!r}."
            )

        step = record.get("step")
        if step != expected_step:
            raise HILError(
                f"HIL trace out of order: expected step {expected_step}, got "
                f"{step!r}. Steps must be contiguous and 0-based."
            )

        payload_keys = set(record) - {"step"}
        if payload_keys != set(schema.names):
            raise HILError(
                f"HIL trace line {expected_step} quantities {sorted(payload_keys)} "
                f"do not match the schema {sorted(schema.names)}."
            )

        for name, shape in schema.quantities:
            value = np.asarray(record[name], dtype=np.float64)
            if value.shape != shape:
                raise HILError(
                    f"HIL trace line {expected_step} quantity {name!r} has shape "
                    f"{value.shape}; schema requires {shape}."
                )
            buffers[name][expected_step] = value

    return buffers
