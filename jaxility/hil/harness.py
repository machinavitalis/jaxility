# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Step-locked HIL parity: run the artifact, compare it to the reference.

This is the top of the HIL stack. Given a reference trajectory (the host
simulation, in JAX, via a :class:`jaxility.testing.sources.Source`) and a
:class:`jaxility.hil.runner.TargetRunner` pointed at the deployed
artifact, :func:`run_hil`:

    1. runs the artifact for ``n_steps`` cycles on the target,
    2. parses its JSONL trace into a step-locked trajectory,
    3. compares the two through the shared equivalence harness under the
       documented ``(target_family, dtype, quantity)`` tolerances, and
    4. returns a :class:`HILReport` whose ``assert_passed`` raises on
       divergence — the same surface the equivalence check uses, so HIL
       parity is the release gate it is meant to be (invariant 6).

The reference is supplied by the caller rather than recomputed here so
the harness stays agnostic to *how* the reference is produced — a zoo
``Source.simulate`` in production, a hand-mirrored recurrence in the
fixture tests. The transport (local subprocess vs. SSH-to-Pi) is the
:class:`TargetRunner`; this module is transport-agnostic.

T-033 scope note: the artifact under test today is the deterministic C
fixture (``test/hil/fixtures/cartpole_hil.c``). The generated acados
controller shim it stands in for is T-034; when it lands it emits the
same trace contract and flows through this exact harness unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..testing.equivalence import EquivalenceReport, Trajectory, compare
from .runner import TargetRunner
from .trace import HIL_TRACE_SCHEMA_V0, StateSchema, parse_trace

HIL_REPORT_SCHEMA_V0 = 0
"""Schema version of the ``HILReport`` payload."""


class HILReport(BaseModel):
    """A HIL parity verdict: the equivalence report plus run provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: int = Field(default=HIL_REPORT_SCHEMA_V0, ge=0)
    trace_schema_version: int = Field(default=HIL_TRACE_SCHEMA_V0, ge=0)

    runner_label: str
    """The transport that produced the candidate trace (``local:...`` / ``ssh:...``)."""

    n_steps: int
    seed: int
    equivalence: EquivalenceReport
    """The underlying step-locked comparison verdict."""

    @property
    def passed(self) -> bool:
        return self.equivalence.overall_passed

    def assert_passed(self) -> None:
        """Raise :class:`~jaxility.errors.EquivalenceError` on divergence.

        Delegates to the equivalence report so HIL parity failures route
        through the same structured-diagnostic path as every other
        equivalence check (PATTERNS §6.1).
        """
        self.equivalence.assert_passed()


def run_hil(
    reference: Trajectory,
    runner: TargetRunner,
    *,
    target_family: str,
    dtype: Literal["float32", "float64"],
    schema: StateSchema,
    n_steps: int,
    seed: int = 0,
) -> HILReport:
    """Run the artifact through ``runner`` and compare it to ``reference``.

    Args
    ----
    reference : Trajectory
        The host reference trajectory (``quantity -> (n_steps, ...)``),
        typically from a zoo ``Source.simulate``. Must carry exactly the
        quantities in ``schema``.
    runner : TargetRunner
        The transport to the deployed artifact (local or SSH).
    target_family : str
        Tolerance-table family key (``"cortex-a76"`` for the Pi 5).
    dtype : {"float32", "float64"}
        The candidate's precision (the embedded artifact's working
        precision — ``"float32"`` for the Cortex-A76 launch path).
    schema : StateSchema
        The quantities and per-step shapes the trace must carry.
    n_steps, seed : int
        Cycle count and deterministic seed, passed through to the target.

    Returns
    -------
    HILReport
        Pass / fail via ``report.passed`` / ``report.assert_passed()``.

    Raises
    ------
    HILError
        On any transport or trace-parse failure (loud, never partial).
    EquivalenceError
        Via the returned report's ``assert_passed`` (not raised here).
    """
    raw = runner.run(n_steps=n_steps, seed=seed)
    candidate = parse_trace(raw, schema, n_steps=n_steps)
    report = compare(reference, candidate, target_family=target_family, dtype=dtype)
    return HILReport(
        runner_label=runner.label,
        n_steps=n_steps,
        seed=seed,
        equivalence=report,
    )
