# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Numerical equivalence check between source simulation and generated artifact.

This module implements the *protocol* — the types and the comparison
function. The *policy* — how tight a divergence is acceptable per
``(target_family, dtype, quantity)`` — lives in
:mod:`jaxility.testing.tolerances` and is documented in
``test/EQUIVALENCE.md``. The two are deliberately separate so a future
tolerance loosening (or tightening) is a single-file change reviewable
against the documented contract.

Invariant 1 (CONTEXT.md) says every generated artifact must pass an
equivalence check against the source simulation before the build is
considered successful. This module is the surface that check is
written against. Tests use mock targets exclusively; real
host / HIL targets land later (T-027 / T-033).

See ``test/EQUIVALENCE.md`` for the tolerance contract, PATTERNS §7.2
for the property-test discipline, and PATTERNS §7.4 for the rule that
tolerances come from the documented table (no magic numbers).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import EquivalenceError
from .tolerances import Tolerance, lookup_tolerance

EQUIVALENCE_SCHEMA_V0 = 0
"""Schema version of the ``EquivalenceReport`` payload."""

QuantityArray = np.ndarray
"""A per-quantity trajectory: shape ``(n_steps, ...)``, dtype-flexible.

The first axis is time; trailing axes are per-step shape (a scalar
quantity is ``(n_steps,)``; a 3-vector is ``(n_steps, 3)``). Both the
source and the candidate must agree on shape; ``compare`` raises if
they do not.
"""

Trajectory = dict[str, QuantityArray]
"""A trajectory is a mapping from quantity name to its per-step array.

Quantity names are free-form strings (``"joint_position"``,
``"joint_velocity"``, ``"actuator_torque"``, ...) but must appear as
keys of the tolerance table for the chosen ``(target_family, dtype)``
or :func:`compare` raises a :class:`KeyError`-style miss (PATTERNS §7.4
— no silent default).
"""


class QuantityDivergence(BaseModel):
    """The divergence between source and candidate for a single quantity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=EQUIVALENCE_SCHEMA_V0, ge=0)
    """Schema version (PATTERNS §3.4); currently v0."""

    quantity: str
    """Quantity name as it appears in the source and candidate trajectories."""

    max_abs_error: float
    """Maximum absolute error across all time steps."""

    max_rel_error: float
    """Maximum relative error (``|a - b| / max(|a|, |b|, eps)``) across steps."""

    first_violation_step: int | None
    """First time step at which the divergence exceeds the tolerance.

    ``None`` when no step violates the bound (``passed`` is then ``True``).
    """

    passed: bool
    """``True`` iff both abs and rel divergence are within the tolerance."""

    tolerance: Tolerance
    """The tolerance row used (``(target_family, dtype, quantity)`` lookup result)."""

    suggestion: str | None
    """Structured hint when ``passed`` is ``False``; ``None`` when ``passed``."""


class EquivalenceReport(BaseModel):
    """Per-trajectory equivalence verdict.

    A report is *passed* iff every quantity in ``per_quantity`` passed.
    Per-quantity failures carry a structured suggestion (PATTERNS §6.3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=EQUIVALENCE_SCHEMA_V0, ge=0)
    """Schema version (PATTERNS §3.4); currently v0."""

    overall_passed: bool
    target_family: str
    dtype: str
    per_quantity: list[QuantityDivergence]

    def assert_passed(self) -> None:
        """Raise :class:`EquivalenceError` when the report failed.

        Routes through the PATTERNS §6.1 hierarchy. Library code
        catches :class:`JaxilityError` for the uniform-handling path
        and :class:`EquivalenceError` for the codegen-divergence path.
        """
        if self.overall_passed:
            return
        failures = [qd for qd in self.per_quantity if not qd.passed]
        total = len(self.per_quantity)
        lines = [f"Equivalence failed for {len(failures)} of {total} quantities:"]
        for qd in failures:
            lines.append(
                f"  - {qd.quantity}: abs={qd.max_abs_error:.3e} "
                f"(tol {qd.tolerance.abs_tol:.3e}), "
                f"rel={qd.max_rel_error:.3e} "
                f"(tol {qd.tolerance.rel_tol:.3e}), "
                f"first violation at step {qd.first_violation_step}; "
                f"{qd.suggestion or 'no suggestion'}"
            )
        raise EquivalenceError("\n".join(lines))


def _suggest(
    quantity: str,
    source: QuantityArray,
    candidate: QuantityArray,
    tolerance: Tolerance,
    first_violation: int | None,
) -> str | None:
    """Pick a structured suggestion for a failed quantity.

    The heuristic is intentionally small for now — three named
    patterns plus a generic fallback. The shape grows as real codegen
    surfaces real bug modes (T-027 onward).
    """
    if first_violation is None:
        return None

    diffs = np.abs(source - candidate)
    abs_max = float(diffs.max(initial=0.0))

    # Actuation-quantity heuristic runs first: when the corruption is on an
    # actuator output, the name of the quantity is a much stronger signal than
    # the step at which the divergence first appears (a constant offset on the
    # actuator output trips at step 0, which would otherwise look like an
    # integration error).
    quant_lower = quantity.lower()
    if any(tag in quant_lower for tag in ("torque", "actuator", "motor")):
        return (
            f"Divergence is in an actuation quantity ({quantity!r}); the "
            "actuator block in the candidate likely differs from the source "
            "(quantisation, friction model, or gain). Inspect the "
            "actuator template parameters."
        )

    early = first_violation < max(1, len(source) // 10)
    if early and abs_max > 100 * tolerance.abs_tol:
        return (
            "Early-step divergence with magnitude >> tolerance suggests an "
            "integration-error or numerical-instability issue; check the "
            "integrator settings and the dtype of the candidate run."
        )

    if tolerance.grade == "ulp-bounded" and abs_max < 10 * tolerance.abs_tol:
        return (
            "Small divergence near ULP-bounded tolerance; check that the "
            "candidate's dtype matches the target profile's expected dtype "
            "(float32 vs. float64)."
        )

    return (
        f"Divergence first exceeded tolerance at step {first_violation}; "
        "no canonical-pattern match. Inspect the quantity's source vs. "
        "candidate trace around that step."
    )


def _compute_divergence(
    quantity: str,
    source: QuantityArray,
    candidate: QuantityArray,
    tolerance: Tolerance,
) -> QuantityDivergence:
    """Per-quantity divergence + suggestion. Pure; no I/O."""
    if source.shape != candidate.shape:
        raise EquivalenceError(
            f"Trajectory shape mismatch for quantity {quantity!r}: "
            f"source {source.shape} vs. candidate {candidate.shape}."
        )

    diff = source.astype(np.float64) - candidate.astype(np.float64)
    abs_err = np.abs(diff)
    # Per-step max across trailing axes (per-step scalar magnitude).
    per_step_abs = (
        abs_err.reshape(abs_err.shape[0], -1).max(axis=1)
        if abs_err.ndim > 1
        else abs_err
    )
    max_abs = float(per_step_abs.max(initial=0.0))

    eps = np.finfo(np.float64).tiny
    denom = np.maximum(
        np.maximum(
            np.abs(source.astype(np.float64)), np.abs(candidate.astype(np.float64))
        ),
        eps,
    )
    rel_err = abs_err / denom
    per_step_rel = (
        rel_err.reshape(rel_err.shape[0], -1).max(axis=1)
        if rel_err.ndim > 1
        else rel_err
    )
    max_rel = float(per_step_rel.max(initial=0.0))

    # A per-step violation is one where BOTH abs and rel exceed their tolerances.
    # Pure rel-overshoot at very small magnitudes is noise; pure abs-overshoot
    # at very large magnitudes is a calibration issue we still want to catch.
    violation_mask = (per_step_abs > tolerance.abs_tol) & (
        per_step_rel > tolerance.rel_tol
    )
    if violation_mask.any():
        first_violation = int(np.argmax(violation_mask))
        passed = False
    else:
        first_violation = None
        passed = True

    suggestion = (
        _suggest(quantity, source, candidate, tolerance, first_violation)
        if not passed
        else None
    )

    return QuantityDivergence(
        quantity=quantity,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        first_violation_step=first_violation,
        passed=passed,
        tolerance=tolerance,
        suggestion=suggestion,
    )


def compare(
    source: Trajectory,
    candidate: Trajectory,
    *,
    target_family: str,
    dtype: Literal["float32", "float64"],
) -> EquivalenceReport:
    """Compare a source trajectory against a candidate trajectory.

    Looks up the tolerance row for each
    ``(target_family, dtype, quantity)`` in the tolerance table
    (PATTERNS §7.4), computes per-quantity divergence, and assembles a
    report. The report's :attr:`~EquivalenceReport.overall_passed` is
    the conjunction of per-quantity passes.

    Args
    ----
    source : Trajectory
        The reference trajectory, typically the JAX source simulation.
    candidate : Trajectory
        The trajectory produced by the candidate (mock-lowered or, in
        later phases, a real cross-compiled artifact run on host or HIL).
    target_family : str
        Target family identifier (``"mock-cortex-a"``, ``"cortex-a76"``,
        ``"cortex-m7"``, ...). Used as the first key of the tolerance
        lookup.
    dtype : Literal["float32", "float64"]
        Precision the candidate was evaluated at. The source is always
        evaluated at the highest available precision; ``dtype`` here
        names the *candidate's* precision.

    Returns
    -------
    EquivalenceReport
        A structured verdict. Pass / fail is exposed as
        ``report.overall_passed``; per-quantity diagnostics are in
        ``report.per_quantity``.

    Raises
    ------
    KeyError
        Propagated from :func:`jaxility.testing.tolerances.lookup_tolerance`
        when the tolerance table has no row for the chosen
        ``(target_family, dtype, quantity)``. There is no silent
        default — adding a quantity to a target is a deliberate
        tolerance-table entry (PATTERNS §7.4).
    EquivalenceError
        If a quantity is in ``source`` but not in ``candidate`` (or
        vice versa), or if their shapes disagree. (Audit N-4 fix: the
        previous docstring claimed ``ValueError`` here; the actual
        raise has always been :class:`~jaxility.errors.EquivalenceError`,
        the PATTERNS §6.1 subclass of :class:`JaxilityError` that
        carries the structured diagnostic.)
    """
    missing_in_candidate = set(source) - set(candidate)
    if missing_in_candidate:
        raise EquivalenceError(
            f"candidate trajectory is missing quantities present in source: "
            f"{sorted(missing_in_candidate)}"
        )
    missing_in_source = set(candidate) - set(source)
    if missing_in_source:
        raise EquivalenceError(
            f"candidate trajectory has quantities not present in source: "
            f"{sorted(missing_in_source)}"
        )

    per_quantity: list[QuantityDivergence] = []
    for quantity in sorted(source):
        tolerance = lookup_tolerance(target_family, dtype, quantity)
        per_quantity.append(
            _compute_divergence(
                quantity, source[quantity], candidate[quantity], tolerance
            )
        )

    overall_passed = all(qd.passed for qd in per_quantity)
    return EquivalenceReport(
        overall_passed=overall_passed,
        target_family=target_family,
        dtype=dtype,
        per_quantity=per_quantity,
    )
