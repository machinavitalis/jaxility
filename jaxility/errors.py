# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Jaxility's structured exception hierarchy (PATTERNS §6.1).

Library code raises one of these. Never :class:`RuntimeError`,
:class:`ValueError`, or :class:`AssertionError` — those are the
loudest possible signal that a library bug slipped past the
exception design (PATTERNS §10).

The full hierarchy lands here. The base +
:class:`CoverageError` (T-013) + :class:`ArtifactError` (T-014) landed
first; the #14 follow-up review lands the rest so the bare
``ValueError`` / ``AssertionError`` raises elsewhere in the package can
route through the hierarchy. Subclasses without callsites yet
(:class:`ToolchainError`, :class:`HILError`, :class:`BenchmarkError`)
appear here so import-by-name works at the package boundary; they will
get specific carrier fields when their first callsite lands.
"""

from __future__ import annotations


class JaxilityError(Exception):
    """Base for all Jaxility library errors (PATTERNS §6.1).

    Every exception raised by Jaxility code derives from this class.
    Catch :class:`JaxilityError` to handle anything from the pipeline
    uniformly; catch a more specific subclass to handle a narrower
    failure mode.
    """


class ArtifactError(JaxilityError):
    """An artifact failed to construct, store, or load."""


class BenchmarkError(JaxilityError):
    """A benchmark run failed (T-035 onward)."""


class CoverageError(JaxilityError):
    """Unsupported JAX op for the chosen ``(dtype, target_family)``.

    Raised by the lowering pipeline when it would emit
    code for a combination the coverage table marks unsupported.
    Currently raised by
    :func:`jaxility.lowering.coverage.assert_supported` from test
    fixtures that exercise the coverage gate.

    The error carries the structured fields the coverage entry
    provides so an agent (or a human) can see exactly what failed and
    what the documented workaround is.
    """

    op: str
    """The JAX op name that triggered the error, e.g. ``"jnp.sin"``."""

    dtype: str
    """The dtype the lowering was asked to support."""

    target_family: str
    """The target family the lowering was asked to support."""

    suggestion: str
    """A documented hint: a smoothing approximation, a different template, or "wait"."""

    def __init__(
        self,
        message: str,
        *,
        op: str,
        dtype: str,
        target_family: str,
        suggestion: str,
    ) -> None:
        super().__init__(message)
        self.op = op
        self.dtype = dtype
        self.target_family = target_family
        self.suggestion = suggestion

    def __str__(self) -> str:  # pragma: no cover - trivial passthrough
        return (
            f"{super().__str__()} "
            f"[op={self.op!r}, dtype={self.dtype!r}, "
            f"target_family={self.target_family!r}] suggestion: {self.suggestion}"
        )


class EquivalenceError(JaxilityError):
    """A numerical equivalence check rejected its inputs or failed.

    Routes through PATTERNS §6.3 — every error message carries enough
    structured context to start debugging. Raised by
    :meth:`jaxility.testing.equivalence.EquivalenceReport.assert_passed`
    on a failed report, by :func:`jaxility.testing.equivalence.compare`
    when the source and candidate trajectories disagree on quantity
    set or per-quantity shape, and by the lowering pipeline
    when a host-build equivalence run diverges beyond tolerance.
    """


class CompositionError(JaxilityError):
    """Dual-path composition is malformed or inconsistent (T-043).

    Raised when a ``CompositionPlan`` arbitration receives dimension-mismatched
    controls / state / envelope, so a misconfigured dual-path deployment fails
    loudly rather than producing an unsafe command (invariant 8).
    """


class HILError(JaxilityError):
    """Hardware-in-the-loop divergence (T-033 onward)."""


class ManifestError(JaxilityError):
    """Manifest schema violation, signature mismatch, or chain break.

    The :class:`jaxility.manifest.ManifestVerificationError`
    derives from this; future schema-migration errors will too.
    """


class SourceError(JaxilityError):
    """A lowering source rejected its inputs or lacks a required state.

    Current callsites:
    :class:`jaxility.testing.MockSource.simulate` (``n_steps < 1``),
    :class:`jaxility.testing.JaxteritySource.from_robot`
    (``require_calibration_state`` mismatch), and
    :meth:`jaxility.testing.JaxteritySource.simulate` (``n_steps < 1``).
    """


class TargetError(JaxilityError):
    """Target profile invalid or its toolchain is missing."""


class ToolchainError(JaxilityError):
    """A subprocess-invoked toolchain failed (PATTERNS §2.1)."""
