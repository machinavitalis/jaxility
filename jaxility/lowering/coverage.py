# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Coverage declaration for the JAX → acados / NPU lowering pipeline.

Invariant 7 (CONTEXT.md) — *no code generation without coverage.* Every
JAX operation emitted into generated code is declared here. Adding new
op support is a documented PR touching the coverage table; the
lowering implementation cannot silently support a new op.

The table is keyed by ``(op_name, dtype, target_family)``:

* ``op_name`` — canonical JAX op identifier
  (``"add"``, ``"jnp.sin"``, ``"matmul"``, ...). For ``jnp.*`` operations
  the table key keeps the ``jnp.`` prefix; for primitive operators the
  table key is the short name (``"add"``, ``"sub"``, ``"mul"``,
  ``"div"``, ``"pow"``).
* ``dtype`` — ``"float32"`` or ``"float64"`` (PATTERNS §7.4 mirrors the
  equivalence-table dtype).
* ``target_family`` — the same family identifier the tolerance table
  uses (``"mock-cortex-a"``, ``"mock-cortex-m"``, ...).

A miss is *not* silent — :func:`assert_supported` raises a structured
:class:`CoverageError` with a documented suggestion. PATTERNS: loud,
not silent.

The lowering pipeline (T-020+) calls :func:`assert_supported` on every
op it would emit. The schema declares the contract; the coverage gate enforces it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import CoverageError

COVERAGE_SCHEMA_V0 = 0
"""Schema version of the ``CoverageEntry`` payload."""

NumericalGrade = Literal["bit-exact", "ulp-bounded", "approximate"]
"""Mirrors :data:`jaxility.testing.tolerances.NumericalGrade`."""


class CoverageEntry(BaseModel):
    """A single ``(op, dtype, target_family)`` coverage row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=COVERAGE_SCHEMA_V0, ge=0)
    """Schema version (PATTERNS §3.4); currently v0."""

    op_name: str
    """Canonical op identifier, e.g. ``"jnp.sin"``."""

    dtype: Literal["float32", "float64"]
    """Precision the row applies to."""

    target_family: str
    """Target family identifier."""

    supported: bool
    """Whether the lowering pipeline may emit code for this combination."""

    implementation_hint: str
    """How the lowering implements (or would implement) this op."""

    grade: NumericalGrade | None
    """Numerical grade when ``supported``; ``None`` when not supported."""

    known_issues: tuple[str, ...] = Field(
        default=(),
        description=(
            "Documented limitations that travel with the entry — surfaces "
            "in coverage CLI output and in CoverageError suggestions."
        ),
    )

    suggestion: str = Field(
        description=(
            "What the user should do when this combination is requested. "
            "For supported ops, a one-line confirmation. For unsupported "
            "ops, a documented workaround: a smoothing approximation, a "
            "different template, or 'wait for release'."
        ),
    )


# The coverage table itself. The current entries cover the canonical
# smooth-op subset acados consumes (per SKILL.md) on both mock targets,
# plus the deliberate "not supported" rows that PATTERNS §6 / SKILL.md
# call out (``lax.cond`` over traced predicates, ``lax.while_loop``,
# dynamic shapes). Real-target rows arrive later.

_SUPPORTED_OPS_SMOOTH = (
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "jnp.sin",
    "jnp.cos",
    "jnp.tan",
    "jnp.exp",
    "jnp.log",
    "jnp.sqrt",
    "matmul",
    "slice[static]",
    "dynamic_slice[static]",
)
"""Smooth ops every supported (dtype, target) row covers.

``slice[static]`` covers JAX's static-index slicing (``x[a:b]`` with
compile-time constants) — distinct from a future ``slice[dynamic]``
row that would track dynamic-index indexing. ``jnp.where[static]`` is
added separately below because the surface keys it on the predicate
shape, not the operand shape.
"""

_DTYPE_GRADE: dict[tuple[str, str], NumericalGrade] = {
    ("mock-cortex-a", "float64"): "ulp-bounded",
    ("mock-cortex-a", "float32"): "approximate",
    ("mock-cortex-m", "float32"): "approximate",
}


def _supported_entry(
    op: str, target_family: str, dtype: str, grade: NumericalGrade
) -> CoverageEntry:
    return CoverageEntry(
        op_name=op,
        dtype=dtype,  # type: ignore[arg-type]
        target_family=target_family,
        supported=True,
        implementation_hint=(f"mock pipeline: pass-through to source JAX at {dtype}."),
        grade=grade,
        suggestion=(f"{op} is supported at {dtype} on {target_family}; no action."),
    )


def _unsupported_entry(
    op: str, target_family: str, dtype: str, *, reason: str, suggestion: str
) -> CoverageEntry:
    return CoverageEntry(
        op_name=op,
        dtype=dtype,  # type: ignore[arg-type]
        target_family=target_family,
        supported=False,
        implementation_hint=f"Not implemented: {reason}",
        grade=None,
        known_issues=(reason,),
        suggestion=suggestion,
    )


def _build_table() -> dict[tuple[str, str, str], CoverageEntry]:
    rows: dict[tuple[str, str, str], CoverageEntry] = {}
    for (target_family, dtype), grade in _DTYPE_GRADE.items():
        for op in _SUPPORTED_OPS_SMOOTH:
            rows[(op, dtype, target_family)] = _supported_entry(
                op, target_family, dtype, grade
            )
        # ``jnp.where`` is supported only over *static* predicates;
        # the lowering distinguishes by inspecting the JAX trace.
        rows[("jnp.where[static]", dtype, target_family)] = _supported_entry(
            "jnp.where[static]", target_family, dtype, grade
        )
        # The deliberate "loud rejection" rows. These cover the cases
        # SKILL.md and CONTEXT.md call out as the unsupported edge.
        rows[("jnp.where[traced]", dtype, target_family)] = _unsupported_entry(
            "jnp.where[traced]",
            target_family,
            dtype,
            reason=(
                "jnp.where over traced predicates produces a non-smooth "
                "graph that acados cannot consume."
            ),
            suggestion=(
                "use a smoothing approximation (sigmoid/softmax with a "
                "named sharpness factor; mirrors Jaxterity invariant 6) "
                "or pick a different template that does not need a "
                "traced predicate."
            ),
        )
        rows[("lax.cond[traced]", dtype, target_family)] = _unsupported_entry(
            "lax.cond[traced]",
            target_family,
            dtype,
            reason=(
                "lax.cond over traced predicates is rejected for the same "
                "non-smoothness reason as jnp.where[traced]."
            ),
            suggestion=(
                "replace with a smoothed branch or restructure the dynamics "
                "to avoid the conditional."
            ),
        )
        rows[("lax.while_loop", dtype, target_family)] = _unsupported_entry(
            "lax.while_loop",
            target_family,
            dtype,
            reason=(
                "acados' OCP formulation does not consume dynamic iteration "
                "counts. Also: MJX's forward-dynamics path runs the "
                "constraint solver under a while_loop unconditionally, so "
                "MJX-driven Jaxterity Robot dynamics cannot translate "
                "directly (see ADR-016)."
            ),
            suggestion=(
                "bound the loop statically (Python ``range``) or fold the "
                "loop body into the OCP horizon. For MJX-emitted while_loops "
                "(constraint solver), supply a closed-form per-robot "
                "dynamics function instead of going through MJX."
            ),
        )
        rows[("dynamic_slice[traced]", dtype, target_family)] = _unsupported_entry(
            "dynamic_slice[traced]",
            target_family,
            dtype,
            reason=(
                "traced start indices require a CasADi conditional / case "
                "expression at every load; this exits the smooth-op subset."
            ),
            suggestion=(
                "use ``jax.lax.dynamic_slice`` with literal start indices "
                "(supported via the ``dynamic_slice[static]`` row) or "
                "rewrite as a plain ``operand[a:b]`` slice."
            ),
        )
        rows[("dynamic_shape", dtype, target_family)] = _unsupported_entry(
            "dynamic_shape",
            target_family,
            dtype,
            reason="dynamic array shapes are incompatible with embedded codegen.",
            suggestion=(
                "fix the shape at trace time, or move the dynamic-shape "
                "code outside the lowered region."
            ),
        )
    return rows


COVERAGE_TABLE: dict[tuple[str, str, str], CoverageEntry] = _build_table()


def lookup(op: str, dtype: str, target_family: str) -> CoverageEntry:
    """Return the coverage entry for ``(op, dtype, target_family)``.

    Raises
    ------
    CoverageError
        If no entry exists for the combination. There is no silent
        default; an unknown op on an unknown target is a deliberate
        loud failure (invariant 7).
    """
    key = (op, dtype, target_family)
    try:
        return COVERAGE_TABLE[key]
    except KeyError:
        raise CoverageError(
            (
                f"no coverage entry for op={op!r}, dtype={dtype!r}, "
                f"target_family={target_family!r}. Add an entry to "
                f"jaxility.lowering.coverage.COVERAGE_TABLE before lowering "
                f"for this combination (invariant 7)."
            ),
            op=op,
            dtype=dtype,
            target_family=target_family,
            suggestion=(
                "add a row to COVERAGE_TABLE with implementation_hint, "
                "grade, and either supported=True (with a tolerance entry "
                "for the same combination) or supported=False (with a "
                "documented suggestion)."
            ),
        ) from None


def assert_supported(op: str, dtype: str, target_family: str) -> CoverageEntry:
    """Return the entry iff ``supported``; otherwise raise.

    Returns
    -------
    CoverageEntry
        The matching row (always ``supported=True`` on return).

    Raises
    ------
    CoverageError
        When the row exists but ``supported=False``, or when no row
        exists (via :func:`lookup`).
    """
    entry = lookup(op, dtype, target_family)
    if entry.supported:
        return entry
    raise CoverageError(
        (
            f"op {op!r} is not supported at dtype={dtype!r} "
            f"on target_family={target_family!r}."
        ),
        op=op,
        dtype=dtype,
        target_family=target_family,
        suggestion=entry.suggestion,
    )


def coverage_markdown(target_family: str | None = None) -> str:
    """Render the coverage table as a Markdown table.

    Args
    ----
    target_family : str | None
        If supplied, only rows for that family appear. ``None``
        emits every row.

    Returns
    -------
    str
        A Markdown table grouped by ``(target_family, dtype)``.
    """
    rows = sorted(
        (
            entry
            for entry in COVERAGE_TABLE.values()
            if target_family is None or entry.target_family == target_family
        ),
        key=lambda e: (e.target_family, e.dtype, e.op_name),
    )

    lines: list[str] = ["# Jaxility coverage matrix", ""]
    if target_family is not None:
        lines.append(f"Filtered to ``target_family = {target_family!r}``.")
        lines.append("")

    last_section: tuple[str, str] | None = None
    for entry in rows:
        section = (entry.target_family, entry.dtype)
        if section != last_section:
            if last_section is not None:
                lines.append("")
            lines.append(f"## `{entry.target_family}` × `{entry.dtype}`")
            lines.append("")
            lines.append(
                "| Op | Supported | Grade | Implementation hint | Suggestion |"
            )
            lines.append(
                "|----|-----------|-------|----------------------|-----------|"
            )
            last_section = section

        grade = entry.grade or "-"
        supported = "yes" if entry.supported else "no"
        lines.append(
            f"| `{entry.op_name}` | {supported} | `{grade}` | "
            f"{entry.implementation_hint} | {entry.suggestion} |"
        )

    return "\n".join(lines) + "\n"
