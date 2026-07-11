# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tolerance contract for the numerical equivalence check.

PATTERNS §7.4 — *"Tolerances come from* ``test/EQUIVALENCE.md``.
*No magic numbers.* :mod:`jaxility.testing.tolerances` *reads from
the documented table."* This module is the source of truth that the
markdown documents; tests cross-check that every row in the table
appears in ``test/EQUIVALENCE.md``.

The lookup key is ``(target_family, dtype, quantity)``:

* ``target_family`` — ``"mock-cortex-a"``, ``"mock-cortex-m"``,
  ``"cortex-a76"``, ``"cortex-m7"``, ... (lookup is exact, not by SoC
  identity — every Cortex-A76 deployment shares the ``"cortex-a76"``
  family bounds).
* ``dtype`` — ``"float32"`` or ``"float64"``; ``float32`` is the
  embedded-default precision and gets correspondingly looser bounds.
* ``quantity`` — the trajectory key
  (``"joint_position"``, ``"actuator_torque"``, ...).

There is no silent default. A miss raises :class:`KeyError`; adding a
quantity to a target is a deliberate table entry that travels through
PR review with a justification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TOLERANCE_SCHEMA_V0 = 0
"""Schema version of the ``Tolerance`` payload."""

NumericalGrade = Literal["bit-exact", "ulp-bounded", "approximate"]
"""How tight the equivalence bound is.

* ``"bit-exact"`` — bit-identical floats. Used only for content hashes
  and integer-coded quantities; not for floating-point quantities
  even on the same machine.
* ``"ulp-bounded"`` — within a small ULP count of the source. The
  embedded-default for properly-implemented codegen on a single
  precision.
* ``"approximate"`` — engineering-tolerance level. Used for
  cross-precision comparisons (source ``float64`` vs. candidate
  ``float32``) and for quantities sensitive to the dtype mix.
"""


class Tolerance(BaseModel):
    """A single ``(target_family, dtype, quantity)`` tolerance row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=TOLERANCE_SCHEMA_V0, ge=0)
    """Schema version (PATTERNS §3.4); currently v0."""

    abs_tol: float
    """Absolute-error bound on a per-step basis."""

    rel_tol: float
    """Relative-error bound on a per-step basis (``|a - b| / max(|a|, |b|)``)."""

    grade: NumericalGrade
    """Categorical grade; documents the *intent* of the bound."""

    rationale: str
    """One-line justification recorded in the contract; surfaces in PR review."""


# The tolerance table. Each entry is human-readable and reviewed.
# Cross-check with ``test/EQUIVALENCE.md``; tests assert the two stay in sync.
#
# The current entries cover the mock targets and the small fixed set of
# quantities the mock pipeline (T-015) exercises. Real targets
# add rows later.
TOLERANCE_TABLE: dict[tuple[str, str, str], Tolerance] = {
    # mock-cortex-a, float64 (the mock pipeline runs at float64 by
    # default; bounds are tight because the candidate just wraps the source).
    ("mock-cortex-a", "float64", "joint_position"): Tolerance(
        abs_tol=1e-12,
        rel_tol=1e-10,
        grade="ulp-bounded",
        rationale=(
            "Mock pipeline wraps the source JAX in float64; bound mirrors "
            "JAX's own jit ULP envelope."
        ),
    ),
    ("mock-cortex-a", "float64", "joint_velocity"): Tolerance(
        abs_tol=1e-10,
        rel_tol=1e-9,
        grade="ulp-bounded",
        rationale=(
            "Velocities accumulate one integration step of position error; "
            "one OOM looser than position."
        ),
    ),
    ("mock-cortex-a", "float64", "actuator_torque"): Tolerance(
        abs_tol=1e-10,
        rel_tol=1e-9,
        grade="ulp-bounded",
        rationale=(
            "Actuator outputs are linear in state at the mock layer; "
            "same envelope as joint_velocity."
        ),
    ),
    # mock-cortex-a, float32 (looser; cross-precision)
    ("mock-cortex-a", "float32", "joint_position"): Tolerance(
        abs_tol=1e-5,
        rel_tol=1e-4,
        grade="approximate",
        rationale=(
            "float32 candidate vs. float64 source; bound reflects "
            "single-precision rounding accumulation."
        ),
    ),
    ("mock-cortex-a", "float32", "joint_velocity"): Tolerance(
        abs_tol=1e-4,
        rel_tol=1e-3,
        grade="approximate",
        rationale=(
            "float32 velocity inherits one extra integration step of float32 noise."
        ),
    ),
    ("mock-cortex-a", "float32", "actuator_torque"): Tolerance(
        abs_tol=1e-4,
        rel_tol=1e-3,
        grade="approximate",
        rationale="float32 actuator outputs; matches velocity bound at mock scale.",
    ),
    # mock-cortex-m, float32 (embedded-default; tighter than cortex-a float32
    # because the mock pipeline still runs the source at full precision).
    ("mock-cortex-m", "float32", "joint_position"): Tolerance(
        abs_tol=1e-5,
        rel_tol=1e-4,
        grade="approximate",
        rationale=(
            "Identical bound to mock-cortex-a float32; the mock layer does "
            "not introduce target-specific drift."
        ),
    ),
    ("mock-cortex-m", "float32", "joint_velocity"): Tolerance(
        abs_tol=1e-4,
        rel_tol=1e-3,
        grade="approximate",
        rationale=(
            "Matches mock-cortex-a float32 joint_velocity; mock target does "
            "not differentiate yet."
        ),
    ),
    ("mock-cortex-m", "float32", "actuator_torque"): Tolerance(
        abs_tol=1e-4,
        rel_tol=1e-3,
        grade="approximate",
        rationale=(
            "Matches mock-cortex-a float32 actuator_torque; mock target does "
            "not differentiate yet."
        ),
    ),
    # host-darwin and host-linux: the host build path runs at float64. The
    # bound reflects the *integrator mismatch* between acados' internal ERK
    # sub-stepping and a single-stage RK4 forward integration of the JAX
    # dynamics — not ULP. acados defaults to multiple sub-steps per OCP
    # stage so its trajectory is more accurate than a coarse single-step
    # JAX integration; engineering-tight bounds reflect what's actually
    # observed at the 1 s, 20-stage Cartpole canonical setup.
    # Tightening to ULP is a later task (T-027 follow-up) where the
    # JAX integrator is configured to match acados' internal sub-step
    # count and the bound drops accordingly.
    ("host-darwin", "float64", "joint_position"): Tolerance(
        abs_tol=1e-2,
        rel_tol=3e-1,
        grade="approximate",
        rationale=(
            "Host build at float64; bound reflects integrator-step "
            "mismatch between acados ERK sub-stepping and the JAX-side "
            "single-stage RK4 reference. Engineering tolerance over a "
            "1 s, 20-stage Cartpole horizon."
        ),
    ),
    ("host-darwin", "float64", "joint_velocity"): Tolerance(
        abs_tol=5e-2,
        rel_tol=5e-1,
        grade="approximate",
        rationale=(
            "Velocity inherits one extra integration step of "
            "integrator-mismatch error; ~10x looser than position."
        ),
    ),
    ("host-darwin", "float64", "actuator_torque"): Tolerance(
        abs_tol=1e-10,
        rel_tol=1e-9,
        grade="ulp-bounded",
        rationale=(
            "Controls round-trip through acados' LINEAR_LS cost without "
            "passing through the integrator; standard ULP envelope at "
            "float64 holds."
        ),
    ),
    ("host-linux", "float64", "joint_position"): Tolerance(
        abs_tol=1e-2,
        rel_tol=3e-1,
        grade="approximate",
        rationale="Same as host-darwin float64 joint_position.",
    ),
    ("host-linux", "float64", "joint_velocity"): Tolerance(
        abs_tol=5e-2,
        rel_tol=5e-1,
        grade="approximate",
        rationale="Same as host-darwin float64 joint_velocity.",
    ),
    ("host-linux", "float64", "actuator_torque"): Tolerance(
        abs_tol=1e-10,
        rel_tol=1e-9,
        grade="ulp-bounded",
        rationale="Same as host-darwin float64 actuator_torque.",
    ),
    # cortex-a76, float32 — HIL (T-033). The candidate is the
    # deployed artifact executing on real Cortex-A76 silicon (Raspberry
    # Pi 5) at the embedded-default float32 precision; the source is the
    # host float64 reference. Bounds reflect single-precision rounding
    # accumulated over a control horizon of a few hundred 1 kHz Euler
    # steps on a stable (decaying) plant — engineering tolerance, not
    # ULP, because the precision mix is the dominant error term. Measured
    # against the deterministic HIL fixture; tighten when the generated
    # acados shim (T-034) replaces the fixture and the integrator is
    # matched.
    ("cortex-a76", "float32", "joint_position"): Tolerance(
        abs_tol=5e-4,
        rel_tol=2e-2,
        grade="approximate",
        rationale=(
            "float32 artifact on Cortex-A76 vs float64 host reference; bound "
            "covers single-precision accumulation over a few hundred 1 kHz "
            "Euler steps on a decaying plant."
        ),
    ),
    ("cortex-a76", "float32", "joint_velocity"): Tolerance(
        abs_tol=1e-3,
        rel_tol=2e-2,
        grade="approximate",
        rationale=(
            "Angular rate carries one extra integration step of float32 "
            "noise; ~2x looser than joint_position."
        ),
    ),
    ("cortex-a76", "float32", "actuator_torque"): Tolerance(
        abs_tol=5e-3,
        rel_tol=2e-2,
        grade="approximate",
        rationale=(
            "Actuator command is a fixed linear gain on the state, so it "
            "amplifies the state's float32 error by the gain magnitude."
        ),
    ),
    # cortex-a76, float64 — HIL for the real acados controller
    # (T-034). The candidate is the generated closed-loop binary: acados OCP
    # control + acados sim (ERK) plant, at acados' native float64. The source
    # is the host closed loop: the same acados control with a JAX ERK4 plant.
    # Measured host-vs-host divergence is ~1e-14 (ULP) — acados' ERK sim
    # integrator matches the JAX ERK4 reference, and the Python and C codegen
    # agree bit-for-bit — so the bound is ULP-bounded, NOT engineering
    # tolerance. The 1e-9 / 1e-7 envelope sits ~5 orders above the measured
    # host floor to carry cross-architecture libm (sin/cos) and FMA-contraction
    # differences on real Cortex-A76; to be re-confirmed when the on-Pi gate
    # lands (T-036). A looser bound would hide real codegen divergence.
    ("cortex-a76", "float64", "joint_position"): Tolerance(
        abs_tol=1e-9,
        rel_tol=1e-7,
        grade="ulp-bounded",
        rationale=(
            "Closed-loop acados controller vs host acados control + JAX ERK4 "
            "plant; measured ~1e-14 host-vs-host, bound carries cross-arch "
            "libm/FMA margin for Cortex-A76 (re-confirm at T-036)."
        ),
    ),
    ("cortex-a76", "float64", "joint_velocity"): Tolerance(
        abs_tol=1e-9,
        rel_tol=1e-7,
        grade="ulp-bounded",
        rationale="Same closed-loop ULP regime as cortex-a76 float64 joint_position.",
    ),
    ("cortex-a76", "float64", "actuator_torque"): Tolerance(
        abs_tol=1e-9,
        rel_tol=1e-7,
        grade="ulp-bounded",
        rationale=(
            "Control round-trips through the acados solve without an extra "
            "integration step; same ULP regime."
        ),
    ),
}


def lookup_tolerance(target_family: str, dtype: str, quantity: str) -> Tolerance:
    """Return the tolerance row for a ``(target_family, dtype, quantity)``.

    Raises
    ------
    KeyError
        If no row exists. There is no silent default; adding support
        for a new quantity on a target is an explicit table entry.
    """
    key = (target_family, dtype, quantity)
    try:
        return TOLERANCE_TABLE[key]
    except KeyError:
        raise KeyError(
            f"No tolerance row for (target_family={target_family!r}, "
            f"dtype={dtype!r}, quantity={quantity!r}). Add an entry to "
            f"jaxility.testing.tolerances.TOLERANCE_TABLE and document it "
            f"in test/EQUIVALENCE.md before lowering for this combination."
        ) from None


def quantities_for(target_family: str, dtype: str) -> list[str]:
    """List every quantity the tolerance table covers for a target / dtype.

    Used by the mock pipeline (T-015) to enumerate the contract
    it must satisfy. Order is sorted-and-stable.
    """
    return sorted(
        q for (tf, dt, q) in TOLERANCE_TABLE if tf == target_family and dt == dtype
    )
