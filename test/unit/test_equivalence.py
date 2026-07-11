# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the equivalence-check protocol (T-010).

Covers the two T-010 acceptance criteria:

1. The mock target produces a trivially-equivalent trajectory and the
   check passes.
2. A deliberately-corrupted target fails with a pinpointed structured
   suggestion (PATTERNS §6.3).

Plus a cross-check that ``test/EQUIVALENCE.md`` documents every row in
``TOLERANCE_TABLE`` — the markdown is the human contract, the Python
dict is the source of truth, and they must stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from jaxility.errors import EquivalenceError
from jaxility.testing import equivalence as eqv
from jaxility.testing import tolerances as tol_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_trajectory(
    n_steps: int = 50, dim: int = 2, *, seed: int = 0
) -> eqv.Trajectory:
    """A small deterministic mock trajectory used by every test."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_steps)
    pos = np.stack([np.sin(2.0 * np.pi * (t + 0.1 * i)) for i in range(dim)], axis=1)
    vel = np.stack(
        [2.0 * np.pi * np.cos(2.0 * np.pi * (t + 0.1 * i)) for i in range(dim)], axis=1
    )
    torque = 0.1 * rng.standard_normal((n_steps, dim))
    return {
        "joint_position": pos,
        "joint_velocity": vel,
        "actuator_torque": torque,
    }


# ---------------------------------------------------------------------------
# Acceptance 1: mock target trivially equivalent.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trivially_equivalent_passes_on_mock_cortex_a_float64() -> None:
    """Source and candidate identical → every quantity passes."""
    source = _make_trajectory()
    candidate = {k: v.copy() for k, v in source.items()}

    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float64"
    )

    assert report.overall_passed is True
    assert report.target_family == "mock-cortex-a"
    assert report.dtype == "float64"
    assert {qd.quantity for qd in report.per_quantity} == set(source)
    for qd in report.per_quantity:
        assert qd.passed
        assert qd.first_violation_step is None
        assert qd.suggestion is None


@pytest.mark.unit
def test_trivially_equivalent_passes_on_mock_cortex_a_float32() -> None:
    """Same trajectory at float32 still passes — bounds are wider."""
    source = _make_trajectory()
    candidate = {k: v.astype(np.float32).astype(np.float64) for k, v in source.items()}

    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float32"
    )

    assert report.overall_passed is True


@pytest.mark.unit
def test_assert_passed_is_silent_on_pass() -> None:
    """``assert_passed`` returns silently when the report passed."""
    source = _make_trajectory()
    candidate = {k: v.copy() for k, v in source.items()}
    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float64"
    )
    report.assert_passed()  # must not raise


# ---------------------------------------------------------------------------
# Acceptance 2: corrupted target fails with a pinpointed suggestion.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_actuator_corruption_fails_with_actuator_suggestion() -> None:
    """Adding a constant offset to actuator_torque trips the actuator suggestion."""
    source = _make_trajectory()
    candidate = {k: v.copy() for k, v in source.items()}
    # 10 N·m offset on every actuator at every step — well above the float64
    # mock-cortex-a actuator tolerance (1e-10).
    candidate["actuator_torque"] = candidate["actuator_torque"] + 10.0

    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float64"
    )

    assert report.overall_passed is False
    actuator = next(
        qd for qd in report.per_quantity if qd.quantity == "actuator_torque"
    )
    assert actuator.passed is False
    assert actuator.first_violation_step == 0
    assert actuator.max_abs_error >= 9.9
    assert actuator.suggestion is not None
    assert "actuat" in actuator.suggestion.lower()
    # The other quantities should still pass — the corruption is local.
    others = [qd for qd in report.per_quantity if qd.quantity != "actuator_torque"]
    assert all(qd.passed for qd in others)


@pytest.mark.unit
def test_early_step_divergence_suggests_integration_error() -> None:
    """Large-magnitude early-step divergence trips the integration hint."""
    source = _make_trajectory(n_steps=200)
    candidate = {k: v.copy() for k, v in source.items()}
    # 5-rad position offset at every step → both early and not actuator-related.
    candidate["joint_position"] = candidate["joint_position"] + 5.0

    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float64"
    )

    assert report.overall_passed is False
    pos = next(qd for qd in report.per_quantity if qd.quantity == "joint_position")
    assert pos.passed is False
    assert pos.first_violation_step == 0
    assert pos.suggestion is not None
    assert (
        "integration" in pos.suggestion.lower()
        or "instability" in pos.suggestion.lower()
    )


@pytest.mark.unit
def test_assert_passed_raises_on_fail() -> None:
    """``assert_passed`` raises with a structured message when the report failed."""
    source = _make_trajectory()
    candidate = {k: v.copy() for k, v in source.items()}
    candidate["actuator_torque"] = candidate["actuator_torque"] + 1.0

    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float64"
    )

    with pytest.raises(EquivalenceError, match="Equivalence failed"):
        report.assert_passed()


# ---------------------------------------------------------------------------
# Shape / lookup miss guards (PATTERNS §7.4 — loud failure).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_tolerance_row_raises_key_error_with_pointer() -> None:
    """Querying an unknown ``(target, dtype, quantity)`` raises with a hint."""
    # A deliberately unsupported target family: no row exists, and none is
    # planned, so this stays a genuine miss as the real-target rows grow.
    with pytest.raises(KeyError, match="TOLERANCE_TABLE"):
        tol_mod.lookup_tolerance("imaginary-soc", "float64", "joint_position")


@pytest.mark.unit
def test_compare_rejects_mismatched_quantity_sets() -> None:
    """Source and candidate must carry the same quantities; loud, not silent."""
    source = _make_trajectory()
    candidate = {"joint_position": source["joint_position"]}  # missing two

    with pytest.raises(EquivalenceError, match="candidate trajectory is missing"):
        eqv.compare(source, candidate, target_family="mock-cortex-a", dtype="float64")


@pytest.mark.unit
def test_compare_rejects_shape_mismatch() -> None:
    """A per-quantity shape disagreement raises."""
    source = _make_trajectory(n_steps=20)
    candidate = {k: v.copy() for k, v in source.items()}
    candidate["joint_position"] = candidate["joint_position"][:10]

    with pytest.raises(EquivalenceError, match="Trajectory shape mismatch"):
        eqv.compare(source, candidate, target_family="mock-cortex-a", dtype="float64")


# ---------------------------------------------------------------------------
# Documentation parity (PATTERNS §7.4 enforcement).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_table_row_is_documented_in_equivalence_md() -> None:
    """``test/EQUIVALENCE.md`` mentions every (target_family, dtype, quantity) row.

    The python table is the source of truth; the markdown is the
    human-readable contract. They must stay in sync — a row added in
    code that is not documented is a CI failure under PATTERNS §7.4.
    """
    doc = (REPO_ROOT / "test" / "EQUIVALENCE.md").read_text()
    for target_family, dtype, quantity in tol_mod.TOLERANCE_TABLE:
        # The doc lists target_family × dtype in section headers and the
        # quantity in the table row; assert both surface in the text.
        assert target_family in doc, (
            f"target_family {target_family!r} is in TOLERANCE_TABLE but "
            f"not mentioned in test/EQUIVALENCE.md"
        )
        assert dtype in doc, (
            f"dtype {dtype!r} is in TOLERANCE_TABLE but not mentioned in "
            f"test/EQUIVALENCE.md"
        )
        assert quantity in doc, (
            f"quantity {quantity!r} is in TOLERANCE_TABLE but not mentioned "
            f"in test/EQUIVALENCE.md"
        )


@pytest.mark.unit
def test_quantity_divergence_has_schema_version() -> None:
    """Per PATTERNS §3.4 every schema record carries a ``schema_version``."""
    qd = eqv.QuantityDivergence(
        quantity="joint_position",
        max_abs_error=0.0,
        max_rel_error=0.0,
        first_violation_step=None,
        passed=True,
        tolerance=tol_mod.lookup_tolerance(
            "mock-cortex-a", "float64", "joint_position"
        ),
        suggestion=None,
    )
    assert qd.schema_version == eqv.EQUIVALENCE_SCHEMA_V0


@pytest.mark.unit
def test_tolerance_has_schema_version() -> None:
    """Per PATTERNS §3.4 every schema record carries a ``schema_version``."""
    tol = tol_mod.lookup_tolerance("mock-cortex-a", "float64", "joint_position")
    assert tol.schema_version == tol_mod.TOLERANCE_SCHEMA_V0


# ---------------------------------------------------------------------------
# Property tests (PATTERNS §7.2 — Hypothesis preferred over example tests for
# equivalence).
# ---------------------------------------------------------------------------


_finite_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@pytest.mark.unit
@given(
    pos=st_np.arrays(
        dtype=np.float64,
        shape=st_np.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=20),
        elements=_finite_floats,
    ),
    vel=st_np.arrays(
        dtype=np.float64,
        shape=st_np.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=20),
        elements=_finite_floats,
    ),
    torque=st_np.arrays(
        dtype=np.float64,
        shape=st_np.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=20),
        elements=_finite_floats,
    ),
)
@hypothesis.settings(max_examples=20, deadline=None)
def test_identical_trajectories_always_pass(
    pos: np.ndarray, vel: np.ndarray, torque: np.ndarray
) -> None:
    """Property: identical trajectories pass equivalence for any input shape."""
    # Hypothesis hands us arbitrary shapes; align them to a common one.
    n_steps = min(pos.shape[0], vel.shape[0], torque.shape[0])
    dim = min(pos.shape[1], vel.shape[1], torque.shape[1])
    source: eqv.Trajectory = {
        "joint_position": pos[:n_steps, :dim],
        "joint_velocity": vel[:n_steps, :dim],
        "actuator_torque": torque[:n_steps, :dim],
    }
    candidate: eqv.Trajectory = {k: v.copy() for k, v in source.items()}
    report = eqv.compare(
        source, candidate, target_family="mock-cortex-a", dtype="float64"
    )
    assert report.overall_passed is True


@pytest.mark.unit
@given(
    offset=st.floats(
        min_value=1.0,
        max_value=1e3,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@hypothesis.settings(max_examples=20, deadline=None)
def test_constant_actuator_offset_always_fails_with_actuator_suggestion(
    offset: float,
) -> None:
    """Property: a >= 1 N·m constant actuator offset always fails."""
    base = _make_trajectory(n_steps=20)
    candidate = {k: v.copy() for k, v in base.items()}
    candidate["actuator_torque"] = candidate["actuator_torque"] + offset

    report = eqv.compare(
        base, candidate, target_family="mock-cortex-a", dtype="float64"
    )
    assert report.overall_passed is False
    actuator = next(
        qd for qd in report.per_quantity if qd.quantity == "actuator_torque"
    )
    assert actuator.passed is False
    assert actuator.suggestion is not None
    assert "actuat" in actuator.suggestion.lower()


@pytest.mark.unit
def test_quantities_for_returns_sorted_list() -> None:
    """``quantities_for`` enumerates the contract for a (target, dtype)."""
    quantities = tol_mod.quantities_for("mock-cortex-a", "float64")
    assert quantities == sorted(quantities)
    assert set(quantities) == {
        "actuator_torque",
        "joint_position",
        "joint_velocity",
    }
