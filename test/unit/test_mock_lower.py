# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the end-to-end mock lowering pipeline (T-015).

T-015 acceptance criteria:

1. End-to-end mock build on the simplest Cartpole passes the
   equivalence check.
2. Manifest verification passes.
3. Changing any source input changes the artifact hash.
"""

from __future__ import annotations

import pytest

from jaxility.errors import CoverageError
from jaxility.manifest import verify_manifest
from jaxility.targets import MOCK_CORTEX_A, MOCK_CORTEX_M
from jaxility.testing import (
    MockArtifactBundle,
    MockSource,
    Source,
    compare,
    mock_lower,
)


def _simple_cartpole() -> MockSource:
    """Cartpole stand-in.

    Two-DOF, fixed initial state. The handle salt + name + initial
    state determine the attestation handle.
    """
    return MockSource(
        name="cartpole",
        initial_state=(0.0, 0.0),
        dim=2,
    )


# ---------------------------------------------------------------------------
# Acceptance 1: end-to-end mock build passes equivalence.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_lower_cartpole_passes_equivalence() -> None:
    """Mock pipeline trivially equivalent to its source (invariant 1)."""
    source = _simple_cartpole()
    bundle = mock_lower(source, MOCK_CORTEX_A, n_steps=50, dtype="float64")

    expected = source.simulate(50)
    candidate = bundle.simulate()

    report = compare(
        expected, candidate, target_family="mock-cortex-a", dtype="float64"
    )
    assert report.overall_passed is True
    report.assert_passed()


@pytest.mark.unit
def test_mock_lower_cartpole_passes_equivalence_on_cortex_m_float32() -> None:
    """The mock pipeline works for both mock targets."""
    source = _simple_cartpole()
    bundle = mock_lower(source, MOCK_CORTEX_M, n_steps=20, dtype="float32")

    report = compare(
        source.simulate(20),
        bundle.simulate(),
        target_family="mock-cortex-m",
        dtype="float32",
    )
    assert report.overall_passed is True


@pytest.mark.unit
def test_mock_lower_records_build_log_stages() -> None:
    """The artifact's build_log spans plan → lower → package."""
    bundle = mock_lower(_simple_cartpole(), MOCK_CORTEX_A)
    stages = [entry.stage for entry in bundle.artifact.build_log]
    assert stages == ["plan", "lower"]
    # ``package`` is logged after the artifact is built so it does not
    # appear in the artifact's own log (the artifact's log is closed
    # before the artifact is constructed) — this is fine and documented.


# ---------------------------------------------------------------------------
# Acceptance 2: manifest verification passes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_lower_manifest_verifies() -> None:
    """The bundle's manifest verifies under the OSS hash-chain signer."""
    bundle = mock_lower(_simple_cartpole(), MOCK_CORTEX_A)
    report = verify_manifest(bundle.manifest)
    assert report.ok is True


@pytest.mark.unit
def test_mock_lower_manifest_verifies_against_expected_hash() -> None:
    """Recomputing the content hash and supplying it as expected → ok."""
    bundle = mock_lower(_simple_cartpole(), MOCK_CORTEX_A)
    report = verify_manifest(
        bundle.manifest,
        expected_content_hash=bundle.manifest.content_hash(),
    )
    assert report.ok is True


@pytest.mark.unit
def test_mock_lower_chain_intact_handle_to_manifest_to_artifact() -> None:
    """The attestation chain is unbroken end-to-end."""
    source = _simple_cartpole()
    bundle = mock_lower(source, MOCK_CORTEX_A)

    # source handle → manifest's source_attestation_handle
    assert bundle.manifest.source_attestation_handle == source.attestation_handle
    # manifest.artifact_content_hash → artifact.content_hash
    assert bundle.manifest.artifact_content_hash == bundle.artifact.content_hash
    # artifact.source_manifest_hash → manifest.content_hash()
    assert bundle.artifact.source_manifest_hash == bundle.manifest.content_hash()
    # target hash agrees on both ends
    assert bundle.manifest.target_profile_hash == MOCK_CORTEX_A.hash
    assert bundle.artifact.target_profile_hash == MOCK_CORTEX_A.hash


# ---------------------------------------------------------------------------
# Acceptance 3: changing any source input changes the artifact hash.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_changing_source_name_changes_artifact_hash() -> None:
    """Two sources with different names produce different artifact hashes."""
    a = mock_lower(MockSource(name="cartpole", initial_state=(0.0, 0.0)), MOCK_CORTEX_A)
    b = mock_lower(
        MockSource(name="cartpole2", initial_state=(0.0, 0.0)), MOCK_CORTEX_A
    )
    assert a.artifact.content_hash != b.artifact.content_hash


@pytest.mark.unit
def test_changing_source_initial_state_changes_artifact_hash() -> None:
    """Different initial state → different handle → different artifact hash."""
    a = mock_lower(MockSource(name="cartpole", initial_state=(0.0, 0.0)), MOCK_CORTEX_A)
    b = mock_lower(MockSource(name="cartpole", initial_state=(0.1, 0.0)), MOCK_CORTEX_A)
    assert a.artifact.content_hash != b.artifact.content_hash


@pytest.mark.unit
def test_changing_source_handle_salt_changes_artifact_hash() -> None:
    """A different upstream recipe salt produces a different artifact hash."""
    a = mock_lower(
        MockSource(name="cartpole", initial_state=(0.0, 0.0), handle_salt=b"salt-a"),
        MOCK_CORTEX_A,
    )
    b = mock_lower(
        MockSource(name="cartpole", initial_state=(0.0, 0.0), handle_salt=b"salt-b"),
        MOCK_CORTEX_A,
    )
    assert a.artifact.content_hash != b.artifact.content_hash


@pytest.mark.unit
def test_changing_target_changes_artifact_hash() -> None:
    """Two different targets produce different artifact hashes."""
    source = _simple_cartpole()
    a = mock_lower(source, MOCK_CORTEX_A, dtype="float32")
    b = mock_lower(source, MOCK_CORTEX_M, dtype="float32")
    assert a.artifact.content_hash != b.artifact.content_hash


@pytest.mark.unit
def test_changing_dtype_changes_artifact_hash() -> None:
    """A dtype change is part of the build identity."""
    source = _simple_cartpole()
    a = mock_lower(source, MOCK_CORTEX_A, dtype="float64")
    b = mock_lower(source, MOCK_CORTEX_A, dtype="float32")
    assert a.artifact.content_hash != b.artifact.content_hash


@pytest.mark.unit
def test_changing_n_steps_changes_artifact_hash() -> None:
    """Trajectory length is part of the build identity."""
    source = _simple_cartpole()
    a = mock_lower(source, MOCK_CORTEX_A, n_steps=10)
    b = mock_lower(source, MOCK_CORTEX_A, n_steps=20)
    assert a.artifact.content_hash != b.artifact.content_hash


@pytest.mark.unit
def test_same_inputs_produce_identical_artifact_hash() -> None:
    """Determinism — invariant 5 at the pipeline level."""
    source = _simple_cartpole()
    a = mock_lower(source, MOCK_CORTEX_A, build_timestamp_utc=1)
    b = mock_lower(source, MOCK_CORTEX_A, build_timestamp_utc=2)
    assert a.artifact.content_hash == b.artifact.content_hash
    # Manifest content hash is timestamp-independent too (T-012 already
    # covers this property; here we cross-check it through the pipeline).
    assert a.manifest.content_hash() == b.manifest.content_hash()


# ---------------------------------------------------------------------------
# Type / surface guards.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_lower_raises_coverage_error_on_uncovered_target_dtype() -> None:
    """B1: an empty tolerance result for ``(family, dtype)`` is a loud failure.

    Pre-review, ``_coverage_assert_quantities`` early-returned when
    ``quantities_for`` was empty — coverage was silently sidestepped on
    any combination with no tolerance rows. This test pins the loud-fail
    contract.
    """
    source = _simple_cartpole()
    # mock-cortex-m has no float64 tolerance rows.
    with pytest.raises(CoverageError, match="tolerance table has no rows"):
        mock_lower(source, MOCK_CORTEX_M, dtype="float64")


@pytest.mark.unit
def test_mock_bundle_source_cannot_be_externally_rebound() -> None:
    """N2: switching to ``PrivateAttr`` keeps the source stable post-construction."""
    source = _simple_cartpole()
    bundle = mock_lower(source, MOCK_CORTEX_A)
    different = MockSource(name="other", initial_state=(1.0, 2.0))

    # PrivateAttr is protected by Pydantic's frozen-model gate: a regular
    # attribute write raises; the object.__setattr__ trick that worked
    # pre-review now hits Pydantic's internal validator.
    with pytest.raises((AttributeError, ValueError)):
        bundle._source = different  # type: ignore[misc]


@pytest.mark.unit
def test_mock_source_satisfies_source_protocol() -> None:
    """``MockSource`` registers as a ``Source`` at runtime."""
    assert isinstance(_simple_cartpole(), Source)


@pytest.mark.unit
def test_bundle_exposes_source_and_simulate() -> None:
    """The bundle holds its source by reference and re-simulates on demand."""
    source = _simple_cartpole()
    bundle = mock_lower(source, MOCK_CORTEX_A)
    assert isinstance(bundle, MockArtifactBundle)
    # Re-simulating gives the same trajectory shape.
    traj = bundle.simulate(10)
    assert set(traj) == {"joint_position", "joint_velocity", "actuator_torque"}
    assert traj["joint_position"].shape == (10, source.dim)
