# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the ``Target`` abstraction + mock profiles (T-011).

T-011 acceptance criteria:

1. The mock targets round-trip through serialisation bit-exactly.
2. The target hash is stable, deterministic, and sensitive to every
   field change.
"""

from __future__ import annotations

import pytest

from jaxility.manifest import canonical_dumps
from jaxility.targets import (
    MOCK_CORTEX_A,
    MOCK_CORTEX_M,
    MOCK_TARGETS,
    MemoryConstraints,
    NPUCapability,
    NPUFamily,
    Quirk,
    RealtimeGuarantee,
    RealtimeKind,
    Scheduler,
    Target,
    ToolchainPin,
    VectorExtension,
)

# ---------------------------------------------------------------------------
# Acceptance 1 — mock targets serialise / round-trip bit-exactly.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("target", [MOCK_CORTEX_A, MOCK_CORTEX_M])
def test_mock_target_round_trips_bit_exactly(target: Target) -> None:
    """``Target`` → JSON → ``Target`` → JSON is byte-identical."""
    first = canonical_dumps(target)
    revived = Target.model_validate_json(target.model_dump_json())
    second = canonical_dumps(revived)
    assert first == second


@pytest.mark.unit
@pytest.mark.parametrize("target", [MOCK_CORTEX_A, MOCK_CORTEX_M])
def test_mock_target_name_matches_family(target: Target) -> None:
    """For mock targets the name equals the family — tolerance lookup uses it.

    Real targets later (e.g. Pi 5) will have ``name="pi5"`` and
    ``family="cortex-a76"``; the equality here is mock-only.
    """
    assert target.name == target.family


@pytest.mark.unit
def test_mock_targets_dict_keys_match_target_names() -> None:
    """``MOCK_TARGETS`` is indexed by ``target.name`` exactly."""
    assert set(MOCK_TARGETS) == {MOCK_CORTEX_A.name, MOCK_CORTEX_M.name}
    for name, target in MOCK_TARGETS.items():
        assert target.name == name


@pytest.mark.unit
def test_mock_targets_supports_capability_queries() -> None:
    """``Target.supports`` is the canonical capability gate (PATTERNS §5.2)."""
    assert MOCK_CORTEX_A.supports(VectorExtension.NEON)
    assert not MOCK_CORTEX_A.supports(VectorExtension.HELIUM)
    assert MOCK_CORTEX_M.supports(VectorExtension.HELIUM)
    assert not MOCK_CORTEX_M.supports(VectorExtension.NEON)


# ---------------------------------------------------------------------------
# Acceptance 2 — hash is stable, deterministic, sensitive to every field.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_target_hashes_are_stable_within_session() -> None:
    """Reading the property twice returns the same digest."""
    assert MOCK_CORTEX_A.hash == MOCK_CORTEX_A.hash
    assert MOCK_CORTEX_A.hash_hex == MOCK_CORTEX_A.hash_hex


@pytest.mark.unit
def test_mock_targets_have_distinct_hashes() -> None:
    """The two mock targets must not collide."""
    assert MOCK_CORTEX_A.hash != MOCK_CORTEX_M.hash


@pytest.mark.unit
def test_target_hash_changes_when_name_changes() -> None:
    """Changing ``name`` produces a different hash."""
    other = MOCK_CORTEX_A.model_copy(update={"name": "mock-cortex-a-prime"})
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_family_changes() -> None:
    """Changing ``family`` produces a different hash."""
    other = MOCK_CORTEX_A.model_copy(update={"family": "different-family"})
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_vector_extensions_change() -> None:
    """Changing the vector-extension tuple produces a different hash."""
    other = MOCK_CORTEX_A.model_copy(
        update={"vector_extensions": (VectorExtension.NEON, VectorExtension.SVE)}
    )
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_memory_changes() -> None:
    """Changing a memory field produces a different hash."""
    new_memory = MOCK_CORTEX_A.memory.model_copy(update={"code_bytes": 1})
    other = MOCK_CORTEX_A.model_copy(update={"memory": new_memory})
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_realtime_changes() -> None:
    """Changing the real-time guarantee produces a different hash."""
    new_rt = MOCK_CORTEX_A.realtime.model_copy(update={"target_cycle_us": 2_000})
    other = MOCK_CORTEX_A.model_copy(update={"realtime": new_rt})
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_toolchain_changes() -> None:
    """Changing the toolchain pin produces a different hash."""
    new_tc = MOCK_CORTEX_A.toolchain.model_copy(update={"version": "13.2.0"})
    other = MOCK_CORTEX_A.model_copy(update={"toolchain": new_tc})
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_npu_changes() -> None:
    """Changing the NPU capability produces a different hash."""
    new_npu = NPUCapability(family=NPUFamily.ETHOS_U55, peak_tops=0.5)
    other = MOCK_CORTEX_A.model_copy(update={"npu": new_npu})
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_quirks_change() -> None:
    """Adding / removing a quirk produces a different hash."""
    new_quirk = Quirk(
        id="new-quirk",
        description="A test quirk added for hash sensitivity.",
    )
    other = MOCK_CORTEX_A.model_copy(
        update={"quirks": MOCK_CORTEX_A.quirks + (new_quirk,)}
    )
    assert other.hash != MOCK_CORTEX_A.hash


@pytest.mark.unit
def test_target_hash_changes_when_vendor_sdk_paths_change() -> None:
    """Changing the vendor-SDK path dict produces a different hash."""
    other = MOCK_CORTEX_A.model_copy(
        update={"vendor_sdk_paths": {"some-sdk": "/opt/some/sdk"}}
    )
    assert other.hash != MOCK_CORTEX_A.hash


# ---------------------------------------------------------------------------
# Schema discipline: extra="forbid", frozen.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_rejects_unknown_field() -> None:
    """``extra="forbid"`` — an unknown field is an error (PATTERNS §3.3)."""
    payload = MOCK_CORTEX_A.model_dump(mode="json")
    payload["unknown_field"] = "x"
    with pytest.raises(ValueError):
        Target.model_validate(payload)


@pytest.mark.unit
def test_target_is_immutable() -> None:
    """``frozen=True`` — field assignment raises."""
    with pytest.raises(ValueError):
        MOCK_CORTEX_A.name = "something-else"  # type: ignore[misc]


@pytest.mark.unit
def test_toolchain_pin_mock_is_well_formed() -> None:
    """The ``ToolchainPin.mock`` factory produces a valid pin."""
    from jaxility.targets import UNVERIFIED_SHA256

    pin = ToolchainPin.mock("foo")
    assert pin.name == "foo-mock-toolchain"
    assert pin.version == "0.0.0-mock"
    assert pin.detect_command == ("foo-mock-toolchain", "--version")
    # M-7: mock pins use the honest "unverified" sentinel rather than
    # ``"0" * 64`` (which looked like a real SHA-256 but meant nothing).
    assert pin.expected_sha256 == UNVERIFIED_SHA256
    assert pin.has_pinned_integrity() is False


@pytest.mark.unit
def test_realtime_guarantee_allows_none_cycle_for_non_rt() -> None:
    """A non-RT target may omit the target cycle."""
    rt = RealtimeGuarantee(
        kind=RealtimeKind.NONE,
        scheduler=Scheduler.NONE,
        target_cycle_us=None,
    )
    assert rt.target_cycle_us is None


@pytest.mark.unit
def test_memory_constraints_reject_zero_or_negative() -> None:
    """Memory fields must be positive — a zero-byte budget is a bug."""
    with pytest.raises(ValueError):
        MemoryConstraints(code_bytes=0, data_bytes=1, stack_bytes=1, has_mmu=False)
