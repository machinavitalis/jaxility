# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the extended ``Target`` profile batch (A4).

The batch adds nine target rows so the build planner has a manifest
contract for every supported SoC before its HIL gate lands. Tests
here cover:

* Basic shape of every new profile (name, family, toolchain pin).
* Capability invariants: Cortex-M4 has no SIMD; A710 has SVE2; Ethos
  profiles carry their NPU family + nonzero TOPS; Apple ANE is
  documented but conservative (TOPS=0).
* Hash distinctness across the *entire* known target set (mock + host
  + Pi 5 + extended). invariant 5: any field change must change the hash.
* Frozen-and-round-trips: every profile round-trips canonical JSON
  byte-identically and rejects mutation.
* Documented quirks: each profile carries the SoC-specific surprise(s)
  the runtime / planner has to respect.
"""

from __future__ import annotations

import pytest

from jaxility.manifest import canonical_dumps
from jaxility.targets import (
    APPLE_SILICON,
    CORTEX_A55,
    CORTEX_A78,
    CORTEX_A710,
    CORTEX_M4,
    ETHOS_U55,
    ETHOS_U65,
    HOST_DARWIN,
    HOST_LINUX,
    MOCK_CORTEX_A,
    MOCK_CORTEX_M,
    NEOVERSE_N1,
    EXTENDED_TARGETS,
    PI5,
    QUALCOMM_IQ10,
    NPUFamily,
    Target,
    VectorExtension,
)

# ---------------------------------------------------------------------------
# Batch shape
# ---------------------------------------------------------------------------

ALL_EXTENDED = [
    CORTEX_A55,
    CORTEX_A78,
    CORTEX_A710,
    NEOVERSE_N1,
    CORTEX_M4,
    ETHOS_U55,
    ETHOS_U65,
    QUALCOMM_IQ10,
    APPLE_SILICON,
]


@pytest.mark.unit
def test_extended_batch_has_nine_targets() -> None:
    assert len(EXTENDED_TARGETS) == 9
    assert len(ALL_EXTENDED) == 9


@pytest.mark.unit
def test_extended_lookup_keys_match_target_names() -> None:
    for t in ALL_EXTENDED:
        assert EXTENDED_TARGETS[t.name] is t


# ---------------------------------------------------------------------------
# Cortex-A family — toolchain pin + vector capability
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "target, expected_family",
    [
        (CORTEX_A55, "cortex-a55"),
        (CORTEX_A78, "cortex-a78"),
        (CORTEX_A710, "cortex-a710"),
        (NEOVERSE_N1, "neoverse-n1"),
    ],
)
def test_arm_a_profile_uses_aarch64_gnu_toolchain(
    target: Target, expected_family: str
) -> None:
    assert target.family == expected_family
    assert target.toolchain.name == "aarch64-none-linux-gnu-gcc"
    assert target.toolchain.version == "15.2.1"
    assert target.supports(VectorExtension.NEON)
    assert target.memory.has_mmu is True


@pytest.mark.unit
def test_cortex_a710_supports_sve2() -> None:
    """Cortex-A710 is Armv9; it gets SVE2 on top of NEON."""
    assert CORTEX_A710.supports(VectorExtension.SVE2)


@pytest.mark.unit
def test_other_a_profiles_do_not_advertise_sve2() -> None:
    """SVE2 is A710 only in this batch."""
    for target in (CORTEX_A55, CORTEX_A78, NEOVERSE_N1):
        assert not target.supports(VectorExtension.SVE2)


@pytest.mark.unit
def test_a_profiles_have_no_npu() -> None:
    """The A-profile rows are CPU-only; paired NPUs are separate Targets."""
    for target in (CORTEX_A55, CORTEX_A78, CORTEX_A710, NEOVERSE_N1):
        assert target.npu.family == NPUFamily.NONE
        assert target.npu.peak_tops == 0.0


# ---------------------------------------------------------------------------
# Cortex-M4
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cortex_m4_uses_baremetal_gnu_toolchain() -> None:
    assert CORTEX_M4.toolchain.name == "arm-none-eabi-gcc"
    assert CORTEX_M4.toolchain.version == "15.2.1"


@pytest.mark.unit
def test_cortex_m4_is_baremetal_hard_realtime() -> None:
    assert CORTEX_M4.memory.has_mmu is False
    from jaxility.targets import RealtimeKind, Scheduler

    assert CORTEX_M4.realtime.kind == RealtimeKind.HARD
    assert CORTEX_M4.realtime.scheduler == Scheduler.CYCLIC_EXECUTIVE


@pytest.mark.unit
def test_cortex_m4_has_no_simd_extension() -> None:
    """M4 ships before Helium (M55+). Vector extensions tuple is (NONE,)."""
    assert CORTEX_M4.vector_extensions == (VectorExtension.NONE,)
    assert not CORTEX_M4.supports(VectorExtension.HELIUM)
    assert not CORTEX_M4.supports(VectorExtension.NEON)


@pytest.mark.unit
def test_cortex_m4_documents_single_precision_fpu_quirk() -> None:
    ids = {q.id for q in CORTEX_M4.quirks}
    assert "single-precision-fpu-only" in ids
    assert "no-dynamic-allocation-after-init" in ids


# ---------------------------------------------------------------------------
# Ethos-U55 / U65
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ethos_u55_pairs_m55_host_with_npu() -> None:
    assert ETHOS_U55.toolchain.name == "arm-none-eabi-gcc"
    assert ETHOS_U55.supports(VectorExtension.HELIUM)
    assert ETHOS_U55.npu.family == NPUFamily.ETHOS_U55
    assert ETHOS_U55.npu.peak_tops > 0


@pytest.mark.unit
def test_ethos_u65_has_more_tops_than_u55() -> None:
    """U65's 512-MAC config doubles U55's 256-MAC headline TOPS."""
    assert ETHOS_U65.npu.peak_tops > ETHOS_U55.npu.peak_tops


@pytest.mark.unit
def test_ethos_profiles_flag_vela_dependency() -> None:
    for target in (ETHOS_U55, ETHOS_U65):
        ids = {q.id for q in target.quirks}
        assert "vela-required-for-npu-codegen" in ids
        assert "int8-only-npu" in ids


# ---------------------------------------------------------------------------
# Qualcomm IQ10
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_qualcomm_iq10_npu_family_and_tops() -> None:
    assert QUALCOMM_IQ10.npu.family == NPUFamily.QUALCOMM
    assert QUALCOMM_IQ10.npu.peak_tops > 0


@pytest.mark.unit
def test_qualcomm_iq10_requires_hexagon_sdk() -> None:
    assert "hexagon-sdk" in QUALCOMM_IQ10.vendor_sdk_paths
    ids = {q.id for q in QUALCOMM_IQ10.quirks}
    assert "hexagon-sdk-required-for-npu" in ids


# ---------------------------------------------------------------------------
# Apple Silicon
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apple_silicon_uses_apple_clang() -> None:
    assert APPLE_SILICON.toolchain.name == "clang"
    assert APPLE_SILICON.toolchain.version == "15.0.0"


@pytest.mark.unit
def test_apple_silicon_ane_is_documented_but_zero_tops() -> None:
    """Pending OQ-4 (ANE programming interface) the TOPS is conservative."""
    assert APPLE_SILICON.npu.family == NPUFamily.APPLE
    assert APPLE_SILICON.npu.peak_tops == 0.0
    ids = {q.id for q in APPLE_SILICON.quirks}
    assert "ane-tops-pending-oq4" in ids


@pytest.mark.unit
def test_apple_silicon_is_not_realtime() -> None:
    from jaxility.targets import RealtimeKind, Scheduler

    assert APPLE_SILICON.realtime.kind == RealtimeKind.NONE
    assert APPLE_SILICON.realtime.scheduler == Scheduler.NONE
    assert APPLE_SILICON.realtime.target_cycle_us is None


# ---------------------------------------------------------------------------
# Hash distinctness — every known profile must hash uniquely (invariant 5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_known_target_profile_has_distinct_hash() -> None:
    """The full known set: 2 mock + 2 host + 1 Pi 5 + 9 extended = 14 unique hashes."""
    all_targets = [
        MOCK_CORTEX_A,
        MOCK_CORTEX_M,
        HOST_DARWIN,
        HOST_LINUX,
        PI5,
        *ALL_EXTENDED,
    ]
    hashes = {t.hash for t in all_targets}
    assert len(hashes) == len(all_targets)


# ---------------------------------------------------------------------------
# Round-trip + freezing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("target", ALL_EXTENDED, ids=[t.name for t in ALL_EXTENDED])
def test_extended_target_round_trips_bit_exactly(target: Target) -> None:
    first = canonical_dumps(target)
    revived = Target.model_validate_json(target.model_dump_json())
    second = canonical_dumps(revived)
    assert first == second


@pytest.mark.unit
@pytest.mark.parametrize("target", ALL_EXTENDED, ids=[t.name for t in ALL_EXTENDED])
def test_extended_target_is_frozen(target: Target) -> None:
    with pytest.raises(ValueError):
        target.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Every profile carries at least one quirk
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("target", ALL_EXTENDED, ids=[t.name for t in ALL_EXTENDED])
def test_extended_target_documents_at_least_one_quirk(target: Target) -> None:
    """invariant 4 / PATTERNS §5: documented surprises travel with the profile."""
    assert len(target.quirks) >= 1
