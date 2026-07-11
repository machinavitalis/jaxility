# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the Pi 5 / Cortex-A76 Target profile (T-030)."""

from __future__ import annotations

import pytest

from jaxility.manifest import canonical_dumps
from jaxility.targets import (
    HOST_DARWIN,
    HOST_LINUX,
    MOCK_CORTEX_A,
    MOCK_CORTEX_M,
    PI5,
    NPUFamily,
    RealtimeKind,
    Scheduler,
    Target,
    VectorExtension,
)

# ---------------------------------------------------------------------------
# Profile shape.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pi5_basics() -> None:
    assert PI5.name == "pi5"
    assert PI5.family == "cortex-a76"


@pytest.mark.unit
def test_pi5_toolchain_pinned() -> None:
    """Arm GNU toolchain 15.2.1 (the launch-baseline pin)."""
    tc = PI5.toolchain
    assert tc.name == "aarch64-none-linux-gnu-gcc"
    assert tc.version == "15.2.1"
    assert "15.2.rel1" in tc.distribution
    assert tc.download_url.startswith("https://developer.arm.com/")


@pytest.mark.unit
def test_pi5_supports_neon() -> None:
    assert PI5.supports(VectorExtension.NEON)
    assert PI5.supports("neon")
    # Pi 5 / Cortex-A76 does NOT have SVE.
    assert not PI5.supports(VectorExtension.SVE)


@pytest.mark.unit
def test_pi5_has_no_npu() -> None:
    assert PI5.npu.family == NPUFamily.NONE
    assert PI5.npu.peak_tops == 0.0


@pytest.mark.unit
def test_pi5_memory_baseline_is_8_gb() -> None:
    """Launch baseline is the 8 GB sku."""
    eight_gib = 8 * 1024 * 1024 * 1024
    assert PI5.memory.code_bytes == eight_gib
    assert PI5.memory.data_bytes == eight_gib
    assert PI5.memory.has_mmu is True


@pytest.mark.unit
def test_pi5_realtime_soft_at_1khz() -> None:
    assert PI5.realtime.kind == RealtimeKind.SOFT
    assert PI5.realtime.scheduler == Scheduler.PREEMPT_RT
    assert PI5.realtime.target_cycle_us == 1_000


# ---------------------------------------------------------------------------
# Quirks documented.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pi5_documents_sku_quirk() -> None:
    """The 4 GB / 16 GB sku quirk is in the profile so the build planner sees it."""
    ids = {q.id for q in PI5.quirks}
    assert "pi5-4gb-sku" in ids


@pytest.mark.unit
def test_pi5_documents_preempt_rt_soft_not_hard_quirk() -> None:
    """The "soft, not hard" RT quirk is the safety-critical disclaimer."""
    ids = {q.id for q in PI5.quirks}
    assert "preempt-rt-soft-not-hard" in ids


@pytest.mark.unit
def test_pi5_documents_dcache_clean_for_codegen_quirk() -> None:
    """D-cache-clean-required-for-codegen-buffers is a runtime-layer contract."""
    ids = {q.id for q in PI5.quirks}
    assert "d-cache-clean-required-for-codegen-buffers" in ids


# ---------------------------------------------------------------------------
# Hash distinctness — Pi 5 must not collide with any prior profile.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pi5_hash_distinct_from_mock_and_host_profiles() -> None:
    """invariant 5: Target hashes are sensitive to every field change."""
    seen = {
        PI5.hash,
        MOCK_CORTEX_A.hash,
        MOCK_CORTEX_M.hash,
        HOST_DARWIN.hash,
        HOST_LINUX.hash,
    }
    assert len(seen) == 5


# ---------------------------------------------------------------------------
# Round-trip: canonical encoding stable.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pi5_round_trips_bit_exactly() -> None:
    """``Target`` → JSON → ``Target`` → JSON byte-identical."""
    first = canonical_dumps(PI5)
    revived = Target.model_validate_json(PI5.model_dump_json())
    second = canonical_dumps(revived)
    assert first == second


@pytest.mark.unit
def test_pi5_is_frozen() -> None:
    with pytest.raises(ValueError):
        PI5.name = "different-pi5"  # type: ignore[misc]
