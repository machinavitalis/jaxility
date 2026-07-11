# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Mock ``Target`` profiles used by the test suite (PATTERNS §5.3).

Mock targets are first-class :class:`Target` instances, not test
fixtures bolted on the side. Their ``toolchain`` carries a
:meth:`~jaxility.targets.models.ToolchainPin.mock` pin so they round
trip through the canonical-JSON serialiser and the manifest hash chain
without invoking any real binary.

Real targets (Pi 5 / Cortex-A76, STM32H7 / Cortex-M7, ...) land
as separate modules under :mod:`jaxility.targets`.
"""

from __future__ import annotations

from .models import (
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


def _mock_cortex_a() -> Target:
    """Construct the ``mock-cortex-a`` profile.

    Stand-in for a Cortex-A class processor: aarch64-ish ABI, MMU
    present, soft real-time scheduling under PREEMPT_RT, NEON for
    SIMD, no NPU. Memory headroom matches a small SBC (Pi-class).
    """
    return Target(
        name="mock-cortex-a",
        family="mock-cortex-a",
        toolchain=ToolchainPin.mock("mock-cortex-a"),
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=64 * 1024 * 1024,  # 64 MiB
            data_bytes=512 * 1024 * 1024,  # 512 MiB
            stack_bytes=8 * 1024 * 1024,  # 8 MiB
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.SOFT,
            scheduler=Scheduler.PREEMPT_RT,
            target_cycle_us=1_000,  # 1 kHz canonical loop
        ),
        vendor_sdk_paths={},
        quirks=(
            Quirk(
                id="float32-rounding-differs-from-host",
                description=(
                    "Mock layer evaluates at float64 then casts; downstream "
                    "real-target codegen at float32 should expect ULP-scale "
                    "drift vs. the host source simulation."
                ),
            ),
        ),
    )


def _mock_cortex_m() -> Target:
    """Construct the ``mock-cortex-m`` profile.

    Stand-in for a Cortex-M class processor: bare-metal, no MMU, hard
    real-time under a cyclic executive, Helium for SIMD, no NPU. Tight
    memory headroom (kibibytes, not mebibytes).
    """
    return Target(
        name="mock-cortex-m",
        family="mock-cortex-m",
        toolchain=ToolchainPin.mock("mock-cortex-m"),
        vector_extensions=(VectorExtension.HELIUM,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=512 * 1024,  # 512 KiB flash budget
            data_bytes=256 * 1024,  # 256 KiB SRAM budget
            stack_bytes=16 * 1024,  # 16 KiB stack budget
            has_mmu=False,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.HARD,
            scheduler=Scheduler.CYCLIC_EXECUTIVE,
            target_cycle_us=500,  # 2 kHz canonical loop on M-class
        ),
        vendor_sdk_paths={},
        quirks=(
            Quirk(
                id="no-dynamic-allocation-after-init",
                description=(
                    "Cortex-M class targets enforce no-malloc-after-init "
                    "(PATTERNS §4.1); the mock target carries the quirk so "
                    "downstream lowering preserves the constraint when it "
                    "moves to real M-class targets."
                ),
            ),
        ),
    )


MOCK_CORTEX_A: Target = _mock_cortex_a()
"""The canonical mock Cortex-A profile. Frozen module-level singleton."""

MOCK_CORTEX_M: Target = _mock_cortex_m()
"""The canonical mock Cortex-M profile. Frozen module-level singleton."""

MOCK_TARGETS: dict[str, Target] = {
    MOCK_CORTEX_A.name: MOCK_CORTEX_A,
    MOCK_CORTEX_M.name: MOCK_CORTEX_M,
}
"""Lookup-by-name for the mock targets."""
