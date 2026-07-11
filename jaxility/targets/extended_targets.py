# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Extended ``Target`` profile batch (A4 / non-blocking pickup).

The Pi 5 launch profile (T-030) covers Cortex-A76. This batch
adds the rest of the supported-SoC list so the build planner has a
target row to attest against well before the corresponding HIL gate
(T-061..T-068) lands.

Each profile here is a *data-only* row: a :class:`Target` with the
toolchain pin, vector extension set, NPU capability, memory budget,
realtime guarantee, and per-SoC quirks. Adding behaviour for any of
these targets is a follow-on PR that registers a strategy keyed on the
capability flag (PATTERNS §5.2 — never a target name check).

Coverage breakdown:

* **Cortex-A55** — LITTLE core in big.LITTLE SoCs; common in mid-range
  robotics SBCs (Khadas VIM, Radxa Zero).
* **Cortex-A78** — perf core in Snapdragon QRB5165, Qualcomm RB5;
  Pi-class headroom with extra ML horsepower.
* **Cortex-A710** — Armv9 perf core; SVE2 vector extension; deployment
  baseline for the late-2025 generation Arm SBC class.
* **Neoverse-N1** — server / edge-server core (AWS Graviton2-class);
  enables Jaxility deployments to edge-gateway SBCs that bridge
  robotics fleets.
* **Cortex-M4** — classic M-profile with single-precision FPU; covers
  the STM32F4 / STM32F7 / nRF52 deployment family that comes after
  the M7 launch target.
* **Ethos-U55** — Cortex-M55 paired with the smaller Ethos NPU
  (256 MAC config; 0.5 TOPS @ INT8). Vela compiler is required for
  NPU codegen — flagged as a vendor-SDK quirk.
* **Ethos-U65** — Cortex-M55 paired with the larger Ethos NPU (512 MAC
  config; 1.0 TOPS @ INT8). Same toolchain as U55; differs only in
  NPU capability and the heap budget for activations.
* **Qualcomm IQ10** — RB3 Gen 2 robotics platform (Cortex-A78 host +
  Hexagon NPU; vendor TOPS spec ≈ 12 TOPS @ INT8). Hexagon SDK is a
  required vendor SDK path.
* **Apple Silicon** — M-series Macs (M1/M2/M3/M4). Apple Neural Engine
  presence is documented (NPUFamily.APPLE) but peak TOPS is
  conservatively 0.0 pending OQ-4 resolution (ANE programming
  interface for non-CoreML codegen).

This batch deliberately leaves Hexagon-side NPU codegen, Vela
integration, and ANE codegen out of scope. The Target rows are the
manifest contract; the dispatcher rows land with each per-target
deployment PR.
"""

from __future__ import annotations

from .models import (
    UNVERIFIED_SHA256,
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
# Pinned toolchains
# ---------------------------------------------------------------------------

# Arm GNU Toolchain 15.2.Rel1 — gnu-linux (aarch64). Same pin as PI5 but
# duplicated as a frozen constant so the extended-targets module is independently
# auditable. The placeholder `expected_sha256` mirrors PI5 until the
# real integrity check ships alongside the install flow.
#
# Banner shape (15.2.Rel1):
#   "...A-profile Architecture 15.2.Rel1 (Build arm-15.86)) 15.2.1 20251203"
# The "15.2.Rel1" inside the parens contains "Rel1" which does not match
# ``\d+\.\d+\.\d+``; we capture the *trailing* GCC-reported semver
# (``15.2.1``) which is followed by the YYYYMMDD build date.
_ARM_GNU_AARCH64: ToolchainPin = ToolchainPin(
    name="aarch64-none-linux-gnu-gcc",
    version="15.2.1",
    distribution="arm-gnu-toolchain-15.2.rel1",
    download_url=(
        "https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/"
        "binrel/arm-gnu-toolchain-15.2.rel1-darwin-arm64-aarch64-none-linux-gnu.tar.xz"
    ),
    expected_sha256=UNVERIFIED_SHA256,
    detect_command=("aarch64-none-linux-gnu-gcc", "--version"),
    version_regex=r"(\d+\.\d+\.\d+)\s+\d{8}",
)

# Arm GNU Toolchain 15.2.Rel1 — bare-metal (arm-none-eabi). M-profile.
_ARM_GNU_EABI: ToolchainPin = ToolchainPin(
    name="arm-none-eabi-gcc",
    version="15.2.1",
    distribution="arm-gnu-toolchain-15.2.rel1",
    download_url=(
        "https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/"
        "binrel/arm-gnu-toolchain-15.2.rel1-darwin-arm64-arm-none-eabi.tar.xz"
    ),
    expected_sha256=UNVERIFIED_SHA256,
    detect_command=("arm-none-eabi-gcc", "--version"),
    version_regex=r"(\d+\.\d+\.\d+)\s+\d{8}",
)

# Apple clang shipped with Xcode 15. ``clang --version`` reports the
# Apple-specific tag; the regex captures the LLVM upstream version that
# the Xcode 15.0 baseline pins.
_APPLE_CLANG: ToolchainPin = ToolchainPin(
    name="clang",
    version="15.0.0",
    distribution="Xcode-15.0",
    download_url="https://developer.apple.com/xcode/",
    expected_sha256=UNVERIFIED_SHA256,
    detect_command=("clang", "--version"),
    version_regex=r"Apple clang version (\d+\.\d+\.\d+)",
)


# ---------------------------------------------------------------------------
# Cortex-A family
# ---------------------------------------------------------------------------


def _cortex_a55() -> Target:
    return Target(
        name="cortex-a55",
        family="cortex-a55",
        toolchain=_ARM_GNU_AARCH64,
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=2 * 1024 * 1024 * 1024,
            data_bytes=2 * 1024 * 1024 * 1024,
            stack_bytes=4 * 1024 * 1024,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.SOFT,
            scheduler=Scheduler.PREEMPT_RT,
            target_cycle_us=2_000,  # 500 Hz baseline for mid-range A55 SBC
        ),
        quirks=(
            Quirk(
                id="big-little-affinity-required",
                description=(
                    "Cortex-A55 is the LITTLE core in big.LITTLE SoCs. "
                    "The Jaxility control thread must be pinned to A55 or "
                    "the perf core (depending on jitter vs. throughput "
                    "trade-off) — the runtime layer enforces affinity."
                ),
            ),
        ),
    )


def _cortex_a78() -> Target:
    return Target(
        name="cortex-a78",
        family="cortex-a78",
        toolchain=_ARM_GNU_AARCH64,
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=4 * 1024 * 1024 * 1024,
            data_bytes=4 * 1024 * 1024 * 1024,
            stack_bytes=8 * 1024 * 1024,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.SOFT,
            scheduler=Scheduler.PREEMPT_RT,
            target_cycle_us=1_000,
        ),
        quirks=(
            Quirk(
                id="paired-npu-handled-separately",
                description=(
                    "A78 ships paired with a vendor NPU (Hexagon on "
                    "Snapdragon; Ethos elsewhere). The NPU is exposed as "
                    "a separate Target profile (QUALCOMM_IQ10 / ETHOS_U65 "
                    "/ ...); this profile is the A78 host CPU only."
                ),
            ),
        ),
    )


def _cortex_a710() -> Target:
    return Target(
        name="cortex-a710",
        family="cortex-a710",
        toolchain=_ARM_GNU_AARCH64,
        vector_extensions=(VectorExtension.NEON, VectorExtension.SVE2),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=4 * 1024 * 1024 * 1024,
            data_bytes=4 * 1024 * 1024 * 1024,
            stack_bytes=8 * 1024 * 1024,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.SOFT,
            scheduler=Scheduler.PREEMPT_RT,
            target_cycle_us=500,  # 2 kHz feasible
        ),
        quirks=(
            Quirk(
                id="sve2-vector-length-detect-at-runtime",
                description=(
                    "SVE2 is variable-length (128..2048 bits). The runtime "
                    "must query the actual vector length at startup and "
                    "fall back to NEON-only paths if SVE2 is unexpectedly "
                    "absent (some SoCs ship A710 with SVE2 disabled at the "
                    "system register level)."
                ),
            ),
        ),
    )


def _neoverse_n1() -> Target:
    return Target(
        name="neoverse-n1",
        family="neoverse-n1",
        toolchain=_ARM_GNU_AARCH64,
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=64 * 1024 * 1024 * 1024,  # 64 GiB — server class
            data_bytes=64 * 1024 * 1024 * 1024,
            stack_bytes=16 * 1024 * 1024,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.SOFT,
            scheduler=Scheduler.PREEMPT_RT,
            target_cycle_us=1_000,
        ),
        quirks=(
            Quirk(
                id="edge-gateway-not-end-effector",
                description=(
                    "Neoverse-N1 deployments are edge-gateway / fleet "
                    "coordinator hosts, not end-effector compute. The "
                    "runtime layer assumes wall-clock latency budgets "
                    "in the millisecond range, not the microsecond range."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Cortex-M family
# ---------------------------------------------------------------------------


def _cortex_m4() -> Target:
    return Target(
        name="cortex-m4",
        family="cortex-m4",
        toolchain=_ARM_GNU_EABI,
        vector_extensions=(VectorExtension.NONE,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=1024 * 1024,  # 1 MiB flash (STM32F407-class)
            data_bytes=192 * 1024,  # 192 KiB SRAM
            stack_bytes=8 * 1024,
            has_mmu=False,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.HARD,
            scheduler=Scheduler.CYCLIC_EXECUTIVE,
            target_cycle_us=500,
        ),
        quirks=(
            Quirk(
                id="single-precision-fpu-only",
                description=(
                    "Cortex-M4 has single-precision FPU (FPv4-SP-D16). "
                    "Double-precision math is software-emulated and "
                    "≥10× slower; the coverage table for cortex-m4 must "
                    "set float64 ops to unsupported."
                ),
            ),
            Quirk(
                id="no-dynamic-allocation-after-init",
                description=(
                    "Bare-metal M-class enforces no-malloc-after-init (PATTERNS §4.1)."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Ethos NPU family (M55 host pairings)
# ---------------------------------------------------------------------------


def _ethos_u55() -> Target:
    return Target(
        name="ethos-u55",
        family="ethos-u55",
        toolchain=_ARM_GNU_EABI,
        vector_extensions=(VectorExtension.HELIUM,),
        npu=NPUCapability(family=NPUFamily.ETHOS_U55, peak_tops=0.5),
        memory=MemoryConstraints(
            code_bytes=2 * 1024 * 1024,  # 2 MiB flash
            data_bytes=512 * 1024,
            stack_bytes=16 * 1024,
            has_mmu=False,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.HARD,
            scheduler=Scheduler.CYCLIC_EXECUTIVE,
            target_cycle_us=1_000,
        ),
        quirks=(
            Quirk(
                id="vela-required-for-npu-codegen",
                description=(
                    "Ethos-U55 NPU code generation requires Arm's Vela "
                    "compiler (separate from arm-none-eabi-gcc). Vela "
                    "consumes TensorFlow Lite Micro models; Jaxility's "
                    "MPC codegen targets the M55 host CPU, with the NPU "
                    "left for the learned-policy lane."
                ),
            ),
            Quirk(
                id="int8-only-npu",
                description=(
                    "Ethos-U55 is INT8-only on the NPU. Float-precision "
                    "policy execution falls back to the M55 host."
                ),
            ),
        ),
    )


def _ethos_u65() -> Target:
    return Target(
        name="ethos-u65",
        family="ethos-u65",
        toolchain=_ARM_GNU_EABI,
        vector_extensions=(VectorExtension.HELIUM,),
        npu=NPUCapability(family=NPUFamily.ETHOS_U65, peak_tops=1.0),
        memory=MemoryConstraints(
            code_bytes=4 * 1024 * 1024,
            data_bytes=1024 * 1024,
            stack_bytes=16 * 1024,
            has_mmu=False,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.HARD,
            scheduler=Scheduler.CYCLIC_EXECUTIVE,
            target_cycle_us=1_000,
        ),
        quirks=(
            Quirk(
                id="vela-required-for-npu-codegen",
                description=(
                    "Same Vela dependency as Ethos-U55; the 512-MAC U65 "
                    "configuration delivers 1.0 TOPS @ INT8."
                ),
            ),
            Quirk(
                id="int8-only-npu",
                description=(
                    "Ethos-U65 NPU is INT8-only; learned policies use INT8 "
                    "quantization recipes (T-042)."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Qualcomm
# ---------------------------------------------------------------------------


def _qualcomm_iq10() -> Target:
    return Target(
        name="qualcomm-iq10",
        family="qualcomm-iq10",
        toolchain=_ARM_GNU_AARCH64,
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.QUALCOMM, peak_tops=12.0),
        memory=MemoryConstraints(
            code_bytes=8 * 1024 * 1024 * 1024,
            data_bytes=8 * 1024 * 1024 * 1024,
            stack_bytes=8 * 1024 * 1024,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.SOFT,
            scheduler=Scheduler.PREEMPT_RT,
            target_cycle_us=1_000,
        ),
        vendor_sdk_paths={
            "hexagon-sdk": "/opt/qcom/Hexagon_SDK",
        },
        quirks=(
            Quirk(
                id="hexagon-sdk-required-for-npu",
                description=(
                    "IQ10 NPU code generation requires the Qualcomm "
                    "Hexagon SDK (separate from the Arm cross-toolchain). "
                    "Jaxility's MPC codegen targets the A78 host; the "
                    "Hexagon NPU is for the learned-policy lane."
                ),
            ),
            Quirk(
                id="vendor-sdk-license-required",
                description=(
                    "Hexagon SDK distribution is licensed; the build "
                    "planner refuses to proceed if the SDK is missing "
                    "and surfaces the licensing URL."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Apple Silicon
# ---------------------------------------------------------------------------


def _apple_silicon() -> Target:
    """Apple M-series profile (M1/M2/M3/M4)."""
    return Target(
        name="apple-silicon",
        family="apple-silicon",
        toolchain=_APPLE_CLANG,
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.APPLE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=16 * 1024 * 1024 * 1024,
            data_bytes=16 * 1024 * 1024 * 1024,
            stack_bytes=8 * 1024 * 1024,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.NONE,
            scheduler=Scheduler.NONE,
            target_cycle_us=None,
        ),
        quirks=(
            Quirk(
                id="ane-tops-pending-oq4",
                description=(
                    "Apple Neural Engine peak TOPS is conservatively 0.0 "
                    "until OQ-4 resolves how Jaxility programs ANE outside "
                    "the CoreML conversion path. NPUFamily.APPLE is set "
                    "so the dispatcher can still capability-key on family."
                ),
            ),
            Quirk(
                id="sip-strips-dyld-library-path",
                description=(
                    "macOS System Integrity Protection strips "
                    "DYLD_LIBRARY_PATH from child processes; the test "
                    "harness uses ctypes.CDLL preload (see "
                    "test/conftest.py) to surface acados shared libs."
                ),
            ),
            Quirk(
                id="user-space-not-realtime",
                description=(
                    "macOS is not a real-time OS; deployments are "
                    "development hosts or non-RT robotics simulators. "
                    "Production deployment targets are PI5 / cortex-* "
                    "Linux."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

CORTEX_A55: Target = _cortex_a55()
"""Cortex-A55 LITTLE-core deployment profile."""

CORTEX_A78: Target = _cortex_a78()
"""Cortex-A78 perf-core deployment profile."""

CORTEX_A710: Target = _cortex_a710()
"""Cortex-A710 (Armv9 with SVE2) deployment profile."""

NEOVERSE_N1: Target = _neoverse_n1()
"""Neoverse-N1 edge-server deployment profile."""

CORTEX_M4: Target = _cortex_m4()
"""Cortex-M4 single-precision-FPU baremetal profile."""

ETHOS_U55: Target = _ethos_u55()
"""Cortex-M55 + Ethos-U55 NPU pair."""

ETHOS_U65: Target = _ethos_u65()
"""Cortex-M55 + Ethos-U65 NPU pair (larger MAC config)."""

QUALCOMM_IQ10: Target = _qualcomm_iq10()
"""Qualcomm RB3 Gen 2 / IQ10 deployment profile."""

APPLE_SILICON: Target = _apple_silicon()
"""Apple M-series (M1/M2/M3/M4) developer-host profile."""


EXTENDED_TARGETS: dict[str, Target] = {
    CORTEX_A55.name: CORTEX_A55,
    CORTEX_A78.name: CORTEX_A78,
    CORTEX_A710.name: CORTEX_A710,
    NEOVERSE_N1.name: NEOVERSE_N1,
    CORTEX_M4.name: CORTEX_M4,
    ETHOS_U55.name: ETHOS_U55,
    ETHOS_U65.name: ETHOS_U65,
    QUALCOMM_IQ10.name: QUALCOMM_IQ10,
    APPLE_SILICON.name: APPLE_SILICON,
}
"""Lookup-by-name for the extended target batch."""
