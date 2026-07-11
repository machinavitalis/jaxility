# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Raspberry Pi 5 / Cortex-A76 ``Target`` profile (T-030).

The Pi 5 is the v0.1 launch target. The full deployment lane is
T-030 (this profile) → T-031 (cross-compile wrapper) → T-032 (Pi 5 C
runtime) → T-033 (HIL parity tests) → T-034 (Cartpole-on-Pi-5 end to
end). This module ships the data half — the Target itself; the
toolchain wrapper, runtime, and HIL gate land in their own
PRs/phases.

Pi 5 specifics:

- **SoC:** Broadcom BCM2712, quad-core Cortex-A76 @ 2.4 GHz.
- **Vector extension:** NEON (advanced SIMD); no SVE.
- **NPU:** none.
- **Memory:** the canonical Pi 5 ships with 4 / 8 / 16 GB DDR4. The
  Target profile pins the 8 GB sku as the launch baseline; the 4 GB
  and 16 GB skus are documented as `quirks` so the build planner
  can warn at deployment if the user picks one.
- **Real-time:** Pi 5 + PREEMPT_RT-patched Raspberry Pi OS reaches
  the 1 kHz canonical Jaxility cycle reliably (~70 µs worst-case
  jitter measured externally; HIL tests will re-measure).
  Configured as ``RealtimeKind.SOFT`` because PREEMPT_RT is not a
  hard real-time guarantee in the EN ISO 13849 sense.
- **Toolchain:** Arm GNU Toolchain
  ``aarch64-none-linux-gnu-gcc 15.2.1`` (acados-template-compatible
  cross-compiler for the gnu-linux ABI). The canonical reproducible
  build runs on Linux x86_64 hosts; Arm does not ship a darwin-arm64
  host build of this chain in 15.2.Rel1, so on Apple Silicon dev
  hosts the Pi 5 cross-compile exercises in CI only (the test in
  ``test/unit/test_cross_compile.py`` skips locally and lights up on
  CI runners). The pin's ``expected_sha256`` is a placeholder
  (``0`` × 64) until the integrity check ships alongside the install
  flow.

ADR-004 fixes Pi 5 as the launch hardware target. ADR-010 says
Cortex-A and Cortex-M share the ``Target`` abstraction but not the
runtime code (`runtime-c/cortex-a/` vs `runtime-c/cortex-m/`).
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


def _pi5() -> Target:
    return Target(
        name="pi5",
        family="cortex-a76",
        toolchain=ToolchainPin(
            name="aarch64-none-linux-gnu-gcc",
            version="15.2.1",
            distribution="arm-gnu-toolchain-15.2.rel1",
            # Canonical reproducible build path is the linux-x86_64
            # archive (CI installs it; .github/workflows/ci.yml). Arm
            # does NOT ship a darwin-arm64 build for this chain in
            # 15.2.Rel1, so local dev on Apple Silicon is documented
            # as a gap in KNOWN_GAPS.md; cross-compile runs in CI.
            download_url=(
                "https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/"
                "binrel/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
            ),
            # Placeholder until the integrity check ships alongside
            # the install flow. CI re-downloads the archive and
            # verifies via cache key, not via expected_sha256 yet.
            expected_sha256=UNVERIFIED_SHA256,
            detect_command=("aarch64-none-linux-gnu-gcc", "--version"),
            # Captures the GCC-reported semver from the trailing
            # ``<version> YYYYMMDD`` banner tail (the 15.2.Rel1 release
            # writes ``15.2.Rel1`` inside the parens which does not
            # parse as semver; the trailing one is canonical).
            version_regex=r"(\d+\.\d+\.\d+)\s+\d{8}",
        ),
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            # 8 GB Pi 5 baseline; quirks document the 4 GB and 16 GB
            # variants. Leave generous code+data budgets — the build
            # planner can detect tighter SKUs at deploy time.
            code_bytes=8 * 1024 * 1024 * 1024,
            data_bytes=8 * 1024 * 1024 * 1024,
            stack_bytes=8 * 1024 * 1024,
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
                id="pi5-4gb-sku",
                description=(
                    "The Pi 5 ships in 4 GB / 8 GB / 16 GB skus. This "
                    "profile pins the 8 GB baseline; the 4 GB sku has "
                    "tighter total memory headroom and the build planner "
                    "may need to refuse builds whose static memory "
                    "footprint exceeds the 4 GB budget. The 16 GB sku is "
                    "additive — same Target hash but more available data "
                    "memory at runtime."
                ),
            ),
            Quirk(
                id="preempt-rt-soft-not-hard",
                description=(
                    "Pi 5 + PREEMPT_RT-patched Raspberry Pi OS reaches "
                    "1 kHz reliably with ~70 µs worst-case jitter "
                    "measured externally, but this is *soft* real-time. "
                    "Deployments to safety-critical contexts (EN ISO "
                    "13849 / IEC 62304) need the certification-grade "
                    "Cortex-M targets, not Pi 5. Documented "
                    "here so the build log surfaces the property."
                ),
            ),
            Quirk(
                id="gicv2-irq-priority-needs-tuning",
                description=(
                    "Pi 5's GIC-400 (GICv2) needs explicit IRQ-priority "
                    "tuning for PREEMPT_RT to give the Jaxility control "
                    "thread deterministic preemption against the kernel's "
                    "soft-IRQ work. The runtime support code "
                    "(T-032) sets this up; this quirk documents the "
                    "dependency so an end-to-end deployer knows the gate "
                    "is at the runtime layer, not the target profile."
                ),
            ),
            Quirk(
                id="d-cache-clean-required-for-codegen-buffers",
                description=(
                    "Generated controller code may live in a buffer "
                    "allocated dynamically at deploy time (the "
                    "``jaxility.builder`` outputs go through the cache "
                    "before execution). Cortex-A76 requires explicit "
                    "D-cache clean (``DC CIVAC``) before the I-cache "
                    "invalidate that lets the CPU execute the new code. "
                    "The runtime support code wraps this; documented "
                    "so a reviewer sees the contract."
                ),
            ),
        ),
    )


PI5: Target = _pi5()
"""The Raspberry Pi 5 / Cortex-A76 deployment target (launch baseline)."""
