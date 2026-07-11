# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Host-platform Target profiles (T-026).

The host build path lets users (and the CI gate) exercise the full
JAX → CasADi → acados pipeline without cross-compilation. The
resulting binary is a normal shared library callable from Python via
``ctypes``; it carries the full manifest chain so verifies under
``jaxility verify`` the same as a cross-compiled artifact.

Two profiles ship at v0.1:

* ``HOST_DARWIN`` — macOS / arm64 with Apple clang as the toolchain.
* ``HOST_LINUX`` — Linux / x86_64 (or arm64) with system gcc.

The right one is selected by :func:`current_host_target` based on
``sys.platform``; callers wanting cross-platform builds (e.g. CI
matrices) construct the explicit profile directly.

These are *real* targets — invariant 6 (HIL gates supportedness)
applies — but the HIL surface for a host target is trivial: the host
*is* the deployment. There is no separate HIL tier.
"""

from __future__ import annotations

import sys

from .models import (
    UNVERIFIED_SHA256,
    MemoryConstraints,
    NPUCapability,
    NPUFamily,
    RealtimeGuarantee,
    RealtimeKind,
    Scheduler,
    Target,
    ToolchainPin,
    VectorExtension,
)


def _darwin_arm64() -> Target:
    return Target(
        name="host-darwin",
        family="host-darwin",
        toolchain=ToolchainPin(
            name="clang",
            version="apple-clang-21",
            distribution="apple-xcode-clt",
            download_url="https://developer.apple.com/xcode/",
            expected_sha256=UNVERIFIED_SHA256,
            detect_command=("clang", "--version"),
            version_regex=r"Apple clang version (\d+\.\d+\.\d+)",
        ),
        vector_extensions=(VectorExtension.NEON,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=1 << 32,  # 4 GiB
            data_bytes=1 << 34,  # 16 GiB
            stack_bytes=1 << 23,  # 8 MiB
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.NONE,
            scheduler=Scheduler.NONE,
            target_cycle_us=None,
        ),
        vendor_sdk_paths={},
    )


def _linux_x86_64() -> Target:
    return Target(
        name="host-linux",
        family="host-linux",
        toolchain=ToolchainPin(
            name="gcc",
            version="system-gcc",
            distribution="system-package",
            download_url="https://gcc.gnu.org/",
            expected_sha256=UNVERIFIED_SHA256,
            detect_command=("gcc", "--version"),
            version_regex=r"gcc \(.*\) (\d+\.\d+\.\d+)",
        ),
        vector_extensions=(VectorExtension.AVX2,),
        npu=NPUCapability(family=NPUFamily.NONE, peak_tops=0.0),
        memory=MemoryConstraints(
            code_bytes=1 << 32,
            data_bytes=1 << 34,
            stack_bytes=1 << 23,
            has_mmu=True,
        ),
        realtime=RealtimeGuarantee(
            kind=RealtimeKind.NONE,
            scheduler=Scheduler.NONE,
            target_cycle_us=None,
        ),
        vendor_sdk_paths={},
    )


HOST_DARWIN: Target = _darwin_arm64()
"""macOS arm64 host target."""

HOST_LINUX: Target = _linux_x86_64()
"""Linux x86_64 / arm64 host target."""

HOST_TARGETS: dict[str, Target] = {
    HOST_DARWIN.name: HOST_DARWIN,
    HOST_LINUX.name: HOST_LINUX,
}
"""Lookup-by-name for host targets."""


def current_host_target() -> Target:
    """Pick the right host target for ``sys.platform``.

    Returns
    -------
    Target
        :data:`HOST_DARWIN` on macOS; :data:`HOST_LINUX` on Linux.

    Raises
    ------
    LookupError
        On Windows or another unsupported host.
    """
    if sys.platform == "darwin":
        return HOST_DARWIN
    if sys.platform.startswith("linux"):
        return HOST_LINUX
    raise LookupError(
        f"no host Target for sys.platform={sys.platform!r}; "
        "Jaxility's host build path supports macOS arm64 and Linux."
    )
