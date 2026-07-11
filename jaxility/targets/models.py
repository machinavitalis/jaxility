# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Pydantic data models for the ``Target`` abstraction (ADR-003).

A :class:`Target` is data, not code. Adding a new SoC is filling out
the model. Target-conditional behaviour lives elsewhere (PATTERNS §5.1
— strategy registrations keyed off capability flags, not virtual
methods on the target).

Every field on every model is reviewed for its contribution to the
target hash, because the hash travels in the manifest (ADR-005) and is
the contract with the attestation chain. Adding a field is a tightly
scoped ADR-grade decision; PATTERNS §3.4 governs schema versioning.

These models are frozen and ``extra="forbid"``. An unknown field is an
error; a mutated target is impossible. Both properties matter for the
deterministic-build invariant (CONTEXT.md invariant 5).
"""

from __future__ import annotations

from enum import Enum

import blake3
from pydantic import BaseModel, ConfigDict, Field

from ..manifest import canonical_dumps


class VectorExtension(str, Enum):
    """SIMD vector ISA extensions a target may expose."""

    NONE = "none"
    NEON = "neon"  # Cortex-A; also baseline on aarch64.
    SVE = "sve"  # Cortex-A; variable-length vectors.
    SVE2 = "sve2"  # Cortex-A; SVE successor.
    HELIUM = "helium"  # Cortex-M; M-profile vector extension.
    SSE = "sse"  # x86; host-only.
    AVX2 = "avx2"  # x86; host-only.


class NPUFamily(str, Enum):
    """Identifiable NPU families across the supported SoC list."""

    NONE = "none"
    ETHOS_U55 = "ethos-u55"
    ETHOS_U65 = "ethos-u65"
    QUALCOMM = "qualcomm"
    APPLE = "apple"  # Apple Neural Engine; deferred (OQ-4).


class NPUCapability(BaseModel):
    """NPU presence and headline capability.

    Mock targets carry ``family=NPUFamily.NONE`` and
    ``peak_tops=0.0``; real-target rows populate both fields from the
    vendor's spec.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: NPUFamily
    peak_tops: float = Field(
        ge=0.0,
        description="Peak TOPS at the published precision; 0.0 for NONE family.",
    )
    # ``integer_precisions`` / ``float_precisions`` tuples were declared
    # speculatively in T-011 but no code reads them. They land alongside
    # the first real NPU target (T-062 — Ethos-U55 / U65).


class MemoryConstraints(BaseModel):
    """Memory headroom the build planner must respect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code_bytes: int = Field(
        gt=0, description="Bytes available for code (.text + .rodata)."
    )
    data_bytes: int = Field(
        gt=0, description="Bytes available for mutable data (.data + .bss + heap)."
    )
    stack_bytes: int = Field(
        gt=0, description="Bytes available for the worst-case call stack."
    )
    has_mmu: bool
    """Whether the target has an MMU (Cortex-A: yes; Cortex-M: no)."""


class RealtimeKind(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


class Scheduler(str, Enum):
    NONE = "none"
    CYCLIC_EXECUTIVE = "cyclic-executive"
    PREEMPT_RT = "preempt-rt"
    FREERTOS = "freertos"


class RealtimeGuarantee(BaseModel):
    """Real-time scheduling category and target cycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RealtimeKind
    scheduler: Scheduler
    target_cycle_us: int | None = Field(
        default=None,
        ge=1,
        description="Target control-loop period in microseconds; ``None`` if non-RT.",
    )


class ToolchainPin(BaseModel):
    """Exact toolchain pin per :doc:`AGENTS/TOOLCHAINS.md`.

    Every external binary Jaxility shells out to is pinned to a
    specific version. The pin records the canonical binary name, the
    upstream release identifier, the official download URL, the
    expected SHA-256, the detect command (used at ``Target`` load),
    and the regex that extracts the version from the detect command's
    stdout (PATTERNS §2.2).

    Mock targets use ``ToolchainPin.mock("mock-cortex-a")`` (or
    similar) so the pin can travel through the manifest without
    pointing at a real binary that does not need to exist.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """Canonical binary name, e.g. ``"aarch64-none-linux-gnu-gcc"``."""

    version: str
    """Exact version string, e.g. ``"13.2.1"``."""

    distribution: str
    """Upstream release identifier, e.g. ``"arm-gnu-toolchain-13.2.rel1"``."""

    download_url: str
    """Official download URL for the pinned distribution."""

    expected_sha256: str = Field(
        description=(
            "Lowercase hex SHA-256 of the downloaded archive, OR the "
            'literal string ``"unverified"`` when no integrity hash '
            "has been pinned yet. The :data:`UNVERIFIED_SHA256` "
            "sentinel is the canonical placeholder; "
            ":meth:`has_pinned_integrity` returns ``False`` for it. "
            "``verify_toolchain_integrity`` raises ToolchainError on "
            "an unverified pin rather than silently passing — invariant "
            '7. The previous ``"0" * 64`` placeholder was a *valid* '
            "hex string and passed every type check while meaning "
            "nothing; the audit M-7 fix renames it."
        ),
    )

    detect_command: tuple[str, ...]
    """Argv used to detect the binary at ``Target`` load time."""

    version_regex: str
    """Regex extracting the version from the detect command's stdout."""

    def has_pinned_integrity(self) -> bool:
        """``True`` iff ``expected_sha256`` is a real hex hash, not the placeholder.

        Used by :func:`jaxility.builder_cross.verify_toolchain_integrity`
        to decide whether to actually run a SHA check or raise an
        "integrity not yet pinned" error.
        """
        return self.expected_sha256 != UNVERIFIED_SHA256

    @classmethod
    def mock(cls, name: str) -> ToolchainPin:
        """Construct a ``ToolchainPin`` for a mock target.

        Mock targets do not invoke a real toolchain (PATTERNS §5.3);
        the pin still appears in the manifest so the canonical-JSON
        serialiser sees a well-typed value.
        """
        return cls(
            name=f"{name}-mock-toolchain",
            version="0.0.0-mock",
            distribution="mock-distribution",
            download_url="https://example.invalid/mock-toolchain",
            expected_sha256=UNVERIFIED_SHA256,
            detect_command=(f"{name}-mock-toolchain", "--version"),
            version_regex=r"mock (\d+\.\d+\.\d+)",
        )


UNVERIFIED_SHA256: str = "unverified"
"""Sentinel placeholder for ``ToolchainPin.expected_sha256`` (M-7).

Pins carrying this value have **not** had their binary integrity
pinned yet. They serialise into the manifest as the literal string
``"unverified"`` so a downstream reader sees the gap honestly rather
than a 64-zero hex digest that looks like a real hash. Use
:meth:`ToolchainPin.has_pinned_integrity` to query.
"""


class Quirk(BaseModel):
    """A documented per-target surprise the build planner must respect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="Short stable identifier, e.g. ``denormals-flushed-to-zero``."
    )
    description: str = Field(
        description="One-paragraph human-readable description for review."
    )
    # ``severity`` (info/warn/error) and ``introduced_in`` (version) were
    # declared speculatively in T-011 but no code reads them; they land
    # alongside the first dispatcher that branches on them.


class Target(BaseModel):
    """A complete profile of a deployment SoC (ADR-003, PATTERNS §5).

    The hash is content-addressed over the canonical-JSON encoding of
    every field. Two ``Target`` instances with the same field values
    produce byte-identical hashes; any field change produces a
    different hash (invariant 5 — deterministic builds).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        description=(
            "Full target identifier, e.g. ``mock-cortex-a``, ``cortex-a76``. "
            "Used by the CLI as the ``--target`` flag value."
        )
    )
    family: str = Field(
        description=(
            "Short family identifier used by the tolerance-table lookup "
            "(PATTERNS §7.4) and by capability-keyed dispatch (PATTERNS §5.2). "
            "Targets in the same family share equivalence bounds."
        )
    )
    schema_version: int = Field(
        default=0,
        ge=0,
        description=(
            "Schema version of this Target record. Changing the schema is "
            "an ADR-grade decision (PATTERNS §3.4)."
        ),
    )
    toolchain: ToolchainPin
    vector_extensions: tuple[VectorExtension, ...]
    npu: NPUCapability
    memory: MemoryConstraints
    realtime: RealtimeGuarantee
    vendor_sdk_paths: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional name → install-path mapping for vendor SDKs the "
            "target requires (e.g. ``{'qualcomm-see': '/opt/qcom/see'}``). "
            "Mock targets leave this empty."
        ),
    )
    quirks: tuple[Quirk, ...] = ()

    @property
    def hash(self) -> bytes:
        """BLAKE3 digest over the canonical-JSON encoding of this target.

        Returns
        -------
        bytes
            32-byte BLAKE3 digest. Manifest field
            ``target_profile_hash`` (ADR-005) is this value.
        """
        return blake3.blake3(canonical_dumps(self)).digest()

    @property
    def hash_hex(self) -> str:
        """Hex-string form of :attr:`hash`. Convenience for logging / CLIs."""
        return self.hash.hex()

    def supports(self, extension: VectorExtension | str) -> bool:
        """Capability query used by capability-keyed dispatch.

        PATTERNS §5.2: target-conditional code reaches for
        ``target.supports(...)``, never for ``if target.name == ...``.

        Accepts either a :class:`VectorExtension` enum member or its
        string value (``"neon"``, ``"helium"``, …) so callers do not
        have to import the enum just to ask. Unknown strings return
        ``False`` rather than raising — the typical caller wants
        "does this target have feature X" semantics.
        """
        if isinstance(extension, str):
            try:
                extension = VectorExtension(extension)
            except ValueError:
                return False
        return extension in self.vector_extensions
