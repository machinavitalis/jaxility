# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Attestation manifest schema v0 (ADR-005).

The manifest is the contract with Jaxterity upstream (attestation
chain) and any compliance consumer downstream. v0 is the OSS minimum
— BLAKE3 hash chain over canonical JSON — with a pluggable
:class:`Signer` interface so the enterprise tier can swap in real
signing without changing the schema.

Schema v0 fields per ADR-005:

* ``schema_version`` — int.
* ``source_attestation_handle`` — bytes; the Jaxterity
  ``Robot.attestation_handle`` decoded from its hex string.
* ``toolchain_versions`` — ``dict[str, str]``; canonical name →
  semver.
* ``target_profile_hash`` — bytes; from
  :attr:`jaxility.targets.Target.hash`.
* ``artifact_content_hash`` — bytes; the deployed artifact's content
  hash (lands in T-014).
* ``build_timestamp_utc`` — int microseconds since epoch; **factored
  out of the content hash** per invariant 5.
* ``signer_identity`` — optional str; the
  :attr:`Signer.identity` of whoever signed.
* ``signature`` — optional bytes; the signature over the content
  hash.

The manifest is *immutable* once constructed. The content hash is the
BLAKE3 digest of the canonical-JSON encoding of the chain-participating
fields (everything *except* timestamp, signer_identity, signature). The
signature, when present, is over the content hash.
"""

from __future__ import annotations

from typing import Annotated, Any

import blake3
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from .canonical import canonical_dumps

SCHEMA_VERSION_V0 = 0
"""The current schema version. v1 will be SLSA-aligned (enterprise)."""


_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _bytes_from_hex(value: Any) -> bytes:
    """Accept ``bytes`` unchanged; decode hex strings to bytes.

    The JSON form of every ``bytes`` field is a lowercase hex string —
    human-readable, round-trippable, and (unlike Pydantic's default
    raw-UTF-8 encoding) safe for arbitrary digests.

    Whitespace and other non-hex characters are rejected explicitly:
    Python's :meth:`bytes.fromhex` silently strips ASCII whitespace
    (``"ab cd"`` → ``b"\xab\xcd"``), which would make two cosmetically
    different JSON manifests parse to identical bytes. The canonical
    form requires byte-identical text, so this validator demands a
    bare, contiguous hex string.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        if any(ch not in _HEX_CHARS for ch in value):
            raise ValueError(
                f"expected a bare hex string for a bytes field; got "
                f"{value!r} (non-hex characters present — whitespace "
                "and separators are not permitted)"
            )
        try:
            return bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(
                f"expected a hex string for a bytes field; got {value!r}"
            ) from exc
    raise TypeError(
        f"bytes field expects bytes or a hex string; got {type(value).__name__}"
    )


HexBytes = Annotated[
    bytes,
    BeforeValidator(_bytes_from_hex),
    PlainSerializer(lambda b: b.hex(), return_type=str, when_used="json"),
]
"""``bytes`` field that serialises to **JSON** as a lowercase hex string.

The ``when_used="json"`` clause is load-bearing: ``model_dump(mode="json")``
and ``model_dump_json()`` emit hex strings (manifests stay human-readable),
but ``model_dump(mode="python")`` still returns raw bytes — which lets
:func:`jaxility.manifest.canonical_dumps` route them through its own
``{"$b16": "<hex>"}`` encoding consistently with
:meth:`Manifest.content_payload`. Without ``when_used="json"`` the two
representations would disagree and a third-party verifier implementing
the documented canonical form would compute a different hash than the
in-process verifier.
"""

_CONTENT_FIELDS = (
    "schema_version",
    "source_attestation_handle",
    "toolchain_versions",
    "target_profile_hash",
    "artifact_content_hash",
)
"""Fields that participate in :meth:`Manifest.content_hash`.

Order is deliberate but immaterial — :func:`canonical_dumps` sorts
keys at every depth. The tuple is the single source of truth so a
review can see at a glance what enters the hash chain.
"""


class Manifest(BaseModel):
    """Attestation manifest schema v0 (ADR-005)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(
        default=SCHEMA_VERSION_V0,
        ge=0,
        description="Schema version; currently v0, v1 will land with SLSA alignment.",
    )
    source_attestation_handle: HexBytes = Field(
        description=(
            "BLAKE3 digest from Jaxterity's ``Robot.attestation_handle``. "
            "Jaxterity exposes the handle as a hex string; manifests carry "
            "the decoded bytes for content-hash stability."
        ),
    )
    toolchain_versions: dict[str, str] = Field(
        description=(
            "Canonical toolchain name → version. Every external binary "
            "Jaxility invoked during the build appears here (invariant 3 / "
            "TOOLCHAINS.md)."
        ),
    )
    target_profile_hash: HexBytes = Field(
        description="BLAKE3 digest of the Target profile (ADR-003, T-011).",
    )
    artifact_content_hash: HexBytes = Field(
        description=(
            "BLAKE3 digest of the deployed artifact. ``Artifact`` lands in "
            "T-014; mock pipelines synthesise the hash from the "
            "source robot."
        ),
    )
    build_timestamp_utc: int = Field(
        ge=0,
        description=(
            "Microseconds since the Unix epoch. **Excluded from the "
            "content hash** per invariant 5 (deterministic builds — same "
            "source + same toolchain + same target → byte-identical hash, "
            "regardless of when the build ran)."
        ),
    )
    signer_identity: str | None = Field(
        default=None,
        description=(
            "Identity of the :class:`Signer` that produced ``signature``. "
            '``"hash-chain-v0"`` for OSS unsigned manifests; an enterprise '
            "signer plugs in its own identity string."
        ),
    )
    signature: HexBytes | None = Field(
        default=None,
        description=(
            "Signature over :meth:`content_hash`. ``None`` for OSS "
            "hash-chain manifests; populated by an enterprise signer."
        ),
    )

    # The two helpers below are the integration surface for verification
    # and signing. They are intentionally side-effect-free.

    def content_payload(self) -> bytes:
        """Canonical-JSON bytes of the chain-participating fields.

        :meth:`content_hash` is BLAKE3 over this payload; a signer is
        a function over this payload. Excludes timestamp, signer
        identity, and signature (ADR-005).
        """
        chain_only = {field: getattr(self, field) for field in _CONTENT_FIELDS}
        return canonical_dumps(chain_only)

    def content_hash(self) -> bytes:
        """BLAKE3 digest of :meth:`content_payload`.

        Two manifests with the same source / toolchain / target /
        artifact produce identical ``content_hash`` regardless of
        ``build_timestamp_utc`` (invariant 5).
        """
        return blake3.blake3(self.content_payload()).digest()
