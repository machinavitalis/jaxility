# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Pluggable signer protocol + OSS ``HashChainSigner`` (ADR-005, ADR-008).

ADR-005 says manifest schema v0 is a BLAKE3 hash chain over canonical
JSON. v0 is OSS-minimum; signing infrastructure is too heavy for the
OSS package, so the OSS signer (:class:`HashChainSigner`) emits
``signature=None`` and relies on the chain alone for verifiability.

ADR-008 makes :class:`Signer` a public protocol so the
``jaxility-enterprise`` package can plug in real signing (sigstore,
GPG, KMS, etc.) without changing the manifest schema.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

HASH_CHAIN_SIGNER_IDENTITY = "hash-chain-v0"
"""The identity string the OSS ``HashChainSigner`` writes to manifests."""


@runtime_checkable
class Signer(Protocol):
    """Pluggable signing interface (ADR-008).

    Implementations sign a content hash (the BLAKE3 digest of the
    manifest's canonical chain payload) and identify themselves so a
    verifier can dispatch.

    The OSS package ships :class:`HashChainSigner` (no signature).
    The ``jaxility-enterprise`` package provides real signing
    implementations behind the same interface.
    """

    @property
    def identity(self) -> str:
        """A short identifier the verifier dispatches on, e.g. ``"sigstore-v1"``."""
        ...

    def sign(self, content_hash: bytes) -> bytes | None:
        """Sign ``content_hash``. Return ``None`` for unsigned hash-chain mode."""
        ...

    def verify(self, content_hash: bytes, signature: bytes | None) -> bool:
        """Return ``True`` iff ``signature`` validates over ``content_hash``."""
        ...


class HashChainSigner:
    """The OSS signer (ADR-005 / ADR-008).

    Does no actual signing — ``signature=None`` — and relies on the
    BLAKE3 hash chain alone. ``verify`` succeeds when ``signature`` is
    ``None`` and ``content_hash`` is a well-formed 32-byte BLAKE3
    digest. The chain itself (source attestation handle → target
    profile hash → artifact content hash) is what carries the
    attestation; the manifest's role is to bind them.

    Enterprises that need real signing replace this with a sigstore /
    GPG / KMS-backed implementation behind the :class:`Signer`
    protocol (ADR-008).
    """

    def __init__(self) -> None:
        # Bind the identity at the instance level so static analysers /
        # protocol-conformance checks see it as a normal attribute (the
        # earlier class-level assignment confused some tooling).
        self.identity: str = HASH_CHAIN_SIGNER_IDENTITY

    def sign(self, content_hash: bytes) -> bytes | None:
        """Return ``None`` — OSS hash-chain manifests carry no signature."""
        if len(content_hash) != 32:
            raise ValueError(
                "HashChainSigner.sign expects a 32-byte BLAKE3 digest; "
                f"got {len(content_hash)} bytes."
            )
        return None

    def verify(self, content_hash: bytes, signature: bytes | None) -> bool:
        """Hash-chain verification: well-formed digest + no signature.

        A non-``None`` ``signature`` means the manifest was produced
        by a different (enterprise) signer; verification routes
        through that signer's ``verify``, not this one.
        """
        return signature is None and len(content_hash) == 32
