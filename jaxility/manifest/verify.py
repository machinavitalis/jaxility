# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Manifest verification — the ``jaxility verify <manifest>`` core.

Verification at v0 walks the BLAKE3 hash chain:

1. The manifest parses against the v0 schema (loud failure on
   schema drift; future v1 manifests must be dispatched on
   ``schema_version`` — that branch lands when v1 lands).
2. The content hash recomputes from the chain-participating fields
   (PATTERNS §3.1 canonical encoding).
3. If an ``expected_content_hash`` was supplied (e.g. from a
   downstream registry), it must match the recomputed hash. This is
   how *tampering* is detected at the OSS level — the registry
   stores the hash; the verifier proves the manifest still produces
   it.
4. The signer's ``verify`` is invoked on ``(content_hash, signature)``.
   For OSS hash-chain manifests this is a structural check; for
   enterprise signers it is a cryptographic check.

The CLI entry point :func:`verify_cli` wires this into
``jaxility verify`` and emits a structured JSON report on stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..errors import ManifestError
from .models import SCHEMA_VERSION_V0, Manifest
from .signer import HASH_CHAIN_SIGNER_IDENTITY, HashChainSigner, Signer

EXIT_OK = 0
EXIT_INVALID_ARGS = 2
EXIT_MANIFEST_ERROR = 40
"""Exit codes per PATTERNS §8.3 — manifest errors map to 40."""


class ChainReport(BaseModel):
    """Structured verdict from :func:`verify_manifest` and the CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    """``True`` iff every check above passed."""

    reason: str
    """One-line human / agent explanation of the verdict."""

    schema_version: int
    """The schema version the manifest reported (``0`` currently)."""

    recomputed_content_hash_hex: str
    """Hex digest of the recomputed content hash. Always present."""

    expected_content_hash_hex: str | None
    """Hex digest of the externally-supplied expected hash, if given."""

    signer_identity: str | None
    """The identity field from the manifest."""

    signature_status: Literal["absent", "verified", "invalid"]
    """Whether a signature was present and, if so, the verifier's verdict."""


class ManifestVerificationError(ManifestError):
    """Raised when a manifest cannot even be parsed against schema v0.

    Routes through PATTERNS §6.1 via :class:`ManifestError`. It once used
    a direct ``ValueError`` parent; the PR #14 follow-up review
    moved it under :class:`JaxilityError` so library callers can catch
    the hierarchy uniformly.
    """


def _signature_status_for_report(
    manifest: Manifest,
) -> Literal["absent", "verified", "invalid"]:
    """Pre-verification signature label that reflects manifest shape.

    Before the signer runs, the report can already truthfully say
    whether a signature is "absent" (None) or "invalid" (present but
    we will not verify it because schema / chain checks failed before
    we got there). The "verified" status is only assigned by the
    signer-acceptance branch.
    """
    if manifest.signature is None:
        return "absent"
    return "invalid"


def _resolve_signer(identity: str | None) -> Signer:
    """Pick a signer to verify with, given the manifest's identity field.

    Currently ships only ``HashChainSigner``. Future enterprise signers
    register here. ``None`` falls back to the hash-chain signer (an
    unsigned manifest implicitly came from a hash-chain producer).
    """
    if identity in (None, HASH_CHAIN_SIGNER_IDENTITY):
        return HashChainSigner()
    raise ManifestVerificationError(
        f"manifest claims signer identity {identity!r}; no signer "
        f"plugged in for that identity. Install the corresponding extra "
        f"or check the manifest source."
    )


def verify_manifest(
    manifest: Manifest,
    *,
    expected_content_hash: bytes | None = None,
    signer: Signer | None = None,
) -> ChainReport:
    """Walk the hash chain and produce a structured verdict.

    Args
    ----
    manifest : Manifest
        Parsed manifest. Use :func:`load_manifest` to parse from a
        path / bytes.
    expected_content_hash : bytes | None
        The hash an out-of-band source (e.g. a benchmark-record
        registry) claims this manifest should produce. ``None`` skips
        the comparison; in that mode verification only confirms the
        manifest parses and the signer is happy.
    signer : Signer | None
        Override the auto-resolved signer (the OSS verifier picks
        :class:`HashChainSigner` by default). Tests use the override
        to inject signer doubles.

    Returns
    -------
    ChainReport
        ``ok=True`` iff schema parses, content hash matches the
        expectation (when supplied), and the signer accepts.
    """
    if manifest.schema_version != SCHEMA_VERSION_V0:
        return ChainReport(
            ok=False,
            reason=(
                f"unsupported schema_version {manifest.schema_version}; "
                f"this verifier only understands v{SCHEMA_VERSION_V0}."
            ),
            schema_version=manifest.schema_version,
            recomputed_content_hash_hex="",
            expected_content_hash_hex=(
                expected_content_hash.hex() if expected_content_hash else None
            ),
            signer_identity=manifest.signer_identity,
            signature_status=_signature_status_for_report(manifest),
        )

    recomputed = manifest.content_hash()
    recomputed_hex = recomputed.hex()
    expected_hex = expected_content_hash.hex() if expected_content_hash else None

    if expected_content_hash is not None and recomputed != expected_content_hash:
        return ChainReport(
            ok=False,
            reason=(
                "recomputed content hash differs from the expected hash — "
                "the manifest has been tampered with relative to the "
                "registered chain entry."
            ),
            schema_version=manifest.schema_version,
            recomputed_content_hash_hex=recomputed_hex,
            expected_content_hash_hex=expected_hex,
            signer_identity=manifest.signer_identity,
            signature_status=_signature_status_for_report(manifest),
        )

    try:
        resolved_signer = (
            signer if signer is not None else _resolve_signer(manifest.signer_identity)
        )
    except ManifestError as exc:
        # Unknown signer identity is a manifest-shape problem, not a
        # crash: surface it as a structured non-OK ChainReport so the
        # CLI returns exit 40 and downstream consumers see the same
        # contract for every verification failure.
        return ChainReport(
            ok=False,
            reason=str(exc),
            schema_version=manifest.schema_version,
            recomputed_content_hash_hex=recomputed_hex,
            expected_content_hash_hex=expected_hex,
            signer_identity=manifest.signer_identity,
            signature_status=_signature_status_for_report(manifest),
        )
    signature_status: Literal["absent", "verified", "invalid"]
    if manifest.signature is None:
        signature_status = "absent"
        signature_ok = resolved_signer.verify(recomputed, None)
    else:
        signature_ok = resolved_signer.verify(recomputed, manifest.signature)
        signature_status = "verified" if signature_ok else "invalid"

    if not signature_ok:
        return ChainReport(
            ok=False,
            reason=(
                f"signer {resolved_signer.identity!r} rejected the "
                f"signature / content hash."
            ),
            schema_version=manifest.schema_version,
            recomputed_content_hash_hex=recomputed_hex,
            expected_content_hash_hex=expected_hex,
            signer_identity=manifest.signer_identity,
            signature_status=signature_status,
        )

    return ChainReport(
        ok=True,
        reason="schema v0; content hash recomputed; signer accepted.",
        schema_version=manifest.schema_version,
        recomputed_content_hash_hex=recomputed_hex,
        expected_content_hash_hex=expected_hex,
        signer_identity=manifest.signer_identity,
        signature_status=signature_status,
    )


def load_manifest(path: Path) -> Manifest:
    """Read and parse a manifest JSON file from disk.

    Raises
    ------
    ManifestVerificationError
        If the file does not exist or fails to parse against the v0
        schema. Routed through this exception rather than Pydantic's
        own so callers (CLI included) get a single failure mode.
    """
    if not path.exists():
        raise ManifestVerificationError(f"manifest file not found: {path}")
    try:
        return Manifest.model_validate_json(path.read_text())
    except ValidationError as exc:
        raise ManifestVerificationError(
            f"manifest {path} failed schema v0 validation: {exc}"
        ) from exc


def verify_cli(
    manifest_path: Path,
    expected_hash_hex: str | None,
) -> int:
    """The ``jaxility verify`` entry point. Emits JSON to stdout.

    Returns the per-PATTERNS §8.3 exit code (``0`` ok / ``40`` manifest
    error / ``2`` invalid args). Used by the CLI dispatcher.
    """
    try:
        manifest = load_manifest(manifest_path)
    except ManifestVerificationError as exc:
        # Emit a structured failure to stdout (PATTERNS §8.2 — JSON on
        # stdout, human-readable on stderr).
        print(json.dumps({"ok": False, "reason": str(exc)}), file=sys.stdout)
        print(str(exc), file=sys.stderr)
        return EXIT_MANIFEST_ERROR

    expected_bytes: bytes | None
    if expected_hash_hex is None:
        expected_bytes = None
    else:
        try:
            expected_bytes = bytes.fromhex(expected_hash_hex)
        except ValueError:
            msg = f"--expected-hash must be a hex string; got {expected_hash_hex!r}"
            print(json.dumps({"ok": False, "reason": msg}), file=sys.stdout)
            print(msg, file=sys.stderr)
            return EXIT_INVALID_ARGS

    report = verify_manifest(manifest, expected_content_hash=expected_bytes)
    print(report.model_dump_json(), file=sys.stdout)
    return EXIT_OK if report.ok else EXIT_MANIFEST_ERROR
