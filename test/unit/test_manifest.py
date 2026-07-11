# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the manifest schema, signer, and verifier (T-012).

T-012 acceptance criteria:

1. Round-trip serialisation is byte-identical.
2. A tampered field is detected by verify.
3. The hash chain is correct.

Plus schema-discipline (extra="forbid", frozen=True), signer
contract (HashChainSigner verify rules), and the CLI's structured
JSON output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxility.cli import main as cli_main
from jaxility.errors import JaxilityError, ManifestError
from jaxility.manifest import (
    HASH_CHAIN_SIGNER_IDENTITY,
    SCHEMA_VERSION_V0,
    ChainReport,
    HashChainSigner,
    Manifest,
    ManifestVerificationError,
    Signer,
    canonical_dumps,
    load_manifest,
    verify_manifest,
)
from jaxility.targets import MOCK_CORTEX_A


def _make_manifest(**overrides: object) -> Manifest:
    """Construct a mock manifest with mostly-default values."""
    defaults: dict[str, object] = {
        "schema_version": SCHEMA_VERSION_V0,
        "source_attestation_handle": bytes.fromhex("aa" * 32),
        "toolchain_versions": {"mock-cortex-a-mock-toolchain": "0.0.0-mock"},
        "target_profile_hash": MOCK_CORTEX_A.hash,
        "artifact_content_hash": bytes.fromhex("bb" * 32),
        "build_timestamp_utc": 1_716_915_600_000_000,  # microseconds since epoch
        "signer_identity": HASH_CHAIN_SIGNER_IDENTITY,
        "signature": None,
    }
    defaults.update(overrides)
    return Manifest(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance 1: round-trip is byte-identical.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manifest_round_trip_is_byte_identical() -> None:
    """``Manifest`` → JSON → ``Manifest`` → JSON is byte-identical.

    Property: the canonical encoding is the source of truth; two
    same-content manifests produce the same canonical bytes.
    """
    m = _make_manifest()
    first = canonical_dumps(m)
    revived = Manifest.model_validate_json(m.model_dump_json())
    second = canonical_dumps(revived)
    assert first == second


@pytest.mark.unit
def test_two_equal_manifests_share_content_hash() -> None:
    """Same field values → same content hash."""
    a = _make_manifest()
    b = _make_manifest()
    assert a.content_hash() == b.content_hash()


# ---------------------------------------------------------------------------
# Acceptance 2: tampered field is detected by verify.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tampered_artifact_hash_detected_by_verify_with_expected_hash() -> None:
    """Changing ``artifact_content_hash`` flips the recomputed hash; verify fails."""
    original = _make_manifest()
    original_hash = original.content_hash()

    tampered = _make_manifest(artifact_content_hash=bytes.fromhex("cc" * 32))
    report = verify_manifest(tampered, expected_content_hash=original_hash)

    assert report.ok is False
    assert "tampered" in report.reason.lower()
    assert report.recomputed_content_hash_hex != original_hash.hex()


@pytest.mark.unit
def test_tampered_source_handle_detected_by_verify() -> None:
    """Changing ``source_attestation_handle`` is detected the same way."""
    original = _make_manifest()
    original_hash = original.content_hash()

    tampered = _make_manifest(source_attestation_handle=bytes.fromhex("99" * 32))
    report = verify_manifest(tampered, expected_content_hash=original_hash)
    assert report.ok is False


@pytest.mark.unit
def test_tampered_toolchain_versions_detected_by_verify() -> None:
    """Changing the toolchain dict flips the hash; verify fails."""
    original = _make_manifest()
    original_hash = original.content_hash()

    tampered = _make_manifest(
        toolchain_versions={"mock-cortex-a-mock-toolchain": "9.9.9-mock"}
    )
    report = verify_manifest(tampered, expected_content_hash=original_hash)
    assert report.ok is False


@pytest.mark.unit
def test_tampered_target_profile_hash_detected_by_verify() -> None:
    """Swapping the target profile hash flips the content hash."""
    original = _make_manifest()
    original_hash = original.content_hash()

    tampered = _make_manifest(target_profile_hash=bytes.fromhex("11" * 32))
    report = verify_manifest(tampered, expected_content_hash=original_hash)
    assert report.ok is False


# ---------------------------------------------------------------------------
# Acceptance 3: the hash chain is correct (timestamp excluded; signer happy).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_timestamp_does_not_enter_content_hash() -> None:
    """Invariant 5: changing ``build_timestamp_utc`` does *not* change the hash."""
    earlier = _make_manifest(build_timestamp_utc=1_000_000_000_000_000)
    later = _make_manifest(build_timestamp_utc=2_000_000_000_000_000)
    assert earlier.content_hash() == later.content_hash()


@pytest.mark.unit
def test_signer_identity_does_not_enter_content_hash() -> None:
    """The signer's identity is not part of the chain (the signature is)."""
    a = _make_manifest(signer_identity="hash-chain-v0")
    b = _make_manifest(signer_identity="enterprise-sigstore-v1")
    assert a.content_hash() == b.content_hash()


@pytest.mark.unit
def test_signature_does_not_enter_content_hash() -> None:
    """The signature signs *over* the hash; it cannot also be inside it."""
    a = _make_manifest(signature=None)
    b = _make_manifest(signature=bytes.fromhex("ff" * 64))
    assert a.content_hash() == b.content_hash()


@pytest.mark.unit
def test_verify_passes_on_unmodified_oss_manifest() -> None:
    """A freshly-built hash-chain manifest verifies cleanly."""
    m = _make_manifest()
    report = verify_manifest(m)
    assert report.ok is True
    assert report.schema_version == SCHEMA_VERSION_V0
    assert report.signer_identity == HASH_CHAIN_SIGNER_IDENTITY
    assert report.signature_status == "absent"
    assert report.recomputed_content_hash_hex == m.content_hash().hex()


@pytest.mark.unit
def test_verify_with_matching_expected_hash_passes() -> None:
    """Supplying the correct expected hash leaves verification ok."""
    m = _make_manifest()
    report = verify_manifest(m, expected_content_hash=m.content_hash())
    assert report.ok is True
    assert report.expected_content_hash_hex == m.content_hash().hex()


# ---------------------------------------------------------------------------
# Signer contract.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manifest_verification_error_is_in_hierarchy() -> None:
    """``ManifestVerificationError`` routes through PATTERNS §6.1."""
    assert issubclass(ManifestVerificationError, ManifestError)
    assert issubclass(ManifestVerificationError, JaxilityError)


@pytest.mark.unit
def test_unknown_signer_identity_returns_chain_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """B3: an unknown signer identity is a structured failure, not a crash."""
    m = _make_manifest(signer_identity="enterprise-sigstore-v1")
    report = verify_manifest(m)
    assert report.ok is False
    assert "signer identity" in report.reason
    assert report.signer_identity == "enterprise-sigstore-v1"


@pytest.mark.unit
def test_cli_verify_returns_40_on_unknown_signer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3: the CLI returns exit 40 instead of crashing on unknown signer."""
    m = _make_manifest(signer_identity="enterprise-sigstore-v1")
    path = tmp_path / "manifest.json"
    path.write_text(m.model_dump_json())
    exit_code = cli_main(["verify", str(path)])
    captured = capsys.readouterr()
    assert exit_code == 40
    report = json.loads(captured.out)
    assert report["ok"] is False


@pytest.mark.unit
def test_canonical_dumps_of_manifest_matches_content_payload() -> None:
    """N1: ``canonical_dumps(manifest)`` and ``manifest.content_payload`` agree.

    Pre-review, the ``HexBytes`` ``PlainSerializer`` ran in
    ``model_dump(mode="python")`` too, so the canonical encoding of
    the full Manifest emitted bare hex strings while
    ``content_payload`` emitted the ``{"$b16": "<hex>"}`` wrapper —
    a third-party verifier implementing the documented canonical form
    computed a different hash than the in-process verifier. The
    ``when_used="json"`` clause fixes this.
    """
    m = _make_manifest()
    chain_fields = {
        "schema_version",
        "source_attestation_handle",
        "toolchain_versions",
        "target_profile_hash",
        "artifact_content_hash",
    }
    # The dict-of-chain-fields encoded canonically must equal content_payload.
    expected_payload = canonical_dumps(
        {field: getattr(m, field) for field in chain_fields}
    )
    assert expected_payload == m.content_payload()


@pytest.mark.unit
def test_hex_bytes_rejects_whitespace() -> None:
    """N4: ``_bytes_from_hex`` rejects whitespace in hex strings."""
    payload = _make_manifest().model_dump(mode="json")
    payload["source_attestation_handle"] = "aa bb" + "aa" * 30
    with pytest.raises(ValueError, match="bare hex"):
        Manifest.model_validate(payload)


@pytest.mark.unit
def test_hash_chain_signer_identity_is_instance_attribute() -> None:
    """Nit: ``identity`` is bound per-instance for typing-tool sanity."""
    signer = HashChainSigner()
    assert "identity" in signer.__dict__
    assert signer.identity == HASH_CHAIN_SIGNER_IDENTITY


@pytest.mark.unit
def test_hash_chain_signer_satisfies_protocol() -> None:
    """``HashChainSigner`` is a ``Signer`` at runtime (Protocol check)."""
    assert isinstance(HashChainSigner(), Signer)
    assert HashChainSigner().identity == HASH_CHAIN_SIGNER_IDENTITY


@pytest.mark.unit
def test_hash_chain_signer_sign_returns_none_for_well_formed_digest() -> None:
    """``HashChainSigner.sign`` returns ``None``; OSS manifests carry no signature."""
    signer = HashChainSigner()
    assert signer.sign(bytes(32)) is None


@pytest.mark.unit
def test_hash_chain_signer_sign_rejects_wrong_length_digest() -> None:
    """A non-32-byte digest raises loudly (PATTERNS: loud failure)."""
    with pytest.raises(ValueError, match="32-byte"):
        HashChainSigner().sign(bytes(16))


@pytest.mark.unit
def test_hash_chain_signer_verify_rejects_unexpected_signature() -> None:
    """An OSS-mode manifest cannot carry a signature; reject if it does."""
    signer = HashChainSigner()
    assert signer.verify(bytes(32), None) is True
    assert signer.verify(bytes(32), b"fake") is False


# ---------------------------------------------------------------------------
# Schema discipline.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manifest_rejects_unknown_field() -> None:
    """``extra="forbid"`` — an unknown field fails validation."""
    payload = _make_manifest().model_dump(mode="json")
    payload["unknown_field"] = "x"
    with pytest.raises(ValueError):
        Manifest.model_validate(payload)


@pytest.mark.unit
def test_manifest_is_immutable() -> None:
    """``frozen=True`` — direct field assignment raises."""
    m = _make_manifest()
    with pytest.raises(ValueError):
        m.build_timestamp_utc = 0  # type: ignore[misc]


@pytest.mark.unit
def test_unknown_schema_version_rejected_by_verify() -> None:
    """A future schema version is reported clearly, not silently accepted."""
    m = _make_manifest(schema_version=99)
    report = verify_manifest(m)
    assert report.ok is False
    assert "schema_version" in report.reason


# ---------------------------------------------------------------------------
# load_manifest + CLI integration.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_manifest_round_trip(tmp_path: Path) -> None:
    """Writing and reading a manifest produces an identical content hash."""
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(m.model_dump_json())
    loaded = load_manifest(path)
    assert loaded.content_hash() == m.content_hash()


@pytest.mark.unit
def test_load_manifest_raises_on_missing_path(tmp_path: Path) -> None:
    """Missing file raises ``ManifestVerificationError``."""
    with pytest.raises(ManifestVerificationError, match="not found"):
        load_manifest(tmp_path / "nope.json")


@pytest.mark.unit
def test_load_manifest_raises_on_invalid_json(tmp_path: Path) -> None:
    """Schema-invalid JSON raises ``ManifestVerificationError``."""
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": "not-an-int"}')
    with pytest.raises(ManifestVerificationError, match="schema v0"):
        load_manifest(path)


@pytest.mark.unit
def test_cli_verify_emits_structured_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``jaxility verify`` writes a ChainReport JSON to stdout (PATTERNS §8.2)."""
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(m.model_dump_json())

    exit_code = cli_main(["verify", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    report = json.loads(captured.out)
    # ChainReport fields are present and consistent with the manifest.
    expected_fields = set(ChainReport.model_fields)
    assert set(report) == expected_fields
    assert report["ok"] is True
    assert report["schema_version"] == SCHEMA_VERSION_V0
    assert report["recomputed_content_hash_hex"] == m.content_hash().hex()


@pytest.mark.unit
def test_cli_verify_returns_40_on_tampered_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI reports exit code 40 on a hash mismatch (PATTERNS §8.3)."""
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(m.model_dump_json())

    wrong_hash = "ff" * 32
    exit_code = cli_main(["verify", str(path), "--expected-hash", wrong_hash])
    captured = capsys.readouterr()

    assert exit_code == 40
    report = json.loads(captured.out)
    assert report["ok"] is False


@pytest.mark.unit
def test_cli_verify_returns_2_on_bad_hex(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid hex on ``--expected-hash`` is an argument error (exit 2)."""
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(m.model_dump_json())

    exit_code = cli_main(["verify", str(path), "--expected-hash", "not-hex"])
    capsys.readouterr()
    assert exit_code == 2
