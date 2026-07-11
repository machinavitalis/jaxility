# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for ``Artifact`` + content-addressed cache (T-014).

T-014 acceptance criteria:

1. Same source + same target + same toolchain → same content hash
   bit-exactly.
2. A different timestamp does not change the hash.

Plus content_hash validation at construction, write-once cache
semantics, round-trip persistence, and tamper detection on load.
"""

from __future__ import annotations

from pathlib import Path

import blake3
import pytest

from jaxility import Artifact as TopLevelArtifact
from jaxility.errors import ArtifactError, JaxilityError
from jaxility.manifest import (
    ARTIFACT_SCHEMA_V0,
    SCHEMA_VERSION_V0,
    Artifact,
    ArtifactCache,
    BuildLogEntry,
    Manifest,
    default_cache_root,
)
from jaxility.targets import MOCK_CORTEX_A


def _make_manifest() -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION_V0,
        source_attestation_handle=bytes.fromhex("aa" * 32),
        toolchain_versions={"mock-cortex-a-mock-toolchain": "0.0.0-mock"},
        target_profile_hash=MOCK_CORTEX_A.hash,
        artifact_content_hash=blake3.blake3(b"mock-payload-bytes").digest(),
        build_timestamp_utc=1_716_915_600_000_000,
    )


def _make_artifact(payload: bytes = b"mock-payload-bytes") -> Artifact:
    manifest = _make_manifest()
    return Artifact.build(
        payload=payload,
        source_manifest_hash=manifest.content_hash(),
        target_profile_hash=MOCK_CORTEX_A.hash,
    )


# ---------------------------------------------------------------------------
# Acceptance 1: same source + same target → identical content hash.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_same_payload_produces_identical_content_hash() -> None:
    """Two artifacts with the same payload share a content hash bit-exactly."""
    a = _make_artifact()
    b = _make_artifact()
    assert a.content_hash == b.content_hash
    assert a.content_hash == blake3.blake3(b"mock-payload-bytes").digest()


@pytest.mark.unit
def test_different_payload_produces_different_content_hash() -> None:
    """A payload change is reflected in the content hash."""
    a = _make_artifact(b"payload-a")
    b = _make_artifact(b"payload-b")
    assert a.content_hash != b.content_hash


# ---------------------------------------------------------------------------
# Acceptance 2: timestamp / log changes do not change the content hash.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_log_does_not_change_content_hash() -> None:
    """Logs are metadata; the artifact's identity is its payload (invariant 5)."""
    base = _make_artifact()

    log_a = (
        BuildLogEntry(
            offset_us=0,
            stage="plan",
            level="info",
            message="started at 1000",
        ),
    )
    log_b = (
        BuildLogEntry(
            offset_us=42,  # different offset
            stage="plan",
            level="info",
            message="started at 9999",
        ),
        BuildLogEntry(
            offset_us=100,
            stage="lower",
            level="info",
            message="lowering done",
        ),
    )

    with_log_a = base.model_copy(update={"build_log": log_a})
    with_log_b = base.model_copy(update={"build_log": log_b})

    assert with_log_a.content_hash == with_log_b.content_hash == base.content_hash


# ---------------------------------------------------------------------------
# Construction guards.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_construction_rejects_mismatched_content_hash() -> None:
    """Passing a content_hash that disagrees with payload raises."""
    with pytest.raises(ArtifactError, match="does not match"):
        Artifact(
            payload=b"some-bytes",
            content_hash=bytes(32),  # zeros — wrong
            source_manifest_hash=bytes(32),
            target_profile_hash=MOCK_CORTEX_A.hash,
        )


@pytest.mark.unit
def test_artifact_is_frozen() -> None:
    """``frozen=True`` — direct field assignment raises."""
    a = _make_artifact()
    with pytest.raises(ValueError):
        a.payload = b"other"  # type: ignore[misc]


@pytest.mark.unit
def test_artifact_rejects_unknown_field() -> None:
    """``extra="forbid"`` — surprise fields fail validation."""
    payload = b"payload"
    base = _make_artifact(payload).model_dump(mode="json")
    base["unknown"] = "x"
    with pytest.raises(ValueError):
        Artifact.model_validate(base)


@pytest.mark.unit
def test_artifact_error_is_subclass_of_jaxility_error() -> None:
    """``ArtifactError`` is part of the PATTERNS §6.1 hierarchy."""
    assert issubclass(ArtifactError, JaxilityError)


@pytest.mark.unit
def test_top_level_artifact_re_export_is_the_same_class() -> None:
    """``jaxility.Artifact`` is the canonical re-export of the manifest type."""
    assert TopLevelArtifact is Artifact


@pytest.mark.unit
def test_artifact_schema_version_is_v0() -> None:
    """Artifacts default to schema v0."""
    assert _make_artifact().schema_version == ARTIFACT_SCHEMA_V0


# ---------------------------------------------------------------------------
# Cache contract — write-once, content-addressed, round-trip.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cache_round_trip_preserves_content_hash(tmp_path: Path) -> None:
    """Store + load returns an Artifact with the same content hash."""
    cache = ArtifactCache(root=tmp_path / "artifacts")
    a = _make_artifact()
    target_dir = cache.store(a)
    assert target_dir.exists()
    assert (target_dir / "payload.bin").read_bytes() == a.payload

    loaded = cache.load(a.content_hash)
    assert loaded.content_hash == a.content_hash
    assert loaded.payload == a.payload
    assert loaded.source_manifest_hash == a.source_manifest_hash
    assert loaded.target_profile_hash == a.target_profile_hash


@pytest.mark.unit
def test_cache_store_is_write_once(tmp_path: Path) -> None:
    """A second store at the same hash raises (invariant 5; loud failure)."""
    cache = ArtifactCache(root=tmp_path / "artifacts")
    a = _make_artifact()
    cache.store(a)
    with pytest.raises(ArtifactError, match="write-once"):
        cache.store(a)


@pytest.mark.unit
def test_cache_load_missing_raises(tmp_path: Path) -> None:
    """Loading a never-stored hash raises a structured error."""
    cache = ArtifactCache(root=tmp_path / "artifacts")
    with pytest.raises(ArtifactError, match="no cache entry"):
        cache.load(bytes(32))


@pytest.mark.unit
def test_cache_has_returns_false_when_missing(tmp_path: Path) -> None:
    """``ArtifactCache.has`` is a fast existence check."""
    cache = ArtifactCache(root=tmp_path / "artifacts")
    a = _make_artifact()
    assert cache.has(a.content_hash) is False
    cache.store(a)
    assert cache.has(a.content_hash) is True


@pytest.mark.unit
def test_cache_detects_on_disk_payload_tampering(tmp_path: Path) -> None:
    """Mutating payload.bin without updating metadata raises on load."""
    cache = ArtifactCache(root=tmp_path / "artifacts")
    a = _make_artifact()
    target_dir = cache.store(a)
    (target_dir / "payload.bin").write_bytes(b"tampered")

    with pytest.raises(ArtifactError, match="mutated"):
        cache.load(a.content_hash)


@pytest.mark.unit
def test_cache_load_rejects_renamed_cache_directory(tmp_path: Path) -> None:
    """B2: ``load(h)`` rejects entries whose stored content_hash != requested.

    If a cache directory is renamed (or forged) so the directory name no
    longer matches its metadata, load must refuse rather than silently
    returning an artifact whose hash differs from what was asked for.
    """
    cache = ArtifactCache(root=tmp_path / "artifacts")
    a = _make_artifact()
    src_dir = cache.store(a)

    # Rename the directory to a different hex string. The metadata still
    # carries the original content_hash; the on-disk layout simulates
    # a renamed-or-forged cache entry.
    forged_hash = bytes(32)
    forged_dir = cache.path_for(forged_hash)
    src_dir.rename(forged_dir)

    with pytest.raises(ArtifactError, match="renamed or forged"):
        cache.load(forged_hash)


@pytest.mark.unit
def test_cache_store_handles_concurrent_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N3: a second ``store`` at the same hash raises ``ArtifactError``,
    not the bare :class:`FileExistsError`."""
    cache = ArtifactCache(root=tmp_path / "artifacts")
    a = _make_artifact()

    # Simulate a TOCTOU race: the directory is created between
    # path_for and mkdir by a hypothetical concurrent writer. The
    # implementation must catch FileExistsError and re-raise as
    # ArtifactError.
    real_exists = type(cache.path_for(a.content_hash)).exists

    def lying_exists(self: Path) -> bool:
        # Pretend the directory doesn't exist even when it does, forcing
        # the mkdir to race.
        if str(self).endswith(a.content_hash.hex()):
            return False
        return real_exists(self)

    cache.path_for(a.content_hash).mkdir(parents=True)
    monkeypatch.setattr(Path, "exists", lying_exists)

    with pytest.raises(ArtifactError, match="write-once"):
        cache.store(a)


@pytest.mark.unit
def test_default_cache_root_respects_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``JAXILITY_CACHE_ROOT`` overrides the default ``~/.cache/jaxility``."""
    monkeypatch.setenv("JAXILITY_CACHE_ROOT", str(tmp_path / "custom"))
    root = default_cache_root()
    assert tmp_path / "custom" / "artifacts" == root


@pytest.mark.unit
def test_default_cache_root_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without override, the cache lives under ``~/.cache/jaxility/artifacts``."""
    monkeypatch.delenv("JAXILITY_CACHE_ROOT", raising=False)
    root = default_cache_root()
    assert root.name == "artifacts"
    assert root.parent.name == "jaxility"
    assert root.parent.parent.name == ".cache"
