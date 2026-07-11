# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""The :class:`Artifact` type and the content-addressed local cache.

An :class:`Artifact` is the output of a build. It carries:

* the payload bytes (the deployed binary at v0; a path to a generated
  C project will land alongside the host build path, T-026),
* the BLAKE3 content hash of the payload (verified at construction),
* the source manifest's :meth:`Manifest.content_hash`,
* the target profile hash (same value as ``Manifest.target_profile_hash``,
  exposed here so an artifact can be linked back to its target without
  re-reading the manifest),
* a structured :class:`BuildLogEntry` log (PATTERNS §9.3).

Artifacts are *content-addressed*. Same source + same target + same
toolchain → byte-identical payload → identical content hash, regardless
of wall-clock time (invariant 5). The local cache lives at
``~/.cache/jaxility/artifacts/<content_hash_hex>/`` and is write-once:
once a directory exists, any further attempt to store at the same hash
raises rather than overwriting (no silent mutation; invariant 5).

The cache is local-only for now. The fleet-update / hosted-cache
extensions are an enterprise concern (ADR-008).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import blake3
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import ArtifactError
from .models import HexBytes

ARTIFACT_SCHEMA_V0 = 0
"""The current artifact schema version."""


class BuildLogEntry(BaseModel):
    """One structured line in the build log (PATTERNS §9.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    offset_us: int = Field(
        ge=0,
        description=(
            "Microseconds since the build started. *Not* a wall-clock "
            "timestamp; using a relative offset keeps two identical "
            "builds at different times producing identical logs and "
            "therefore identical artifacts (invariant 5)."
        ),
    )
    stage: Literal[
        "plan",
        "lower",
        "compile",
        "link",
        "package",
        "verify",
    ]
    level: Literal["info", "warn", "error"]
    message: str
    detail: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional key/value detail (toolchain output, file paths, "
            "exit codes). Free-form keys, string-only values for "
            "canonical-JSON stability."
        ),
    )


class Artifact(BaseModel):
    """A built artifact with verifiable provenance (T-014, ADR-005)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(
        default=ARTIFACT_SCHEMA_V0,
        ge=0,
        description="Artifact schema version; currently v0.",
    )
    payload: HexBytes = Field(
        description=(
            "Deployed-artifact bytes. v0 carries the binary inline; a "
            "path-to-generated-C-project variant lands with the host "
            "build path (T-026)."
        ),
    )
    content_hash: HexBytes = Field(
        description=(
            "BLAKE3 digest of :attr:`payload`. Verified at construction "
            "and at load; mismatch raises :class:`ArtifactError`."
        ),
    )
    source_manifest_hash: HexBytes = Field(
        description=(
            "The source :meth:`Manifest.content_hash` — the hop in the "
            "attestation chain that points from this artifact back to "
            "the manifest that produced it."
        ),
    )
    target_profile_hash: HexBytes = Field(
        description=(
            "The :attr:`jaxility.targets.Target.hash` of the target "
            "this artifact was built for. Redundant with the manifest "
            "but kept here so an artifact alone can be linked back to "
            "its target without a manifest lookup."
        ),
    )
    build_log: tuple[BuildLogEntry, ...] = Field(
        default=(),
        description=(
            "Structured build log (PATTERNS §9.3). Excluded from "
            ":attr:`content_hash` — the artifact's identity is its "
            "payload, not the log of how the payload was built."
        ),
    )

    @model_validator(mode="after")
    def _check_content_hash(self) -> Artifact:
        expected = blake3.blake3(self.payload).digest()
        if self.content_hash != expected:
            raise ArtifactError(
                "Artifact content_hash does not match BLAKE3(payload); "
                f"declared {self.content_hash.hex()!r} vs. computed "
                f"{expected.hex()!r}. Refusing to construct an artifact "
                "whose hash doesn't agree with its payload."
            )
        return self

    @classmethod
    def build(
        cls,
        *,
        payload: bytes,
        source_manifest_hash: bytes,
        target_profile_hash: bytes,
        build_log: tuple[BuildLogEntry, ...] = (),
    ) -> Artifact:
        """Construct an Artifact, computing the content hash from the payload."""
        return cls(
            payload=payload,
            content_hash=blake3.blake3(payload).digest(),
            source_manifest_hash=source_manifest_hash,
            target_profile_hash=target_profile_hash,
            build_log=build_log,
        )


def default_cache_root() -> Path:
    """Return the default artifact cache root.

    ``~/.cache/jaxility/artifacts/`` by default; overridable via the
    ``JAXILITY_CACHE_ROOT`` environment variable for CI / tests / users
    who deliberately move the cache.
    """
    override = os.environ.get("JAXILITY_CACHE_ROOT")
    if override:
        return Path(override).expanduser().resolve() / "artifacts"
    return Path("~/.cache/jaxility/artifacts").expanduser().resolve()


_PAYLOAD_NAME = "payload.bin"
_METADATA_NAME = "artifact.json"


class ArtifactCache:
    """Write-once content-addressed local cache for built artifacts.

    Layout::

        <root>/<content_hash_hex>/
            payload.bin     # the artifact's bytes
            artifact.json   # the full Artifact model (without payload duplication)

    The cache is *write-once*: a second store at an existing hash
    raises :class:`ArtifactError` rather than mutating. This matters
    for the deterministic-build invariant — a divergent rebuild must
    fail loudly, not silently overwrite the older identical-hash
    record.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = root if root is not None else default_cache_root()

    def path_for(self, content_hash: bytes) -> Path:
        """Return the directory the artifact lives at (may not yet exist)."""
        return self.root / content_hash.hex()

    def store(self, artifact: Artifact) -> Path:
        """Store an artifact. Returns the directory it landed in.

        Atomic with respect to concurrent ``store`` calls at the same
        content hash: the loser raises :class:`ArtifactError` rather
        than ``FileExistsError`` (so the loud-structured-failure
        contract holds across the TOCTOU window).

        Raises
        ------
        ArtifactError
            When a directory already exists at this artifact's hash —
            the cache is write-once.
        """
        target_dir = self.path_for(artifact.content_hash)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            target_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactError(
                f"refusing to overwrite existing cache entry at {target_dir}. "
                "The artifact cache is write-once (invariant 5); if you "
                "intentionally want to replace, delete the directory first."
            ) from exc
        (target_dir / _PAYLOAD_NAME).write_bytes(artifact.payload)
        # Store the full Artifact model in JSON form for round-trip. Uses
        # ``model_dump_json`` (not :func:`canonical_dumps`) because the
        # storage form needs to load back through Pydantic — the
        # ``HexBytes`` JSON encoding is hex strings, which the load
        # validator decodes. Canonical-JSON is only the *hash* form,
        # not the *storage* form (the two were unified before the
        # ``when_used="json"`` PlainSerializer landed and have to stay
        # separate now).
        (target_dir / _METADATA_NAME).write_text(artifact.model_dump_json())
        return target_dir

    def load(self, content_hash: bytes) -> Artifact:
        """Load a previously-stored artifact by content hash.

        Re-validates that:

        1. A cache entry exists at the requested hash.
        2. The stored artifact's ``content_hash`` equals the requested
           hash (catches renamed / forged cache directories).
        3. The on-disk payload bytes still hash to ``content_hash``
           (catches manual tampering with ``payload.bin``).

        Any mismatch raises :class:`ArtifactError`; content addressing
        is the safety property, so a silent mismatch would defeat the
        whole cache.
        """
        target_dir = self.path_for(content_hash)
        metadata_path = target_dir / _METADATA_NAME
        if not metadata_path.exists():
            raise ArtifactError(
                f"no cache entry for content hash {content_hash.hex()} "
                f"under {self.root}. Check the hash or rebuild the artifact."
            )
        artifact = Artifact.model_validate_json(metadata_path.read_text())
        if artifact.content_hash != content_hash:
            raise ArtifactError(
                f"cache entry at {target_dir} carries content_hash "
                f"{artifact.content_hash.hex()} but was requested as "
                f"{content_hash.hex()}; the cache directory has been "
                "renamed or forged."
            )
        # Cross-check the on-disk payload bytes against the stored hash;
        # catches manual tampering with payload.bin without touching the
        # metadata.
        payload_on_disk = (target_dir / _PAYLOAD_NAME).read_bytes()
        if blake3.blake3(payload_on_disk).digest() != artifact.content_hash:
            raise ArtifactError(
                f"on-disk payload at {target_dir / _PAYLOAD_NAME} no longer "
                "hashes to the recorded content_hash; the cache entry has "
                "been mutated outside ArtifactCache."
            )
        return artifact

    def has(self, content_hash: bytes) -> bool:
        """``True`` iff a cache entry for ``content_hash`` exists."""
        return (self.path_for(content_hash) / _METADATA_NAME).exists()
