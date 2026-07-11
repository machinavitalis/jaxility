# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""End-to-end mock lowering pipeline (T-015).

:func:`mock_lower` exercises every contract — coverage check
(T-013), target dispatch (T-011), manifest construction (T-012),
artifact production (T-014) — without invoking any real toolchain.
The returned :class:`MockArtifactBundle` carries the artifact, the
manifest, and a :meth:`simulate` method that re-runs the source's
trajectory through the (trivially identical) mock implementation, so
the equivalence check (T-010) passes by construction.

The bundle is the integration point tests use. T-016 plugs a
real :class:`jaxterity.robot.Robot` into the same surface; the
pipeline does not care whether its source is a
:class:`~jaxility.testing.sources.MockSource` or a real Robot.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..errors import CoverageError
from ..lowering import coverage as coverage_module
from ..manifest import (
    SCHEMA_VERSION_V0,
    Artifact,
    BuildLogEntry,
    Manifest,
    canonical_dumps,
)
from ..targets import Target
from .equivalence import Trajectory
from .sources import Source
from .tolerances import quantities_for

MOCK_PIPELINE_VERSION = "v0"
"""Mock-pipeline identifier that travels in the manifest and the payload."""


class MockArtifactBundle(BaseModel):
    """Wrap-up of one ``mock_lower`` call (T-015)."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    source_name: str
    target_name: str
    dtype: Literal["float32", "float64"]
    n_steps: int
    artifact: Artifact
    manifest: Manifest

    # Held by reference so the bundle can re-run the source's trajectory.
    # Pydantic ``PrivateAttr`` is the supported way to attach non-schema
    # state to a frozen ``BaseModel`` — it sidesteps both the
    # canonical-JSON encoding (private attrs are not in ``model_dump``)
    # and the freeze (Pydantic gates the write through its own
    # initialisation machinery, so the object.__setattr__ trick the
    # earlier bundle used is no longer needed and external callers
    # cannot rebind the source post-construction).
    _source: Source = PrivateAttr()
    _source_locked: bool = PrivateAttr(default=False)

    def __init__(self, /, *, source: Source, **data: object) -> None:
        super().__init__(**data)  # type: ignore[arg-type]
        self._source = source
        self._source_locked = True

    def __setattr__(self, name: str, value: object) -> None:
        # Pydantic ``PrivateAttr`` is rebindable on a frozen model by
        # default; we tighten that here so external callers cannot
        # silently decouple the bundle's behaviour from its manifest
        # by reassigning ``_source``.
        if name == "_source" and getattr(self, "_source_locked", False):
            raise AttributeError(
                "MockArtifactBundle._source is locked after construction; "
                "build a new bundle if you need a different source."
            )
        super().__setattr__(name, value)

    @property
    def source(self) -> Source:
        return self._source

    def simulate(self, n_steps: int | None = None) -> Trajectory:
        """Re-run the source's trajectory through the mock pipeline.

        The mock layer adds no transformation, so the
        candidate trajectory is byte-identical to the source's. T-027
        replaces this with the host-build path and real codegen.
        """
        return self._source.simulate(n_steps if n_steps is not None else self.n_steps)


def _coverage_assert_for_build(target_family: str, dtype: str) -> None:
    """Run the coverage gate for a single mock build (invariant 7).

    Two checks fire, both unconditional:

    1. The tolerance table must have *at least one* row for
       ``(target_family, dtype)``; an empty result means the build is
       running on an unsupported precision / family combination and
       the gate refuses it loudly. (Pre-review, this was a silent
       early-return — see B1 in the initial review.)
    2. The coverage table must declare the canonical smooth op
       ``add`` as supported on the same ``(family, dtype)``. This is
       the load-bearing parity check between the tolerance contract
       and the coverage contract — if the two ever drift, the build
       fails before producing an artifact.

    The function does *not* iterate per-quantity: the mock pipeline
    does not emit per-op codegen, so a per-quantity loop would do the
    same work N times. The real lowering will call
    :func:`jaxility.lowering.coverage.assert_supported` once per op
    it actually emits.
    """
    quantities = quantities_for(target_family, dtype)
    if not quantities:
        raise CoverageError(
            (
                f"the tolerance table has no rows for "
                f"target_family={target_family!r}, dtype={dtype!r}; "
                "the build cannot proceed because the equivalence "
                "check has no documented bound."
            ),
            op="<mock build>",
            dtype=dtype,
            target_family=target_family,
            suggestion=(
                "add rows to jaxility.testing.tolerances.TOLERANCE_TABLE "
                "and document them in test/EQUIVALENCE.md before lowering "
                "for this (target_family, dtype)."
            ),
        )
    coverage_module.assert_supported(op="add", dtype=dtype, target_family=target_family)


def _build_payload(
    *,
    source: Source,
    target: Target,
    dtype: str,
    n_steps: int,
) -> bytes:
    """Canonical-JSON payload that fingerprints the mock build.

    The payload is the bytes-identity of the artifact; per invariant
    5, equal source + target + toolchain + dtype + n_steps must
    produce the byte-identical payload. The mock pipeline therefore
    only hashes the inputs that *change* the build.
    """
    return canonical_dumps(
        {
            "schema_version": SCHEMA_VERSION_V0,
            "mock_pipeline_version": MOCK_PIPELINE_VERSION,
            "source_name": source.name,
            "source_attestation_handle": source.attestation_handle,
            "target_profile_hash": target.hash,
            "toolchain_name": target.toolchain.name,
            "toolchain_version": target.toolchain.version,
            "dtype": dtype,
            "n_steps": n_steps,
        }
    )


def mock_lower(
    source: Source,
    target: Target,
    *,
    dtype: Literal["float32", "float64"] = "float64",
    n_steps: int = 50,
    build_timestamp_utc: int | None = None,
) -> MockArtifactBundle:
    """Run the mock lowering pipeline end-to-end.

    Steps (each maps to a build-log entry):

    1. **plan** — record the inputs and confirm the
       ``(target_family, dtype)`` is covered (T-013).
    2. **lower** — assemble the canonical payload (the "binary"
       analogue at v0).
    3. **package** — build the manifest, the artifact, and the
       bundle.

    Args
    ----
    source : Source
        Anything satisfying the :class:`~jaxility.testing.sources.Source`
        Protocol. :class:`~jaxility.testing.sources.MockSource` for
        tests; a Jaxterity adapter for T-016.
    target : Target
        Mock targets only. Real targets land later.
    dtype : "float32" | "float64"
        Precision the mock pipeline reports in the manifest /
        coverage check. ``float64`` by default; mirrors the
        tolerance contract.
    n_steps : int
        Trajectory length the bundle will simulate. Default 50.
    build_timestamp_utc : int | None
        Override for the manifest's wall-clock build timestamp
        (microseconds since epoch). Defaults to the current time;
        callers pin it from tests to keep test assertions stable.

    Returns
    -------
    MockArtifactBundle
        Artifact + Manifest + ``simulate`` callable.
    """
    build_log: list[BuildLogEntry] = []
    start_offset_us = 0

    build_log.append(
        BuildLogEntry(
            offset_us=start_offset_us,
            stage="plan",
            level="info",
            message=(
                f"mock_lower {source.name!r} on target {target.name!r} "
                f"at dtype={dtype} for {n_steps} steps"
            ),
            detail={
                "source_handle_hex": source.attestation_handle.hex(),
                "target_profile_hash_hex": target.hash.hex(),
            },
        )
    )

    _coverage_assert_for_build(target.family, dtype)

    payload = _build_payload(source=source, target=target, dtype=dtype, n_steps=n_steps)
    build_log.append(
        BuildLogEntry(
            offset_us=1,
            stage="lower",
            level="info",
            message="canonical payload assembled",
            detail={"payload_bytes": str(len(payload))},
        )
    )

    if build_timestamp_utc is None:
        build_timestamp_utc = int(time.time() * 1_000_000)

    # Manifest first (artifact_content_hash depends on the payload's BLAKE3,
    # which is identical to Artifact.content_hash once the artifact is built).
    import blake3

    payload_hash = blake3.blake3(payload).digest()

    manifest = Manifest(
        schema_version=SCHEMA_VERSION_V0,
        source_attestation_handle=source.attestation_handle,
        toolchain_versions={
            target.toolchain.name: target.toolchain.version,
            "jaxility-mock-pipeline": MOCK_PIPELINE_VERSION,
        },
        target_profile_hash=target.hash,
        artifact_content_hash=payload_hash,
        build_timestamp_utc=build_timestamp_utc,
    )

    artifact = Artifact.build(
        payload=payload,
        source_manifest_hash=manifest.content_hash(),
        target_profile_hash=target.hash,
        build_log=tuple(build_log),
    )

    build_log.append(
        BuildLogEntry(
            offset_us=2,
            stage="package",
            level="info",
            message="manifest + artifact constructed",
            detail={
                "manifest_content_hash_hex": manifest.content_hash().hex(),
                "artifact_content_hash_hex": artifact.content_hash.hex(),
            },
        )
    )

    return MockArtifactBundle(
        source=source,
        source_name=source.name,
        target_name=target.name,
        dtype=dtype,
        n_steps=n_steps,
        artifact=artifact,
        manifest=manifest,
    )
