# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Host-only build path (T-026).

End-to-end glue from a translated dynamics + OCP spec all the way to
a content-addressed :class:`jaxility.Artifact` carrying the
shared-library bytes acados generated and compiled. The same code path
is the basis for the cross-compilation targets (T-031 / Pi 5; T-051 /
STM32H7) — those will reuse :func:`build_for_target` and just swap the
target profile.

The host build path is deliberately small at v0:

1. Construct the :class:`AcadosOcp` from the translated dynamics + spec.
2. Construct :class:`AcadosOcpSolver` — acados generates a C project
   under ``<work>/c_generated_code/<model>/`` and compiles it.
3. Locate the resulting shared library, read its bytes, and wrap
   them in an :class:`Artifact` with a chain-linked manifest.
4. Return both the artifact and the live solver — the latter is what
   the equivalence-check tier (T-027) calls to evaluate the
   compiled controller.

The work directory is a caller-supplied path so callers in tests can
isolate. The default ``jaxility build`` CLI lands artifacts in
``~/.cache/jaxility/builds/<artifact-hash>``.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

from .errors import TargetError
from .lowering import CasadiFunction, OcpTemplateSpec, build_ocp
from .manifest import (
    SCHEMA_VERSION_V0,
    Artifact,
    BuildLogEntry,
    Manifest,
)
from .manifest.toolchain_detect import detect_toolchain_versions
from .targets import Target


@dataclass(frozen=True)
class BuildBundle:
    """End-to-end product of :func:`build_for_target`.

    Carries the artifact (for verification / cache storage), the
    manifest (for chain checks), and the live :class:`AcadosOcpSolver`
    instance the equivalence-check tier uses to evaluate the compiled
    controller without re-loading the shared library.
    """

    artifact: Artifact
    manifest: Manifest
    target: Target
    solver: Any  # acados_template.AcadosOcpSolver
    shared_library_path: Path


def _locate_shared_library(work_dir: Path, model_name: str) -> Path:
    """Find the shared library acados just emitted under ``work_dir``."""
    candidates: list[Path] = []
    for ext in (".dylib", ".so"):
        candidates.extend((work_dir / "c_generated_code").rglob(f"*{model_name}*{ext}"))
    if not candidates:
        raise TargetError(
            f"host build for model {model_name!r} did not produce a "
            f"shared library under {work_dir}/c_generated_code/. "
            "Check that AcadosOcpSolver constructed without error."
        )
    return candidates[0].resolve()


def build_for_target(
    *,
    dynamics: CasadiFunction,
    spec: OcpTemplateSpec,
    target: Target,
    source_attestation_handle: bytes,
    work_dir: Path,
    build_timestamp_utc: int | None = None,
    extra_toolchain_versions: dict[str, str] | None = None,
) -> BuildBundle:
    """Run the full lowering pipeline and produce a content-addressed Artifact.

    Args
    ----
    dynamics : CasadiFunction
        From :func:`jaxility.lowering.translate` or
        :func:`jaxility.lowering.generate_dynamics`.
    spec : OcpTemplateSpec
        From a template factory (LQR / TrackingMPC / WBC /
        CentroidalMPC) or hand-rolled.
    target : Target
        ``HOST_DARWIN`` / ``HOST_LINUX`` for the host path;
        cross-compile targets land later.
    source_attestation_handle : bytes
        BLAKE3 digest from the upstream Source (Jaxterity Robot or
        MockSource). Used as ``Manifest.source_attestation_handle``.
    work_dir : Path
        Empty directory acados writes its generated code into. The
        caller owns its lifecycle; tests use a ``tmp_path``.
    build_timestamp_utc : int | None
        Override for the manifest timestamp; defaults to current time.
    extra_toolchain_versions : dict[str, str] | None
        Extra entries merged into the manifest's ``toolchain_versions``
        (over the detected defaults) — the honest provenance record for
        code-generation tools that participated but aren't detected per
        target, e.g. ``{"pinocchio": ...}`` when the dynamics came from
        :func:`jaxility.lowering.generate_dynamics`.

    Returns
    -------
    BuildBundle
        Artifact + manifest + live solver + library path.
    """
    from acados_template import AcadosOcpSolver

    work_dir.mkdir(parents=True, exist_ok=True)

    log: list[BuildLogEntry] = []
    log.append(
        BuildLogEntry(
            offset_us=0,
            stage="plan",
            level="info",
            message=(
                f"build_for_target target={target.name!r} spec.name={spec.name!r} "
                f"horizon={spec.horizon_steps} nx={dynamics.input_shapes[0][0]}"
            ),
            detail={
                "target_profile_hash_hex": target.hash.hex(),
                "source_attestation_handle_hex": source_attestation_handle.hex(),
            },
        )
    )

    ocp = build_ocp(dynamics, spec)
    log.append(
        BuildLogEntry(
            offset_us=1,
            stage="lower",
            level="info",
            message="acados OCP constructed",
            detail={"model_name": ocp.model.name},
        )
    )

    # acados writes JSON + generated C under the cwd; force it into
    # ``work_dir`` by chdir'ing for the construction call.
    json_filename = f"{ocp.model.name}.json"
    old_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        solver = AcadosOcpSolver(ocp, json_file=json_filename, verbose=False)
    finally:
        os.chdir(old_cwd)

    log.append(
        BuildLogEntry(
            offset_us=2,
            stage="compile",
            level="info",
            message="acados generated + compiled shared library",
            detail={
                "host_python": platform.python_version(),
                "host_machine": platform.machine(),
            },
        )
    )

    library_path = _locate_shared_library(work_dir, ocp.model.name)
    payload = library_path.read_bytes()
    log.append(
        BuildLogEntry(
            offset_us=3,
            stage="package",
            level="info",
            message="shared library packaged into artifact",
            detail={
                "library_path": str(library_path),
                "payload_bytes": str(len(payload)),
            },
        )
    )

    if build_timestamp_utc is None:
        build_timestamp_utc = int(time.time() * 1_000_000)

    toolchain_versions = detect_toolchain_versions(target)
    if extra_toolchain_versions:
        toolchain_versions = {**toolchain_versions, **extra_toolchain_versions}

    manifest = Manifest(
        schema_version=SCHEMA_VERSION_V0,
        source_attestation_handle=source_attestation_handle,
        toolchain_versions=toolchain_versions,
        target_profile_hash=target.hash,
        artifact_content_hash=blake3.blake3(payload).digest(),
        build_timestamp_utc=build_timestamp_utc,
    )

    artifact = Artifact.build(
        payload=payload,
        source_manifest_hash=manifest.content_hash(),
        target_profile_hash=target.hash,
        build_log=tuple(log),
    )

    return BuildBundle(
        artifact=artifact,
        manifest=manifest,
        target=target,
        solver=solver,
        shared_library_path=library_path,
    )


# The previous ``_detect_acados_version`` / ``_detect_casadi_version``
# shims in this module preserved a silent ``"unknown"`` fallback that
# A6 explicitly claimed to remove. They were dead code (no callers in
# the tree) and were deleted in the M-5 audit fix. Use the strict
# detectors in :mod:`jaxility.manifest.toolchain_detect` directly.
