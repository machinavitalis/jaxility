# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""``jaxility build-urdf`` — build directly from a URDF/MJCF (T-124).

The zoo path (``jaxility build <name>``) needs a hand-written dynamics factory
per robot. This path takes *any* fixed-base URDF/MJCF, generates its rigid-body
dynamics with :func:`jaxility.lowering.generate_dynamics` (Pinocchio parse +
CasADi ABA), wraps a joint-space **regulation** controller around it, and runs
the same ``build_for_target`` pipeline — artifact + signed manifest included.

The manifest's ``source_attestation_handle`` anchors to a BLAKE3 hash of the
description bytes (there is no upstream calibrated Robot to anchor to), and the
Pinocchio version is recorded in ``toolchain_versions['pinocchio']`` — the honest
provenance that these dynamics were generated, not lowered from JAX.

Scope matches ``generate_dynamics``: fixed-base, 1-DoF joints. The controller is
a single quadratic regulation to the zero pose; richer controllers stay on the
zoo/template path (a URDF carries no task specification).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, cast


def _select_target(name: str):
    from ..targets import HOST_TARGETS, current_host_target

    if name == "host":
        return current_host_target()
    if name in HOST_TARGETS:
        return HOST_TARGETS[name]
    raise SystemExit(
        f"unknown target {name!r}; build-urdf supports "
        "``host`` / ``host-darwin`` / ``host-linux``."
    )


def _infer_format(source: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return "mjcf" if source.suffix.lower() in {".xml", ".mjcf"} else "urdf"


def _emit(payload: dict) -> None:
    print(json.dumps(payload), file=sys.stdout)


def run_build_urdf(
    *,
    source: str,
    source_format: str | None = None,
    backend: str = "auto",
    target_name: str = "host",
    work_dir: str | None = None,
    horizon_steps: int = 20,
    time_horizon_s: float = 1.0,
    state_cost: float = 1.0,
    input_cost: float = 0.1,
    name: str | None = None,
) -> int:
    """Implement ``jaxility build-urdf <path> [--target ...] [--backend ...]``.

    Emits a structured JSON report on stdout (PATTERNS §8.2). Returns ``0`` on
    success, ``2`` on bad arguments / missing file, ``1`` on a build failure.
    """
    import blake3

    from ..builder import build_for_target
    from ..errors import ToolchainError
    from ..lowering import OcpTemplateSpec, generate_dynamics

    src_path = Path(source).expanduser()
    if not src_path.is_file():
        _emit({"ok": False, "reason": f"no such description file: {src_path}"})
        return 2

    fmt = _infer_format(src_path, source_format)
    model_name = name or src_path.stem.replace("-", "_")

    try:
        target = _select_target(target_name)
    except SystemExit as exc:
        _emit({"ok": False, "reason": str(exc)})
        return 2

    try:
        # argparse ``choices`` already constrains these to the valid literals.
        dynamics = generate_dynamics(
            str(src_path),
            source_format=cast(Literal["urdf", "mjcf"], fmt),
            backend=cast(Literal["auto", "pinocchio-casadi", "jaxility-aba"], backend),
            name=model_name,
        )
    except ToolchainError as exc:
        _emit({"ok": False, "reason": f"generate_dynamics failed: {exc}"})
        return 1

    nx = dynamics.input_shapes[0][0]
    nu = dynamics.input_shapes[1][0]
    spec = OcpTemplateSpec(
        name=model_name,
        horizon_steps=horizon_steps,
        time_horizon_s=time_horizon_s,
        state_cost=tuple([float(state_cost)] * nx),
        input_cost=tuple([float(input_cost)] * nu),
        terminal_state_cost=tuple([10.0 * float(state_cost)] * nx),
        state_reference=tuple([0.0] * nx),
        input_reference=tuple([0.0] * nu),
        initial_state=tuple([0.0] * nx),
    )

    # The chain anchor is the description itself (no upstream Robot handle).
    handle = blake3.blake3(b"urdf-source:" + src_path.read_bytes()).digest()

    # Honest provenance: these dynamics were *generated* via Pinocchio.
    from ..manifest import detect_pinocchio_version

    extra_toolchain = {"pinocchio": detect_pinocchio_version()}

    if work_dir is None:
        work_dir_path = Path("~/.cache/jaxility/builds").expanduser() / model_name
    else:
        work_dir_path = Path(work_dir).expanduser()

    try:
        bundle = build_for_target(
            dynamics=dynamics,
            spec=spec,
            target=target,
            source_attestation_handle=handle,
            work_dir=work_dir_path,
            extra_toolchain_versions=extra_toolchain,
        )
    except Exception as exc:  # noqa: BLE001 - surface the build failure as JSON
        _emit({"ok": False, "reason": f"build failed: {exc!r}"})
        return 1

    _emit(
        {
            "ok": True,
            "source": str(src_path),
            "source_format": fmt,
            "backend": backend,
            "model_name": model_name,
            "nx": nx,
            "nu": nu,
            "target": target.name,
            "artifact_content_hash_hex": bundle.artifact.content_hash.hex(),
            "manifest_content_hash_hex": bundle.manifest.content_hash().hex(),
            "source_attestation_handle_hex": handle.hex(),
            "pinocchio_version": extra_toolchain["pinocchio"],
            "shared_library_path": str(bundle.shared_library_path),
            "payload_bytes": len(bundle.artifact.payload),
        }
    )
    return 0


__all__ = ["run_build_urdf"]
