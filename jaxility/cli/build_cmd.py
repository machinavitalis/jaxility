# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""``jaxility build`` subcommand implementation (T-026 + dynamics adapter)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import blake3

from ..targets import HOST_TARGETS, Target, current_host_target


def _select_target(name: str) -> Target:
    if name == "host":
        return current_host_target()
    if name in HOST_TARGETS:
        return HOST_TARGETS[name]
    raise SystemExit(
        f"unknown target {name!r}; build supports "
        "``host`` / ``host-darwin`` / ``host-linux``."
    )


def _select_template_spec(zoo_entry, dynamics):
    """Pick the right template factory for a zoo entry's ``template`` field.

    Pulls keyword arguments from ``zoo_entry.template_options`` so users
    can override per-entry without touching CLI code. Missing keys fall
    back to the template's own defaults.
    """
    from ..templates import WBCTask, centroidal_mpc, lqr, tracking_mpc, wbc

    options = dict(zoo_entry.template_options)
    options.setdefault("name", zoo_entry.name)

    if zoo_entry.template == "LQR":
        return lqr(dynamics, **options)
    if zoo_entry.template == "TrackingMPC":
        return tracking_mpc(dynamics, **options)
    if zoo_entry.template == "WBC":
        # Auto-construct a single regulation task from Q / R if no
        # explicit ``tasks`` are supplied. Default behaviour:
        # one task per zoo entry until the upstream Task DSL lands.
        if "tasks" not in options:
            Q = options.pop("Q")
            R = options.pop("R")
            tasks = [
                WBCTask(
                    name="regulate",
                    priority=1.0,
                    state_weight=Q,
                    input_weight=R,
                    state_reference=tuple([0.0] * len(Q)),
                    input_reference=tuple([0.0] * len(R)),
                )
            ]
            options["tasks"] = tasks
        return wbc(dynamics, **options)
    if zoo_entry.template == "CentroidalMPC":
        if "initial_com_state" not in options:
            options["initial_com_state"] = options.pop("initial_state")
        return centroidal_mpc(dynamics, **options)
    raise SystemExit(
        f"unknown template {zoo_entry.template!r} on zoo entry {zoo_entry.name!r}."
    )


def _emit(payload: dict) -> None:
    # Resolve sys.stdout at call time, not at definition time — acados
    # subprocess invocations and pytest's capsys both replace stdout
    # mid-test, and a default-argument capture would point at a stale
    # handle by the second build in the same process.
    print(json.dumps(payload), file=sys.stdout)


def run_build(
    *,
    zoo_name: str,
    target_name: str,
    work_dir: str | None,
) -> int:
    """Implement ``jaxility build <zoo_name> --target <target>``.

    Calls the zoo entry's ``jax_dynamics_factory`` to get a JAX
    callable, translates it via T-020, picks the right template via
    ``zoo_entry.template_options``, and runs ``build_for_target``.
    Emits a structured JSON report on stdout (PATTERNS §8.2).
    """
    from ..builder import build_for_target
    from ..lowering import translate
    from ..zoo import load as load_zoo

    try:
        entry = load_zoo(zoo_name)
    except KeyError as exc:
        _emit({"ok": False, "reason": str(exc)})
        return 2

    if entry.casadi_dynamics_factory is None and entry.jax_dynamics_factory is None:
        _emit(
            {
                "ok": False,
                "reason": (
                    f"zoo entry {zoo_name!r} has no dynamics factory "
                    f"(upstream_status={entry.upstream_status!r}); the host build "
                    "path needs real dynamics (a JAX function or a generated "
                    "CasADi function). Promote upstream first or supply a factory."
                ),
            }
        )
        return 1

    try:
        target = _select_target(target_name)
    except SystemExit as exc:
        _emit({"ok": False, "reason": str(exc)})
        return 2

    # Generator-sourced entries (T-126) hand back a CasadiFunction directly and
    # skip the JAX-translate path; otherwise translate the entry's JAX dynamics.
    if entry.casadi_dynamics_factory is not None:
        try:
            dynamics = entry.casadi_dynamics_factory()
        except Exception as exc:
            _emit({"ok": False, "reason": f"failed to generate dynamics: {exc!r}"})
            return 1
    else:
        # The early return above guarantees a factory exists; in this branch
        # casadi_dynamics_factory is None, so jax_dynamics_factory is not.
        assert entry.jax_dynamics_factory is not None
        try:
            jax_fn, state_shape, control_shape = entry.jax_dynamics_factory()
        except Exception as exc:
            # Audit N-5 fix: the previous ``# pragma: no cover`` on this
            # branch was justified by "no test exercises it". The test in
            # ``test/unit/test_cli_zoo_build.py`` now drives a factory that
            # deliberately raises, so the pragma is gone and coverage
            # tracks this branch normally.
            _emit(
                {
                    "ok": False,
                    "reason": f"failed to extract JAX dynamics: {exc!r}",
                }
            )
            return 1
        dynamics = translate(
            jax_fn, in_shapes=(state_shape, control_shape), name=zoo_name
        )

    try:
        spec = _select_template_spec(entry, dynamics)
    except Exception as exc:
        # The dynamics may lower cleanly yet still not be end-to-end buildable —
        # e.g. a zoo entry whose controller template is not wired yet. Surface
        # that as a structured report (PATTERNS §8.2), never an uncaught crash.
        _emit(
            {
                "ok": False,
                "reason": (
                    f"zoo entry {zoo_name!r} is not yet end-to-end buildable: "
                    f"{type(exc).__name__}: {exc}. The dynamics lower, but the "
                    f"{entry.template!r} template is not wired for this entry — "
                    "see the entry's remaining_work."
                ),
            }
        )
        return 1

    # Source attestation handle. Audit M-6 close: the previous version
    # silently swallowed every exception from ``source_factory()`` and
    # forged 32 zero bytes as the chain anchor, contradicting the
    # CLAIMS.md "Verification cryptographically validates the chain
    # end-to-end" guarantee. The new contract:
    #
    # * If ``source_factory`` raises, we surface the failure in the
    #   CLI output (and the build log) rather than burying it.
    # * If the constructed source has no ``attestation_handle``, that
    #   is a Source-contract violation; we name it explicitly.
    #
    # Both paths still produce a 32-byte handle so the manifest can
    # populate ``source_attestation_handle``, but the handle is
    # derived from a self-explaining marker (``"source-unavailable:"``
    # + reason, BLAKE3-hashed) so a verifier sees the gap.
    handle: bytes
    try:
        source = entry.source_factory()
    except Exception as exc:
        marker = f"source-unavailable:source_factory:{type(exc).__name__}:{exc}"
        handle = blake3.blake3(marker.encode("utf-8")).digest()
        _emit({"warning": marker})
    else:
        attestation = getattr(source, "attestation_handle", None)
        if isinstance(attestation, bytes) and len(attestation) == 32:
            handle = attestation
        else:
            marker = f"source-unavailable:no-attestation-handle:{type(source).__name__}"
            handle = blake3.blake3(marker.encode("utf-8")).digest()
            _emit({"warning": marker})

    if work_dir is None:
        work_dir_path = Path("~/.cache/jaxility/builds").expanduser() / zoo_name
    else:
        work_dir_path = Path(work_dir).expanduser()

    bundle = build_for_target(
        dynamics=dynamics,
        spec=spec,
        target=target,
        source_attestation_handle=handle,
        work_dir=work_dir_path,
    )

    _emit(
        {
            "ok": True,
            "zoo_name": zoo_name,
            "target": target.name,
            "artifact_content_hash_hex": bundle.artifact.content_hash.hex(),
            "manifest_content_hash_hex": bundle.manifest.content_hash().hex(),
            # M-6: surface the actual chain anchor so a verifier can
            # see whether it is a real Source-supplied handle or one
            # of the documented self-explaining markers.
            "source_attestation_handle_hex": handle.hex(),
            "shared_library_path": str(bundle.shared_library_path),
            "payload_bytes": len(bundle.artifact.payload),
        }
    )
    return 0


__all__ = ["run_build", "_select_target", "_select_template_spec"]
