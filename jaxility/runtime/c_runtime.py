# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Build orchestrator for the on-target C runtime (T-032 / T-052).

The C runtime sources under ``runtime-c/`` are cross-compiled per
:class:`~jaxility.targets.Target` into a static archive
``libjaxility_runtime_<family>.a`` that the deployment artifact links
against. This module is the Python-side glue between the runtime
source tree and the cross-toolchain selection in
:mod:`jaxility.builder_cross`.

The build splits into:

* :func:`runtime_sources_for_family` — which ``.c`` files apply to a
  given target family. Cortex-A picks up ``arena.c`` +
  ``cycle_posix.c`` + ``dcache_aarch64.c``; Cortex-M picks up
  ``arena.c`` + ``dcache_thumb.c`` (the cycle scheduler ships
  per-MCU, not per-family).
* :func:`plan_runtime_build` — composes the ``(compile_argvs, ar_argv,
  output_path)`` for a deterministic build. Pure data; runs no
  subprocess.
* :func:`build_runtime_archive` — runs the plan. Returns a
  :class:`RuntimeArchive` carrying the archive path and a BLAKE3
  content hash for manifest binding.

The archive's content hash is what the deployment manifest's
``toolchain_versions['runtime-c']`` row records — the runtime is a
build-time toolchain ingredient alongside acados and CasADi
(invariant 5: byte-identical inputs → byte-identical artifact).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import blake3

from ..builder_cross import cflags_for_family
from ..errors import TargetError, ToolchainError
from ..targets import Target

# ---------------------------------------------------------------------------
# Source-tree layout
# ---------------------------------------------------------------------------

# The runtime source tree lives at the repo root. The path is computed
# at import time so the orchestrator works from any cwd. We discover
# the runtime tree by walking up from this file's location until we
# find a directory named ``runtime-c``.


def _discover_runtime_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "runtime-c"
        if candidate.is_dir():
            return candidate
    raise TargetError(
        "could not locate runtime-c/ source tree by walking up from "
        f"{here}; the runtime sources are required for the build "
        "orchestrator. Re-clone the repository or set "
        "$JAXILITY_RUNTIME_C_DIR."
    )


def runtime_root() -> Path:
    """Return the absolute path to the ``runtime-c/`` source tree."""
    env = os.environ.get("JAXILITY_RUNTIME_C_DIR")
    if env:
        return Path(env).expanduser()
    return _discover_runtime_root()


# ---------------------------------------------------------------------------
# Per-family source filter
# ---------------------------------------------------------------------------

# Keyed by the family-group name we use internally: ``"cortex-a"`` for
# the A-profile linux targets, ``"cortex-m"`` for M-profile baremetal.
# The user-facing :class:`Target` family (``"cortex-a76"``,
# ``"cortex-m4"``) maps onto these via prefix.
_GROUP_SOURCES: dict[str, tuple[str, ...]] = {
    # ``rt_posix.c`` + ``deploy_posix.c`` (T-032) are Cortex-A only:
    # rt_posix pins the control thread under PREEMPT_RT Linux, and
    # deploy_posix dlopen()s the controller .so and drives it from the
    # cycle loop. The bare-metal Cortex-M runtime has no OS scheduler
    # and no dlopen, so both are excluded from that group. Note:
    # ``deploy_main.c`` is the launcher *entry point* (carries main())
    # and is deliberately NOT in the archive — it is compiled directly
    # into the launcher binary by jaxility.runtime.deploy.
    "cortex-a": (
        "arena.c",
        "cycle_posix.c",
        "dcache_aarch64.c",
        "rt_posix.c",
        "deploy_posix.c",
    ),
    "cortex-m": ("arena.c", "dcache_thumb.c"),
}


def _family_group(family: str) -> str:
    """Map a :class:`Target.family` to the runtime source-group name."""
    if family.startswith("cortex-a") or family.startswith("neoverse"):
        return "cortex-a"
    if family.startswith("cortex-m") or family.startswith("ethos-"):
        return "cortex-m"
    raise TargetError(
        f"no runtime source group registered for target family {family!r}. "
        f"Known families: cortex-a* / neoverse-* (POSIX runtime), "
        f"cortex-m* / ethos-* (baremetal runtime). Add a row to "
        f"jaxility.runtime.c_runtime._GROUP_SOURCES + the per-family "
        "branch in _family_group if a new SoC class lands."
    )


def runtime_sources_for_family(family: str) -> tuple[Path, ...]:
    """Return the absolute paths of the runtime ``.c`` sources for ``family``.

    Raises
    ------
    TargetError
        Family is unknown — there is no silent default (invariant 7).
    """
    group = _family_group(family)
    src_dir = runtime_root() / "src"
    paths = tuple((src_dir / name).resolve() for name in _GROUP_SOURCES[group])
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise TargetError(
            f"runtime sources for family {family!r} are missing: "
            f"{[str(p) for p in missing]}. The runtime-c/src/ tree is "
            "incomplete; re-clone the repository."
        )
    return paths


# ---------------------------------------------------------------------------
# Build plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeBuildPlan:
    """A composed, deterministic plan for cross-compiling the runtime.

    Pure data — constructing it does no I/O. Same inputs (target,
    runtime_root, work_dir) produce a byte-identical plan, which is
    the foundation of the manifest's reproducible-build invariant.
    """

    target: Target
    """Deployment target the runtime is built for."""

    sources: tuple[Path, ...]
    """Per-target runtime ``.c`` sources, in stable order."""

    object_paths: tuple[Path, ...]
    """Per-source ``.o`` output paths under ``work_dir``."""

    compile_argvs: tuple[tuple[str, ...], ...]
    """One ``compiler_argv`` tuple per source, in the same order as
    ``sources`` / ``object_paths``. The compiler is invoked with ``-c``
    (compile-only) and the per-family flags from
    :func:`jaxility.builder_cross.cflags_for_family`, with ``-shared``
    / ``-fPIC`` filtered out (A-profile family flags include them for
    the OCP shared-object build; the runtime needs object files
    instead)."""

    ar_argv: tuple[str, ...]
    """``ar rcs`` argv that assembles ``object_paths`` into
    ``archive_path``."""

    archive_path: Path
    """Where the produced ``.a`` lands. Conventionally
    ``work_dir / 'libjaxility_runtime_<family>.a'``."""


def _ar_tool_name(target: Target) -> str:
    """Derive the ``ar`` binary name from the toolchain's gcc name.

    Arm GNU ships ``aarch64-none-linux-gnu-ar`` next to
    ``aarch64-none-linux-gnu-gcc``; the convention is to swap the
    ``-gcc`` suffix for ``-ar``. Same for ``arm-none-eabi-{gcc,ar}``.
    """
    name = target.toolchain.name
    if name.endswith("-gcc"):
        return name[: -len("-gcc")] + "-ar"
    raise ToolchainError(
        f"toolchain name {name!r} does not end with '-gcc'; cannot derive "
        f"the matching 'ar' binary. Add a target-specific override if a "
        "non-GNU toolchain lands."
    )


def plan_runtime_build(
    *, target: Target, work_dir: Path, archive_path: Path | None = None
) -> RuntimeBuildPlan:
    """Compose a :class:`RuntimeBuildPlan` for ``target``.

    The plan is deterministic — same inputs → byte-identical argv
    tuples. That property travels into the manifest content hash.
    """
    family = target.family
    sources = runtime_sources_for_family(family)
    work_dir = work_dir.resolve()

    # Borrow the per-family flag composition from the cross-compile
    # tier. Strip the OCP-shared-library flags (``-shared``, ``-fPIC``)
    # because the runtime is a *static* archive of relocatable
    # objects. Keep everything else (ABI / ISA / -O3 / warnings).
    family_flags = tuple(f for f in cflags_for_family(family) if f not in ("-shared",))
    # ``-c`` is the compile-only flag; deduplicate if already present.
    if "-c" not in family_flags:
        family_flags = family_flags + ("-c",)

    include_dir = runtime_root() / "include"

    cc = target.toolchain.name
    object_paths: list[Path] = []
    compile_argvs: list[tuple[str, ...]] = []
    for src in sources:
        obj = work_dir / (src.stem + ".o")
        object_paths.append(obj)
        argv = [
            cc,
            *family_flags,
            f"-I{include_dir}",
            "-o",
            str(obj),
            str(src),
        ]
        compile_argvs.append(tuple(argv))

    if archive_path is None:
        archive_path = work_dir / f"libjaxility_runtime_{family}.a"

    ar = _ar_tool_name(target)
    # Audit M-4: ``D`` (deterministic) zeroes mtime / uid / gid / mode
    # in member headers so two builds of the same sources produce
    # byte-identical archives. ``rcs`` = create + add + write index.
    ar_argv = (ar, "rcsD", str(archive_path), *(str(p) for p in object_paths))

    return RuntimeBuildPlan(
        target=target,
        sources=sources,
        object_paths=tuple(object_paths),
        compile_argvs=tuple(compile_argvs),
        ar_argv=ar_argv,
        archive_path=archive_path,
    )


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeArchive:
    """Output of :func:`build_runtime_archive` — the produced ``.a``.

    Carries the path + content hash so the manifest can record exactly
    which runtime archive a deployment artifact linked against. The
    hash is BLAKE3 over the archive bytes (same digest family as
    :class:`jaxility.manifest.Manifest.artifact_content_hash`).
    """

    target: Target
    archive_path: Path
    content_hash: bytes
    object_paths: tuple[Path, ...]


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def _run_or_raise(argv: tuple[str, ...], *, stage: str, timeout_s: float) -> None:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise ToolchainError(
            f"runtime build {stage} failed to launch {argv[0]!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(
            f"runtime build {stage} timed out after {timeout_s}s running {argv[0]!r}."
        ) from exc
    if completed.returncode != 0:
        raise ToolchainError(
            f"runtime build {stage} failed with exit code "
            f"{completed.returncode}.\nargv: {argv!r}\n"
            f"stderr:\n{completed.stderr}"
        )


def build_runtime_archive(
    *,
    target: Target,
    work_dir: Path,
    archive_path: Path | None = None,
    timeout_s: float = 120.0,
) -> RuntimeArchive:
    """Cross-compile the runtime sources for ``target`` and pack as ``.a``.

    The work directory is created if it doesn't exist. Existing
    ``.o`` / ``.a`` files are clobbered (the build is idempotent;
    we don't carry a Make-style mtime cache).

    Raises
    ------
    ToolchainError
        Toolchain binary missing or returned non-zero on compile / ar.
    TargetError
        Unknown family or runtime sources missing.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    cc = target.toolchain.name
    if shutil.which(cc) is None:
        raise ToolchainError(
            f"runtime build for target {target.name!r} requires {cc!r} on "
            f"PATH (not found). Install Arm GNU Toolchain "
            f"{target.toolchain.version} from {target.toolchain.download_url}."
        )

    plan = plan_runtime_build(
        target=target, work_dir=work_dir, archive_path=archive_path
    )

    ar = plan.ar_argv[0]
    if shutil.which(ar) is None:
        raise ToolchainError(
            f"runtime build for target {target.name!r} requires {ar!r} on "
            "PATH (sibling of the gcc binary; ships with the same Arm GNU "
            "toolchain release). Not found."
        )

    # Clean stale outputs so the archive does not pick up leftovers.
    for obj in plan.object_paths:
        if obj.exists():
            obj.unlink()
    if plan.archive_path.exists():
        plan.archive_path.unlink()

    for argv in plan.compile_argvs:
        _run_or_raise(argv, stage="compile", timeout_s=timeout_s)
    _run_or_raise(plan.ar_argv, stage="ar", timeout_s=timeout_s)

    if not plan.archive_path.exists():
        raise ToolchainError(
            f"runtime ar reported success but no archive at {plan.archive_path}."
        )

    payload = plan.archive_path.read_bytes()
    return RuntimeArchive(
        target=target,
        archive_path=plan.archive_path,
        content_hash=blake3.blake3(payload).digest(),
        object_paths=plan.object_paths,
    )
