# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Deployment-launcher build orchestrator (T-032).

The deployment glue's two halves are the on-target C (``deploy_posix.c``
/ ``deploy.h``, shipped in the runtime archive) and *this* build-time
Python that produces the launcher executable. The launcher
(``jx_deploy_<family>``) is ``deploy_main.c`` cross-compiled against the
runtime archive ``libjaxility_runtime_<family>.a``; at run time it
dlopen()s the cross-compiled controller ``.so`` and drives it from the
cycle loop (see ``jaxility_runtime/deploy.h``).

The launcher binary does **not** link against acados — the controller
is loaded dynamically — so the launcher depends only on the runtime
archive plus ``-ldl`` / ``-pthread``. It is built ``-rdynamic`` so the
controller can resolve runtime symbols (e.g. ``jx_arena_alloc``) from
the launcher's exported symbol table.

Same plan/execute split as the rest of the runtime tooling: a
deterministic :class:`DeployLauncherPlan` (pure data) and an
:func:`execute_deploy_launcher` that runs it.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import blake3

from ..builder_cross import cflags_for_family
from ..errors import TargetError, ToolchainError
from ..targets import Target
from .c_runtime import runtime_root, runtime_sources_for_family

# The launcher entry-point source. Lives under runtime-c/src/ but is NOT
# part of any runtime-archive source group (it carries ``main``).
LAUNCHER_SOURCE_NAME = "deploy_main.c"

# The deploy glue object that must be present in the runtime archive for
# the launcher to link. Used as the per-family capability gate.
_DEPLOY_RUNTIME_SOURCE = "deploy_posix.c"


def _launcher_source() -> Path:
    src = (runtime_root() / "src" / LAUNCHER_SOURCE_NAME).resolve()
    if not src.exists():
        raise TargetError(
            f"launcher source {src} is missing; the runtime-c/src tree is "
            "incomplete. Re-clone the repository."
        )
    return src


@dataclass(frozen=True)
class DeployLauncherPlan:
    """A composed, deterministic plan to build the deployment launcher.

    Pure data — constructing it does no I/O and needs no toolchain.
    """

    target: Target
    launcher_source: Path
    runtime_archive: Path
    include_dir: Path
    output_path: Path
    compile_argv: tuple[str, ...]


def plan_deploy_launcher(
    *,
    target: Target,
    runtime_archive: Path,
    work_dir: Path,
    output_path: Path | None = None,
) -> DeployLauncherPlan:
    """Compose a :class:`DeployLauncherPlan` for ``target``.

    Args
    ----
    target : Target
        Deployment target. Must be a POSIX (Cortex-A) family — the
        launcher uses dlopen, which the bare-metal Cortex-M runtime has
        no counterpart for. A Cortex-M target raises ``TargetError``.
    runtime_archive : Path
        The ``libjaxility_runtime_<family>.a`` from
        :func:`jaxility.runtime.build_runtime_archive`. It must carry
        ``deploy_posix.o`` (the cortex-a group does).
    work_dir : Path
        Where the launcher binary lands (unless ``output_path`` given).
    output_path : Path | None
        Override for the launcher binary path. Defaults to
        ``work_dir / f"jx_deploy_{family}"``.

    Raises
    ------
    TargetError
        Unknown family, a non-POSIX (no-dlopen) family, or a missing
        launcher source.
    """
    family = target.family
    # Capability gate: the launcher only makes sense where the runtime
    # group carries the dlopen-based deploy glue (cortex-a). This also
    # rejects unknown families (runtime_sources_for_family raises).
    source_names = {p.name for p in runtime_sources_for_family(family)}
    if _DEPLOY_RUNTIME_SOURCE not in source_names:
        raise TargetError(
            f"deployment launcher is unsupported for target family "
            f"{family!r}: its runtime group has no {_DEPLOY_RUNTIME_SOURCE} "
            "(dlopen-based launcher). The launcher is a POSIX / Cortex-A "
            "facility; bare-metal Cortex-M deployment is a different "
            "mechanism (T-052)."
        )

    launcher_source = _launcher_source()
    include_dir = (runtime_root() / "include").resolve()
    runtime_archive = runtime_archive.resolve()

    # Reuse the per-family flag composition; strip ``-shared`` (we link
    # an executable, not a shared object). Keep -fPIC / ISA / -O3.
    flags = tuple(f for f in cflags_for_family(family) if f != "-shared")

    if output_path is None:
        output_path = (work_dir.resolve() / f"jx_deploy_{family}").resolve()
    else:
        output_path = output_path.resolve()

    cc = target.toolchain.name
    argv: list[str] = [cc, *flags]
    # -rdynamic exports the launcher's symbols so the dlopen()ed
    # controller can resolve runtime functions (jx_arena_alloc, ...).
    argv.append("-rdynamic")
    argv.append(f"-I{include_dir}")
    argv.extend(("-o", str(output_path)))
    argv.append(str(launcher_source))
    # Archive after the source (it satisfies the source's references);
    # -ldl for dlopen, -pthread for the SCHED_FIFO path in rt_posix.
    argv.append(str(runtime_archive))
    argv.extend(("-ldl", "-pthread"))

    return DeployLauncherPlan(
        target=target,
        launcher_source=launcher_source,
        runtime_archive=runtime_archive,
        include_dir=include_dir,
        output_path=output_path,
        compile_argv=tuple(argv),
    )


@dataclass(frozen=True)
class DeployLauncher:
    """Output of :func:`execute_deploy_launcher` — the launcher binary."""

    target: Target
    launcher_path: Path
    content_hash: bytes


def execute_deploy_launcher(
    plan: DeployLauncherPlan, *, timeout_s: float = 120.0
) -> DeployLauncher:
    """Build the launcher binary from ``plan``.

    Raises
    ------
    ToolchainError
        Toolchain missing, runtime archive absent, compile failed, or
        the compiler reported success but wrote no binary.
    """
    cc = plan.target.toolchain.name
    if shutil.which(cc) is None:
        raise ToolchainError(
            f"deployment launcher build for target {plan.target.name!r} "
            f"requires {cc!r} on PATH (not found). Install Arm GNU "
            f"Toolchain {plan.target.toolchain.version} from "
            f"{plan.target.toolchain.download_url}."
        )
    if not plan.runtime_archive.exists():
        raise ToolchainError(
            f"runtime archive {plan.runtime_archive} does not exist; build "
            "it first with jaxility.runtime.build_runtime_archive."
        )

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    if plan.output_path.exists():
        plan.output_path.unlink()

    try:
        completed = subprocess.run(
            list(plan.compile_argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise ToolchainError(
            f"failed to launch {plan.compile_argv[0]!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(
            f"deployment launcher build timed out after {timeout_s}s "
            f"(target={plan.target.name!r})."
        ) from exc

    if completed.returncode != 0:
        raise ToolchainError(
            f"deployment launcher build failed for target "
            f"{plan.target.name!r} with exit code {completed.returncode}.\n"
            f"argv: {plan.compile_argv!r}\nstderr:\n{completed.stderr}"
        )
    if not plan.output_path.exists():
        raise ToolchainError(
            f"deployment launcher build reported success but no binary at "
            f"{plan.output_path}.\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    payload = plan.output_path.read_bytes()
    return DeployLauncher(
        target=plan.target,
        launcher_path=plan.output_path,
        content_hash=blake3.blake3(payload).digest(),
    )
