# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Cross-build orchestrator for the acados / blasfeo / hpipm static
archives a deployment artifact links against (T-031 follow-up).

The structural cross-compile wrapper (:mod:`jaxility.builder_cross`)
compiles the acados-generated controller C for an Arm target, but the
resulting objects reference acados / hpipm / blasfeo symbols. On the
host those live in the shared libraries acados builds by default; for a
cross target they do not exist until they are cross-built *as static
archives* and fed into the link step through the ``extra_link_args`` /
``extra_include_dirs`` seam in :func:`~jaxility.builder_cross.plan_cross_compile`.

This module closes that gap by **building the dependencies from
source** for the target, rather than vendoring prebuilt binary blobs
(ADR-018). It reuses the acados upstream CMake build the host path
already depends on, routed through the same Arm GNU toolchain via a
CMake toolchain file (``cmake/toolchains/``), with three deltas from
the host configure:

* ``-DBUILD_SHARED_LIBS=OFF`` — emit ``.a`` archives, not ``.so`` /
  ``.dylib``. (The host default is shared; that is precisely why the
  cross archives don't exist yet.)
* ``-DBLASFEO_TARGET=<per-family>`` — pin blasfeo's micro-arch kernel
  selection to the deployment CPU (Cortex-A76 for the Pi 5).
* ``-DCMAKE_TOOLCHAIN_FILE=<target toolchain file>`` — route every
  compiler invocation through the cross toolchain.

The module mirrors the plan/execute split used everywhere in Jaxility:

* :class:`DepBuildPlan` — pure data; constructing it does no I/O and
  needs no toolchain. The configure / build argv, install prefix,
  expected archive paths, include dirs, and link args are all
  derivable structurally, so the composition is tested on any host.
* :func:`plan_dep_build` — composes the plan.
* :func:`link_args_for_prefix` — the consumer-facing helper: given an
  install prefix, returns ``(include_dirs, link_args)`` ready to pass
  into :func:`~jaxility.builder_cross.plan_cross_compile` or
  :func:`~jaxility.builder_cross.cross_build_for_target`.
* :func:`execute_dep_build` / :func:`build_cross_deps` — subprocess
  bound; run cmake configure + build + install. Tier-B (skipped unless
  both the cross toolchain and an acados source tree are present).

The static-archive link order matters: acados depends on hpipm which
depends on blasfeo, and the three have mutual references that GNU ld
resolves only inside a ``--start-group`` / ``--end-group`` wrapper.
:func:`link_args_for_prefix` emits that wrapper.

Provenance: :class:`CrossBuiltDeps` carries a BLAKE3 content hash per
archive so the deployment manifest can record exactly which dependency
binaries an artifact linked against. Note that upstream acados/blasfeo
do not currently emit byte-deterministic archives (their ``ar``
invocations are not ``D``-mode), so the recorded hashes are provenance,
not a reproducibility guarantee — see KNOWN_GAPS.md.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import blake3

from .errors import TargetError, ToolchainError
from .targets import Target

# Default ``cmake --build --parallel`` width. Fixed (not derived from
# ``os.cpu_count()``) so :class:`DepBuildPlan` is host-independent and
# the plan-composition tests are stable.
DEFAULT_BUILD_JOBS = 4

# ---------------------------------------------------------------------------
# Per-family blasfeo micro-arch target
# ---------------------------------------------------------------------------

# blasfeo selects hand-tuned linear-algebra kernels by ``TARGET``. The
# names are blasfeo's, verified against external/blasfeo/CMakeLists.txt
# in the pinned acados checkout. Adding a family is an explicit row —
# no silent ``GENERIC`` fallback (invariant 7); a wrong guess would
# silently ship slow kernels.
_BLASFEO_TARGET_FOR_FAMILY: dict[str, str] = {
    "cortex-a76": "ARMV8A_ARM_CORTEX_A76",
}


def blasfeo_target_for_family(family: str) -> str:
    """Return the blasfeo ``TARGET`` kernel selection for ``family``.

    Raises
    ------
    TargetError
        Family has no registered blasfeo target. Add a row to
        ``_BLASFEO_TARGET_FOR_FAMILY`` (the value must be one of the
        ``TARGET`` options in ``external/blasfeo/CMakeLists.txt``).
    """
    try:
        return _BLASFEO_TARGET_FOR_FAMILY[family]
    except KeyError:
        known = ", ".join(sorted(_BLASFEO_TARGET_FOR_FAMILY)) or "(none)"
        raise TargetError(
            f"no blasfeo TARGET registered for target family {family!r}; "
            "add a row to jaxility.builder_deps._BLASFEO_TARGET_FOR_FAMILY "
            "(value must be a blasfeo TARGET from external/blasfeo/"
            f"CMakeLists.txt). Known families: {known}."
        ) from None


# ---------------------------------------------------------------------------
# CMake toolchain-file discovery
# ---------------------------------------------------------------------------

# The toolchain files live at the repo root under ``cmake/toolchains/``,
# discovered by walking up from this file (same pattern as
# ``jaxility.runtime.c_runtime.runtime_root``). ``$JAXILITY_CMAKE_DIR``
# overrides the search for out-of-tree installs.

_TOOLCHAIN_FILE_FOR_BINARY: dict[str, str] = {
    "aarch64-none-linux-gnu-gcc": "aarch64-none-linux-gnu.cmake",
}


def _discover_cmake_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "cmake" / "toolchains"
        if candidate.is_dir():
            return candidate
    raise TargetError(
        "could not locate cmake/toolchains/ by walking up from "
        f"{here}; the CMake toolchain files are required to cross-build "
        "the acados dependencies. Re-clone the repository or set "
        "$JAXILITY_CMAKE_DIR."
    )


def cmake_toolchains_dir() -> Path:
    """Return the absolute path to the ``cmake/toolchains/`` directory."""
    env = os.environ.get("JAXILITY_CMAKE_DIR")
    if env:
        return Path(env).expanduser()
    return _discover_cmake_dir()


def toolchain_file_for_target(target: Target) -> Path:
    """Return the CMake toolchain file for ``target``'s cross toolchain.

    Raises
    ------
    TargetError
        No toolchain file registered for the target's compiler, or the
        registered file does not exist on disk.
    """
    binary = target.toolchain.name
    try:
        filename = _TOOLCHAIN_FILE_FOR_BINARY[binary]
    except KeyError:
        known = ", ".join(sorted(_TOOLCHAIN_FILE_FOR_BINARY)) or "(none)"
        raise TargetError(
            f"no CMake toolchain file registered for cross toolchain "
            f"{binary!r}; add a row to jaxility.builder_deps."
            f"_TOOLCHAIN_FILE_FOR_BINARY and ship the file under "
            f"cmake/toolchains/. Known toolchains: {known}."
        ) from None
    path = (cmake_toolchains_dir() / filename).resolve()
    if not path.exists():
        raise TargetError(
            f"CMake toolchain file {path} for {binary!r} is missing; the "
            "cmake/toolchains/ tree is incomplete. Re-clone the repository."
        )
    return path


# ---------------------------------------------------------------------------
# Archive set
# ---------------------------------------------------------------------------

# Core acados static archives, in *link order* (dependents before their
# dependencies): acados → hpipm → blasfeo. qpOASES_e is appended only
# when built (it is a C++ archive and drags in libstdc++; the Pi 5
# launch controllers use HPIPM, so it is OFF by default).
_CORE_ARCHIVES: tuple[str, ...] = ("acados", "hpipm", "blasfeo")

# acados installs headers under include/{acados,acados_c,blasfeo,hpipm}.
# The generated controller C includes e.g. ``acados_c/ocp_nlp_interface.h``
# and ``blasfeo/blasfeo_target.h``; the include root plus the per-lib
# subdirs cover both spellings.
_INCLUDE_SUBDIRS: tuple[str, ...] = ("", "acados", "acados_c", "blasfeo", "hpipm")


def link_args_for_prefix(
    install_prefix: Path, *, with_qpoases: bool = False
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Return ``(include_dirs, link_args)`` for an acados install prefix.

    The result plugs directly into the ``extra_include_dirs`` /
    ``extra_link_args`` parameters of
    :func:`~jaxility.builder_cross.plan_cross_compile` /
    :func:`~jaxility.builder_cross.cross_build_for_target`.

    The static archives are interdependent, so the link args wrap them
    in a GNU ``--start-group`` / ``--end-group`` so the linker iterates
    to a fixed point regardless of command-line order. ``-lm`` follows
    the group (blasfeo / hpipm call into libm).

    This is pure path arithmetic — it does not check the archives exist
    (planning is I/O-free; :func:`execute_dep_build` verifies presence).
    """
    install_prefix = install_prefix.resolve()
    lib_dir = install_prefix / "lib"
    include_root = install_prefix / "include"
    include_dirs = tuple(
        (include_root / sub).resolve() if sub else include_root
        for sub in _INCLUDE_SUBDIRS
    )

    archive_libs = list(_CORE_ARCHIVES)
    if with_qpoases:
        archive_libs.append("qpOASES_e")

    group = [
        "-Wl,--start-group",
        *(f"-l{name}" for name in archive_libs),
        "-Wl,--end-group",
    ]
    link_args: tuple[str, ...] = (f"-L{lib_dir}", *group, "-lm")
    if with_qpoases:
        # qpOASES_e is C++; pull in the C++ runtime after the group.
        link_args = (*link_args, "-lstdc++")
    return include_dirs, link_args


def _expected_archive_paths(
    install_prefix: Path, *, with_qpoases: bool
) -> tuple[Path, ...]:
    lib_dir = (install_prefix / "lib").resolve()
    names = list(_CORE_ARCHIVES)
    if with_qpoases:
        names.append("qpOASES_e")
    return tuple(lib_dir / f"lib{name}.a" for name in names)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepBuildPlan:
    """A composed, deterministic plan to cross-build the acados deps.

    Pure data: constructing it does no I/O and does not require cmake or
    the cross toolchain. ``configure_argv`` / ``build_argv`` are stable
    for a fixed set of inputs.
    """

    target: Target
    """Deployment target the archives are built for."""

    acados_source_dir: Path
    """Root of the acados source checkout (with blasfeo / hpipm
    submodules under ``external/``)."""

    build_dir: Path
    """CMake build directory (out-of-source)."""

    install_prefix: Path
    """``-DACADOS_INSTALL_DIR``; archives land under ``<prefix>/lib``,
    headers under ``<prefix>/include``."""

    toolchain_file: Path
    """``-DCMAKE_TOOLCHAIN_FILE``; routes every compile through the
    cross toolchain."""

    blasfeo_target: str
    """blasfeo ``TARGET`` kernel selection for the deployment CPU."""

    with_qpoases: bool
    """Whether the qpOASES_e solver is built (and linked)."""

    configure_argv: tuple[str, ...]
    """``cmake -S ... -B ... -G 'Unix Makefiles' -D...`` argv."""

    build_argv: tuple[str, ...]
    """``cmake --build ... --target install --parallel N`` argv."""

    archive_paths: tuple[Path, ...]
    """Expected ``.a`` outputs, in link order (acados, hpipm, blasfeo
    [, qpOASES_e])."""

    include_dirs: tuple[Path, ...]
    """Header search roots for the consumer's cross-compile."""

    link_args: tuple[str, ...]
    """Link args (``-L`` + ``--start-group`` archive group + ``-lm``)
    for the consumer's cross-compile."""


def plan_dep_build(
    *,
    target: Target,
    acados_source_dir: Path,
    build_dir: Path,
    install_prefix: Path,
    toolchain_file: Path | None = None,
    with_qpoases: bool = False,
    jobs: int = DEFAULT_BUILD_JOBS,
) -> DepBuildPlan:
    """Compose a :class:`DepBuildPlan` to cross-build acados deps for ``target``.

    Args
    ----
    target : Target
        Deployment target. Its ``family`` selects the blasfeo kernel
        target; its ``toolchain.name`` selects the CMake toolchain file.
    acados_source_dir : Path
        acados source checkout root (must contain ``CMakeLists.txt`` and
        the blasfeo / hpipm submodules under ``external/``).
    build_dir : Path
        Out-of-source CMake build directory.
    install_prefix : Path
        Where the archives + headers install.
    toolchain_file : Path | None
        Override the CMake toolchain file. Defaults to the one
        registered for ``target.toolchain.name``.
    with_qpoases : bool
        Build + link the qpOASES_e solver. Default ``False`` — the Pi 5
        launch controllers use HPIPM and qpOASES_e drags in libstdc++.
    jobs : int
        ``cmake --build --parallel`` width. Fixed default keeps the plan
        host-independent.

    Raises
    ------
    TargetError
        Unknown family / toolchain, missing toolchain file, or the
        acados source dir has no ``CMakeLists.txt``.
    """
    if jobs < 1:
        raise TargetError(f"jobs must be >= 1, got {jobs}.")

    acados_source_dir = acados_source_dir.resolve()
    if not (acados_source_dir / "CMakeLists.txt").exists():
        raise TargetError(
            f"acados source dir {acados_source_dir} has no CMakeLists.txt; "
            "point at a recursive acados checkout (blasfeo / hpipm under "
            "external/). See AGENTS/TOOLCHAINS.md 'acados (T-021+)'."
        )
    build_dir = build_dir.resolve()
    install_prefix = install_prefix.resolve()
    blasfeo_target = blasfeo_target_for_family(target.family)
    if toolchain_file is None:
        toolchain_file = toolchain_file_for_target(target)
    else:
        toolchain_file = toolchain_file.resolve()

    configure_argv: tuple[str, ...] = (
        "cmake",
        "-S",
        str(acados_source_dir),
        "-B",
        str(build_dir),
        "-G",
        "Unix Makefiles",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DACADOS_INSTALL_DIR={install_prefix}",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        "-DBUILD_SHARED_LIBS=OFF",
        f"-DBLASFEO_TARGET={blasfeo_target}",
        f"-DACADOS_WITH_QPOASES={'ON' if with_qpoases else 'OFF'}",
        "-DACADOS_SILENT=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    build_argv: tuple[str, ...] = (
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "install",
        "--parallel",
        str(jobs),
    )

    include_dirs, link_args = link_args_for_prefix(
        install_prefix, with_qpoases=with_qpoases
    )

    return DepBuildPlan(
        target=target,
        acados_source_dir=acados_source_dir,
        build_dir=build_dir,
        install_prefix=install_prefix,
        toolchain_file=toolchain_file,
        blasfeo_target=blasfeo_target,
        with_qpoases=with_qpoases,
        configure_argv=configure_argv,
        build_argv=build_argv,
        archive_paths=_expected_archive_paths(
            install_prefix, with_qpoases=with_qpoases
        ),
        include_dirs=include_dirs,
        link_args=link_args,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossBuiltDeps:
    """Output of :func:`execute_dep_build` — the cross-built archives.

    Carries the include dirs + link args to feed into the controller
    cross-compile, plus a BLAKE3 content hash per archive so the
    deployment manifest can record dependency-binary provenance.
    """

    target: Target
    install_prefix: Path
    include_dirs: tuple[Path, ...]
    link_args: tuple[str, ...]
    archive_hashes: tuple[tuple[str, bytes], ...]
    """``(archive filename, blake3 digest)`` pairs, in link order."""


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
            f"acados dep build {stage} failed to launch {argv[0]!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(
            f"acados dep build {stage} timed out after {timeout_s}s "
            f"running {argv[0]!r}."
        ) from exc
    if completed.returncode != 0:
        raise ToolchainError(
            f"acados dep build {stage} failed with exit code "
            f"{completed.returncode}.\nargv: {argv!r}\n"
            f"stderr:\n{completed.stderr}"
        )


def execute_dep_build(
    plan: DepBuildPlan, *, timeout_s: float = 1800.0
) -> CrossBuiltDeps:
    """Run ``plan``: cmake configure + build + install the cross archives.

    The cross toolchain and ``cmake`` must be on PATH. The build dir is
    created if needed; an existing one is reused (CMake handles
    incremental reconfigure).

    Raises
    ------
    ToolchainError
        ``cmake`` or the cross toolchain missing; configure / build /
        install returned non-zero; or an expected archive is absent
        after a reported-successful install.
    """
    cc = plan.target.toolchain.name
    if shutil.which("cmake") is None:
        raise ToolchainError(
            "acados dep build requires 'cmake' on PATH (not found). "
            "Install CMake >= 3.x."
        )
    if shutil.which(cc) is None:
        raise ToolchainError(
            f"acados dep build for target {plan.target.name!r} requires "
            f"{cc!r} on PATH (not found). Install Arm GNU Toolchain "
            f"{plan.target.toolchain.version} from "
            f"{plan.target.toolchain.download_url}."
        )

    plan.build_dir.mkdir(parents=True, exist_ok=True)
    plan.install_prefix.mkdir(parents=True, exist_ok=True)

    _run_or_raise(plan.configure_argv, stage="configure", timeout_s=timeout_s)
    _run_or_raise(plan.build_argv, stage="build+install", timeout_s=timeout_s)

    missing = [p for p in plan.archive_paths if not p.exists()]
    if missing:
        raise ToolchainError(
            f"acados dep build reported success but these archives are "
            f"absent: {[str(p) for p in missing]}. Expected under "
            f"{plan.install_prefix / 'lib'}; the configure may have "
            "silently disabled a component (check that BUILD_SHARED_LIBS "
            "is OFF and the blasfeo / hpipm submodules are present)."
        )

    hashes = tuple(
        (p.name, blake3.blake3(p.read_bytes()).digest()) for p in plan.archive_paths
    )
    return CrossBuiltDeps(
        target=plan.target,
        install_prefix=plan.install_prefix,
        include_dirs=plan.include_dirs,
        link_args=plan.link_args,
        archive_hashes=hashes,
    )


def build_cross_deps(
    *,
    target: Target,
    acados_source_dir: Path,
    build_dir: Path,
    install_prefix: Path,
    toolchain_file: Path | None = None,
    with_qpoases: bool = False,
    jobs: int = DEFAULT_BUILD_JOBS,
    timeout_s: float = 1800.0,
) -> CrossBuiltDeps:
    """Plan + execute the acados dep cross-build in one call.

    Convenience wrapper over :func:`plan_dep_build` +
    :func:`execute_dep_build`. See those for the argument semantics and
    the exceptions raised.
    """
    plan = plan_dep_build(
        target=target,
        acados_source_dir=acados_source_dir,
        build_dir=build_dir,
        install_prefix=install_prefix,
        toolchain_file=toolchain_file,
        with_qpoases=with_qpoases,
        jobs=jobs,
    )
    return execute_dep_build(plan, timeout_s=timeout_s)
