# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the on-target C runtime build orchestrator (T-032 / T-052).

Three tiers, increasing in dependency surface:

1. **Plan composition** (pure, no I/O) — runs everywhere. Tests the
   per-family source filter, ar-binary derivation, argv composition,
   determinism, and the loud-fail surface (unknown family).

2. **Host arena unit tests** (runs ``test/test_arena.c`` via the
   host ``cc``) — verifies the portable C contract on every dev
   host. Gated only on the host compiler being present.

3. **Tier B real cross-compile** — drives the actual cross-toolchain
   to produce a real ``libjaxility_runtime_<family>.a``. Gated on
   the toolchain (``arm-none-eabi-gcc`` locally, plus
   ``aarch64-none-linux-gnu-gcc`` on CI Linux runners).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from jaxility.errors import TargetError, ToolchainError
from jaxility.runtime import (
    RuntimeArchive,
    RuntimeBuildPlan,
    build_runtime_archive,
    plan_runtime_build,
    runtime_root,
    runtime_sources_for_family,
)
from jaxility.runtime.c_runtime import _ar_tool_name, _family_group
from jaxility.targets import (
    APPLE_SILICON,
    CORTEX_A55,
    CORTEX_A78,
    CORTEX_M4,
    ETHOS_U55,
    NEOVERSE_N1,
    PI5,
)

# ---------------------------------------------------------------------------
# Layout discovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runtime_root_locates_runtime_c_tree() -> None:
    root = runtime_root()
    assert root.is_dir()
    assert (root / "src").is_dir()
    assert (root / "include" / "jaxility_runtime").is_dir()
    assert (root / "src" / "arena.c").exists()
    assert (root / "include" / "jaxility_runtime" / "arena.h").exists()


# ---------------------------------------------------------------------------
# Family -> source-group mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "target, expected_group",
    [
        (PI5, "cortex-a"),
        (CORTEX_A55, "cortex-a"),
        (CORTEX_A78, "cortex-a"),
        (NEOVERSE_N1, "cortex-a"),
        (CORTEX_M4, "cortex-m"),
        (ETHOS_U55, "cortex-m"),
    ],
)
def test_family_group_dispatch(target, expected_group: str) -> None:
    assert _family_group(target.family) == expected_group


@pytest.mark.unit
def test_family_group_unknown_family_raises() -> None:
    with pytest.raises(TargetError, match="no runtime source group"):
        _family_group("apple-silicon")


@pytest.mark.unit
def test_runtime_sources_for_cortex_a_includes_cycle_and_dcache() -> None:
    sources = runtime_sources_for_family("cortex-a76")
    names = {p.name for p in sources}
    assert names == {
        "arena.c",
        "cycle_posix.c",
        "dcache_aarch64.c",
        "rt_posix.c",
        "deploy_posix.c",
    }


@pytest.mark.unit
def test_runtime_sources_for_cortex_m_skips_posix_scheduler() -> None:
    sources = runtime_sources_for_family("cortex-m4")
    names = {p.name for p in sources}
    assert names == {"arena.c", "dcache_thumb.c"}


# ---------------------------------------------------------------------------
# ar-binary derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ar_tool_name_for_pi5() -> None:
    assert _ar_tool_name(PI5) == "aarch64-none-linux-gnu-ar"


@pytest.mark.unit
def test_ar_tool_name_for_cortex_m4() -> None:
    assert _ar_tool_name(CORTEX_M4) == "arm-none-eabi-ar"


@pytest.mark.unit
def test_ar_tool_name_rejects_non_gcc_suffix() -> None:
    # Apple Silicon's toolchain is "clang" — no -gcc suffix to swap.
    with pytest.raises(ToolchainError, match="does not end with '-gcc'"):
        _ar_tool_name(APPLE_SILICON)


# ---------------------------------------------------------------------------
# Plan composition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plan_for_pi5_carries_aarch64_argv(tmp_path: Path) -> None:
    plan = plan_runtime_build(target=PI5, work_dir=tmp_path)
    assert isinstance(plan, RuntimeBuildPlan)
    # Each compile_argv starts with the cross-gcc and ends with the
    # ``.c`` source.
    for src, argv in zip(plan.sources, plan.compile_argvs):
        assert argv[0] == "aarch64-none-linux-gnu-gcc"
        assert argv[-1] == str(src)
        assert "-c" in argv
        assert "-mcpu=cortex-a76" in argv
        # ``-shared`` must be filtered out: runtime is a static
        # archive, not a shared library.
        assert "-shared" not in argv
    assert plan.ar_argv[0] == "aarch64-none-linux-gnu-ar"
    # M-4: ``rcsD`` deterministic mode zeroes mtime / uid / gid in the
    # member headers so the archive bytes reproduce across builds.
    assert plan.ar_argv[1] == "rcsD"
    assert plan.archive_path.name == "libjaxility_runtime_cortex-a76.a"


@pytest.mark.unit
def test_plan_for_cortex_m4_includes_thumb_and_c_only(tmp_path: Path) -> None:
    plan = plan_runtime_build(target=CORTEX_M4, work_dir=tmp_path)
    for argv in plan.compile_argvs:
        assert argv[0] == "arm-none-eabi-gcc"
        assert "-mcpu=cortex-m4" in argv
        assert "-mthumb" in argv
        assert "-c" in argv
    assert plan.ar_argv[0] == "arm-none-eabi-ar"


@pytest.mark.unit
def test_plan_includes_runtime_include_dir(tmp_path: Path) -> None:
    plan = plan_runtime_build(target=PI5, work_dir=tmp_path)
    include_flag = f"-I{runtime_root() / 'include'}"
    assert all(include_flag in argv for argv in plan.compile_argvs)


@pytest.mark.unit
def test_plan_is_deterministic(tmp_path: Path) -> None:
    p1 = plan_runtime_build(target=PI5, work_dir=tmp_path)
    p2 = plan_runtime_build(target=PI5, work_dir=tmp_path)
    assert p1.compile_argvs == p2.compile_argvs
    assert p1.ar_argv == p2.ar_argv
    assert p1.archive_path == p2.archive_path


@pytest.mark.unit
def test_plan_unknown_family_raises(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="no runtime source group"):
        plan_runtime_build(target=APPLE_SILICON, work_dir=tmp_path)


@pytest.mark.unit
def test_plan_object_paths_under_work_dir(tmp_path: Path) -> None:
    plan = plan_runtime_build(target=PI5, work_dir=tmp_path)
    for obj in plan.object_paths:
        assert obj.parent == tmp_path.resolve()
        assert obj.suffix == ".o"


# ---------------------------------------------------------------------------
# build_runtime_archive — missing toolchain raises loud, not silent
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_missing_toolchain_raises_toolchain_error(tmp_path: Path) -> None:
    """If the cross-toolchain isn't on PATH, the wrapper raises ToolchainError."""
    if shutil.which("aarch64-none-linux-gnu-gcc") is not None:
        pytest.skip("aarch64 toolchain on PATH; verifying the *missing* branch")
    with pytest.raises(ToolchainError) as exc_info:
        build_runtime_archive(target=PI5, work_dir=tmp_path)
    msg = str(exc_info.value)
    assert "aarch64-none-linux-gnu-gcc" in msg
    assert "developer.arm.com" in msg


# ---------------------------------------------------------------------------
# Host arena unit tests — runs runtime-c/test/test_arena.c via host cc
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc")
def test_arena_host_unit_tests_pass(tmp_path: Path) -> None:
    """Build and run runtime-c/test/test_arena.c with the host compiler."""
    root = runtime_root()
    bin_path = tmp_path / "test_arena"
    compile_argv = [
        "cc",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        f"-I{root / 'include'}",
        str(root / "src" / "arena.c"),
        str(root / "test" / "test_arena.c"),
        "-o",
        str(bin_path),
    ]
    compiled = subprocess.run(compile_argv, capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr

    run = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all 13 tests passed" in run.stdout


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc")
def test_cycle_posix_host_unit_tests_pass(tmp_path: Path) -> None:
    """Audit M-8: build and run cycle_posix.c tests with the host cc."""
    root = runtime_root()
    bin_path = tmp_path / "test_cycle_posix"
    compile_argv = [
        "cc",
        "-std=c99",
        "-D_POSIX_C_SOURCE=200809L",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        f"-I{root / 'include'}",
        str(root / "src" / "arena.c"),
        str(root / "src" / "cycle_posix.c"),
        str(root / "test" / "test_cycle_posix.c"),
        "-o",
        str(bin_path),
    ]
    compiled = subprocess.run(compile_argv, capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all 12 tests passed" in run.stdout


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc")
def test_rt_host_unit_tests_pass(tmp_path: Path) -> None:
    """T-032: build and run rt_posix.c tests with the host cc.

    On Linux (CI x86_64; the Pi 5 production path) the real
    sched_setaffinity / SCHED_FIFO calls run; the C test accepts
    {OK, PRIVILEGE} so it passes whether or not the runner grants
    CAP_SYS_NICE. On macOS the functions return JX_RT_ERR_UNSUPPORTED
    and the C test asserts that branch. Either way the binary exits 0
    and prints a "tests passed." line — that's the contract.
    """
    root = runtime_root()
    bin_path = tmp_path / "test_rt"
    compile_argv = [
        "cc",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-pthread",  # pthread_setschedparam on glibc
        f"-I{root / 'include'}",
        str(root / "src" / "rt_posix.c"),
        str(root / "test" / "test_rt.c"),
        "-o",
        str(bin_path),
    ]
    compiled = subprocess.run(compile_argv, capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "tests passed." in run.stdout


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc")
def test_dcache_host_unit_tests_pass(tmp_path: Path) -> None:
    """Audit M-8: build and run dcache tests with the host cc.

    On aarch64 hosts (Apple Silicon, Pi 5, Linux aarch64 runners) the
    aarch64 implementation is exercised end-to-end (4 contract tests).
    On x86_64 hosts the entry point is platform-gated and the main()
    prints a "skipped at C level" line. The Python driver only
    asserts the binary exits 0 in both cases — that's the contract.
    """
    root = runtime_root()
    bin_path = tmp_path / "test_dcache"
    compile_argv = [
        "cc",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        f"-I{root / 'include'}",
        # Always include both implementations; the ``#if`` gates pick
        # the right one for the host arch (and emit nothing on x86_64).
        str(root / "src" / "dcache_aarch64.c"),
        str(root / "src" / "dcache_thumb.c"),
        str(root / "test" / "test_dcache.c"),
        "-o",
        str(bin_path),
    ]
    compiled = subprocess.run(compile_argv, capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr


# ---------------------------------------------------------------------------
# Tier B — real cross-compiled runtime archive
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("arm-none-eabi-gcc") is None,
    reason="arm-none-eabi-gcc not on PATH; Cortex-M Tier-B runtime build skipped",
)
def test_build_real_cortex_m4_runtime_archive(tmp_path: Path) -> None:
    """Tier B (Cortex-M) — produce a real libjaxility_runtime_cortex-m4.a."""
    import blake3 as _blake3

    result = build_runtime_archive(target=CORTEX_M4, work_dir=tmp_path)
    assert isinstance(result, RuntimeArchive)
    assert result.archive_path.exists()
    assert result.archive_path.name == "libjaxility_runtime_cortex-m4.a"
    assert len(result.content_hash) == 32  # BLAKE3
    # GNU ar archives start with the magic "!<arch>\n".
    assert result.archive_path.read_bytes().startswith(b"!<arch>\n")
    # Each declared object should exist on disk.
    for obj in result.object_paths:
        assert obj.exists()
    # Audit M-11: content_hash is actually BLAKE3 of the archive bytes,
    # not a random or hardcoded value.
    assert (
        result.content_hash == _blake3.blake3(result.archive_path.read_bytes()).digest()
    )


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("arm-none-eabi-gcc") is None,
    reason="arm-none-eabi-gcc not on PATH; Cortex-M Tier-B runtime build skipped",
)
def test_cortex_m4_runtime_archive_is_byte_deterministic(tmp_path: Path) -> None:
    """Audit M-4: two builds of the same sources produce a byte-identical archive.

    invariant 5 wired all the way through: ``ar -D`` zeroes the
    member-header timestamps so the produced archive bytes (and
    therefore the BLAKE3 content_hash) reproduce across runs.
    """
    archive_a = build_runtime_archive(target=CORTEX_M4, work_dir=tmp_path / "a")
    archive_b = build_runtime_archive(target=CORTEX_M4, work_dir=tmp_path / "b")
    assert archive_a.archive_path.read_bytes() == archive_b.archive_path.read_bytes()
    assert archive_a.content_hash == archive_b.content_hash


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("aarch64-none-linux-gnu-gcc") is None,
    reason=(
        "aarch64-none-linux-gnu-gcc not on PATH; Pi 5 Tier-B runtime build "
        "runs in CI (Linux x86_64 runners; no darwin-arm64 host build "
        "in Arm GNU 15.2.Rel1)."
    ),
)
def test_build_real_pi5_runtime_archive(tmp_path: Path) -> None:
    """Tier B (Pi 5) — produce a real libjaxility_runtime_cortex-a76.a."""
    import blake3 as _blake3

    result = build_runtime_archive(target=PI5, work_dir=tmp_path)
    assert isinstance(result, RuntimeArchive)
    assert result.archive_path.exists()
    assert result.archive_path.name == "libjaxility_runtime_cortex-a76.a"
    assert len(result.content_hash) == 32
    assert result.archive_path.read_bytes().startswith(b"!<arch>\n")
    for obj in result.object_paths:
        assert obj.exists()
    # Audit M-11: content_hash is actually BLAKE3 of the archive bytes.
    assert (
        result.content_hash == _blake3.blake3(result.archive_path.read_bytes()).digest()
    )


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("aarch64-none-linux-gnu-gcc") is None,
    reason="aarch64-none-linux-gnu-gcc not on PATH; determinism check runs in CI",
)
def test_pi5_runtime_archive_is_deterministic(tmp_path: Path) -> None:
    """Audit M-4: same inputs -> byte-identical archive AND content_hash.

    invariant 5 closure. ``ar -D`` deterministic mode zeroes the
    member-header timestamps; the resulting archive bytes (and
    therefore the BLAKE3 content_hash) reproduce across builds.
    """
    work_a = tmp_path / "build-a"
    work_b = tmp_path / "build-b"
    archive_a = build_runtime_archive(target=PI5, work_dir=work_a)
    archive_b = build_runtime_archive(target=PI5, work_dir=work_b)
    assert archive_a.archive_path.read_bytes() == archive_b.archive_path.read_bytes()
    assert archive_a.content_hash == archive_b.content_hash
