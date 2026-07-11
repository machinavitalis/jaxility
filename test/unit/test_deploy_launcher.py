# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the deployment launcher (T-032).

Three tiers:

1. **Plan composition** (pure, no I/O) — the launcher cross-link argv,
   the POSIX-only capability gate, determinism. Runs everywhere.

2. **Host dlopen end-to-end** — compiles the cortex-a runtime sources +
   ``test_deploy.c`` with the host ``cc``, builds a fake controller
   ``.so``, and runs the launcher harness against it. Verifies the
   dlopen / dlsym / arena-threading / cycle-loop wiring without
   hardware or acados. Gated only on the host compiler.

3. **Tier B real cross-link** — builds the runtime archive then
   cross-links the real ``deploy_main.c`` launcher against it. Gated on
   the aarch64 cross toolchain (CI Linux runners).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from jaxility.errors import TargetError, ToolchainError
from jaxility.runtime import (
    DeployLauncherPlan,
    build_runtime_archive,
    execute_deploy_launcher,
    plan_deploy_launcher,
    runtime_root,
)
from jaxility.targets import CORTEX_M4, PI5

# ---------------------------------------------------------------------------
# Plan composition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plan_argv_links_archive_and_dl_pthread(tmp_path: Path) -> None:
    archive = tmp_path / "libjaxility_runtime_cortex-a76.a"
    archive.write_bytes(b"!<arch>\n")  # plan does no I/O on the archive
    plan = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    assert isinstance(plan, DeployLauncherPlan)
    argv = plan.compile_argv
    assert argv[0] == "aarch64-none-linux-gnu-gcc"
    assert "-rdynamic" in argv  # controller resolves runtime symbols
    assert "-ldl" in argv  # dlopen
    assert "-pthread" in argv  # SCHED_FIFO path in rt_posix
    # ``-shared`` is stripped: the launcher is an executable, not a .so.
    assert "-shared" not in argv
    # The archive comes after the launcher source (satisfies its refs).
    src_idx = argv.index(str(plan.launcher_source))
    ar_idx = argv.index(str(archive.resolve()))
    assert src_idx < ar_idx


@pytest.mark.unit
def test_plan_output_name_defaults_to_family(tmp_path: Path) -> None:
    archive = tmp_path / "rt.a"
    archive.write_bytes(b"!<arch>\n")
    plan = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    assert plan.output_path.name == "jx_deploy_cortex-a76"


@pytest.mark.unit
def test_plan_includes_runtime_include_dir(tmp_path: Path) -> None:
    archive = tmp_path / "rt.a"
    archive.write_bytes(b"!<arch>\n")
    plan = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    assert f"-I{runtime_root() / 'include'}" in plan.compile_argv


@pytest.mark.unit
def test_plan_is_deterministic(tmp_path: Path) -> None:
    archive = tmp_path / "rt.a"
    archive.write_bytes(b"!<arch>\n")
    p1 = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    p2 = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    assert p1.compile_argv == p2.compile_argv


@pytest.mark.unit
def test_plan_is_frozen(tmp_path: Path) -> None:
    archive = tmp_path / "rt.a"
    archive.write_bytes(b"!<arch>\n")
    plan = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    with pytest.raises(Exception):
        plan.target = CORTEX_M4  # type: ignore[misc]


@pytest.mark.unit
def test_plan_cortex_m_rejected_no_dlopen(tmp_path: Path) -> None:
    """Bare-metal Cortex-M has no dlopen — the launcher is POSIX-only."""
    archive = tmp_path / "rt.a"
    archive.write_bytes(b"!<arch>\n")
    with pytest.raises(TargetError, match="unsupported for target family"):
        plan_deploy_launcher(
            target=CORTEX_M4, runtime_archive=archive, work_dir=tmp_path
        )


# ---------------------------------------------------------------------------
# execute — missing toolchain / archive raise loud
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_missing_archive_raises(tmp_path: Path) -> None:
    archive = tmp_path / "nonexistent.a"
    # Build the plan with a placeholder that exists, then point execute
    # at a missing archive by removing it.
    archive.write_bytes(b"!<arch>\n")
    plan = plan_deploy_launcher(target=PI5, runtime_archive=archive, work_dir=tmp_path)
    archive.unlink()
    if shutil.which("aarch64-none-linux-gnu-gcc") is None:
        # Toolchain check fires first; either error is acceptable, both
        # are ToolchainError.
        with pytest.raises(ToolchainError):
            execute_deploy_launcher(plan)
    else:
        with pytest.raises(ToolchainError, match="does not exist"):
            execute_deploy_launcher(plan)


# ---------------------------------------------------------------------------
# Host dlopen end-to-end — runtime-c/test/test_deploy.c via host cc
# ---------------------------------------------------------------------------

_FAKE_CONTROLLER_C = """\
/* Fake controller .so for the deploy host test. Reads the public arena
 * struct to prove the arena was threaded in; stops after a few steps. */
#include "jaxility_runtime/arena.h"
#include <stdint.h>

static int calls = 0;

int jx_controller_init(jx_arena_t *arena) {
    return (arena != 0 && arena->base != 0 && arena->capacity > 0) ? 0 : 1;
}

int jx_controller_step(void) {
    calls += 1;
    return (calls >= 5) ? 1 : 0; /* non-zero => clean stop */
}

uint64_t jx_controller_period_ns(void) { return 200000ULL; } /* 0.2 ms */
"""


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc")
def test_deploy_glue_host_end_to_end(tmp_path: Path) -> None:
    """Build a fake controller .so + the launcher harness and run them.

    Exercises dlopen, dlsym of the three ABI symbols, arena threading,
    the cycle loop, and the clean-stop path — all on the host, no
    hardware. The cortex-a runtime sources compile and run on the dev
    host (POSIX) the same way they do on the Pi.
    """
    root = runtime_root()
    inc = root / "include"

    # 1. Fake controller .so.
    controller_c = tmp_path / "fake_controller.c"
    controller_c.write_text(_FAKE_CONTROLLER_C)
    controller_so = tmp_path / "fake_controller.so"
    build_so = subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            f"-I{inc}",
            str(controller_c),
            "-o",
            str(controller_so),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert build_so.returncode == 0, build_so.stderr

    # 2. Launcher harness (test_deploy.c) over the cortex-a runtime srcs.
    harness = tmp_path / "test_deploy"
    build_harness = subprocess.run(
        [
            "cc",
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-pthread",
            f"-I{inc}",
            str(root / "src" / "arena.c"),
            str(root / "src" / "cycle_posix.c"),
            str(root / "src" / "rt_posix.c"),
            str(root / "src" / "deploy_posix.c"),
            str(root / "test" / "test_deploy.c"),
            "-o",
            str(harness),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert build_harness.returncode == 0, build_harness.stderr

    # 3. Run the harness against the fake controller.
    run = subprocess.run(
        [str(harness), str(controller_so)], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "deploy test passed." in run.stdout


# ---------------------------------------------------------------------------
# Tier B — real cross-link of the launcher against the runtime archive.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("aarch64-none-linux-gnu-gcc") is None,
    reason="aarch64-none-linux-gnu-gcc not on PATH; Pi 5 Tier-B runs in CI",
)
def test_cross_link_launcher_pi5(tmp_path: Path) -> None:
    """Tier B — build the runtime archive then cross-link the launcher."""
    archive = build_runtime_archive(target=PI5, work_dir=tmp_path / "rt")
    plan = plan_deploy_launcher(
        target=PI5, runtime_archive=archive.archive_path, work_dir=tmp_path / "out"
    )
    launcher = execute_deploy_launcher(plan)
    assert launcher.launcher_path.exists()
    payload = launcher.launcher_path.read_bytes()
    assert payload.startswith(b"\x7fELF")  # aarch64 ELF executable
    assert len(launcher.content_hash) == 32  # BLAKE3
