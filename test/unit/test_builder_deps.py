# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the acados-dependency cross-build orchestrator (T-031 follow-up).

The orchestrator splits into a *plan* (pure data; no I/O, no toolchain)
and an *execute* step (cmake configure + build + install against the
cross toolchain). Tier A — what runs on every CI host — covers plan
composition, the blasfeo-target / toolchain-file mappings, and the
link-arg / include-dir derivation. Tier B (skipped unless both the
aarch64 cross toolchain *and* an acados source tree are present) drives
a real cross-build and links a controller against the produced
archives.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from jaxility import (
    DepBuildPlan,
    blasfeo_target_for_family,
    link_args_for_prefix,
    plan_dep_build,
    toolchain_file_for_target,
)
from jaxility.builder_deps import _CORE_ARCHIVES, build_cross_deps
from jaxility.errors import TargetError
from jaxility.targets import PI5, ToolchainPin


def _fake_acados(tmp_path: Path) -> Path:
    """A directory that looks enough like an acados checkout for planning."""
    root = tmp_path / "acados"
    root.mkdir()
    (root / "CMakeLists.txt").write_text("# fake acados root\n")
    return root


# ---------------------------------------------------------------------------
# blasfeo target mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_blasfeo_target_for_cortex_a76() -> None:
    assert blasfeo_target_for_family("cortex-a76") == "ARMV8A_ARM_CORTEX_A76"


@pytest.mark.unit
def test_blasfeo_target_unknown_family_raises() -> None:
    with pytest.raises(TargetError) as exc_info:
        blasfeo_target_for_family("totally-fake-family")
    msg = str(exc_info.value)
    assert "totally-fake-family" in msg
    assert "_BLASFEO_TARGET_FOR_FAMILY" in msg  # points at where to add the row


# ---------------------------------------------------------------------------
# CMake toolchain-file resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_toolchain_file_for_pi5_resolves_to_existing_file() -> None:
    path = toolchain_file_for_target(PI5)
    assert path.exists()
    assert path.name == "aarch64-none-linux-gnu.cmake"


@pytest.mark.unit
def test_toolchain_file_unknown_toolchain_raises() -> None:
    bogus = PI5.model_copy(
        update={
            "toolchain": ToolchainPin(
                **{**PI5.toolchain.model_dump(), "name": "some-other-gcc"}
            )
        }
    )
    with pytest.raises(TargetError, match="no CMake toolchain file"):
        toolchain_file_for_target(bogus)


# ---------------------------------------------------------------------------
# link args / include dirs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_link_args_wrap_archives_in_group_and_order(tmp_path: Path) -> None:
    _, link_args = link_args_for_prefix(tmp_path / "prefix")
    # -L comes first.
    assert link_args[0] == f"-L{(tmp_path / 'prefix' / 'lib').resolve()}"
    # The interdependent archives are wrapped in a linker group, in link
    # order (acados -> hpipm -> blasfeo), and -lm trails the group.
    assert "-Wl,--start-group" in link_args
    assert "-Wl,--end-group" in link_args
    start = link_args.index("-Wl,--start-group")
    end = link_args.index("-Wl,--end-group")
    assert link_args[start + 1 : end] == ("-lacados", "-lhpipm", "-lblasfeo")
    assert link_args[-1] == "-lm"


@pytest.mark.unit
def test_link_args_qpoases_adds_archive_and_stdcxx(tmp_path: Path) -> None:
    _, link_args = link_args_for_prefix(tmp_path / "p", with_qpoases=True)
    start = link_args.index("-Wl,--start-group")
    end = link_args.index("-Wl,--end-group")
    assert "-lqpOASES_e" in link_args[start + 1 : end]
    # qpOASES_e is C++; the C++ runtime is pulled in after the group.
    assert "-lstdc++" in link_args


@pytest.mark.unit
def test_include_dirs_cover_root_and_subdirs(tmp_path: Path) -> None:
    include_dirs, _ = link_args_for_prefix(tmp_path / "p")
    inc_root = (tmp_path / "p" / "include").resolve()
    assert inc_root in include_dirs
    for sub in ("acados", "acados_c", "blasfeo", "hpipm"):
        assert (inc_root / sub).resolve() in include_dirs


# ---------------------------------------------------------------------------
# Plan composition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plan_configure_argv_pins_static_and_blasfeo_target(tmp_path: Path) -> None:
    plan = plan_dep_build(
        target=PI5,
        acados_source_dir=_fake_acados(tmp_path),
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
    )
    argv = plan.configure_argv
    assert "-DBUILD_SHARED_LIBS=OFF" in argv
    assert "-DBLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76" in argv
    assert "-DACADOS_WITH_QPOASES=OFF" in argv
    # Toolchain file routes the cross compiler.
    tc = toolchain_file_for_target(PI5)
    assert f"-DCMAKE_TOOLCHAIN_FILE={tc}" in argv
    assert f"-DACADOS_INSTALL_DIR={(tmp_path / 'prefix').resolve()}" in argv


@pytest.mark.unit
def test_plan_build_argv_installs_in_parallel(tmp_path: Path) -> None:
    plan = plan_dep_build(
        target=PI5,
        acados_source_dir=_fake_acados(tmp_path),
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
        jobs=3,
    )
    assert plan.build_argv[:5] == (
        "cmake",
        "--build",
        str((tmp_path / "build").resolve()),
        "--target",
        "install",
    )
    assert plan.build_argv[-2:] == ("--parallel", "3")


@pytest.mark.unit
def test_plan_archive_paths_in_link_order(tmp_path: Path) -> None:
    plan = plan_dep_build(
        target=PI5,
        acados_source_dir=_fake_acados(tmp_path),
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
    )
    names = [p.name for p in plan.archive_paths]
    assert names == [f"lib{n}.a" for n in _CORE_ARCHIVES]
    lib_dir = (tmp_path / "prefix" / "lib").resolve()
    assert all(p.parent == lib_dir for p in plan.archive_paths)


@pytest.mark.unit
def test_plan_with_qpoases_enables_and_adds_archive(tmp_path: Path) -> None:
    plan = plan_dep_build(
        target=PI5,
        acados_source_dir=_fake_acados(tmp_path),
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
        with_qpoases=True,
    )
    assert "-DACADOS_WITH_QPOASES=ON" in plan.configure_argv
    assert any(p.name == "libqpOASES_e.a" for p in plan.archive_paths)


@pytest.mark.unit
def test_plan_is_deterministic(tmp_path: Path) -> None:
    src = _fake_acados(tmp_path)
    kw = dict(
        target=PI5,
        acados_source_dir=src,
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
    )
    p1 = plan_dep_build(**kw)  # type: ignore[arg-type]
    p2 = plan_dep_build(**kw)  # type: ignore[arg-type]
    assert p1.configure_argv == p2.configure_argv
    assert p1.build_argv == p2.build_argv
    assert p1.link_args == p2.link_args


@pytest.mark.unit
def test_plan_is_frozen(tmp_path: Path) -> None:
    plan = plan_dep_build(
        target=PI5,
        acados_source_dir=_fake_acados(tmp_path),
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
    )
    assert isinstance(plan, DepBuildPlan)
    with pytest.raises(Exception):
        plan.with_qpoases = True  # type: ignore[misc]


@pytest.mark.unit
def test_plan_missing_cmakelists_raises(tmp_path: Path) -> None:
    (tmp_path / "not-acados").mkdir()
    with pytest.raises(TargetError, match="no CMakeLists.txt"):
        plan_dep_build(
            target=PI5,
            acados_source_dir=tmp_path / "not-acados",
            build_dir=tmp_path / "build",
            install_prefix=tmp_path / "prefix",
        )


@pytest.mark.unit
def test_plan_bad_jobs_raises(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="jobs must be"):
        plan_dep_build(
            target=PI5,
            acados_source_dir=_fake_acados(tmp_path),
            build_dir=tmp_path / "build",
            install_prefix=tmp_path / "prefix",
            jobs=0,
        )


@pytest.mark.unit
def test_plan_unknown_family_raises(tmp_path: Path) -> None:
    bogus = PI5.model_copy(update={"family": "totally-fake-family"})
    with pytest.raises(TargetError, match="no blasfeo TARGET"):
        plan_dep_build(
            target=bogus,
            acados_source_dir=_fake_acados(tmp_path),
            build_dir=tmp_path / "build",
            install_prefix=tmp_path / "prefix",
        )


# ---------------------------------------------------------------------------
# Tier B — real cross-build (needs aarch64 toolchain + acados source)
# ---------------------------------------------------------------------------


def _acados_source_dir() -> Path | None:
    env = os.environ.get("JAXILITY_ACADOS_DIR")
    candidates = [Path(env)] if env else []
    candidates.append(Path.home() / "Dev" / "acados")
    for c in candidates:
        if (c / "CMakeLists.txt").exists():
            return c
    return None


_HAVE_AARCH64 = shutil.which("aarch64-none-linux-gnu-gcc") is not None
_HAVE_CMAKE = shutil.which("cmake") is not None
_ACADOS_SRC = _acados_source_dir()
_TIER_B_REASON = (
    "Tier B needs aarch64-none-linux-gnu-gcc + cmake + an acados source "
    "tree (set $JAXILITY_ACADOS_DIR); skips on Apple Silicon dev hosts "
    "and on CI runners without an acados checkout."
)


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAVE_AARCH64 and _HAVE_CMAKE and _ACADOS_SRC is not None),
    reason=_TIER_B_REASON,
)
def test_build_cross_deps_pi5_produces_archives(tmp_path: Path) -> None:
    """Tier B — cross-build the acados deps for Pi 5 and check the archives."""
    assert _ACADOS_SRC is not None
    deps = build_cross_deps(
        target=PI5,
        acados_source_dir=_ACADOS_SRC,
        build_dir=tmp_path / "build",
        install_prefix=tmp_path / "prefix",
    )
    names = {name for name, _ in deps.archive_hashes}
    assert names == {f"lib{n}.a" for n in _CORE_ARCHIVES}
    for _, digest in deps.archive_hashes:
        assert len(digest) == 32  # BLAKE3 default digest length


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAVE_AARCH64 and _HAVE_CMAKE and _ACADOS_SRC is not None),
    reason=_TIER_B_REASON,
)
def test_cross_build_for_target_links_with_deps(tmp_path: Path) -> None:
    """Tier B — the controller links cleanly once the deps are supplied.

    This is the close of the T-031 'Linker gap': with the cross-built
    archives wired through ``deps=``, ``cross_build_for_target`` produces
    a real ELF shared object instead of dying with a ToolchainError at
    the link step.
    """
    pytest.importorskip("acados_template")
    pytest.importorskip("casadi")
    import jax.numpy as jnp  # noqa: PLC0415

    from jaxility import cross_build_for_target
    from jaxility.lowering import translate
    from jaxility.templates import lqr

    assert _ACADOS_SRC is not None
    deps = build_cross_deps(
        target=PI5,
        acados_source_dir=_ACADOS_SRC,
        build_dir=tmp_path / "deps_build",
        install_prefix=tmp_path / "deps_prefix",
    )

    def cartpole(x, u):
        return jnp.stack(
            (x[1], -9.81 * jnp.sin(x[0]) + u[0], x[3], -0.1 * x[3]), axis=0
        )

    cf = translate(cartpole, in_shapes=((4,), (1,)), name="t031_demo")
    spec = lqr(
        cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.1, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="t031_demo",
        horizon_steps=5,
        time_horizon_s=0.5,
    )
    bundle = cross_build_for_target(
        dynamics=cf,
        spec=spec,
        target=PI5,
        source_attestation_handle=bytes(32),
        work_dir=tmp_path / "build",
        deps=deps,
    )
    payload = bundle.shared_library_path.read_bytes()
    assert payload.startswith(b"\x7fELF")
    # Dep provenance traveled into the manifest.
    assert any(k.startswith("dep-archive:") for k in bundle.manifest.toolchain_versions)
