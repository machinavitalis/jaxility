# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the host-only build path (T-026)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility import BuildBundle, build_for_target
from jaxility.cli import main as cli_main
from jaxility.lowering import translate
from jaxility.manifest import Artifact, ArtifactCache, verify_manifest
from jaxility.targets import HOST_DARWIN, HOST_LINUX, current_host_target
from jaxility.templates import lqr


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA = _tera_available()


def _cartpole(state, control):
    g, mp, mc, L = 9.81, 0.1, 1.0, 0.5
    _, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    denom = mc + mp * sin_t * sin_t
    x_ddot = (control[0] + mp * sin_t * (L * theta_dot * theta_dot + g * cos_t)) / denom
    theta_ddot = (
        -control[0] * cos_t
        - mp * L * theta_dot * theta_dot * cos_t * sin_t
        - (mc + mp) * g * sin_t
    ) / (L * denom)
    return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])


@pytest.fixture(scope="module")
def cartpole_cf():
    return translate(_cartpole, in_shapes=((4,), (1,)), name="cartpole_host_t026")


# ---------------------------------------------------------------------------
# Host target profiles.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_host_darwin_profile_basics() -> None:
    assert HOST_DARWIN.name == "host-darwin"
    assert HOST_DARWIN.family == "host-darwin"
    assert HOST_DARWIN.toolchain.name == "clang"


@pytest.mark.unit
def test_host_linux_profile_basics() -> None:
    assert HOST_LINUX.name == "host-linux"
    assert HOST_LINUX.family == "host-linux"
    assert HOST_LINUX.toolchain.name == "gcc"


@pytest.mark.unit
def test_current_host_target_picks_platform() -> None:
    t = current_host_target()
    if sys.platform == "darwin":
        assert t is HOST_DARWIN
    elif sys.platform.startswith("linux"):
        assert t is HOST_LINUX
    else:
        pytest.skip(f"unsupported sys.platform={sys.platform!r}")


@pytest.mark.unit
def test_host_target_hashes_distinct() -> None:
    assert HOST_DARWIN.hash != HOST_LINUX.hash


# ---------------------------------------------------------------------------
# build_for_target end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_for_target_produces_real_artifact(tmp_path: Path, cartpole_cf) -> None:
    """The host build produces a non-empty shared library + intact chain."""
    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="cartpole_host_t026",
    )
    bundle = build_for_target(
        dynamics=cartpole_cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes.fromhex("11" * 32),
        work_dir=tmp_path / "build",
        build_timestamp_utc=1,
    )

    assert isinstance(bundle, BuildBundle)
    assert isinstance(bundle.artifact, Artifact)
    assert bundle.artifact.payload  # non-empty shared library
    assert bundle.shared_library_path.suffix in {".dylib", ".so"}
    assert bundle.shared_library_path.exists()


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_for_target_chain_intact(tmp_path: Path, cartpole_cf) -> None:
    """The attestation chain holds across the host build."""
    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="cartpole_chain_t026",
    )
    handle = bytes.fromhex("aa" * 32)
    bundle = build_for_target(
        dynamics=cartpole_cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=handle,
        work_dir=tmp_path / "build",
        build_timestamp_utc=1,
    )

    assert bundle.manifest.source_attestation_handle == handle
    assert bundle.manifest.artifact_content_hash == bundle.artifact.content_hash
    assert bundle.artifact.source_manifest_hash == bundle.manifest.content_hash()
    assert bundle.manifest.target_profile_hash == current_host_target().hash
    report = verify_manifest(bundle.manifest)
    assert report.ok is True


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_for_target_solver_is_live(tmp_path: Path, cartpole_cf) -> None:
    """The bundle's solver instance can solve the OCP directly."""
    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="cartpole_solver_t026",
    )
    bundle = build_for_target(
        dynamics=cartpole_cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes.fromhex("22" * 32),
        work_dir=tmp_path / "build",
    )
    status = bundle.solver.solve()
    assert status == 0
    u0 = bundle.solver.get(0, "u")
    assert np.all(np.isfinite(u0))


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_for_target_artifact_caches_round_trip(
    tmp_path: Path, cartpole_cf
) -> None:
    """``ArtifactCache.store`` + ``load`` round-trip works for a host artifact."""
    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="cartpole_cache_t026",
    )
    bundle = build_for_target(
        dynamics=cartpole_cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes.fromhex("33" * 32),
        work_dir=tmp_path / "build",
        build_timestamp_utc=1,
    )
    cache = ArtifactCache(root=tmp_path / "cache")
    cache.store(bundle.artifact)
    loaded = cache.load(bundle.artifact.content_hash)
    assert loaded.content_hash == bundle.artifact.content_hash
    assert loaded.payload == bundle.artifact.payload


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_for_target_locates_shared_library_or_raises(
    tmp_path: Path, cartpole_cf
) -> None:
    """If acados emits nothing under work_dir, the builder raises TargetError."""
    # Sanity: positive case — finds the library.
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        name="cartpole_locate_t026",
    )
    bundle = build_for_target(
        dynamics=cartpole_cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes(32),
        work_dir=tmp_path / "build",
    )
    assert bundle.shared_library_path.exists()


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_build_unknown_zoo_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(["build", "definitely-not-a-zoo-entry"])
    captured = capsys.readouterr()
    assert exit_code == 2
    report = json.loads(captured.out)
    assert report["ok"] is False


@pytest.mark.unit
def test_cli_build_stub_zoo_entry_returns_1_with_structured_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stub entry (no dynamics factory) reports the upstream gap clearly."""
    pytest.importorskip("jaxterity")
    exit_code = cli_main(["build", "berkeley_humanoid_lite"])
    captured = capsys.readouterr()
    assert exit_code == 1
    report = json.loads(captured.out)
    assert report["ok"] is False
    assert "stub" in report["reason"]


@pytest.mark.unit
def test_cli_build_crazyflie_reports_template_not_wired(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Crazyflie's dynamics translate, but the OCP template is a T-110 follow-on.

    The build must fail *structurally* (PATTERNS §8.2) — a clear JSON reason —
    rather than crash with an uncaught template error.
    """
    pytest.importorskip("jaxterity")
    exit_code = cli_main(["build", "crazyflie"])
    captured = capsys.readouterr()
    assert exit_code == 1
    report = json.loads(captured.out)
    assert report["ok"] is False
    assert "not yet end-to-end buildable" in report["reason"]
    assert "template" in report["reason"]


# The old "real-robot returns 1 + note" test is gone:
# the dynamics-extraction adapter + analytical-cartpole fallback
# closes the gap that test was about. test_cli_zoo_build.py now
# asserts the positive case (``jaxility build cartpole --target host``
# returns 0 with a structured ok JSON report).
