# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the CLI ↔ zoo dynamics-extraction adapter."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.cli import main as cli_main


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA = _tera_available()


# ---------------------------------------------------------------------------
# Dynamics adapter (no Jaxterity needed beyond the import skip).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_jax_dynamics_from_robot_returns_jit_traceable_callable() -> None:
    """The adapter produces a JAX-traceable ``f(state, control) -> dx``."""
    pytest.importorskip("jaxterity")
    from jaxterity.zoo import load

    from jaxility.cli.dynamics_adapter import jax_dynamics_from_robot

    robot = load("cartpole")
    jax_fn, state_shape, control_shape = jax_dynamics_from_robot(
        robot, q0=[0.0, 0.0], qd0=[0.0, 0.0], nu=1
    )

    assert state_shape == (4,)
    assert control_shape == (1,)

    x = jnp.array([0.3, 0.0, 0.0, 0.0])
    u = jnp.array([1.0])

    direct = jax_fn(x, u)
    assert direct.shape == (4,)
    assert np.all(np.isfinite(np.asarray(direct)))

    jitted = jax.jit(jax_fn)
    jit_result = jitted(x, u)
    np.testing.assert_allclose(np.asarray(direct), np.asarray(jit_result))


# ---------------------------------------------------------------------------
# Zoo entries expose the dynamics factory + template_options.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cartpole_entry_has_dynamics_factory_and_options() -> None:
    pytest.importorskip("jaxterity")
    from jaxility.zoo import load

    entry = load("cartpole")
    assert entry.jax_dynamics_factory is not None
    opts = entry.template_options
    assert "Q" in opts
    assert "R" in opts
    assert "initial_state" in opts


@pytest.mark.unit
def test_crazyflie_entry_has_dynamics_factory() -> None:
    """Crazyflie ships a closed-form Newton-Euler factory (T-110)."""
    pytest.importorskip("jaxterity")
    from jaxility.zoo import load

    entry = load("crazyflie")
    assert entry.jax_dynamics_factory is not None
    dynamics, state_shape, control_shape = entry.jax_dynamics_factory()
    assert state_shape == (13,)
    assert control_shape == (4,)


@pytest.mark.unit
def test_so100_entry_has_no_dynamics_factory() -> None:
    """SO-100 surfaces the MJX gap structurally — no factory."""
    pytest.importorskip("jaxterity")
    from jaxility.zoo import load

    entry = load("so100")
    assert entry.jax_dynamics_factory is None


@pytest.mark.unit
def test_stub_entries_have_no_dynamics_factory() -> None:
    pytest.importorskip("jaxterity")
    from jaxility.zoo import load

    for name in ("berkeley_humanoid_lite",):
        assert load(name).jax_dynamics_factory is None


# ---------------------------------------------------------------------------
# CLI: real-robot path now compiles end-to-end on host.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_cli_build_cartpole_host_succeeds_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``jaxility build cartpole --target host`` produces a real artifact."""
    pytest.importorskip("jaxterity")
    exit_code = cli_main(
        ["build", "cartpole", "--target", "host", "--work-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.out
    report = json.loads(captured.out)
    assert report["ok"] is True
    assert report["zoo_name"] == "cartpole"
    assert report["target"] in ("host-darwin", "host-linux")
    assert report["payload_bytes"] > 0
    assert Path(report["shared_library_path"]).exists()
    # Chain hashes are 64-char hex BLAKE3 digests.
    assert len(report["artifact_content_hash_hex"]) == 64
    assert len(report["manifest_content_hash_hex"]) == 64


# ---------------------------------------------------------------------------
# Audit M-6: CLI emits a self-explaining marker when the Source factory
# fails instead of silently forging a zero attestation handle.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_cli_build_surfaces_source_factory_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M-6: if ``source_factory()`` raises, the CLI emits a warning,
    derives the attestation handle from a documented marker, and does NOT
    silently zero the handle.
    """
    pytest.importorskip("jaxterity")
    import dataclasses

    from jaxility import zoo

    original = zoo.load("cartpole")

    def boom() -> object:
        raise RuntimeError("synthetic source failure")

    broken = dataclasses.replace(original, source_factory=boom)
    monkeypatch.setattr(
        zoo, "load", lambda name: broken if name == "cartpole" else original
    )

    exit_code = cli_main(
        ["build", "cartpole", "--target", "host", "--work-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out

    assert exit_code == 0, out
    # The marker travels through the CLI's NDJSON stream as a warning row.
    assert "source-unavailable:source_factory" in out
    assert "synthetic source failure" in out

    # The handle is non-zero: it is BLAKE3 of the marker, NOT bytes(32).
    final_report = json.loads(out.splitlines()[-1])
    assert final_report["ok"] is True
    assert len(final_report["source_attestation_handle_hex"]) == 64
    assert final_report["source_attestation_handle_hex"] != "00" * 32


@pytest.mark.unit
def test_cli_build_surfaces_failing_jax_dynamics_factory(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit N-5: when a zoo entry's ``jax_dynamics_factory()`` raises,
    the CLI emits a structured ``{"ok": false, "reason": "failed to
    extract JAX dynamics: ..."}`` report and returns exit code 1.

    Before this test the branch carried a ``# pragma: no cover``
    marker; the pragma is gone and the branch is now exercised.
    """
    pytest.importorskip("jaxterity")
    import dataclasses

    from jaxility import zoo

    original = zoo.load("cartpole")

    def boom_factory() -> object:
        raise RuntimeError("synthetic JAX factory failure")

    broken = dataclasses.replace(original, jax_dynamics_factory=boom_factory)
    monkeypatch.setattr(
        zoo, "load", lambda name: broken if name == "cartpole" else original
    )

    exit_code = cli_main(["build", "cartpole", "--target", "host"])
    out = capsys.readouterr().out

    assert exit_code == 1, out
    report = json.loads(out)
    assert report["ok"] is False
    assert "failed to extract JAX dynamics" in report["reason"]
    assert "synthetic JAX factory failure" in report["reason"]


@pytest.mark.unit
def test_cli_build_so100_returns_structured_no_factory_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SO-100 has no dynamics factory; CLI surfaces the gap clearly."""
    pytest.importorskip("jaxterity")
    exit_code = cli_main(["build", "so100", "--target", "host"])
    captured = capsys.readouterr()

    assert exit_code == 1
    report = json.loads(captured.out)
    assert report["ok"] is False
    assert "jax_dynamics_factory" in report["reason"]


@pytest.mark.unit
def test_cli_build_crazyflie_reports_template_not_wired(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Crazyflie's closed-form dynamics translate, but its OCP template is a
    T-110 follow-on, so the build fails with a clear structured reason."""
    pytest.importorskip("jaxterity")
    exit_code = cli_main(["build", "crazyflie", "--target", "host"])
    captured = capsys.readouterr()

    assert exit_code == 1
    report = json.loads(captured.out)
    assert report["ok"] is False
    assert "not yet end-to-end buildable" in report["reason"]


@pytest.mark.unit
def test_cli_build_unknown_zoo_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(["build", "definitely-not-an-entry"])
    captured = capsys.readouterr()
    assert exit_code == 2
    report = json.loads(captured.out)
    assert report["ok"] is False
