# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Unitree G1 zoo entry — the first branched, generator-sourced robot (T-126).

Verifies the entry is registered with generator-sourced dynamics, that the
factory produces a valid 29-DoF branched ``CasadiFunction``, that it threads
through the WBC template into an ``AcadosOcp``, and (slow) that
``jaxility build unitree_g1`` compiles a real artifact end-to-end.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from jaxility.cli import main as cli_main
from jaxility.zoo import available, load


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA = _tera_available()


@pytest.mark.unit
def test_unitree_g1_registered_as_generator_sourced() -> None:
    """G1 is in the zoo with a casadi_dynamics_factory (not a JAX factory)."""
    assert "unitree_g1" in available()
    entry = load("unitree_g1")
    assert entry.upstream_status == "real-robot"
    assert entry.template == "WBC"
    assert entry.casadi_dynamics_factory is not None
    assert entry.jax_dynamics_factory is None  # generator-sourced, not JAX


@pytest.mark.unit
def test_unitree_g1_factory_produces_branched_dynamics() -> None:
    """The factory generates a 29-DoF (branched) CasadiFunction with an audit trail."""
    pytest.importorskip("mujoco")
    pytest.importorskip("jaxterity")
    entry = load("unitree_g1")
    dyn = entry.casadi_dynamics_factory()
    assert dyn.input_shapes == ((58,), (29,))  # nx = 2*29, nu = 29
    assert dyn.output_shapes == ((58,),)
    assert {"casadi:sin", "casadi:cos", "casadi:mul"} <= dyn.primitives_used


@pytest.mark.unit
def test_unitree_g1_threads_into_wbc_ocp() -> None:
    """The generated dynamics + WBC template build a valid AcadosOcp (no compile)."""
    pytest.importorskip("mujoco")
    pytest.importorskip("jaxterity")
    pytest.importorskip("acados_template")
    from jaxility.cli.build_cmd import _select_template_spec
    from jaxility.lowering import build_ocp

    entry = load("unitree_g1")
    dyn = entry.casadi_dynamics_factory()
    spec = _select_template_spec(entry, dyn)
    assert len(spec.state_cost) == 58 and len(spec.input_cost) == 29
    ocp = build_ocp(dyn, spec)
    assert ocp.model.f_expl_expr is not None


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_unitree_g1_builds_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``jaxility build unitree_g1`` compiles the branched humanoid to an artifact.

    Slow: acados codegen + compile of a 58-state / 29-input OCP takes minutes.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("jaxterity")
    exit_code = cli_main(
        ["build", "unitree_g1", "--target", "host", "--work-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    report = json.loads(captured.out.splitlines()[-1])
    assert report["ok"] is True
    assert report["zoo_name"] == "unitree_g1"
    assert report["payload_bytes"] > 0
    assert Path(report["shared_library_path"]).exists()
    assert len(report["manifest_content_hash_hex"]) == 64
