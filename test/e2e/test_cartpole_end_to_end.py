# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Smoke test for ``examples/cartpole_end_to_end.py`` — the T-102 host dry-run.

Exercises the integrated stack the example demonstrates: a calibrated Jaxterity
Robot → export → host build → closed-loop HIL parity → attestation manifest →
the T-101 "recalibration moves the deployed artifact" invariant.

Gated like the T-034 controller-HIL test (acados ``t_renderer`` + host ``cc``);
``test/conftest.py`` puts the acados shared libs on ``DYLD_LIBRARY_PATH`` so this
runs wherever the controller HIL test runs. Marked ``slow`` (two acados builds).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("jaxterity")
_jt_cartpole = pytest.importorskip("jaxterity.zoo.cartpole")
if not hasattr(_jt_cartpole, "reduced_params"):
    pytest.skip(
        "installed jaxterity predates T-101 (no reduced_params)",
        allow_module_level=True,
    )


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_NEEDS_BUILD = pytest.mark.skipif(
    not _tera_available() or shutil.which("cc") is None,
    reason="acados t_renderer + host cc required for the end-to-end build",
)

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "cartpole_end_to_end.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("cartpole_e2e_demo", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass needs the module registered
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


def _calibrated_cartpole():
    import jaxterity.zoo as zoo

    return zoo.load("cartpole").with_provenance(
        ("t102-e2e-test", "v0", "telemetry-hash"), calibrated=True
    )


@_NEEDS_BUILD
@pytest.mark.slow
def test_end_to_end_builds_runs_and_is_robot_rooted(demo, tmp_path) -> None:
    """The compiled controller stabilises the robot-faithful plant, and the
    manifest is rooted on the robot's real attestation handle."""
    robot = _calibrated_cartpole()
    built = demo.build_controller(robot, tmp_path / "nominal", "cartpole_e2e_test")

    # The compiled controller passes step-locked closed-loop HIL parity.
    report = demo.hil_parity(built, n_steps=20)
    assert report.passed

    # The manifest chain verifies and is rooted on the robot's real handle.
    assert demo.verify_manifest(built.bundle.manifest).ok
    assert built.bundle.manifest.source_attestation_handle == bytes.fromhex(
        robot.attestation_handle
    )

    # The solve-time bench returns a 1 kHz verdict (host indicator).
    bench = demo.bench_host(built.bundle.solver, n_cycles=50)
    assert isinstance(bench["meets_1khz"], bool)
    assert bench["mean_us"] > 0.0


@_NEEDS_BUILD
@pytest.mark.slow
def test_recalibration_moves_the_deployed_artifact(demo, tmp_path) -> None:
    """T-101 through the full pipeline: a heavier pole → a different binary.

    The deployed artifact tracks calibration, not just the attestation handle —
    the whole point of sourcing the closed-form scalars from the robot.
    """
    robot = _calibrated_cartpole()
    heavy = robot.with_parameters({"pole.mass": robot.parameters()["pole.mass"] * 2.0})
    assert robot.attestation_handle != heavy.attestation_handle

    nominal = demo.build_controller(robot, tmp_path / "n", "cartpole_e2e_n")
    heavier = demo.build_controller(heavy, tmp_path / "h", "cartpole_e2e_h")
    assert nominal.bundle.artifact.content_hash != heavier.bundle.artifact.content_hash
