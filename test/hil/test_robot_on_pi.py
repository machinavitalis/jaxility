# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""On-silicon parity for the **robot-rooted** controller (T-102 hardware finish).

Mirrors ``test_on_pi_controller.py`` but builds the controller from a calibrated
Jaxterity Robot — dynamics via ``jaxterity.zoo.cartpole.reduced_params`` and the
manifest rooted on the robot's real ``attestation_handle`` (T-101) — using the
``examples/cartpole_end_to_end.py`` builders, ships it to a tethered Pi 5, builds
it natively on the Cortex-A76, and asserts on-Pi HIL parity + the 1 kHz budget.

Opt-in and self-skipping: requires the host acados ``t_renderer``,
``JAXILITY_HIL_SSH_HOST`` (e.g. ``pi@raspberrypi.local``), and a target acados
(``JAXILITY_HIL_ACADOS``, default ``$HOME/acados``). A no-op in hardware-free CI.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
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

_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")
_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "cartpole_end_to_end.py"


def _tera() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    return bool(root and (Path(root) / "bin" / "t_renderer").exists()) or (
        shutil.which("t_renderer") is not None
    )


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *_SSH_OPTS, host, cmd], capture_output=True, text=True, timeout=15
    )


def _load_demo():
    spec = importlib.util.spec_from_file_location("cartpole_e2e_demo", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass needs the module registered
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


@pytest.mark.hil
def test_robot_rooted_controller_on_pi(demo, tmp_path) -> None:
    """The controller built from a calibrated Robot runs on real Cortex-A76
    silicon: on-Pi HIL parity passes and the solve meets 1 kHz."""
    if not _tera():
        pytest.skip("host acados t_renderer required to build the controller")
    host = os.environ.get("JAXILITY_HIL_SSH_HOST")
    if not host:
        pytest.skip("JAXILITY_HIL_SSH_HOST not set; on-Pi robot tier skipped")
    if _ssh(host, "true").returncode != 0:
        pytest.skip(f"HIL ssh host {host!r} not reachable")

    import jaxterity.zoo as zoo

    robot = zoo.load("cartpole").with_provenance(
        ("t102-onpi-test", "v0", "telemetry-hash"), calibrated=True
    )
    built = demo.build_controller(robot, tmp_path / "build", "cartpole_robot_onpi_test")

    # The manifest the artifact carries is rooted on *this* robot's handle.
    assert built.bundle.manifest.source_attestation_handle == bytes.fromhex(
        robot.attestation_handle
    )

    report, record = demo.run_on_pi(built, host, n_steps=40, n_cycles=500)
    assert report.passed
    assert report.runner_label.startswith("ssh:")
    assert record.meets_1khz, (
        f"worst-case on-Pi solve {record.solve.max_ns / 1e3:.1f} us exceeds the "
        f"1 kHz budget"
    )
