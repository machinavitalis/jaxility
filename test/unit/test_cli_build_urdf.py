# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""``jaxility build-urdf`` — build directly from a URDF/MJCF (T-124).

Covers the arg-handling paths (always run) and the acados-gated end-to-end build
that turns a raw URDF into an attested artifact whose manifest records Pinocchio
provenance. Self-skips without ``pin`` / ``t_renderer``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from jaxility.cli import main as cli_main


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA = _tera_available()


def _link(name: str, com: str, mass: float, i: float) -> str:
    return (
        f'<link name="{name}"><inertial>'
        f'<origin xyz="{com}"/><mass value="{mass}"/>'
        f'<inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/>'
        "</inertial></link>"
    )


def _rev(name: str, parent: str, child: str, xyz: str, axis: str) -> str:
    return (
        f'<joint name="{name}" type="revolute">'
        f'<parent link="{parent}"/><child link="{child}"/>'
        f'<origin xyz="{xyz}"/><axis xyz="{axis}"/>'
        '<limit lower="-2" upper="2" effort="5" velocity="5"/></joint>'
    )


_TWO_LINK_URDF = f"""<?xml version="1.0"?>
<robot name="twolink">
  {_link("base", "0 0 0", 1.0, 0.01)}
  {_link("l1", "0.1 0 0", 0.5, 0.005)}
  {_link("l2", "0.1 0 0", 0.3, 0.003)}
  {_rev("j1", "base", "l1", "0 0 0.1", "0 0 1")}
  {_rev("j2", "l1", "l2", "0.2 0 0", "0 1 0")}
</robot>
"""

_FLOATING_URDF = f"""<?xml version="1.0"?>
<robot name="floater">
  {_link("base", "0 0 0", 1.0, 0.01)}
  {_link("body", "0 0 0", 0.5, 0.005)}
  <joint name="free" type="floating"><parent link="base"/><child link="body"/></joint>
</robot>
"""


def _write(tmp_path: Path, text: str, name: str = "robot.urdf") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- #
# Arg handling — always run.                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_urdf_missing_file_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(["build-urdf", "/no/such/file.urdf"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["ok"] is False
    assert "no such description file" in report["reason"]


@pytest.mark.unit
def test_build_urdf_format_inference() -> None:
    from jaxility.cli.build_urdf_cmd import _infer_format

    assert _infer_format(Path("arm.urdf"), None) == "urdf"
    assert _infer_format(Path("arm.xml"), None) == "mjcf"
    assert _infer_format(Path("arm.mjcf"), None) == "mjcf"
    assert _infer_format(Path("arm.xml"), "urdf") == "urdf"  # explicit wins


@pytest.mark.unit
def test_build_urdf_floating_base_surfaces_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A floating-base description fails loudly with a structured reason (exit 1)."""
    pytest.importorskip("pinocchio")
    src = _write(tmp_path, _FLOATING_URDF, "floater.urdf")
    exit_code = cli_main(["build-urdf", str(src), "--backend", "jaxility-aba"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["ok"] is False
    assert "generate_dynamics failed" in report["reason"]


# --------------------------------------------------------------------------- #
# End-to-end — needs Pinocchio + acados' t_renderer.                            #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_urdf_host_succeeds_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``jaxility build-urdf <2-link.urdf>`` produces a real attested artifact."""
    pytest.importorskip("pinocchio")
    src = _write(tmp_path, _TWO_LINK_URDF, "twolink.urdf")
    exit_code = cli_main(
        [
            "build-urdf",
            str(src),
            "--backend",
            "jaxility-aba",
            "--target",
            "host",
            "--work-dir",
            str(tmp_path / "build"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out
    report = json.loads(captured.out)
    assert report["ok"] is True
    assert report["nx"] == 4 and report["nu"] == 2
    assert report["target"] in ("host-darwin", "host-linux")
    assert report["payload_bytes"] > 0
    assert Path(report["shared_library_path"]).exists()
    assert len(report["artifact_content_hash_hex"]) == 64
    assert len(report["manifest_content_hash_hex"]) == 64
    assert report["pinocchio_version"]  # non-empty provenance string


@pytest.mark.unit
@pytest.mark.skipif(not _TERA, reason="t_renderer required for host build")
def test_build_urdf_manifest_records_pinocchio_provenance(tmp_path: Path) -> None:
    """The built manifest carries ``toolchain_versions['pinocchio']`` (provenance)."""
    pytest.importorskip("pinocchio")
    from jaxility.builder import build_for_target
    from jaxility.lowering import OcpTemplateSpec, generate_dynamics
    from jaxility.manifest import detect_pinocchio_version
    from jaxility.targets import current_host_target

    dyn = generate_dynamics(_TWO_LINK_URDF, backend="jaxility-aba", name="twolink")
    nx, nu = 4, 2
    spec = OcpTemplateSpec(
        name="twolink",
        horizon_steps=10,
        time_horizon_s=0.2,
        state_cost=tuple([1.0] * nx),
        input_cost=tuple([0.1] * nu),
        terminal_state_cost=tuple([10.0] * nx),
        state_reference=tuple([0.0] * nx),
        input_reference=tuple([0.0] * nu),
        initial_state=tuple([0.0] * nx),
    )
    bundle = build_for_target(
        dynamics=dyn,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=b"\x00" * 32,
        work_dir=tmp_path / "b",
        extra_toolchain_versions={"pinocchio": detect_pinocchio_version()},
    )
    tv = bundle.manifest.toolchain_versions
    assert tv["pinocchio"] == detect_pinocchio_version()
    assert "casadi" in tv  # detected defaults still present (merge, not replace)
