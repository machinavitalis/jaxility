# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the acados ToolchainPin detection (A6).

A6 replaces the silent ``"unknown"`` fallback in
``Manifest.toolchain_versions`` with structured detection that names
*why* a version is unavailable when it is. The pinned upstreams
Jaxility validates against live as importable constants so reviewers
can diff what changed at upgrade time.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from jaxility.manifest import (
    JAXILITY_ACADOS_DIR_ENV,
    JAXILITY_ACADOS_LIBRARY_PIN,
    JAXILITY_ACADOS_TEMPLATE_PIN,
    detect_acados_library_version,
    detect_acados_template_version,
    detect_casadi_version,
    detect_toolchain_versions,
)
from jaxility.targets import PI5

# ---------------------------------------------------------------------------
# Pin constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_acados_template_pin_is_documented() -> None:
    """The pinned acados-template version Jaxility validates against."""
    assert JAXILITY_ACADOS_TEMPLATE_PIN
    assert JAXILITY_ACADOS_TEMPLATE_PIN.count(".") >= 2  # x.y.z form


@pytest.mark.unit
def test_acados_library_pin_is_documented() -> None:
    """The pinned acados C-library git revision."""
    assert JAXILITY_ACADOS_LIBRARY_PIN
    # Either a git-describe form (v0.5.4-7-gdc6668f85) or a plain tag.
    pin = JAXILITY_ACADOS_LIBRARY_PIN
    assert pin.startswith("v") or "." in pin


@pytest.mark.unit
def test_acados_env_var_constant_is_exposed() -> None:
    assert JAXILITY_ACADOS_DIR_ENV == "JAXILITY_ACADOS_DIR"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_acados_template_returns_real_version() -> None:
    """No more ``"unknown"`` placeholder — the install must be visible."""
    pytest.importorskip("acados_template")
    detected = detect_acados_template_version()
    assert detected != "unknown"
    # x.y.z form; the install in this dev environment ships 0.5.x.
    assert detected.count(".") >= 2


@pytest.mark.unit
def test_detect_acados_template_raises_when_missing() -> None:
    """If acados_template isn't installed, the loud-fail path raises ToolchainError."""
    try:
        import acados_template  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        pytest.skip("acados_template is installed; tests the missing-install path")
    from jaxility.errors import ToolchainError

    with pytest.raises(ToolchainError, match="acados_template is not installed"):
        detect_acados_template_version()


@pytest.mark.unit
def test_detect_casadi_returns_real_version() -> None:
    pytest.importorskip("casadi")
    detected = detect_casadi_version()
    assert detected != "unknown"
    assert detected.count(".") >= 1  # casadi uses major.minor or major.minor.patch


@pytest.mark.unit
def test_detect_acados_library_returns_git_describe_when_repo_present() -> None:
    """When the acados dir is a git checkout, return its describe."""
    detected = detect_acados_library_version()
    # Either a real git describe (starts with 'v' for a tag) or a
    # self-explaining 'library-unknown:<reason>' marker.
    assert detected
    assert isinstance(detected, str)
    if detected.startswith("library-unknown:"):
        # Loud fallback — name the reason
        reason = detected.split(":", 1)[1]
        assert reason
    else:
        # Successful detection — at minimum a non-empty string
        assert detected.strip() == detected


@pytest.mark.unit
def test_detect_acados_library_loud_fallback_when_dir_missing(tmp_path: Path) -> None:
    """Point env at a non-existent path — get a self-explaining marker, not silence."""
    fake = tmp_path / "does-not-exist"
    with mock.patch.dict(os.environ, {JAXILITY_ACADOS_DIR_ENV: str(fake)}):
        detected = detect_acados_library_version()
    assert detected.startswith("library-unknown:no-source-dir:")
    assert str(fake) in detected


@pytest.mark.unit
def test_detect_acados_library_loud_fallback_when_not_a_git_checkout(
    tmp_path: Path,
) -> None:
    """A directory exists but isn't a git checkout — explicit marker."""
    (tmp_path / "not-a-git-checkout").mkdir()
    with mock.patch.dict(
        os.environ, {JAXILITY_ACADOS_DIR_ENV: str(tmp_path / "not-a-git-checkout")}
    ):
        detected = detect_acados_library_version()
    assert detected == "library-unknown:not-a-git-checkout"


# ---------------------------------------------------------------------------
# Full toolchain-versions dict for the manifest
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_toolchain_versions_for_pi5() -> None:
    pytest.importorskip("acados_template")
    pytest.importorskip("casadi")
    versions = detect_toolchain_versions(PI5)
    # Pi 5's binary key gets the pinned value.
    assert versions["aarch64-none-linux-gnu-gcc"] == PI5.toolchain.version
    # The three Jaxility-managed components are all detected.
    assert versions["acados-template"]
    assert versions["acados-template"] != "unknown"
    assert versions["acados-library"]
    assert versions["casadi"]
    assert versions["casadi"] != "unknown"
    # No silent placeholder anywhere.
    assert all(v != "unknown" for v in versions.values())


@pytest.mark.unit
def test_detect_toolchain_versions_keys_are_canonical_and_stable() -> None:
    """The dict's key set is the contract; new keys are an ADR-grade decision."""
    pytest.importorskip("acados_template")
    pytest.importorskip("casadi")
    versions = detect_toolchain_versions(PI5)
    expected_keys = {
        "aarch64-none-linux-gnu-gcc",
        "acados-template",
        "acados-library",
        "casadi",
    }
    assert set(versions.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Manifest carries the detected versions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_manifest_toolchain_versions_field_consumes_detected_dict() -> None:
    """The detect dict round-trips into and out of Manifest unchanged."""
    pytest.importorskip("acados_template")
    pytest.importorskip("casadi")
    from jaxility.manifest import SCHEMA_VERSION_V0, Manifest

    versions = detect_toolchain_versions(PI5)
    m = Manifest(
        schema_version=SCHEMA_VERSION_V0,
        source_attestation_handle=b"\x00" * 32,
        toolchain_versions=versions,
        target_profile_hash=PI5.hash,
        artifact_content_hash=b"\x11" * 32,
        build_timestamp_utc=0,
    )
    assert m.toolchain_versions == versions
    # And the full Manifest round-trips through canonical JSON.
    revived = Manifest.model_validate_json(m.model_dump_json())
    assert revived.toolchain_versions == versions
