# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Dual-path attestation tests (T-045) — the manifest covers both artifacts."""

from __future__ import annotations

import blake3
import pytest

from jaxility.compose import (
    CompositionPlan,
    DualPathAttestation,
    SafetyEnvelope,
    attest_dual_path,
)
from jaxility.manifest.models import Manifest


def _manifest(handle_byte: str = "11") -> Manifest:
    return Manifest(
        source_attestation_handle=bytes.fromhex(handle_byte * 32),
        toolchain_versions={"aarch64-none-linux-gnu-gcc": "15.2.1"},
        target_profile_hash=bytes.fromhex("22" * 32),
        artifact_content_hash=bytes.fromhex("33" * 32),
        build_timestamp_utc=1,
    )


def _plan(name: str = "cp") -> CompositionPlan:
    return CompositionPlan(
        name=name,
        mpc_period_ns=1_000_000,
        policy_period_ns=20_000_000,
        safety_envelope=SafetyEnvelope(
            name="cp",
            state_lower=(-1.0,),
            state_upper=(1.0,),
            input_lower=(-1.0,),
            input_upper=(1.0,),
        ),
    )


@pytest.mark.unit
def test_attestation_binds_both_artifacts_and_plan() -> None:
    manifest = _manifest()
    policy_bytes = b"a-policy-model-flatbuffer"
    plan = _plan()
    att = attest_dual_path(
        controller_manifest=manifest, policy_artifact_bytes=policy_bytes, plan=plan
    )
    assert att.controller_manifest_hash == manifest.content_hash()
    assert att.policy_artifact_hash == blake3.blake3(policy_bytes).digest()
    assert isinstance(att, DualPathAttestation)
    assert len(att.content_hash()) == 32


@pytest.mark.unit
def test_attestation_changes_when_any_artifact_changes() -> None:
    base = attest_dual_path(
        controller_manifest=_manifest("11"),
        policy_artifact_bytes=b"policy-v1",
        plan=_plan("cp"),
    )
    # A different policy -> different attestation.
    diff_policy = attest_dual_path(
        controller_manifest=_manifest("11"),
        policy_artifact_bytes=b"policy-v2",
        plan=_plan("cp"),
    )
    # A different controller (recalibrated) -> different attestation.
    diff_ctrl = attest_dual_path(
        controller_manifest=_manifest("99"),
        policy_artifact_bytes=b"policy-v1",
        plan=_plan("cp"),
    )
    # A different safety envelope / plan -> different attestation.
    diff_plan = attest_dual_path(
        controller_manifest=_manifest("11"),
        policy_artifact_bytes=b"policy-v1",
        plan=_plan("cp-tighter"),
    )
    hashes = {
        base.content_hash(),
        diff_policy.content_hash(),
        diff_ctrl.content_hash(),
        diff_plan.content_hash(),
    }
    assert len(hashes) == 4  # every change moves the attestation


@pytest.mark.unit
def test_attestation_is_deterministic() -> None:
    a = attest_dual_path(
        controller_manifest=_manifest(), policy_artifact_bytes=b"p", plan=_plan()
    )
    b = attest_dual_path(
        controller_manifest=_manifest(), policy_artifact_bytes=b"p", plan=_plan()
    )
    assert a.content_hash() == b.content_hash()
