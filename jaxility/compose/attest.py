# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Dual-path attestation: one record binding both deployed artifacts (T-045).

A dual-path deployment ships *two* artifacts — the acados controller (a
content-addressed `Artifact` + chain-linked `Manifest`) and the learned policy
(ONNX / `.tflite` bytes). The dual-path gate requires the attestation to cover
**both**. :class:`DualPathAttestation` is that binding: the controller's manifest
hash, the policy artifact's BLAKE3 hash, and the `CompositionPlan` hash (the
safety surface), folded into a single verifiable `content_hash`. Recalibrating
the robot moves the controller manifest; retraining the policy moves the policy
hash; changing the safety envelope moves the plan hash — any of which changes the
dual-path attestation, exactly as attestation should.
"""

from __future__ import annotations

import blake3
from pydantic import BaseModel, ConfigDict, Field

from ..manifest.canonical import canonical_dumps
from ..manifest.models import HexBytes, Manifest
from .plan import CompositionPlan

DUAL_PATH_ATTEST_SCHEMA_V0 = 0
"""Schema version of the dual-path attestation payload."""


class DualPathAttestation(BaseModel):
    """Binds the controller manifest, the policy artifact, and the plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=DUAL_PATH_ATTEST_SCHEMA_V0, ge=0)
    controller_manifest_hash: HexBytes
    """BLAKE3 ``content_hash`` of the acados controller's :class:`Manifest`."""
    policy_artifact_hash: HexBytes
    """BLAKE3 of the policy artifact bytes (ONNX / LiteRT)."""
    composition_plan_hash: HexBytes
    """BLAKE3 of the canonical :class:`CompositionPlan` (the safety surface)."""

    def content_hash(self) -> bytes:
        """BLAKE3 digest binding both artifacts + the plan (invariant 5 stable)."""
        return blake3.blake3(canonical_dumps(self.model_dump(mode="json"))).digest()


def attest_dual_path(
    *,
    controller_manifest: Manifest,
    policy_artifact_bytes: bytes,
    plan: CompositionPlan,
) -> DualPathAttestation:
    """Produce the dual-path attestation covering both artifacts + the plan."""
    return DualPathAttestation(
        controller_manifest_hash=controller_manifest.content_hash(),
        policy_artifact_hash=blake3.blake3(policy_artifact_bytes).digest(),
        composition_plan_hash=blake3.blake3(
            canonical_dumps(plan.model_dump(mode="json"))
        ).digest(),
    )
