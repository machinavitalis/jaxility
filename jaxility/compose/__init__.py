# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Dual-path runtime composition: acados MPC + learned-policy arbitration.

Per ADR-002, control and learning are deployed on two paths sharing a
declarative :class:`CompositionPlan` (rate, priority, safety envelope,
fallback). The plan, not buried code, is the safety surface.

Public surface (T-043):

* :class:`CompositionPlan` / :class:`SafetyEnvelope` / :class:`ArbitrationMode`
  — the declarative dual-path contract.
* :func:`arbitrate` / :class:`ArbitrationResult` — the reference one-step
  arbiter (the on-target C runtime mirrors it); always clamps into the envelope
  and falls back to the MPC on timeout / envelope breach.
"""

from __future__ import annotations

from .attest import DualPathAttestation, attest_dual_path
from .codegen import (
    DenseLayer,
    MLPPolicy,
    generate_dual_path_bench_source,
    generate_dual_path_hil_source,
)
from .plan import (
    ArbitrationMode,
    ArbitrationResult,
    CompositionPlan,
    SafetyEnvelope,
    arbitrate,
)

__all__ = [
    "ArbitrationMode",
    "ArbitrationResult",
    "CompositionPlan",
    "DenseLayer",
    "DualPathAttestation",
    "MLPPolicy",
    "SafetyEnvelope",
    "arbitrate",
    "attest_dual_path",
    "generate_dual_path_bench_source",
    "generate_dual_path_hil_source",
]
