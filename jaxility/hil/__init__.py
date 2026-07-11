# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Hardware-in-the-loop co-simulation.

Source simulation runs on the host; the deployed artifact runs on the
target (or on Arm FVP); each cycle is step-locked and compared.
Divergence reports point at the offending code-generation step
(invariant 6). HIL parity is the release gate for a target (ADR-011).

Public surface (T-033):

* :func:`run_hil` / :class:`HILReport` — the step-locked parity check.
* :class:`TargetRunner` / :class:`LocalRunner` / :class:`SshRunner` —
  the transport to the deployed artifact (local subprocess or SSH to a
  real Cortex-A target such as the Raspberry Pi 5).
* :class:`StateSchema` / :func:`parse_trace` / :data:`CARTPOLE_SCHEMA` —
  the JSONL trace contract the on-target artifact emits.
"""

from __future__ import annotations

from .controller import (
    CARTPOLE_LQR_TRACE,
    TraceQuantity,
    build_controller_hil_binary,
    generate_controller_hil_source,
)
from .harness import HIL_REPORT_SCHEMA_V0, HILReport, run_hil
from .remote import build_controller_on_target
from .runner import (
    DEFAULT_TIMEOUT_S,
    LocalRunner,
    SshRunner,
    TargetRunner,
)
from .trace import (
    CARTPOLE_LQR_SCHEMA,
    CARTPOLE_SCHEMA,
    HIL_TRACE_SCHEMA_V0,
    StateSchema,
    parse_trace,
)

__all__ = [
    "CARTPOLE_LQR_SCHEMA",
    "CARTPOLE_LQR_TRACE",
    "CARTPOLE_SCHEMA",
    "DEFAULT_TIMEOUT_S",
    "HIL_REPORT_SCHEMA_V0",
    "HIL_TRACE_SCHEMA_V0",
    "HILReport",
    "LocalRunner",
    "SshRunner",
    "StateSchema",
    "TargetRunner",
    "TraceQuantity",
    "build_controller_hil_binary",
    "build_controller_on_target",
    "generate_controller_hil_source",
    "parse_trace",
    "run_hil",
]
