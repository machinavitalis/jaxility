# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""``Target`` abstraction and per-SoC profiles.

Per ADR-003, a :class:`Target` is a Pydantic data model, not a subclass
hierarchy. Adding a new SoC is filling out the model. Target-conditional
behaviour will live in :mod:`jaxility.targets.dispatch` (PATTERNS §5,
strategy registrations keyed off capability flags) — that module lands
with the first real-target dispatch path (T-030 onward); for now
capability queries are handled in-line via
:meth:`Target.supports`.

The abstraction landed in T-011; the Pi 5 profile lands in T-030.
"""

from .extended_targets import (
    APPLE_SILICON,
    CORTEX_A55,
    CORTEX_A78,
    CORTEX_A710,
    CORTEX_M4,
    ETHOS_U55,
    ETHOS_U65,
    EXTENDED_TARGETS,
    NEOVERSE_N1,
    QUALCOMM_IQ10,
)
from .host import HOST_DARWIN, HOST_LINUX, HOST_TARGETS, current_host_target
from .mock import MOCK_CORTEX_A, MOCK_CORTEX_M, MOCK_TARGETS
from .models import (
    UNVERIFIED_SHA256,
    MemoryConstraints,
    NPUCapability,
    NPUFamily,
    Quirk,
    RealtimeGuarantee,
    RealtimeKind,
    Scheduler,
    Target,
    ToolchainPin,
    VectorExtension,
)
from .pi5 import PI5

__all__ = [
    "APPLE_SILICON",
    "CORTEX_A55",
    "CORTEX_A78",
    "CORTEX_A710",
    "CORTEX_M4",
    "ETHOS_U55",
    "ETHOS_U65",
    "HOST_DARWIN",
    "HOST_LINUX",
    "HOST_TARGETS",
    "MOCK_CORTEX_A",
    "MOCK_CORTEX_M",
    "MOCK_TARGETS",
    "MemoryConstraints",
    "NEOVERSE_N1",
    "NPUCapability",
    "NPUFamily",
    "EXTENDED_TARGETS",
    "PI5",
    "QUALCOMM_IQ10",
    "Quirk",
    "RealtimeGuarantee",
    "RealtimeKind",
    "Scheduler",
    "Target",
    "ToolchainPin",
    "UNVERIFIED_SHA256",
    "VectorExtension",
    "current_host_target",
]
