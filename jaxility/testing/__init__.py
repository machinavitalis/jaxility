# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Test utilities: mock targets, equivalence helpers, golden-artifact tools.

:func:`mock_lower` (T-015) exercises every
contract step (coverage → target dispatch → manifest → artifact) with a
Python-only mock backend, so the contracts are testable without any
cross-compilation infrastructure.
"""

from .equivalence import (
    EquivalenceReport,
    QuantityDivergence,
    Trajectory,
    compare,
)
from .jaxterity_source import JaxteritySource
from .mock_lower import MOCK_PIPELINE_VERSION, MockArtifactBundle, mock_lower
from .sources import MockSource, Source
from .tolerances import (
    Tolerance,
    lookup_tolerance,
    quantities_for,
)

__all__ = [
    "EquivalenceReport",
    "JaxteritySource",
    "MOCK_PIPELINE_VERSION",
    "MockArtifactBundle",
    "MockSource",
    "QuantityDivergence",
    "Source",
    "Tolerance",
    "Trajectory",
    "compare",
    "lookup_tolerance",
    "mock_lower",
    "quantities_for",
]
