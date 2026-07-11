# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Smoke test: every subpackage is importable.

This is the T-001 acceptance test ("``pytest test/`` exits 0") and the
floor the rest of the suite stands on. Each subpackage is enumerated
explicitly rather than discovered so a removed subpackage shows up as a
test failure, not a silent gap. PATTERNS §7.1 tier 1 (unit).
"""

from __future__ import annotations

import importlib

import pytest

JAXILITY_SUBPACKAGES = [
    "jaxility",
    "jaxility.lowering",
    "jaxility.templates",
    "jaxility.policy",
    "jaxility.targets",
    "jaxility.runtime",
    "jaxility.compose",
    "jaxility.manifest",
    "jaxility.bench",
    "jaxility.hil",
    "jaxility.cli",
    "jaxility.mcp",
    "jaxility.testing",
]


@pytest.mark.unit
@pytest.mark.parametrize("name", JAXILITY_SUBPACKAGES)
def test_subpackage_importable(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.unit
def test_version_attribute_present() -> None:
    import jaxility

    assert isinstance(jaxility.__version__, str)
    assert jaxility.__version__  # non-empty
