# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Reference robot zoo deployment configs (T-017).

T-017 acceptance: each zoo robot mock-builds and passes the
equivalence check. Plus structural guarantees: registry round-trips,
artifact hashes are distinct per entry, real-vs-stub status is
exposed, and each entry's README is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxility.manifest import verify_manifest
from jaxility.testing import compare
from jaxility.zoo import (
    CONFIGS,
    ZooDeploymentConfig,
    available,
    load,
    mock_build,
)

EXPECTED_ENTRIES = ("berkeley_humanoid_lite", "cartpole", "crazyflie", "so100")
"""The zoo entries; alphabetical."""


@pytest.fixture(scope="module")
def configs() -> dict[str, ZooDeploymentConfig]:
    return CONFIGS()


# ---------------------------------------------------------------------------
# Registry surface.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_zoo_registers_expected_entries(
    configs: dict[str, ZooDeploymentConfig],
) -> None:
    """The zoo ships exactly the four documented entries."""
    assert sorted(configs) == list(EXPECTED_ENTRIES)
    assert sorted(available()) == list(EXPECTED_ENTRIES)


@pytest.mark.unit
@pytest.mark.parametrize("name", EXPECTED_ENTRIES)
def test_load_returns_named_entry(name: str) -> None:
    """``load(name)`` is the canonical lookup."""
    entry = load(name)
    assert entry.name == name


@pytest.mark.unit
def test_load_raises_on_unknown_entry() -> None:
    """An unknown zoo name raises loudly."""
    with pytest.raises(KeyError, match="unknown zoo entry"):
        load("definitely-not-here")


@pytest.mark.unit
def test_real_and_stub_status_distinction(
    configs: dict[str, ZooDeploymentConfig],
) -> None:
    """Cartpole + SO-100 are real robots; Crazyflie + humanoid are stubs."""
    real = {name for name, c in configs.items() if c.upstream_status == "real-robot"}
    stub = {
        name
        for name, c in configs.items()
        if c.upstream_status == "stub-pending-jaxterity"
    }
    assert real == {"cartpole", "so100"}
    assert stub == {"crazyflie", "berkeley_humanoid_lite"}


# ---------------------------------------------------------------------------
# Acceptance: every entry mock-builds and passes equivalence.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", EXPECTED_ENTRIES)
def test_entry_mock_builds_passes_equivalence(name: str) -> None:
    """Each zoo entry mock-builds end-to-end and its equivalence check passes."""
    if name == "cartpole" or name == "so100":
        pytest.importorskip("jaxterity")

    entry = load(name)
    bundle = mock_build(entry)

    source = entry.source_factory()
    report = compare(
        source.simulate(entry.n_steps),
        bundle.simulate(),
        target_family=entry.target.family,
        dtype=entry.dtype,
    )
    assert report.overall_passed is True


@pytest.mark.unit
@pytest.mark.parametrize("name", EXPECTED_ENTRIES)
def test_entry_manifest_verifies(name: str) -> None:
    """Each zoo entry's manifest verifies under the OSS signer."""
    if name == "cartpole" or name == "so100":
        pytest.importorskip("jaxterity")

    bundle = mock_build(load(name))
    report = verify_manifest(bundle.manifest)
    assert report.ok is True


@pytest.mark.unit
@pytest.mark.parametrize("name", EXPECTED_ENTRIES)
def test_entry_propagates_target_profile_hash(name: str) -> None:
    """N8: ``bundle.manifest.target_profile_hash == config.target.hash``."""
    if name == "cartpole" or name == "so100":
        pytest.importorskip("jaxterity")

    entry = load(name)
    bundle = mock_build(entry)
    assert bundle.manifest.target_profile_hash == entry.target.hash
    assert bundle.artifact.target_profile_hash == entry.target.hash


@pytest.mark.unit
def test_every_zoo_entry_has_a_distinct_artifact_hash(
    configs: dict[str, ZooDeploymentConfig],
) -> None:
    """No two entries collide — each carries a unique source / target / dtype."""
    pytest.importorskip("jaxterity")

    hashes = {
        name: mock_build(entry).artifact.content_hash for name, entry in configs.items()
    }
    assert len(set(hashes.values())) == len(hashes)


# ---------------------------------------------------------------------------
# Documentation: every entry has a README per T-017.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", EXPECTED_ENTRIES)
def test_each_entry_has_a_readme(name: str) -> None:
    """T-017 spec: each zoo entry README documents source / license / remaining work."""
    readme = (
        Path(__file__).resolve().parents[2] / "jaxility" / "zoo" / name / "README.md"
    )
    assert readme.exists(), f"missing {readme}"
    text = readme.read_text()
    assert "License" in text
    assert "Remaining work" in text or "Upstream gap" in text


# ---------------------------------------------------------------------------
# Target sanity per entry.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cartpole_targets_mock_cortex_a(
    configs: dict[str, ZooDeploymentConfig],
) -> None:
    assert configs["cartpole"].target.family == "mock-cortex-a"
    assert configs["cartpole"].template == "LQR"


@pytest.mark.unit
def test_so100_targets_mock_cortex_a(configs: dict[str, ZooDeploymentConfig]) -> None:
    assert configs["so100"].target.family == "mock-cortex-a"
    assert configs["so100"].template == "WBC"


@pytest.mark.unit
def test_crazyflie_targets_mock_cortex_m(
    configs: dict[str, ZooDeploymentConfig],
) -> None:
    assert configs["crazyflie"].target.family == "mock-cortex-m"
    assert configs["crazyflie"].template == "TrackingMPC"
    assert configs["crazyflie"].dtype == "float32"


@pytest.mark.unit
def test_humanoid_targets_mock_cortex_a(
    configs: dict[str, ZooDeploymentConfig],
) -> None:
    assert configs["berkeley_humanoid_lite"].target.family == "mock-cortex-a"
    assert configs["berkeley_humanoid_lite"].template == "CentroidalMPC"
