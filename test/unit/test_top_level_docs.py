# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the top-level CLAIMS.md and KNOWN_GAPS.md docs (A5).

These are *contract* docs: CLAIMS.md says what the library guarantees
right now; KNOWN_GAPS.md says what it explicitly does not. They are
symmetric — a claim that lands or a gap that closes should move text
between them, not leave one stale.

These tests catch the common drift mode: a file gets deleted, a
cross-reference breaks, or the README forgets to point at them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CLAIMS = REPO / "CLAIMS.md"
KNOWN_GAPS = REPO / "KNOWN_GAPS.md"
README = REPO / "README.md"


@pytest.mark.unit
def test_claims_file_exists() -> None:
    assert CLAIMS.exists(), "CLAIMS.md missing at repo root"


@pytest.mark.unit
def test_known_gaps_file_exists() -> None:
    assert KNOWN_GAPS.exists(), "KNOWN_GAPS.md missing at repo root"


@pytest.mark.unit
def test_claims_cross_references_known_gaps() -> None:
    """Symmetric pair: CLAIMS must point at KNOWN_GAPS."""
    text = CLAIMS.read_text()
    assert "KNOWN_GAPS.md" in text, (
        "CLAIMS.md should cross-reference KNOWN_GAPS.md "
        "so readers see both halves of the contract."
    )


@pytest.mark.unit
def test_known_gaps_cross_references_claims() -> None:
    """Symmetric pair: KNOWN_GAPS must point back at CLAIMS."""
    text = KNOWN_GAPS.read_text()
    assert "CLAIMS.md" in text, (
        "KNOWN_GAPS.md should cross-reference CLAIMS.md "
        "so readers see both halves of the contract."
    )


@pytest.mark.unit
def test_known_gaps_names_mjx_close() -> None:
    """The single most surprising gap — MJX as source — must be there by name."""
    text = KNOWN_GAPS.read_text()
    assert "MJX" in text
    assert "ADR-016" in text


@pytest.mark.unit
def test_known_gaps_names_tier_b_cross_compile() -> None:
    """The current end-of-the-pipe gap — Tier B / real Arm GCC."""
    text = KNOWN_GAPS.read_text()
    assert "T-031" in text or "Tier B" in text
    assert "aarch64-none-linux-gnu-gcc" in text or "Arm GNU" in text


@pytest.mark.unit
def test_claims_lists_all_14_target_profiles() -> None:
    """CLAIMS' targets table must enumerate every shipped Target name."""
    text = CLAIMS.read_text()
    expected_names = [
        "MOCK_CORTEX_A",
        "MOCK_CORTEX_M",
        "HOST_DARWIN",
        "HOST_LINUX",
        "PI5",
        "CORTEX_A55",
        "CORTEX_A78",
        "CORTEX_A710",
        "NEOVERSE_N1",
        "CORTEX_M4",
        "ETHOS_U55",
        "ETHOS_U65",
        "QUALCOMM_IQ10",
        "APPLE_SILICON",
    ]
    for name in expected_names:
        assert name in text, f"CLAIMS.md targets table missing {name!r}"


@pytest.mark.unit
def test_claims_names_the_four_templates() -> None:
    """LQR / TrackingMPC / WBC / Centroidal MPC — the shipped templates."""
    text = CLAIMS.read_text()
    for name in ("LQR", "TrackingMPC", "WBC", "Centroidal"):
        assert name in text, f"CLAIMS.md is missing template {name!r}"


@pytest.mark.unit
def test_claims_toolchain_versions_match_code() -> None:
    """Audit M-2: prevent CLAIMS.md from drifting away from the pinned versions.

    The targets table in CLAIMS.md records the pinned toolchain version
    for every shipped :class:`Target`. When a pin bumps in code the
    doc must bump too — the previous drift slipped through because no
    test enforced agreement.
    """
    from jaxility.targets import (
        APPLE_SILICON,
        CORTEX_A55,
        CORTEX_A78,
        CORTEX_A710,
        CORTEX_M4,
        ETHOS_U55,
        ETHOS_U65,
        NEOVERSE_N1,
        PI5,
        QUALCOMM_IQ10,
    )

    text = CLAIMS.read_text()
    pinned_real_targets = [
        PI5,
        CORTEX_A55,
        CORTEX_A78,
        CORTEX_A710,
        NEOVERSE_N1,
        CORTEX_M4,
        ETHOS_U55,
        ETHOS_U65,
        QUALCOMM_IQ10,
        APPLE_SILICON,
    ]
    for target in pinned_real_targets:
        # Every real Target's row must list its current pinned version.
        pin = f"{target.toolchain.name} {target.toolchain.version}"
        assert pin in text, (
            f"CLAIMS.md targets table is stale: expected {pin!r} "
            f"for {target.name!r} but did not find it."
        )


# ---------------------------------------------------------------------------
# README support table consistency. The README ships a human-readable
# quick-reference table that must stay aligned with CLAIMS.md and
# KNOWN_GAPS.md. These tests catch the common drift modes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_readme_cross_references_claims_and_known_gaps() -> None:
    """README must point readers at both contract docs."""
    text = README.read_text()
    assert "CLAIMS.md" in text, "README should cross-reference CLAIMS.md"
    assert "KNOWN_GAPS.md" in text, "README should cross-reference KNOWN_GAPS.md"


@pytest.mark.unit
def test_readme_status_is_not_stale_bootstrap() -> None:
    """The bootstrap status string was stale at the time the support
    table landed; future drift in either direction should be caught."""
    text = README.read_text()
    # Either Phase 2 closed (current) or a later phase — Phase 0 alone
    # is the stale string.
    assert "Phase 0" not in text or "Phase 2" in text or "Phase 3" in text


@pytest.mark.unit
def test_readme_lists_the_four_templates() -> None:
    """README support table must name every template that ships."""
    text = README.read_text()
    for name in ("LQR", "TrackingMPC", "WBC", "Centroidal MPC"):
        assert name in text, f"README support table missing template {name!r}"


@pytest.mark.unit
def test_readme_lists_every_shipped_target_profile() -> None:
    """Every shipped :class:`Target` must appear in the README targets table.

    Mirrors the CLAIMS.md anti-drift test so the two stay in lockstep.
    """
    from jaxility.targets import (
        APPLE_SILICON,
        CORTEX_A55,
        CORTEX_A78,
        CORTEX_A710,
        CORTEX_M4,
        ETHOS_U55,
        ETHOS_U65,
        HOST_DARWIN,
        HOST_LINUX,
        MOCK_CORTEX_A,
        MOCK_CORTEX_M,
        NEOVERSE_N1,
        PI5,
        QUALCOMM_IQ10,
    )

    text = README.read_text()
    all_targets = [
        PI5,
        CORTEX_A55,
        CORTEX_A78,
        CORTEX_A710,
        NEOVERSE_N1,
        CORTEX_M4,
        ETHOS_U55,
        ETHOS_U65,
        QUALCOMM_IQ10,
        APPLE_SILICON,
        HOST_DARWIN,
        HOST_LINUX,
        MOCK_CORTEX_A,
        MOCK_CORTEX_M,
    ]
    for target in all_targets:
        symbol = target.name.upper().replace("-", "_")
        assert symbol in text, f"README targets table missing {symbol!r}"


@pytest.mark.unit
def test_readme_pi5_pin_version_matches_code() -> None:
    """The PI5 row in the README must reflect the current pin."""
    from jaxility.targets import PI5

    text = README.read_text()
    pin = f"`{PI5.toolchain.name} {PI5.toolchain.version}`"
    assert pin in text, (
        f"README PI5 row is stale: expected {pin!r} but did not find it."
    )


@pytest.mark.unit
def test_readme_lists_mjx_as_rejected_dynamics() -> None:
    """The dynamics-shapes table must name the MJX rejection (ADR-016).

    This is the single largest surprise for downstream users; the README
    must not bury it.
    """
    text = README.read_text()
    assert "MJX" in text, "README should name the MJX gap by name"
    assert "ADR-016" in text or "closed-form" in text


@pytest.mark.unit
def test_readme_names_acados_smooth_op_subset() -> None:
    """The README's primitive table must use the canonical phrasing
    so a search for 'smooth-op' from any doc finds the right place."""
    text = README.read_text()
    assert "smooth-op" in text or "smooth op" in text


# ---------------------------------------------------------------------------
# Doc-drift governance tests — catch the most likely future drift modes.
#
# Each test is a tripwire for a specific drift scenario that the
# audit pass surfaced. They cannot prevent every drift (a closed gap
# still needs an agent to move the text), but they catch the
# mechanical ones.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_claims_carries_doc_drift_rule_banner() -> None:
    """The maintainer-note banner at the top of CLAIMS.md must be
    present. If a future agent rewrites CLAIMS and drops the banner,
    the rule disappears — this catches it."""
    text = CLAIMS.read_text()
    assert "doc-drift rule" in text or "three-document contract" in text
    assert "test_top_level_docs.py" in text


@pytest.mark.unit
def test_known_gaps_carries_doc_drift_rule_banner() -> None:
    """Same for KNOWN_GAPS.md."""
    text = KNOWN_GAPS.read_text()
    assert "doc-drift rule" in text
    assert "test_top_level_docs.py" in text


@pytest.mark.unit
def test_context_md_documents_doc_drift_rule() -> None:
    """AGENTS/CONTEXT.md must document the doc-drift governance so
    future agents reading the orientation see the rule."""
    text = (REPO / "AGENTS" / "CONTEXT.md").read_text()
    assert "Contract docs and the doc-drift rule" in text


@pytest.mark.unit
def test_readme_supported_primitive_categories_match_coverage_table() -> None:
    """The README's primitive table must reflect every supported op in
    ``COVERAGE_TABLE``'s smooth subset. If a new primitive lands as a
    handler + coverage row but the README isn't updated, the
    representative-op check below fails."""
    from jaxility.lowering.coverage import _SUPPORTED_OPS_SMOOTH

    text = README.read_text()
    # Pick a representative op from each category to spot-check the
    # README mentions it. Adding a whole new category to
    # _SUPPORTED_OPS_SMOOTH should add a category row to the README.
    representatives = {
        "add": "Arithmetic",
        "jnp.sin": "Transcendentals",
        "matmul": "Linear algebra",  # matmul is the user-facing alias for dot_general
        "slice[static]": "Indexing",
        "dynamic_slice[static]": "Indexing",
    }
    for op, _category in representatives.items():
        assert op in _SUPPORTED_OPS_SMOOTH, (
            f"{op!r} is no longer in _SUPPORTED_OPS_SMOOTH; either restore "
            "it or update this test."
        )
    # The README must surface the existence of the smooth-op subset
    # AND mention representative op names from each main category.
    assert "smooth-op" in text or "smooth op" in text
    assert "add" in text
    assert "sin" in text  # any transcendental mention
    assert "dot_general" in text or "matmul" in text


@pytest.mark.unit
def test_readme_template_names_resolve_to_real_factories() -> None:
    """The four templates the README names must be importable.

    Catches the drift mode where a template gets renamed in code but
    the README still lists the old name.
    """
    from jaxility import templates

    for symbol in ("lqr", "tracking_mpc", "wbc", "centroidal_mpc"):
        assert hasattr(templates, symbol), (
            f"jaxility.templates is missing {symbol!r}; "
            "the README + CLAIMS table is stale."
        )


@pytest.mark.unit
def test_ci_install_step_matches_cortex_m_live_claim() -> None:
    """Audit M-1 closed by installing arm-none-eabi-gcc on CI.

    If a future PR removes the install step but the CLAIMS / README
    still claim "Cortex-M lane live in CI", this test catches the
    drift.
    """
    ci_yml = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    claims_text = CLAIMS.read_text()
    cortex_m_live_in_ci = "Cortex-M lane" in claims_text and (
        "locally and in CI" in claims_text or "local + CI" in claims_text
    )
    if cortex_m_live_in_ci:
        assert "arm-none-eabi-gcc" in ci_yml, (
            "CLAIMS.md says the Cortex-M lane runs in CI but ci.yml does "
            "not install arm-none-eabi-gcc — move the claim back to "
            "KNOWN_GAPS.md or restore the install step."
        )


@pytest.mark.unit
def test_ci_install_step_matches_pi5_live_claim() -> None:
    """Same shape as the Cortex-M check, for the Pi 5 lane."""
    ci_yml = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    claims_text = CLAIMS.read_text()
    pi5_live_in_ci = "Pi 5 lane" in claims_text and "CI" in claims_text
    if pi5_live_in_ci:
        assert "aarch64-none-linux-gnu-gcc" in ci_yml, (
            "CLAIMS.md says the Pi 5 lane runs in CI but ci.yml does "
            "not install aarch64-none-linux-gnu-gcc — move the claim "
            "back to KNOWN_GAPS.md or restore the install step."
        )


@pytest.mark.unit
def test_decisions_md_records_adr_017_for_l4casadi_integration() -> None:
    """ADR-017 governs the L4CasADi integration for embedded
    learned functions. If a future PR ships T-043 without citing
    ADR-017, the implementation has skipped the architectural-decision
    step. This test asserts the ADR text exists; the *acceptance* status
    is checked separately below."""
    text = (REPO / "AGENTS" / "DECISIONS.md").read_text()
    assert "ADR-017" in text, "DECISIONS.md is missing the ADR-017 entry"
    assert "L4CasADi" in text, "ADR-017 must name L4CasADi"
    assert "model_external_shared_lib" in text, (
        "ADR-017 must name the load-bearing acados FFI seam"
    )


@pytest.mark.unit
def test_known_gaps_names_l4casadi_as_prior_art() -> None:
    """The KNOWN_GAPS learned-policy section must reference L4CasADi
    as the prior art so a future agent picking up T-043 sees the
    integration shape immediately."""
    text = KNOWN_GAPS.read_text()
    assert "L4CasADi" in text
    assert "ADR-017" in text


@pytest.mark.unit
def test_readme_contains_mermaid_lowering_diagram() -> None:
    """The README's lowering-landscape diagrams are the visual map of
    where Jaxility sits among the JAX-lowering paths. They must use
    the Mermaid format GitHub renders natively AND must be split into
    multiple focused diagrams rather than one tangled flowchart.

    The single-diagram approach (first attempt) was visually
    unreadable; the three-diagram split (Jaxility's lane / broader
    ecosystem / L4CasADi sidetrack) replaced it. This test locks the
    split in place — a future PR that collapses back to one giant
    diagram will fail here."""
    text = README.read_text()
    # At least three separate Mermaid code fences. The 3-diagram
    # split is the load-bearing readability decision; collapsing to
    # one is the regression to catch.
    mermaid_fence_count = text.count("```mermaid")
    assert mermaid_fence_count >= 3, (
        f"README must contain at least 3 separate Mermaid diagrams "
        f"(found {mermaid_fence_count}). The 3-diagram split — "
        "Jaxility's lane / broader ecosystem / L4CasADi sidetrack — "
        "replaced a single tangled flowchart that was unreadable."
    )
    # The diagrams must reference every load-bearing IR / backend
    # symbol so a reader sees the full landscape.
    for symbol in (
        "jaxpr",
        "HLO",
        "ONNX",
        "CasADi",
        "acados",
        "LiteRT",
        "L4CasADi",
        "XLA",
    ):
        assert symbol in text, f"README lowering diagram missing {symbol!r}"


@pytest.mark.unit
def test_readme_lowering_paths_table_names_learned_policy_lane() -> None:
    """The 'when to use which path' table must point at the L4CasADi
    integration so a downstream consumer evaluating the learned-policy
    lane sees it. Once that lane ships this test will move with the docs."""
    text = README.read_text()
    assert "L4CasADi" in text


@pytest.mark.unit
def test_pypi_placeholder_stays_minimal() -> None:
    """The ``pypi-placeholder/`` directory exists to claim the
    ``jaxility`` name on PyPI. It must stay minimal:

    - ``version == "0.0.0"`` (mirrors jaxonomy / jaxterity);
    - no runtime dependencies (a placeholder shouldn't pull anything);
    - the package name matches the project root's package name;
    - the LICENSE file ships alongside.

    The placeholder is a one-shot upload. PyPI does not allow
    re-uploads of the same version, so any drift here would
    silently corrupt the next release attempt.
    """
    import sys

    placeholder = REPO / "pypi-placeholder"
    assert placeholder.is_dir(), (
        "pypi-placeholder/ directory missing — the PyPI name-reservation "
        "package should ship with the repo as a re-issuable backup until "
        "the v0.0.1 release."
    )
    assert (placeholder / "LICENSE").exists()
    assert (placeholder / "README.md").exists()
    assert (placeholder / "UPLOAD.md").exists()
    assert (placeholder / "jaxility" / "__init__.py").exists()

    if sys.version_info < (3, 11):
        return  # tomllib is 3.11+; older runners skip the deeper checks
    import tomllib as _tomllib  # noqa: PLC0415

    cfg = _tomllib.loads((placeholder / "pyproject.toml").read_text())
    project = cfg["project"]
    assert project["name"] == "jaxility"
    assert project["version"] == "0.0.0", (
        f"placeholder version is {project['version']!r}; must stay 0.0.0. "
        "PyPI does not allow re-uploads of the same version, so bumping "
        "the placeholder breaks the one-shot upload contract."
    )
    assert not project.get("dependencies"), (
        "placeholder must be dependency-free (matches jaxonomy / jaxterity); "
        f"found {project.get('dependencies')!r}."
    )
    assert not project.get("optional-dependencies"), (
        "placeholder must not declare optional dependencies; "
        f"found {project.get('optional-dependencies')!r}."
    )


@pytest.mark.unit
def test_known_gaps_pending_phrases_have_grep_anchors() -> None:
    """KNOWN_GAPS.md uses canonical phrases ('pending', 'deferred',
    'not yet', 'still out of scope', 'queued') as grep targets for
    future agents looking for gaps to close. If the file loses every
    one of those phrases the rule has eroded."""
    text = KNOWN_GAPS.read_text().lower()
    canonical = ("pending", "deferred", "not yet", "still", "queued")
    assert any(phrase in text for phrase in canonical), (
        "KNOWN_GAPS.md has lost every canonical 'this gap exists' "
        "phrase — either everything shipped (unlikely) or the grep "
        "anchors got rewritten."
    )
