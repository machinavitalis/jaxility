# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the coverage declaration + ``jaxility coverage`` CLI (T-013).

T-013 acceptance criteria:

1. Coverage table parses and validates.
2. A lowering attempt on an unsupported op raises a structured
   ``CoverageError`` referencing the coverage entry.
"""

from __future__ import annotations

import pytest

from jaxility.cli import main as cli_main
from jaxility.errors import CoverageError, JaxilityError
from jaxility.lowering.coverage import (
    COVERAGE_TABLE,
    CoverageEntry,
    assert_supported,
    coverage_markdown,
    lookup,
)

# ---------------------------------------------------------------------------
# Acceptance 1: table parses + validates; structural guarantees.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coverage_table_is_non_empty() -> None:
    """The table is non-empty — the test fixture for the CI gate."""
    assert len(COVERAGE_TABLE) > 0


@pytest.mark.unit
def test_every_entry_is_a_coverage_entry() -> None:
    """Schema discipline: every value is a frozen ``CoverageEntry``."""
    for entry in COVERAGE_TABLE.values():
        assert isinstance(entry, CoverageEntry)


@pytest.mark.unit
def test_table_keys_match_entry_fields() -> None:
    """``(op, dtype, target_family)`` key matches the entry's fields."""
    for (op, dtype, target_family), entry in COVERAGE_TABLE.items():
        assert entry.op_name == op
        assert entry.dtype == dtype
        assert entry.target_family == target_family


@pytest.mark.unit
def test_coverage_entry_has_schema_version() -> None:
    """Per PATTERNS §3.4 every schema record carries a ``schema_version``."""
    from jaxility.lowering.coverage import COVERAGE_SCHEMA_V0

    entry = lookup("add", "float64", "mock-cortex-a")
    assert entry.schema_version == COVERAGE_SCHEMA_V0


@pytest.mark.unit
def test_static_index_slicing_is_supported() -> None:
    """SKILL.md advertises static-index slicing; coverage table mirrors it."""
    entry = lookup("slice[static]", "float64", "mock-cortex-a")
    assert entry.supported is True


@pytest.mark.unit
def test_unsupported_entries_have_no_grade() -> None:
    """When ``supported`` is ``False``, ``grade`` is ``None`` — invariant 7."""
    for entry in COVERAGE_TABLE.values():
        if not entry.supported:
            assert entry.grade is None
        else:
            assert entry.grade is not None


# ---------------------------------------------------------------------------
# Acceptance 2: assert_supported raises structured CoverageError.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assert_supported_returns_entry_for_supported_combo() -> None:
    """The canonical smooth op is supported on the mock target."""
    entry = assert_supported(
        op="jnp.sin", dtype="float64", target_family="mock-cortex-a"
    )
    assert entry.supported is True
    assert entry.op_name == "jnp.sin"


@pytest.mark.unit
def test_assert_supported_raises_on_unsupported_op_with_documented_suggestion() -> None:
    """An unsupported op raises ``CoverageError`` with a structured payload."""
    with pytest.raises(CoverageError) as excinfo:
        assert_supported(
            op="lax.while_loop",
            dtype="float64",
            target_family="mock-cortex-a",
        )
    err = excinfo.value
    assert err.op == "lax.while_loop"
    assert err.dtype == "float64"
    assert err.target_family == "mock-cortex-a"
    assert "horizon" in err.suggestion.lower() or "loop" in err.suggestion.lower()
    # Structured: the suggestion references the documented workaround.
    assert err.suggestion != ""


@pytest.mark.unit
def test_assert_supported_raises_on_unknown_combo_with_table_pointer() -> None:
    """An unknown ``(op, dtype, target)`` raises with a pointer at the table."""
    with pytest.raises(CoverageError) as excinfo:
        assert_supported(
            op="jnp.imaginary_op", dtype="float64", target_family="mock-cortex-a"
        )
    err = excinfo.value
    assert "COVERAGE_TABLE" in err.suggestion


@pytest.mark.unit
def test_coverage_error_is_subclass_of_jaxility_error() -> None:
    """``CoverageError`` derives from the canonical base (PATTERNS §6.1)."""
    assert issubclass(CoverageError, JaxilityError)


@pytest.mark.unit
def test_coverage_error_str_contains_structured_fields() -> None:
    """``str(err)`` surfaces op + dtype + target_family + suggestion."""
    err = CoverageError(
        "test message",
        op="jnp.sin",
        dtype="float32",
        target_family="cortex-m7",
        suggestion="use the smoothed variant",
    )
    text = str(err)
    assert "test message" in text
    assert "jnp.sin" in text
    assert "float32" in text
    assert "cortex-m7" in text
    assert "smoothed" in text


# ---------------------------------------------------------------------------
# coverage_markdown rendering.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coverage_markdown_lists_a_section_per_target_and_dtype() -> None:
    """One ``## target × dtype`` section per group present in the table."""
    md = coverage_markdown()
    expected_groups = {(e.target_family, e.dtype) for e in COVERAGE_TABLE.values()}
    for target_family, dtype in expected_groups:
        assert f"## `{target_family}` × `{dtype}`" in md


@pytest.mark.unit
def test_coverage_markdown_target_filter_narrows_output() -> None:
    """The ``--target`` filter restricts to one family."""
    md = coverage_markdown(target_family="mock-cortex-a")
    assert "mock-cortex-a" in md
    assert "mock-cortex-m" not in md


# ---------------------------------------------------------------------------
# Sanity: the supported smooth-op subset matches SKILL.md.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "jnp.sin",
        "jnp.cos",
        "jnp.tan",
        "jnp.exp",
        "jnp.log",
        "jnp.sqrt",
        "matmul",
        "jnp.where[static]",
        "slice[static]",
    ],
)
def test_smooth_op_supported_on_mock_cortex_a_float64(op: str) -> None:
    """Every smooth op listed in SKILL.md is supported on the mock target."""
    entry = lookup(op, "float64", "mock-cortex-a")
    assert entry.supported is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "op",
    ["jnp.where[traced]", "lax.cond[traced]", "lax.while_loop", "dynamic_shape"],
)
def test_documented_unsupported_op_is_unsupported(op: str) -> None:
    """The documented unsupported edge surfaces in the table."""
    entry = lookup(op, "float64", "mock-cortex-a")
    assert entry.supported is False
    assert entry.grade is None
    assert entry.suggestion != ""


# ---------------------------------------------------------------------------
# CLI integration.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_coverage_emits_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    """``jaxility coverage`` emits the full coverage matrix on stdout."""
    exit_code = cli_main(["coverage"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Jaxility coverage matrix" in captured.out
    assert "mock-cortex-a" in captured.out
    assert "mock-cortex-m" in captured.out


@pytest.mark.unit
def test_cli_coverage_target_filter(capsys: pytest.CaptureFixture[str]) -> None:
    """``--target`` narrows the table."""
    exit_code = cli_main(["coverage", "--target", "mock-cortex-m"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "mock-cortex-m" in captured.out
    assert "mock-cortex-a" not in captured.out
