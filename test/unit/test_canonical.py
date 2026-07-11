# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the canonical-JSON serialiser (T-011 supporting infrastructure).

The serialiser is in :mod:`jaxility.manifest.canonical`; everything
that gets hashed (target profiles, manifests, benchmark records) flows
through it (PATTERNS §3.1). These tests pin the canonical form so the
manifest hash chain stays stable across Pydantic / Python minor
versions.
"""

from __future__ import annotations

import math

import pytest
from pydantic import BaseModel, ConfigDict

from jaxility.manifest import canonical_dumps


class _Sample(BaseModel):
    """Tiny model used to pin nested-Pydantic encoding."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    value: float
    flags: tuple[str, ...]


@pytest.mark.unit
def test_int_bool_none_pass_through() -> None:
    """Primitives encode as their JSON counterparts."""
    assert canonical_dumps(0) == b"0"
    assert canonical_dumps(True) == b"true"
    assert canonical_dumps(False) == b"false"
    assert canonical_dumps(None) == b"null"


@pytest.mark.unit
def test_string_passes_through_as_json_string() -> None:
    """Strings become JSON strings (UTF-8, escaped as needed)."""
    assert canonical_dumps("ok") == b'"ok"'


@pytest.mark.unit
def test_float_encoded_as_repr_string() -> None:
    """Floats become strings (PATTERNS §3.1 'no floats' rule)."""
    assert canonical_dumps(1e-12) == b'"1e-12"'
    assert canonical_dumps(3.14) == b'"3.14"'


@pytest.mark.unit
def test_nan_and_inf_raise() -> None:
    """NaN / Inf raise ValueError — schemas use a documented string form."""
    with pytest.raises(ValueError, match="NaN"):
        canonical_dumps(float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        canonical_dumps(math.inf)
    with pytest.raises(ValueError, match="NaN"):
        canonical_dumps(-math.inf)


@pytest.mark.unit
def test_bytes_encoded_as_hex_wrapper() -> None:
    """Bytes become ``{"$b16": "<hex>"}`` so the reader recovers them."""
    out = canonical_dumps(b"\xde\xad\xbe\xef")
    assert out == b'{"$b16":"deadbeef"}'


@pytest.mark.unit
def test_dict_keys_are_sorted_at_every_depth() -> None:
    """Lexicographic key sort at every depth (PATTERNS §3.1)."""
    nested = {"z": {"b": 1, "a": 2}, "a": 1}
    out = canonical_dumps(nested)
    assert out == b'{"a":1,"z":{"a":2,"b":1}}'


@pytest.mark.unit
def test_no_whitespace_between_separators() -> None:
    """Compact form (PATTERNS §3.1)."""
    out = canonical_dumps({"a": [1, 2, 3]})
    assert b" " not in out
    assert out == b'{"a":[1,2,3]}'


@pytest.mark.unit
def test_tuple_and_list_encode_identically() -> None:
    """Tuples and lists are both JSON arrays."""
    assert canonical_dumps((1, 2, 3)) == b"[1,2,3]"
    assert canonical_dumps([1, 2, 3]) == b"[1,2,3]"


@pytest.mark.unit
def test_pydantic_model_encoded_through_model_dump() -> None:
    """A Pydantic model encodes as if it were its ``model_dump`` dict."""
    sample = _Sample(name="x", value=2.5, flags=("hot", "cold"))
    out = canonical_dumps(sample)
    # Keys sorted: "flags" < "name" < "value"; "value" is a stringified float.
    assert out == b'{"flags":["hot","cold"],"name":"x","value":"2.5"}'


@pytest.mark.unit
def test_unsupported_type_raises_type_error() -> None:
    """Unknown types fail loudly (PATTERNS: no silent default)."""

    class Opaque:
        pass

    with pytest.raises(TypeError, match="canonical_dumps cannot encode"):
        canonical_dumps(Opaque())


@pytest.mark.unit
def test_equal_inputs_produce_byte_identical_output() -> None:
    """Two equal-content models produce byte-identical canonical encodings.

    This is the load-bearing property for the manifest hash chain
    (ADR-005). The test is the gate.
    """
    a = _Sample(name="x", value=1.5, flags=("a", "b"))
    b = _Sample(name="x", value=1.5, flags=("a", "b"))
    assert canonical_dumps(a) == canonical_dumps(b)


@pytest.mark.unit
def test_any_field_change_changes_encoding() -> None:
    """A field change produces a different byte string.

    Per PATTERNS §3.1 / §3.4 — a content-hashed payload must be
    sensitive to every byte of the schema.
    """
    base = _Sample(name="x", value=1.5, flags=("a",))
    changed_name = _Sample(name="y", value=1.5, flags=("a",))
    changed_value = _Sample(name="x", value=1.6, flags=("a",))
    changed_flags = _Sample(name="x", value=1.5, flags=("b",))

    enc = canonical_dumps(base)
    assert canonical_dumps(changed_name) != enc
    assert canonical_dumps(changed_value) != enc
    assert canonical_dumps(changed_flags) != enc
