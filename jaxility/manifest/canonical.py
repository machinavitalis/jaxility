# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Canonical JSON serialisation for everything that gets hashed.

PATTERNS §3.1 — *"All manifests, target profiles, coverage entries,
and benchmark records use a single canonical JSON serialiser:
``jaxility.manifest.canonical_dumps``."* This module is that
serialiser. Hashes are taken over ``canonical_dumps(x)``; ``json.dumps``
on a hashed payload is a CI failure (PATTERNS §10).

The canonical form is:

* UTF-8.
* Keys sorted lexicographically at every depth.
* No whitespace between separators (compact form).
* No floats. Numeric values that arrive as Python ``float`` are
  emitted as **strings** using :func:`repr`, which produces the
  shortest round-trippable decimal (e.g. ``1e-12`` → ``"1e-12"``,
  ``3.14159`` → ``"3.14159"``). The reader of a canonical payload
  knows the schema, so a string-encoded number is no harder to parse
  than a JSON number — and the encoding sidesteps every JSON-library
  float-formatting divergence.
* Bytes values are emitted as the hex string of the bytes (lower
  case, no separator) wrapped in a single-key object ``{"$b16": "..."}``
  so the reader can recover them unambiguously. Plain string fields
  do not collide because they are emitted as bare strings, not as
  ``{"$b16": ...}`` wrappers.
* Integers, booleans, and ``None`` pass through as JSON ``integer``,
  ``true`` / ``false``, and ``null``.
* ``NaN``, ``Infinity``, and ``-Infinity`` raise :class:`ValueError` —
  schemas that legitimately need to express them use the documented
  string forms (e.g. ``"+inf"``) at the field level.

The serialiser accepts Python primitives and Pydantic v2 models. For
a model, the serialiser dumps to a dict via
``model.model_dump(mode="python")`` and then applies the canonical
rules above. ``extra="forbid"`` on every Pydantic model (PATTERNS §3.3)
guarantees no surprise field leaks in.

This module is dependency-free beyond the standard library, so it is
importable before any target / lowering / signing extra is installed.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel

_BYTES_KEY = "$b16"
"""Canonical key for the hex-string encoding of a ``bytes`` payload."""


def _coerce(value: Any) -> Any:
    """Recursively coerce a Python value into a canonical-JSON-safe form.

    Pure; no I/O. Order of cases matters — ``bool`` is checked before
    ``int`` because ``bool`` is a subclass of ``int`` in Python.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                "Canonical JSON does not encode NaN / Infinity directly. "
                "Use a documented string form at the schema level."
            )
        # repr gives the shortest decimal that round-trips back to the
        # same float; that is the property the hash relies on.
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {_BYTES_KEY: value.hex()}
    if isinstance(value, BaseModel):
        # Pydantic v2: model_dump(mode="python") preserves bytes as bytes,
        # tuples as tuples, etc. (mode="json" would already string-ify).
        return _coerce(value.model_dump(mode="python"))
    if isinstance(value, dict):
        # Keys must be strings for JSON. Sort them lexicographically.
        return {
            str(k): _coerce(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    raise TypeError(
        f"canonical_dumps cannot encode value of type {type(value).__name__!r}: "
        f"{value!r}. Add an explicit schema field or convert at the call site."
    )


def canonical_dumps(value: Any) -> bytes:
    """Return the canonical-JSON UTF-8 byte string for ``value``.

    Equal Python values must yield byte-identical output across hosts,
    Python minor versions, and Pydantic minor versions (the
    cross-host stability is the load-bearing property for the
    manifest hash chain).

    Args
    ----
    value : Any
        A Pydantic model, dict, list, tuple, str, bytes, int, bool,
        or ``None`` — recursively. Floats are emitted as strings;
        bytes as ``{"$b16": "<hex>"}``.

    Returns
    -------
    bytes
        Compact UTF-8 JSON with keys sorted at every depth.

    Raises
    ------
    ValueError
        On ``NaN`` / ``Infinity`` floats. Schemas needing these use a
        documented string form at the field level.
    TypeError
        On unsupported types (anything outside the small primitives
        list above). Adding new types requires extending this module
        with a documented encoding.
    """
    coerced = _coerce(value)
    return json.dumps(
        coerced,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
