# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Shared pytest fixtures for the Jaxility test suite.

Three test tiers live under ``test/`` (PATTERNS §7.1):

* ``test/unit/`` — pure Python, no toolchains, no hardware.
* ``test/host/`` — host-only builds (Linux x86); requires CasADi and acados.
* ``test/hil/`` — cross-compiled builds on FVP or hardware.

The initial suite only exercises tier 1. Tier 2 lands with the host
build path (T-026); tier 3 with HIL (T-033).

Once the host build path lands, the OCP-construction tests need the acados runtime
library on ``DYLD_LIBRARY_PATH`` (or ``LD_LIBRARY_PATH`` on Linux) and
the source headers on ``ACADOS_SOURCE_DIR``. We set both here from a
canonical local install at ``~/Dev/acados`` if the env vars
are not already set — that keeps ``pytest test/`` working without
manual shell setup. Users who installed acados elsewhere can export
``JAXILITY_ACADOS_DIR`` before running pytest to override.
"""

from __future__ import annotations

import os
from pathlib import Path


def _default_acados_root() -> Path | None:
    override = os.environ.get("JAXILITY_ACADOS_DIR")
    if override:
        return Path(override)
    canonical = Path.home() / "Dev" / "acados"
    if canonical.exists():
        return canonical
    return None


_ACADOS_ROOT = _default_acados_root()
if _ACADOS_ROOT is not None:
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(_ACADOS_ROOT))
    _lib_dir = _ACADOS_ROOT / "lib"
    _lib_dir_s = str(_lib_dir)
    if os.uname().sysname == "Darwin":
        existing = os.environ.get("DYLD_LIBRARY_PATH", "")
        if _lib_dir_s not in existing.split(":"):
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{_lib_dir_s}:{existing}" if existing else _lib_dir_s
            )
    else:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if _lib_dir_s not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{_lib_dir_s}:{existing}" if existing else _lib_dir_s
            )

    # macOS SIP strips DYLD_LIBRARY_PATH when invoking SIP-protected
    # subprocesses, so the env-var path above doesn't always reach the
    # dyld linker when ``libacados.dylib`` is first loaded inside a
    # subprocess. Preloading the dylibs with explicit absolute paths
    # via ctypes routes around SIP — once a library is in the process
    # address space, subsequent dlopen requests find it without
    # consulting the search path.
    import ctypes

    _ACADOS_DYLIBS = (
        "libblasfeo.dylib",
        "libhpipm.dylib",
        "libqpOASES_e.dylib",
        "libacados.dylib",
    )
    for _name in _ACADOS_DYLIBS:
        _candidate = _lib_dir / _name
        if _candidate.exists():
            try:
                ctypes.CDLL(str(_candidate))
            except OSError:
                # Best-effort: surface any real load failure when the
                # test that needs the library actually runs.
                pass
