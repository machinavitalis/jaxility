# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Toolchain version detection for manifest fields (A6).

``Manifest.toolchain_versions`` records the exact version of every
external code-generation toolchain Jaxility shelled out to during the
build. The dictionary is canonical-JSON serialised into the manifest,
so its values must be deterministic and detectable — *not* placeholder
strings like ``"unknown"`` (PATTERNS §6 — no silent fallback).

This module centralises the detection. Callers reach for:

* :func:`detect_acados_template_version` — the Python interface
  package's version, from :mod:`importlib.metadata`.
* :func:`detect_acados_library_version` — the C-library snapshot, by
  shelling out to ``git describe`` against ``$JAXILITY_ACADOS_DIR``
  (or its default ``~/Dev/acados``). Falls back to a
  marker that *names the gap* rather than papering over it.
* :func:`detect_casadi_version` — :attr:`casadi.__version__`.
* :func:`detect_toolchain_versions` — the full ``dict`` the manifest
  carries; binds the deployment target's binary version too.

The pinned upstreams Jaxility is currently validated against live as
constants (``JAXILITY_ACADOS_TEMPLATE_PIN``, ``JAXILITY_ACADOS_LIBRARY_PIN``).
A detected version that disagrees with the pin **does not** raise —
users may legitimately upgrade and re-validate — but the pin lives in
code so a reviewer can diff what changed in the manifest at upgrade
time.
"""

from __future__ import annotations

import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..errors import ToolchainError
from ..targets import Target

# ---------------------------------------------------------------------------
# Pinned upstreams (the versions Jaxility's test suite currently runs against)
# ---------------------------------------------------------------------------

JAXILITY_ACADOS_TEMPLATE_PIN: str = "0.5.1"
"""acados-template Python package version Jaxility is validated against."""

JAXILITY_ACADOS_LIBRARY_PIN: str = "v0.5.4-7-gdc6668f85"
"""acados C library snapshot Jaxility is validated against.

``git describe --tags --always`` output from ``$JAXILITY_ACADOS_DIR`` at
the time this pin landed. Upgrading the local acados build past this
revision is allowed; the manifest will record the new version. The
pin's job is to make the diff observable at review time.
"""

JAXILITY_ACADOS_DIR_ENV: str = "JAXILITY_ACADOS_DIR"
"""Environment variable Jaxility honours for the local acados source tree."""

DEFAULT_ACADOS_DIR: Path = Path.home() / "Dev" / "acados"
"""Default location Jaxility looks in if ``JAXILITY_ACADOS_DIR`` is unset."""


# ---------------------------------------------------------------------------
# acados-template (Python interface)
# ---------------------------------------------------------------------------


def detect_acados_template_version() -> str:
    """Return ``acados_template``'s installed package version.

    Raises
    ------
    ToolchainError
        ``acados_template`` is not importable / not installed. No
        ``"unknown"`` placeholder — the absence is loud (invariant 7).
    """
    try:
        return version("acados_template")
    except PackageNotFoundError as exc:
        raise ToolchainError(
            "acados_template is not installed. The acados Python interface "
            "is required for codegen. Install via "
            "`pip install -e $JAXILITY_ACADOS_DIR/interfaces/acados_template` "
            "after building libacados (see AGENTS/TOOLCHAINS.md)."
        ) from exc


# ---------------------------------------------------------------------------
# acados C library
# ---------------------------------------------------------------------------


def _acados_source_dir() -> Path:
    raw = os.environ.get(JAXILITY_ACADOS_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_ACADOS_DIR


def detect_acados_library_version() -> str:
    """Return the acados C library version string.

    Strategy: ``git describe --tags --always`` against
    ``$JAXILITY_ACADOS_DIR`` (default ``~/Dev/acados``).
    The acados C library does not expose its version through a macro
    or symbol, so the git tag of the build tree is the truth.

    Falls back to ``"library-unknown:<reason>"`` if introspection
    fails. The fallback string is *self-explaining* — it names why we
    could not detect — so the manifest never silently records a
    placeholder (invariant 7 / PATTERNS §6).
    """
    source = _acados_source_dir()
    if not source.exists():
        return f"library-unknown:no-source-dir:{source}"
    if not (source / ".git").exists():
        return "library-unknown:not-a-git-checkout"
    try:
        completed = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=str(source),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except OSError as exc:
        return f"library-unknown:git-not-on-path:{exc}"
    if completed.returncode != 0:
        return f"library-unknown:git-describe-rc{completed.returncode}"
    out = completed.stdout.strip()
    if not out:
        return "library-unknown:empty-git-describe"
    return out


# ---------------------------------------------------------------------------
# CasADi
# ---------------------------------------------------------------------------


def detect_casadi_version() -> str:
    """Return ``casadi.__version__``.

    Raises
    ------
    ToolchainError
        CasADi is not importable. As with acados, the absence is loud.
    """
    try:
        import casadi as ca  # noqa: PLC0415
    except Exception as exc:
        raise ToolchainError(
            "casadi is not installed. CasADi is the JAX → acados symbolic "
            "boundary (ADR-001) and required for any lowering. Install via "
            "`pip install casadi`."
        ) from exc
    return ca.__version__


# ---------------------------------------------------------------------------
# Pinocchio (rigid-body-dynamics generator, T-124)
# ---------------------------------------------------------------------------


def detect_pinocchio_version() -> str:
    """Return Pinocchio's installed version.

    Recorded in ``Manifest.toolchain_versions['pinocchio']`` for builds whose
    dynamics were generated from a URDF/MJCF via
    :func:`jaxility.lowering.generate_dynamics` (T-124). A Pinocchio-generated
    graph carries no jaxpr coverage audit trail, so this version string is the
    provenance record that the model came from Pinocchio's codegen rather than
    the JAX-lowering path.

    Raises
    ------
    ToolchainError
        Pinocchio is not importable. As with acados / casadi, the absence is
        loud rather than a ``"unknown"`` placeholder (invariant 7).
    """
    try:
        import pinocchio  # noqa: PLC0415
    except Exception as exc:
        raise ToolchainError(
            "pinocchio is not installed. It is required to record the "
            "provenance of dynamics generated by "
            "`jaxility.lowering.generate_dynamics`. Install the optional extra: "
            "`pip install 'jaxility[rbd]'` (or `pip install pin`)."
        ) from exc
    return pinocchio.__version__


# ---------------------------------------------------------------------------
# Full toolchain-versions dict for the manifest
# ---------------------------------------------------------------------------


def detect_toolchain_versions(target: Target) -> dict[str, str]:
    """Build the full ``toolchain_versions`` dict for the manifest.

    Keys:

    * ``target.toolchain.name`` — pinned binary version recorded on the
      :class:`Target` profile. The cross-compile path may overwrite
      this with the *detected* version from
      :func:`jaxility.builder_cross.verify_toolchain_installed`; the
      host path uses the pinned value (the host's compiler is not part
      of the deployment contract — its version travels separately via
      :func:`platform.python_version` in the build log).
    * ``"acados-template"`` — Python interface package version.
    * ``"acados-library"`` — local C-library git revision.
    * ``"casadi"`` — CasADi package version.

    Raises
    ------
    ToolchainError
        ``acados_template`` or ``casadi`` is missing. The C-library
        detection is *informational* — its absence yields a self-
        explaining placeholder rather than raising.
    """
    return {
        target.toolchain.name: target.toolchain.version,
        "acados-template": detect_acados_template_version(),
        "acados-library": detect_acados_library_version(),
        "casadi": detect_casadi_version(),
    }
