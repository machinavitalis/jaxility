# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""JAX → CasADi → acados lowering pipeline.

The single biggest technical complexity in Jaxility lives here. The
pipeline is structured as three distinct passes (ADR-001) so a future
JAX → MLIR substrate can replace ``jax_to_casadi`` without touching the
rest of the package. Concrete implementations land later; the
coverage declaration (T-013) landed first.
"""

from .casadi_to_acados import (
    CASADI_TO_ACADOS_SCHEMA_V0,
    OcpTemplateSpec,
    build_ocp,
)
from .coverage import (
    COVERAGE_TABLE,
    CoverageEntry,
    assert_supported,
    coverage_markdown,
    lookup,
)
from .jax_to_casadi import (
    JAX_TO_CASADI_SCHEMA_V0,
    CasadiFunction,
    translate,
)

__all__ = [
    "CASADI_TO_ACADOS_SCHEMA_V0",
    "COVERAGE_TABLE",
    "CasadiFunction",
    "CoverageEntry",
    "JAX_TO_CASADI_SCHEMA_V0",
    "OcpTemplateSpec",
    "assert_supported",
    "build_ocp",
    "coverage_markdown",
    "lookup",
    "translate",
]
