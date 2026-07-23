# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Jaxility — JAX-to-Arm robotics deployment compiler with signed manifests.

Top-level package. The subpackages map one-to-one to the architecture in
``AGENTS/CONTEXT.md``:

* :mod:`jaxility.lowering` — JAX → CasADi → acados translation pipeline.
* :mod:`jaxility.templates` — acados problem templates (LQR, MPC, WBC, ...).
* :mod:`jaxility.policy` — learned-policy deployment (ONNX/LiteRT/ExecuTorch).
* :mod:`jaxility.targets` — ``Target`` profiles and per-SoC dispatch.
* :mod:`jaxility.runtime` — process-side runtime support (subprocess runner).
* :mod:`jaxility.compose` — dual-path MPC + policy composition.
* :mod:`jaxility.manifest` — attestation manifest schema, signing, verify.
* :mod:`jaxility.bench` — benchmark harness and result schema.
* :mod:`jaxility.hil` — hardware-in-the-loop co-simulation.
* :mod:`jaxility.cli` — top-level CLI entry points.
* :mod:`jaxility.mcp` — FastMCP server exposing build/verify/bench/hil.
* :mod:`jaxility.testing` — test utilities and mock targets.

Concrete contracts land in T-010..T-018.
"""

from .builder import BuildBundle, build_for_target
from .builder_cross import (
    CrossCompilePlan,
    cflags_for_family,
    cross_build_for_target,
    execute_cross_compile,
    plan_cross_compile,
    resolve_toolchain_integrity,
    verify_toolchain_installed,
    verify_toolchain_integrity,
)
from .builder_deps import (
    CrossBuiltDeps,
    DepBuildPlan,
    blasfeo_target_for_family,
    build_cross_deps,
    execute_dep_build,
    link_args_for_prefix,
    plan_dep_build,
    toolchain_file_for_target,
)
from .manifest import Artifact

__version__ = "1.1.0"
__all__ = [
    "Artifact",
    "BuildBundle",
    "CrossBuiltDeps",
    "CrossCompilePlan",
    "DepBuildPlan",
    "__version__",
    "blasfeo_target_for_family",
    "build_cross_deps",
    "build_for_target",
    "cflags_for_family",
    "cross_build_for_target",
    "execute_cross_compile",
    "execute_dep_build",
    "link_args_for_prefix",
    "plan_cross_compile",
    "plan_dep_build",
    "toolchain_file_for_target",
    "verify_toolchain_installed",
    "resolve_toolchain_integrity",
    "verify_toolchain_integrity",
]
