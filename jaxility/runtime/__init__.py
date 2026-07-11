# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Build-time runtime support + on-target C runtime build orchestrator.

This subpackage owns:

* The *one* allowed wrapper around :func:`subprocess.run` in Jaxility
  (:mod:`jaxility.runtime.subprocess_runner`, PATTERNS §2.1). Direct
  calls to ``subprocess.run`` / ``Popen`` / ``os.system`` from library
  code are forbidden and CI-gated.
* The cross-compile orchestrator for the on-target C runtime
  (T-032 / T-052, :mod:`jaxility.runtime.c_runtime`). The C source
  tree lives under ``runtime-c/`` at the repo root; this module
  produces a static archive ``libjaxility_runtime_<family>.a`` per
  :class:`~jaxility.targets.Target` that downstream deployment
  artifacts link against.
"""

from .c_runtime import (
    RuntimeArchive,
    RuntimeBuildPlan,
    build_runtime_archive,
    plan_runtime_build,
    runtime_root,
    runtime_sources_for_family,
)
from .deploy import (
    DeployLauncher,
    DeployLauncherPlan,
    execute_deploy_launcher,
    plan_deploy_launcher,
)

__all__ = [
    "DeployLauncher",
    "DeployLauncherPlan",
    "RuntimeArchive",
    "RuntimeBuildPlan",
    "build_runtime_archive",
    "execute_deploy_launcher",
    "plan_deploy_launcher",
    "plan_runtime_build",
    "runtime_root",
    "runtime_sources_for_family",
]
