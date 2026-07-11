# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Benchmark harness and structured result schema.

``jaxility bench <robot> --target <soc>`` measures cycle time, jitter, and
memory footprint of a deployed controller on the target. Records are JSON and
carry the source manifest hash so a benchmark is always tied to the artifact it
measured (ADR-009); the public ``jaxility-benchmarks`` page is generated from
these.

Public surface (T-035):

* :class:`BenchRecord` / :class:`SolveTiming` — the structured result.
* :func:`generate_controller_bench_source` / :func:`run_controller_bench` —
  generate the timing binary for an acados controller and run it on the target.
"""

from __future__ import annotations

from .controller_bench import (
    generate_controller_bench_source,
    run_controller_bench,
)
from .record import (
    BENCH_HARNESS_VERSION,
    BENCH_SCHEMA_V0,
    BenchRecord,
    SolveTiming,
)
from .run import BenchTarget, run_cartpole_benchmark

__all__ = [
    "BENCH_HARNESS_VERSION",
    "BENCH_SCHEMA_V0",
    "BenchRecord",
    "BenchTarget",
    "SolveTiming",
    "generate_controller_bench_source",
    "run_cartpole_benchmark",
    "run_controller_bench",
]
