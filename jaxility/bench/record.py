# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Structured benchmark record for a deployed controller (T-035).

A :class:`BenchRecord` captures what `jaxility bench <robot> --target <soc>`
measures: the per-cycle controller solve time (the quantity that must fit
inside the control period), its jitter, and the peak resident memory — plus
the provenance needed to reproduce it. Records are JSON-serialisable and carry
the source manifest hash so a benchmark is always tied to the exact artifact it
measured (ADR-009); the public benchmark page is generated from these.

Timing is measured on the target around the bare acados solve (warm-started
RTI), excluding trace I/O, so the numbers reflect the control computation a
real deployment would run — not the harness overhead.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

BENCH_SCHEMA_V0 = 0
"""Schema version of the ``BenchRecord`` payload."""

BENCH_HARNESS_VERSION = "0.1.0"
"""Version of the measurement harness; bumped when the methodology changes."""


class SolveTiming(BaseModel):
    """Summary statistics of the per-cycle solve time, in nanoseconds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int = Field(ge=1)
    min_ns: float
    max_ns: float
    mean_ns: float
    p50_ns: float
    p99_ns: float
    stddev_ns: float

    @property
    def jitter_ns(self) -> float:
        """Peak-to-peak jitter (max - min) — the worst-case timing spread."""
        return self.max_ns - self.min_ns

    @property
    def sustainable_rate_hz(self) -> float:
        """Control rate the *worst-case* cycle sustains (1e9 / max_ns)."""
        return 1e9 / self.max_ns if self.max_ns > 0 else float("inf")


class BenchRecord(BaseModel):
    """A reproducible benchmark of a deployed controller on a target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=BENCH_SCHEMA_V0, ge=0)
    harness_version: str = BENCH_HARNESS_VERSION

    robot: str
    target_family: str
    target_name: str
    runner_label: str
    """Transport the benchmark ran through (``ssh:...`` for on-Pi)."""

    n_warmup: int = Field(ge=0)
    n_cycles: int = Field(ge=1)
    solve: SolveTiming
    max_rss_kib: int = Field(ge=0)
    """Peak resident set size (KiB) of the benchmark process."""

    source_manifest_hash: str | None = None
    """Hex BLAKE3 of the source manifest; ties the record to its artifact."""

    @property
    def meets_1khz(self) -> bool:
        """Whether every measured cycle fits the 1 kHz budget (< 1e6 ns)."""
        return self.solve.max_ns < 1e6

    @classmethod
    def from_solve_times(
        cls,
        solve_ns: list[float],
        *,
        robot: str,
        target_family: str,
        target_name: str,
        runner_label: str,
        n_warmup: int,
        max_rss_kib: int,
        source_manifest_hash: str | None = None,
    ) -> BenchRecord:
        """Build a record from the raw per-cycle solve times (ns)."""
        if not solve_ns:
            from ..errors import BenchmarkError

            raise BenchmarkError(
                "benchmark produced no solve-time samples; the target run "
                "emitted an empty timing array."
            )
        arr = np.asarray(solve_ns, dtype=np.float64)
        timing = SolveTiming(
            count=int(arr.size),
            min_ns=float(arr.min()),
            max_ns=float(arr.max()),
            mean_ns=float(arr.mean()),
            p50_ns=float(np.percentile(arr, 50)),
            p99_ns=float(np.percentile(arr, 99)),
            stddev_ns=float(arr.std()),
        )
        return cls(
            robot=robot,
            target_family=target_family,
            target_name=target_name,
            runner_label=runner_label,
            n_warmup=n_warmup,
            n_cycles=int(arr.size),
            solve=timing,
            max_rss_kib=max_rss_kib,
            source_manifest_hash=source_manifest_hash,
        )
