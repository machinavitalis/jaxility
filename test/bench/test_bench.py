# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Unit tests for the benchmark record + generator (T-035).

Pure-Python: no acados, no compiler, no hardware. The on-silicon
measurement path is exercised by test/hil/test_on_pi_controller.py.
"""

from __future__ import annotations

import pytest

from jaxility.bench import (
    BenchRecord,
    SolveTiming,
    generate_controller_bench_source,
)
from jaxility.errors import BenchmarkError


@pytest.mark.unit
def test_solve_timing_derived_fields() -> None:
    t = SolveTiming(
        count=3,
        min_ns=100.0,
        max_ns=200.0,
        mean_ns=150.0,
        p50_ns=150.0,
        p99_ns=199.0,
        stddev_ns=40.0,
    )
    assert t.jitter_ns == pytest.approx(100.0)
    assert t.sustainable_rate_hz == pytest.approx(1e9 / 200.0)


@pytest.mark.unit
def test_bench_record_from_solve_times_stats() -> None:
    samples = [100.0, 200.0, 300.0, 400.0, 1000.0]
    rec = BenchRecord.from_solve_times(
        samples,
        robot="cartpole",
        target_family="cortex-a76",
        target_name="pi5",
        runner_label="ssh:pi",
        n_warmup=50,
        max_rss_kib=6000,
    )
    assert rec.n_cycles == 5
    assert rec.solve.min_ns == 100.0
    assert rec.solve.max_ns == 1000.0
    assert rec.solve.mean_ns == pytest.approx(400.0)
    # max 1000 ns << 1e6 ns budget
    assert rec.meets_1khz is True
    assert rec.max_rss_kib == 6000


@pytest.mark.unit
def test_bench_record_detects_1khz_miss() -> None:
    # A worst-case cycle slower than 1 ms fails the 1 kHz budget.
    rec = BenchRecord.from_solve_times(
        [500.0, 1_500_000.0],
        robot="cartpole",
        target_family="cortex-a76",
        target_name="pi5",
        runner_label="ssh:pi",
        n_warmup=0,
        max_rss_kib=1,
    )
    assert rec.meets_1khz is False


@pytest.mark.unit
def test_bench_record_empty_samples_raises() -> None:
    with pytest.raises(BenchmarkError, match="no solve-time samples"):
        BenchRecord.from_solve_times(
            [],
            robot="cartpole",
            target_family="cortex-a76",
            target_name="pi5",
            runner_label="x",
            n_warmup=0,
            max_rss_kib=0,
        )


@pytest.mark.unit
def test_bench_record_round_trips_json() -> None:
    rec = BenchRecord.from_solve_times(
        [94000.0, 97000.0, 135000.0],
        robot="cartpole",
        target_family="cortex-a76",
        target_name="pi5",
        runner_label="ssh:pi",
        n_warmup=100,
        max_rss_kib=5888,
        source_manifest_hash="ab" * 32,
    )
    restored = BenchRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


@pytest.mark.unit
def test_generate_bench_source_times_the_solve() -> None:
    src = generate_controller_bench_source(
        model_name="demo", nx=4, nu=1, initial_state=(0.3, 0.0, 0.0, 0.0), n_warmup=64
    )
    assert "clock_gettime(CLOCK_MONOTONIC" in src
    assert "demo_acados_solve(ocp)" in src
    assert "JX_N_WARMUP 64L" in src
    assert "solve_ns" in src  # emitted as the escaped C literal \"solve_ns\"
    assert "getrusage(RUSAGE_SELF" in src


@pytest.mark.unit
def test_generate_bench_source_rejects_bad_initial_state() -> None:
    with pytest.raises(BenchmarkError, match="nx=4"):
        generate_controller_bench_source(
            model_name="demo", nx=4, nu=1, initial_state=(0.0,)
        )
