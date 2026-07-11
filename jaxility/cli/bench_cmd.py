# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""``jaxility bench`` subcommand implementation (T-035)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SUPPORTED_ROBOTS = ("cartpole",)
_SUPPORTED_TARGETS = ("host", "pi5")


def run_bench(
    *,
    robot: str,
    target_name: str,
    n_cycles: int,
    n_warmup: int,
    seed: int,
    out: str | None,
) -> int:
    """Implement ``jaxility bench <robot> --target <host|pi5>``.

    Builds the controller, runs the timing binary on the target, and writes
    the :class:`jaxility.bench.BenchRecord` as JSON (to ``--out`` or stdout).
    Exit codes: ``0`` success, ``2`` invalid arguments, ``1`` benchmark failure.
    """
    if robot not in _SUPPORTED_ROBOTS:
        print(
            f"unknown robot {robot!r}; supported: {', '.join(_SUPPORTED_ROBOTS)}",
            file=sys.stderr,
        )
        return 2
    if target_name not in _SUPPORTED_TARGETS:
        print(
            f"unknown target {target_name!r}; supported: "
            f"{', '.join(_SUPPORTED_TARGETS)}",
            file=sys.stderr,
        )
        return 2
    if n_cycles < 1:
        print("--cycles must be >= 1", file=sys.stderr)
        return 2

    from ..bench import run_cartpole_benchmark
    from ..errors import JaxilityError

    try:
        with tempfile.TemporaryDirectory(prefix="jaxility-bench-") as tmp:
            record = run_cartpole_benchmark(
                target=target_name,  # type: ignore[arg-type]
                work_dir=Path(tmp),
                n_cycles=n_cycles,
                n_warmup=n_warmup,
                seed=seed,
            )
    except JaxilityError as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1

    payload = record.model_dump_json(indent=2)
    if out is not None:
        Path(out).expanduser().write_text(payload + "\n")
    else:
        print(payload, file=sys.stdout)
    return 0


__all__ = ["run_bench"]
