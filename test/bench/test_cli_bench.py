# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""``jaxility bench`` CLI tests (T-035)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from jaxility.bench import BenchRecord
from jaxility.cli import main


@pytest.mark.unit
def test_bench_cli_rejects_unknown_robot(capsys) -> None:
    rc = main(["bench", "spaceship", "--target", "host"])
    assert rc == 2
    assert "unknown robot" in capsys.readouterr().err


@pytest.mark.unit
def test_bench_cli_rejects_unknown_target(capsys) -> None:
    rc = main(["bench", "cartpole", "--target", "mars"])
    assert rc == 2
    assert "unknown target" in capsys.readouterr().err


def _tera() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    return bool(root and (Path(root) / "bin" / "t_renderer").exists()) or (
        shutil.which("t_renderer") is not None
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not _tera() or shutil.which("cc") is None,
    reason="acados t_renderer + host cc required for the host benchmark",
)
def test_bench_cli_host_emits_valid_record(capsys) -> None:
    rc = main(
        ["bench", "cartpole", "--target", "host", "--cycles", "50", "--warmup", "20"]
    )
    assert rc == 0
    record = BenchRecord.model_validate(json.loads(capsys.readouterr().out))
    assert record.robot == "cartpole"
    assert record.target_name == "host"
    assert record.n_cycles == 50
    assert record.solve.min_ns > 0
    assert record.source_manifest_hash is not None
