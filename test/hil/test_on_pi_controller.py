# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""On-silicon controller HIL parity + benchmark (T-034 completion, T-035).

Builds the real acados Cartpole LQR controller on the host, ships the generated
C to a tethered Pi 5, compiles the HIL and benchmark binaries there against a
target-native acados, and:

* asserts step-locked HIL parity of the on-Pi controller against the host
  reference under `cortex-a76` x float64 (T-034 on real silicon), and
* measures per-cycle solve time + jitter + memory, asserting the 1 kHz budget
  (T-035).

Opt-in and self-skipping: requires the host acados toolchain (t_renderer),
`JAXILITY_HIL_SSH_HOST` set to a reachable target, and a target acados install
(``JAXILITY_HIL_ACADOS``, default ``$HOME/acados``) with its ``lib`` on the
target loader path. A no-op in hardware-free CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cartpole_lqr_reference import (
    MODEL_NAME,
    build_cartpole_lqr,
    closed_loop_reference,
)

from jaxility.bench import generate_controller_bench_source, run_controller_bench
from jaxility.hil import (
    CARTPOLE_LQR_SCHEMA,
    CARTPOLE_LQR_TRACE,
    build_controller_on_target,
    generate_controller_hil_source,
    run_hil,
)

_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")
_N_WARMUP = 100


def _tera() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    return bool(root and (Path(root) / "bin" / "t_renderer").exists()) or (
        shutil.which("t_renderer") is not None
    )


def _ssh(
    host: str, cmd: str, timeout: float = 15.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *_SSH_OPTS, host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def on_pi_runners(tmp_path_factory: pytest.TempPathFactory):
    """Build on host, ship+compile on the Pi; yield (hil_runner, bench_runner)."""
    if not _tera():
        pytest.skip("host acados t_renderer required to build the controller")
    host = os.environ.get("JAXILITY_HIL_SSH_HOST")
    if not host:
        pytest.skip("JAXILITY_HIL_SSH_HOST not set; on-Pi controller tier skipped")
    if _ssh(host, "true").returncode != 0:
        pytest.skip(f"HIL ssh host {host!r} not reachable")

    acados = os.environ.get("JAXILITY_HIL_ACADOS")
    if not acados:
        probe = _ssh(host, "echo $HOME/acados")
        acados = probe.stdout.strip() if probe.returncode == 0 else ""
    if not acados or _ssh(host, f"test -d {acados}/lib").returncode != 0:
        pytest.skip(f"target acados install not found on {host} ({acados!r})")

    bundle = build_cartpole_lqr(tmp_path_factory.mktemp("on_pi") / "build")
    gen_dir = bundle.shared_library_path.parent

    hil_runner = build_controller_on_target(
        host=host,
        generated_code_dir=gen_dir,
        model_name=MODEL_NAME,
        source=generate_controller_hil_source(
            model_name=MODEL_NAME,
            nx=4,
            nu=1,
            trace=CARTPOLE_LQR_TRACE,
            initial_state=(0.3, 0.0, 0.0, 0.0),
        ),
        source_name=f"{MODEL_NAME}_hil_main.c",
        remote_dir="/tmp/jaxility-onpi",
        remote_acados=acados,
    )
    bench_runner = build_controller_on_target(
        host=host,
        generated_code_dir=gen_dir,
        model_name=MODEL_NAME,
        source=generate_controller_bench_source(
            model_name=MODEL_NAME,
            nx=4,
            nu=1,
            initial_state=(0.3, 0.0, 0.0, 0.0),
            n_warmup=_N_WARMUP,
        ),
        source_name=f"{MODEL_NAME}_bench_main.c",
        remote_dir="/tmp/jaxility-onpi-bench",
        remote_acados=acados,
    )
    return bundle, hil_runner, bench_runner


@pytest.mark.hil
def test_controller_hil_parity_on_pi(on_pi_runners) -> None:
    """T-034: the acados controller on real Cortex-A76 matches the host reference."""
    bundle, hil_runner, _ = on_pi_runners
    n_steps, seed = 40, 0
    reference = closed_loop_reference(bundle.solver, n_steps, seed)
    report = run_hil(
        reference,
        hil_runner,
        target_family="cortex-a76",
        dtype="float64",
        schema=CARTPOLE_LQR_SCHEMA,
        n_steps=n_steps,
        seed=seed,
    )
    report.assert_passed()
    assert report.passed
    assert report.runner_label.startswith("ssh:")


@pytest.mark.hil
def test_controller_benchmark_on_pi(on_pi_runners) -> None:
    """T-035: per-cycle solve time on the Pi 5 meets the 1 kHz budget."""
    _, _, bench_runner = on_pi_runners
    record = run_controller_bench(
        bench_runner,
        robot="cartpole",
        target_family="cortex-a76",
        target_name="pi5",
        n_cycles=1000,
        n_warmup=_N_WARMUP,
        seed=0,
    )
    assert record.n_cycles == 1000
    assert record.solve.min_ns > 0
    assert record.solve.max_ns >= record.solve.min_ns
    # The Cartpole LQR is comfortably real-time on the Pi 5.
    assert record.meets_1khz, (
        f"worst-case solve {record.solve.max_ns / 1e3:.1f} us exceeds the "
        f"1 kHz (1000 us) budget"
    )
    assert record.max_rss_kib > 0
