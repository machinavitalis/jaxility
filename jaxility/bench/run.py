# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""End-to-end benchmark orchestration (T-035).

`run_cartpole_benchmark` builds the Cartpole LQR controller, generates the
timing binary, runs it on the chosen target, and returns a
:class:`jaxility.bench.BenchRecord`. Two targets:

* ``"host"`` — compile + run locally (`LocalRunner`). The everywhere-available
  path for CI and for checking the harness; host timings are not a deployment
  claim, only a smoke check.
* ``"pi5"`` — ship + build on a tethered Pi 5 (`build_controller_on_target`) and
  run over SSH. Configured by ``JAXILITY_HIL_SSH_HOST`` and the target acados
  install (``JAXILITY_HIL_ACADOS``, default ``$HOME/acados``). This is the
  real on-silicon measurement.

This is the build-on-target proof path (the on-Pi numbers), not the attested
cross-compiled artifact (T-036). The returned record carries the source
manifest hash so the measurement is tied to the exact controller it ran.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from ..errors import BenchmarkError
from ..hil import (
    LocalRunner,
    build_controller_hil_binary,
    build_controller_on_target,
)
from . import cartpole as _cp
from .controller_bench import generate_controller_bench_source, run_controller_bench
from .record import BenchRecord

BenchTarget = Literal["host", "pi5"]

_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")


def _manifest_hash(bundle) -> str | None:
    manifest = getattr(bundle, "manifest", None)
    if manifest is None:
        return None
    content_hash = getattr(manifest, "content_hash", None)
    # ``Manifest.content_hash`` is a method (build_cmd uses it as such).
    digest = content_hash() if callable(content_hash) else content_hash
    if isinstance(digest, (bytes, bytearray)):
        return bytes(digest).hex()
    if isinstance(digest, str):
        return digest
    return None


def run_cartpole_benchmark(
    *,
    target: BenchTarget,
    work_dir: Path,
    n_cycles: int = 1000,
    n_warmup: int = 100,
    seed: int = 0,
) -> BenchRecord:
    """Build + run the Cartpole benchmark on ``target``; return the record."""
    bundle = _cp.build_cartpole_controller(work_dir / "build")
    gen_dir = bundle.shared_library_path.parent
    source = generate_controller_bench_source(
        model_name=_cp.MODEL_NAME,
        nx=_cp.NX,
        nu=_cp.NU,
        initial_state=_cp.INITIAL_STATE,
        n_warmup=n_warmup,
    )
    manifest_hash = _manifest_hash(bundle)

    if target == "host":
        exe = build_controller_hil_binary(
            generated_code_dir=gen_dir,
            model_name=_cp.MODEL_NAME,
            source=source,
            out_path=gen_dir / f"{_cp.MODEL_NAME}_bench",
        )
        runner: object = LocalRunner(exe)
        target_family, target_name = "host", "host"
    elif target == "pi5":
        host = os.environ.get("JAXILITY_HIL_SSH_HOST")
        if not host:
            raise BenchmarkError(
                "target 'pi5' needs JAXILITY_HIL_SSH_HOST set to the tethered Pi "
                "(e.g. pi@raspberrypi.local)."
            )
        acados = os.environ.get("JAXILITY_HIL_ACADOS")
        if not acados:
            probe = subprocess.run(
                ["ssh", *_SSH_OPTS, host, "echo $HOME/acados"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            acados = probe.stdout.strip() if probe.returncode == 0 else ""
        if not acados:
            raise BenchmarkError(
                "could not resolve the target acados install; set JAXILITY_HIL_ACADOS."
            )
        runner = build_controller_on_target(
            host=host,
            generated_code_dir=gen_dir,
            model_name=_cp.MODEL_NAME,
            source=source,
            source_name=f"{_cp.MODEL_NAME}_bench_main.c",
            remote_dir="/tmp/jaxility-bench",
            remote_acados=acados,
        )
        target_family, target_name = "cortex-a76", "pi5"
    else:  # pragma: no cover - Literal guards this
        raise BenchmarkError(f"unknown benchmark target {target!r}")

    return run_controller_bench(
        runner,  # type: ignore[arg-type]
        robot="cartpole",
        target_family=target_family,
        target_name=target_name,
        n_cycles=n_cycles,
        n_warmup=n_warmup,
        seed=seed,
        source_manifest_hash=manifest_hash,
    )
