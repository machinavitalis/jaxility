# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Fixtures for the HIL tier (T-033).

Two transports are exercised:

* **local** — the fixture is built with the host ``cc`` and run as a
  subprocess. Available wherever a C compiler is (CI, dev box).
* **ssh** — the fixture is built natively on a tethered target and run
  over SSH. Opt-in and self-skipping: it runs only when
  ``JAXILITY_HIL_SSH_HOST`` is set to a reachable destination (e.g.
  ``pi@raspberrypi.local`` for the Pi 5 launch target). CI without
  hardware skips it cleanly; it never hangs on a missing key because the
  probe uses ``BatchMode=yes`` + a short ``ConnectTimeout``.

The on-Pi build mirrors the smoke-test path: copy the single C source
over, compile with the Pi's native ``gcc`` at the Cortex-A76 ISA flags,
and run the resulting binary. When T-034 produces a real cross-compiled
controller, ``SshRunner.deploy_binary`` ships that artifact instead and
this build step falls away.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_FIXTURE_SRC = Path(__file__).parent / "fixtures" / "cartpole_hil.c"

_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")

# Cortex-A76 ISA flags — the same selection jaxility.builder_cross uses
# for the cortex-a76 family, minus the shared-object flags (this is an
# executable). Native gcc on the Pi accepts -mcpu=cortex-a76.
_PI_CFLAGS = ("-mcpu=cortex-a76", "-O2", "-std=c99", "-Wall", "-Wextra")


@pytest.fixture(scope="session")
def local_hil_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the HIL fixture with the host ``cc``; skip if none present."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("no host cc to build the HIL fixture")
    out = tmp_path_factory.mktemp("hil") / "cartpole_hil"
    argv = [
        cc,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        str(_FIXTURE_SRC),
        "-o",
        str(out),
    ]
    compiled = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    return out


def _ssh_host() -> str | None:
    return os.environ.get("JAXILITY_HIL_SSH_HOST") or None


def _reachable(host: str) -> bool:
    try:
        probe = subprocess.run(
            ["ssh", *_SSH_OPTS, host, "true"],
            capture_output=True,
            text=True,
            timeout=12,
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


@pytest.fixture(scope="session")
def remote_hil_binary() -> tuple[str, str]:
    """Build the fixture on the tethered target; return ``(host, remote_path)``.

    Skips unless ``JAXILITY_HIL_SSH_HOST`` names a reachable host with a
    working ``gcc``.
    """
    host = _ssh_host()
    if host is None:
        pytest.skip("JAXILITY_HIL_SSH_HOST not set; HIL ssh tier skipped")
    if not _reachable(host):
        pytest.skip(f"HIL ssh host {host!r} not reachable; tier skipped")

    remote_dir = "/tmp/jaxility-hil"
    remote_src = f"{remote_dir}/cartpole_hil.c"
    remote_bin = f"{remote_dir}/cartpole_hil"

    mk = subprocess.run(
        ["ssh", *_SSH_OPTS, host, f"mkdir -p {remote_dir}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert mk.returncode == 0, mk.stderr

    cp = subprocess.run(
        ["scp", *_SSH_OPTS, str(_FIXTURE_SRC), f"{host}:{remote_src}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cp.returncode == 0, cp.stderr

    build_cmd = f"gcc {' '.join(_PI_CFLAGS)} {remote_src} -o {remote_bin}"
    build = subprocess.run(
        ["ssh", *_SSH_OPTS, host, build_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr
    return host, remote_bin
