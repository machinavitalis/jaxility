# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""HIL parity on a tethered target over SSH (T-033).

The same step-locked parity check as ``test_hil_local.py``, but the
artifact runs on real silicon — the Raspberry Pi 5 / Cortex-A76 launch
target, or any host named by ``JAXILITY_HIL_SSH_HOST``. The
``remote_hil_binary`` fixture builds the fixture natively on the target
and self-skips when no reachable host is configured, so this is a no-op
in hardware-free CI and a real on-silicon gate when a Pi is tethered.

This is the closest thing in the suite to invariant 6 (HIL parity is the
release gate): the float32 artifact running on the actual deployment
chip must match the host float64 reference within the documented
``cortex-a76`` tolerances.
"""

from __future__ import annotations

import pytest
from cartpole_reference import cartpole_reference

from jaxility.hil import CARTPOLE_SCHEMA, SshRunner, run_hil


@pytest.mark.hil
def test_ssh_hil_parity_on_target(remote_hil_binary) -> None:
    host, remote_bin = remote_hil_binary
    n_steps, seed = 200, 0
    reference = cartpole_reference(n_steps, seed)
    runner = SshRunner(host=host, remote_executable=remote_bin)
    report = run_hil(
        reference,
        runner,
        target_family="cortex-a76",
        dtype="float32",
        schema=CARTPOLE_SCHEMA,
        n_steps=n_steps,
        seed=seed,
    )
    report.assert_passed()
    assert report.passed
    assert report.runner_label.startswith(f"ssh:{host}:")


@pytest.mark.hil
def test_ssh_hil_is_deterministic_on_target(remote_hil_binary) -> None:
    host, remote_bin = remote_hil_binary
    runner = SshRunner(host=host, remote_executable=remote_bin)
    a = runner.run(n_steps=64, seed=2)
    b = runner.run(n_steps=64, seed=2)
    assert a == b
