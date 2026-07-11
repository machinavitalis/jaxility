# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""HIL harness tests via the local-subprocess transport (T-033).

Two layers:

* **trace parser** — pure-Python contract tests for
  :func:`jaxility.hil.parse_trace`: the happy path and every loud-fail
  mode (malformed JSON, wrong step order, schema mismatch, wrong count).
* **end-to-end local parity** — build the deterministic fixture with the
  host ``cc`` (the ``local_hil_binary`` fixture), run it through
  :class:`LocalRunner` + :func:`run_hil`, and assert step-locked parity
  against the float64 host reference under the ``cortex-a76`` × float32
  tolerances. This validates the whole harness without hardware; the
  ssh tier (``test_hil_ssh.py``) re-runs the same check on real silicon.
"""

from __future__ import annotations

import numpy as np
import pytest
from cartpole_reference import cartpole_reference

from jaxility.errors import EquivalenceError, HILError
from jaxility.hil import (
    CARTPOLE_SCHEMA,
    LocalRunner,
    parse_trace,
    run_hil,
)

# ---------------------------------------------------------------------------
# Trace parser contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_trace_happy_path() -> None:
    stdout = (
        '{"step": 0, "joint_position": [0.2], "joint_velocity": [0.0], '
        '"actuator_torque": [-1.6]}\n'
        '{"step": 1, "joint_position": [0.1], "joint_velocity": [-0.5], '
        '"actuator_torque": [-0.3]}\n'
    )
    traj = parse_trace(stdout, CARTPOLE_SCHEMA, n_steps=2)
    assert set(traj) == {"joint_position", "joint_velocity", "actuator_torque"}
    assert traj["joint_position"].shape == (2, 1)
    assert traj["joint_position"][0, 0] == pytest.approx(0.2)
    assert traj["actuator_torque"][1, 0] == pytest.approx(-0.3)


@pytest.mark.unit
def test_parse_trace_rejects_wrong_count() -> None:
    stdout = (
        '{"step": 0, "joint_position": [0.2], "joint_velocity": [0.0], '
        '"actuator_torque": [0.0]}\n'
    )
    with pytest.raises(HILError, match="expected 3"):
        parse_trace(stdout, CARTPOLE_SCHEMA, n_steps=3)


@pytest.mark.unit
def test_parse_trace_rejects_out_of_order() -> None:
    stdout = (
        '{"step": 0, "joint_position": [0.2], "joint_velocity": [0.0], '
        '"actuator_torque": [0.0]}\n'
        '{"step": 5, "joint_position": [0.1], "joint_velocity": [0.0], '
        '"actuator_torque": [0.0]}\n'
    )
    with pytest.raises(HILError, match="out of order"):
        parse_trace(stdout, CARTPOLE_SCHEMA, n_steps=2)


@pytest.mark.unit
def test_parse_trace_rejects_schema_mismatch() -> None:
    stdout = '{"step": 0, "joint_position": [0.2], "joint_velocity": [0.0]}\n'
    with pytest.raises(HILError, match="do not match the schema"):
        parse_trace(stdout, CARTPOLE_SCHEMA, n_steps=1)


@pytest.mark.unit
def test_parse_trace_rejects_bad_shape() -> None:
    stdout = (
        '{"step": 0, "joint_position": [0.2, 0.3], "joint_velocity": [0.0], '
        '"actuator_torque": [0.0]}\n'
    )
    with pytest.raises(HILError, match="has shape"):
        parse_trace(stdout, CARTPOLE_SCHEMA, n_steps=1)


@pytest.mark.unit
def test_parse_trace_rejects_non_json() -> None:
    with pytest.raises(HILError, match="not valid JSON"):
        parse_trace("Segmentation fault\n", CARTPOLE_SCHEMA, n_steps=1)


# ---------------------------------------------------------------------------
# End-to-end local parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_local_hil_parity_passes(local_hil_binary) -> None:
    n_steps, seed = 200, 0
    reference = cartpole_reference(n_steps, seed)
    runner = LocalRunner(local_hil_binary)
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
    assert report.runner_label.startswith("local:")
    assert all(qd.passed for qd in report.equivalence.per_quantity)


@pytest.mark.unit
def test_local_hil_is_deterministic(local_hil_binary) -> None:
    runner = LocalRunner(local_hil_binary)
    a = runner.run(n_steps=50, seed=3)
    b = runner.run(n_steps=50, seed=3)
    assert a == b


@pytest.mark.unit
def test_local_hil_seed_changes_trajectory(local_hil_binary) -> None:
    runner = LocalRunner(local_hil_binary)
    a = parse_trace(runner.run(n_steps=20, seed=0), CARTPOLE_SCHEMA, n_steps=20)
    b = parse_trace(runner.run(n_steps=20, seed=7), CARTPOLE_SCHEMA, n_steps=20)
    assert not np.allclose(a["joint_position"], b["joint_position"])


@pytest.mark.unit
def test_local_hil_detects_divergence(local_hil_binary) -> None:
    """A reference that does not match the artifact must fail parity loudly."""
    n_steps, seed = 200, 0
    reference = cartpole_reference(n_steps, seed)
    # Corrupt the reference well beyond tolerance.
    reference["joint_position"] = reference["joint_position"] + 1.0
    runner = LocalRunner(local_hil_binary)
    report = run_hil(
        reference,
        runner,
        target_family="cortex-a76",
        dtype="float32",
        schema=CARTPOLE_SCHEMA,
        n_steps=n_steps,
        seed=seed,
    )
    assert not report.passed
    with pytest.raises(EquivalenceError):
        report.assert_passed()
