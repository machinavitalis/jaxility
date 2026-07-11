# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the WBC template (T-024)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.errors import TargetError
from jaxility.lowering import OcpTemplateSpec, build_ocp, translate
from jaxility.templates import WBCTask, lqr, wbc
from jaxility.templates.wbc import WBC_SCHEMA_V0


def _manipulator_2dof(state, control):
    """A trivial 2-DOF manipulator: dx = [qd; u] (no gravity)."""
    return jnp.array([state[2], state[3], control[0], control[1]])


@pytest.fixture(scope="module")
def manip_cf():
    return translate(_manipulator_2dof, in_shapes=((4,), (2,)), name="manip")


def _tera_renderer_available() -> bool:
    import os
    import shutil

    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if acados_root and (Path(acados_root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA_AVAILABLE = _tera_renderer_available()


# ---------------------------------------------------------------------------
# WBCTask schema.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wbc_task_has_schema_version() -> None:
    task = WBCTask(
        name="t1",
        priority=1.0,
        state_weight=(1.0, 1.0),
        input_weight=(0.1,),
        state_reference=(0.0, 0.0),
        input_reference=(0.0,),
    )
    assert task.schema_version == WBC_SCHEMA_V0


@pytest.mark.unit
def test_wbc_task_rejects_inconsistent_state_lengths() -> None:
    with pytest.raises(ValueError, match="state_weight length"):
        WBCTask(
            name="bad",
            priority=1.0,
            state_weight=(1.0, 1.0),  # length 2
            input_weight=(0.1,),
            state_reference=(0.0,),  # length 1 — mismatch
            input_reference=(0.0,),
        )


@pytest.mark.unit
def test_wbc_task_is_frozen() -> None:
    task = WBCTask(
        name="t",
        priority=1.0,
        state_weight=(1.0,),
        input_weight=(0.1,),
        state_reference=(0.0,),
        input_reference=(0.0,),
    )
    with pytest.raises(ValueError):
        task.priority = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# wbc(): combination rules.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wbc_single_task_reduces_to_lqr_shape(manip_cf) -> None:
    """A single task with weight 1 reproduces an LQR with the same diagonal."""
    spec_wbc = wbc(
        manip_cf,
        tasks=[
            WBCTask(
                name="track",
                priority=1.0,
                state_weight=(10.0, 10.0, 1.0, 1.0),
                input_weight=(0.1, 0.1),
                state_reference=(0.5, 0.5, 0.0, 0.0),
                input_reference=(0.0, 0.0),
            )
        ],
        initial_state=(0.0, 0.0, 0.0, 0.0),
        name="single-task-wbc",
    )
    spec_lqr = lqr(
        manip_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1, 0.1),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        state_reference=(0.5, 0.5, 0.0, 0.0),
        name="lqr",
    )
    assert spec_wbc.state_cost == spec_lqr.state_cost
    assert spec_wbc.input_cost == spec_lqr.input_cost
    np.testing.assert_allclose(spec_wbc.state_reference, spec_lqr.state_reference)
    np.testing.assert_allclose(spec_wbc.input_reference, spec_lqr.input_reference)


@pytest.mark.unit
def test_wbc_two_tasks_priority_weighted_combination(manip_cf) -> None:
    """Two tasks combine: Q = sum(prio * sw); yref weighted average."""
    task_a = WBCTask(
        name="A",
        priority=1.0,
        state_weight=(2.0, 0.0, 0.0, 0.0),
        input_weight=(0.0, 0.0),
        state_reference=(1.0, 0.0, 0.0, 0.0),
        input_reference=(0.0, 0.0),
    )
    task_b = WBCTask(
        name="B",
        priority=3.0,
        state_weight=(4.0, 0.0, 0.0, 0.0),
        input_weight=(0.0, 0.0),
        state_reference=(-1.0, 0.0, 0.0, 0.0),
        input_reference=(0.0, 0.0),
    )
    spec = wbc(
        manip_cf,
        tasks=[task_a, task_b],
        initial_state=(0.0, 0.0, 0.0, 0.0),
    )
    # Q[0] = 1*2 + 3*4 = 14.
    assert spec.state_cost[0] == pytest.approx(14.0)
    # yref[0] = (1*2*1.0 + 3*4*-1.0) / 14 = -10/14.
    assert spec.state_reference[0] == pytest.approx(-10.0 / 14.0)
    # Components with no weight collapse to 1e-9.
    assert spec.state_cost[1] == pytest.approx(1e-9)
    assert spec.state_reference[1] == 0.0


@pytest.mark.unit
def test_wbc_empty_tasks_rejected(manip_cf) -> None:
    with pytest.raises(TargetError, match="at least one WBCTask"):
        wbc(
            manip_cf,
            tasks=[],
            initial_state=(0.0, 0.0, 0.0, 0.0),
        )


@pytest.mark.unit
def test_wbc_rejects_task_state_weight_length_mismatch(manip_cf) -> None:
    bad_task = WBCTask(
        name="bad",
        priority=1.0,
        state_weight=(1.0, 1.0),  # nx=4 expected
        input_weight=(0.1, 0.1),
        state_reference=(0.0, 0.0),
        input_reference=(0.0, 0.0),
    )
    with pytest.raises(TargetError, match="state_weight length"):
        wbc(manip_cf, tasks=[bad_task], initial_state=(0.0, 0.0, 0.0, 0.0))


@pytest.mark.unit
def test_wbc_default_q_terminal(manip_cf) -> None:
    spec = wbc(
        manip_cf,
        tasks=[
            WBCTask(
                name="t",
                priority=1.0,
                state_weight=(2.0, 2.0, 1.0, 1.0),
                input_weight=(0.1, 0.1),
                state_reference=(0.0, 0.0, 0.0, 0.0),
                input_reference=(0.0, 0.0),
            )
        ],
        initial_state=(0.0, 0.0, 0.0, 0.0),
        Q_terminal_factor=5.0,
    )
    assert spec.terminal_state_cost == (10.0, 10.0, 5.0, 5.0)


@pytest.mark.unit
def test_wbc_input_bounds_propagate(manip_cf) -> None:
    spec = wbc(
        manip_cf,
        tasks=[
            WBCTask(
                name="t",
                priority=1.0,
                state_weight=(1.0, 1.0, 1.0, 1.0),
                input_weight=(0.1, 0.1),
                state_reference=(0.0, 0.0, 0.0, 0.0),
                input_reference=(0.0, 0.0),
            )
        ],
        initial_state=(0.0, 0.0, 0.0, 0.0),
        input_bounds=((-1.0, -1.0), (1.0, 1.0)),
    )
    assert spec.input_lower == (-1.0, -1.0)
    assert spec.input_upper == (1.0, 1.0)


# ---------------------------------------------------------------------------
# End-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wbc_returns_ocp_template_spec(manip_cf) -> None:
    spec = wbc(
        manip_cf,
        tasks=[
            WBCTask(
                name="t",
                priority=1.0,
                state_weight=(10.0, 10.0, 1.0, 1.0),
                input_weight=(0.1, 0.1),
                state_reference=(0.0, 0.0, 0.0, 0.0),
                input_reference=(0.0, 0.0),
            )
        ],
        initial_state=(0.3, 0.3, 0.0, 0.0),
    )
    assert isinstance(spec, OcpTemplateSpec)


@pytest.mark.unit
@pytest.mark.skipif(not _TERA_AVAILABLE, reason="t_renderer required.")
def test_wbc_manipulator_solves(
    tmp_path: Path, manip_cf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two tasks on a 2-DOF manipulator: stabilise and damp velocity."""
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    stabilise = WBCTask(
        name="stabilise",
        priority=10.0,
        state_weight=(1.0, 1.0, 0.0, 0.0),
        input_weight=(0.0, 0.0),
        state_reference=(0.0, 0.0, 0.0, 0.0),
        input_reference=(0.0, 0.0),
    )
    damp = WBCTask(
        name="damp",
        priority=1.0,
        state_weight=(0.0, 0.0, 1.0, 1.0),
        input_weight=(0.05, 0.05),
        state_reference=(0.0, 0.0, 0.0, 0.0),
        input_reference=(0.0, 0.0),
    )
    spec = wbc(
        manip_cf,
        tasks=[stabilise, damp],
        initial_state=(0.5, -0.3, 0.0, 0.0),
        horizon_steps=10,
        time_horizon_s=0.5,
        input_bounds=((-5.0, -5.0), (5.0, 5.0)),
        name="wbc_e2e",
    )
    ocp = build_ocp(manip_cf, spec)
    solver = AcadosOcpSolver(ocp, json_file="wbc_e2e.json", verbose=False)
    status = solver.solve()
    assert status == 0
    u0 = solver.get(0, "u")
    assert np.all(np.isfinite(u0))
    # Controller should drive joint positions back toward zero — final
    # |q| smaller than initial.
    x_final = solver.get(spec.horizon_steps, "x")
    assert np.all(np.isfinite(x_final))
    init_norm = np.linalg.norm([0.5, -0.3])
    final_norm = float(np.linalg.norm(x_final[:2]))
    assert final_norm < init_norm
