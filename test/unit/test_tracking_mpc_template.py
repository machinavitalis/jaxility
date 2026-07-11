# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the TrackingMPC template (T-023)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.errors import TargetError
from jaxility.lowering import OcpTemplateSpec, build_ocp, translate
from jaxility.templates import set_reference_trajectory, tracking_mpc


def _double_integrator(state, control):
    # 2-state double integrator: dx = [v; u]
    return jnp.array([state[1], control[0]])


@pytest.fixture(scope="module")
def double_int_cf():
    return translate(_double_integrator, in_shapes=((2,), (1,)), name="doubleint")


def _tera_renderer_available() -> bool:
    import os
    import shutil

    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if acados_root and (Path(acados_root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA_AVAILABLE = _tera_renderer_available()


# ---------------------------------------------------------------------------
# Spec shape and defaults.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tracking_mpc_returns_ocp_template_spec(double_int_cf) -> None:
    spec = tracking_mpc(
        double_int_cf,
        Q=(10.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0),
    )
    assert isinstance(spec, OcpTemplateSpec)


@pytest.mark.unit
def test_tracking_mpc_defaults_match_lqr(double_int_cf) -> None:
    """Without a reference_trajectory, behaviour is LQR-equivalent."""
    spec = tracking_mpc(
        double_int_cf,
        Q=(1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0),
    )
    assert spec.state_reference == (0.0, 0.0)
    assert spec.input_reference == (0.0,)
    assert spec.terminal_state_cost == (10.0, 10.0)


@pytest.mark.unit
def test_tracking_mpc_seeds_yref_from_first_reference(double_int_cf) -> None:
    """The first trajectory stage seeds the build-time ``state_reference``."""
    traj = [(0.5, 0.1), (0.6, 0.2), (0.7, 0.3)]
    spec = tracking_mpc(
        double_int_cf,
        Q=(1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0),
        reference_trajectory=traj,
    )
    assert spec.state_reference == (0.5, 0.1)


@pytest.mark.unit
def test_tracking_mpc_rejects_empty_trajectory(double_int_cf) -> None:
    with pytest.raises(TargetError, match="reference_trajectory is empty"):
        tracking_mpc(
            double_int_cf,
            Q=(1.0, 1.0),
            R=(0.1,),
            initial_state=(0.0, 0.0),
            reference_trajectory=[],
        )


@pytest.mark.unit
def test_tracking_mpc_propagates_horizon_options(double_int_cf) -> None:
    spec = tracking_mpc(
        double_int_cf,
        Q=(1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0),
        horizon_steps=40,
        time_horizon_s=2.0,
        Q_terminal_factor=20.0,
        name="custom-tracker",
    )
    assert spec.horizon_steps == 40
    assert spec.time_horizon_s == 2.0
    assert spec.terminal_state_cost == (20.0, 20.0)
    assert spec.name == "custom-tracker"


# ---------------------------------------------------------------------------
# set_reference_trajectory: runtime updater.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    not _TERA_AVAILABLE,
    reason="t_renderer required to construct AcadosOcpSolver and test cost_set.",
)
def test_set_reference_trajectory_updates_solver_yref(
    tmp_path: Path, double_int_cf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage yref = concat(state_ref[k], input_ref); terminal yref state-only."""
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    spec = tracking_mpc(
        double_int_cf,
        Q=(10.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0),
        horizon_steps=5,
        time_horizon_s=0.5,
        name="di_runtime_update",
    )
    ocp = build_ocp(double_int_cf, spec)
    solver = AcadosOcpSolver(ocp, json_file="di_runtime.json", verbose=False)

    # Reference trajectory length N + 1 = 6.
    traj = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0), (0.6, 0.0)]
    set_reference_trajectory(solver, traj)

    # Cross-check via cost_get on a couple of stages.
    yref_0 = solver.cost_get(0, "yref")
    yref_2 = solver.cost_get(2, "yref")
    yref_e = solver.cost_get(5, "yref")
    np.testing.assert_allclose(yref_0, [0.1, 0.0, 0.0])  # state + zero input
    np.testing.assert_allclose(yref_2, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(yref_e, [0.6, 0.0])  # terminal: state-only


@pytest.mark.unit
@pytest.mark.skipif(not _TERA_AVAILABLE, reason="t_renderer required.")
def test_set_reference_trajectory_rejects_short_trajectory(
    tmp_path: Path, double_int_cf, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    spec = tracking_mpc(
        double_int_cf,
        Q=(10.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0),
        horizon_steps=5,
        time_horizon_s=0.5,
        name="di_short_traj",
    )
    ocp = build_ocp(double_int_cf, spec)
    solver = AcadosOcpSolver(ocp, json_file="di_short.json", verbose=False)

    too_short = [(0.1, 0.0)] * 3  # need 6
    with pytest.raises(TargetError, match="too short"):
        set_reference_trajectory(solver, too_short)


# ---------------------------------------------------------------------------
# End-to-end: tracking MPC follows a sinusoidal reference.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _TERA_AVAILABLE, reason="t_renderer required.")
def test_tracking_mpc_double_integrator_follows_step_reference(
    tmp_path: Path, double_int_cf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The controller drives the double integrator toward a step reference."""
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    n = 10
    spec = tracking_mpc(
        double_int_cf,
        Q=(50.0, 1.0),
        R=(0.01,),
        initial_state=(0.0, 0.0),
        horizon_steps=n,
        time_horizon_s=1.0,
        name="di_step_track",
        input_bounds=((-5.0,), (5.0,)),
    )
    ocp = build_ocp(double_int_cf, spec)
    solver = AcadosOcpSolver(ocp, json_file="di_step.json", verbose=False)

    # Target step: position = 1.0 from stage 0 onward.
    traj = [(1.0, 0.0)] * (n + 1)
    set_reference_trajectory(solver, traj)

    status = solver.solve()
    assert status == 0

    # First control input should be positive (push toward x = 1.0).
    u0 = solver.get(0, "u")
    assert np.all(np.isfinite(u0))
    assert u0[0] > 0.0

    # Final predicted state should be closer to (1.0, 0.0) than the initial (0, 0).
    x_final = solver.get(n, "x")
    assert np.all(np.isfinite(x_final))
    assert abs(x_final[0] - 1.0) < abs(0.0 - 1.0)
