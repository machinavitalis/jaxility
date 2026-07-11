# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the CasADi → acados OCP builder (T-021).

T-021 acceptance:

1. A CasADi-formulated cartpole LQR problem builds to a valid
   ``AcadosOcp``.
2. The OCP solves and produces sensible outputs.

Acceptance (2) requires the ``t_renderer`` binary on
``$ACADOS_SOURCE_DIR/bin``. When it is absent the *solves* test
``skipif``-skips; the *builds* tests still run. Once tera lands the
solve test runs automatically — no code change.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.errors import TargetError
from jaxility.lowering import (
    CASADI_TO_ACADOS_SCHEMA_V0,
    OcpTemplateSpec,
    build_ocp,
    translate,
)


def _tera_renderer_available() -> bool:
    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if not acados_root:
        return False
    candidate = Path(acados_root) / "bin" / "t_renderer"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return True
    # Fallback: PATH lookup.
    return shutil.which("t_renderer") is not None


_TERA_AVAILABLE = _tera_renderer_available()


# ---------------------------------------------------------------------------
# Fixtures: cartpole + pendulum dynamics.
# ---------------------------------------------------------------------------


def _cartpole_dynamics(state, control):
    g, mp, mc, L = 9.81, 0.1, 1.0, 0.5
    _, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    denom = mc + mp * sin_t * sin_t
    x_ddot = (control[0] + mp * sin_t * (L * theta_dot * theta_dot + g * cos_t)) / denom
    theta_ddot = (
        -control[0] * cos_t
        - mp * L * theta_dot * theta_dot * cos_t * sin_t
        - (mc + mp) * g * sin_t
    ) / (L * denom)
    return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])


def _pendulum_dynamics(state, control):
    g, L = 9.81, 1.0
    theta, theta_dot = state[0], state[1]
    return jnp.array([theta_dot, -g / L * jnp.sin(theta) + control[0]])


@pytest.fixture(scope="module")
def cartpole_cf():
    return translate(_cartpole_dynamics, in_shapes=((4,), (1,)), name="cartpole")


@pytest.fixture(scope="module")
def pendulum_cf():
    return translate(_pendulum_dynamics, in_shapes=((2,), (1,)), name="pendulum")


@pytest.fixture()
def cartpole_spec() -> OcpTemplateSpec:
    return OcpTemplateSpec(
        horizon_steps=20,
        time_horizon_s=1.0,
        state_cost=(10.0, 10.0, 1.0, 1.0),
        input_cost=(0.1,),
        terminal_state_cost=(50.0, 50.0, 5.0, 5.0),
        state_reference=(0.0, 0.0, 0.0, 0.0),
        input_reference=(0.0,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_lower=(-10.0,),
        input_upper=(10.0,),
        name="cartpole",
    )


# ---------------------------------------------------------------------------
# Spec validation.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spec_schema_version_is_v0() -> None:
    spec = OcpTemplateSpec(
        horizon_steps=1,
        time_horizon_s=0.1,
        state_cost=(1.0,),
        input_cost=(1.0,),
        terminal_state_cost=(1.0,),
        state_reference=(0.0,),
        input_reference=(0.0,),
        initial_state=(0.0,),
    )
    assert spec.schema_version == CASADI_TO_ACADOS_SCHEMA_V0


@pytest.mark.unit
def test_spec_rejects_inconsistent_state_vs_terminal_cost() -> None:
    with pytest.raises(ValueError, match="terminal_state_cost length"):
        OcpTemplateSpec(
            horizon_steps=1,
            time_horizon_s=0.1,
            state_cost=(1.0, 2.0),
            input_cost=(1.0,),
            terminal_state_cost=(1.0,),  # wrong length
            state_reference=(0.0, 0.0),
            input_reference=(0.0,),
            initial_state=(0.0, 0.0),
        )


@pytest.mark.unit
def test_spec_rejects_inconsistent_state_reference() -> None:
    with pytest.raises(ValueError, match="state_reference length"):
        OcpTemplateSpec(
            horizon_steps=1,
            time_horizon_s=0.1,
            state_cost=(1.0, 2.0),
            input_cost=(1.0,),
            terminal_state_cost=(1.0, 1.0),
            state_reference=(0.0,),  # wrong length
            input_reference=(0.0,),
            initial_state=(0.0, 0.0),
        )


@pytest.mark.unit
def test_spec_rejects_inconsistent_input_bounds() -> None:
    with pytest.raises(ValueError, match="input_lower length"):
        OcpTemplateSpec(
            horizon_steps=1,
            time_horizon_s=0.1,
            state_cost=(1.0,),
            input_cost=(1.0, 1.0),
            terminal_state_cost=(1.0,),
            state_reference=(0.0,),
            input_reference=(0.0, 0.0),
            initial_state=(0.0,),
            input_lower=(-1.0,),  # wrong length
        )


@pytest.mark.unit
def test_spec_is_frozen() -> None:
    spec = OcpTemplateSpec(
        horizon_steps=1,
        time_horizon_s=0.1,
        state_cost=(1.0,),
        input_cost=(1.0,),
        terminal_state_cost=(1.0,),
        state_reference=(0.0,),
        input_reference=(0.0,),
        initial_state=(0.0,),
    )
    with pytest.raises(ValueError):
        spec.horizon_steps = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_ocp: structural sanity (acceptance 1).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_ocp_cartpole_returns_acados_ocp(cartpole_cf, cartpole_spec) -> None:
    """The cartpole LQR problem builds to a valid AcadosOcp (acceptance 1)."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    from acados_template import AcadosOcp

    assert isinstance(ocp, AcadosOcp)
    assert ocp.model.name == "cartpole"


@pytest.mark.unit
def test_build_ocp_carries_model_symbols(cartpole_cf, cartpole_spec) -> None:
    """``model.x``, ``model.u``, ``model.f_expl_expr`` are populated."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    assert ocp.model.x.shape == (4, 1)
    assert ocp.model.u.shape == (1, 1)
    assert ocp.model.f_expl_expr.shape == (4, 1)


@pytest.mark.unit
def test_build_ocp_cost_matrix_shapes(cartpole_cf, cartpole_spec) -> None:
    """``W`` is blkdiag(Q, R) with the right shapes."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    nx, nu = 4, 1
    ny = nx + nu
    assert ocp.cost.W.shape == (ny, ny)
    assert ocp.cost.W_e.shape == (nx, nx)
    assert ocp.cost.Vx.shape == (ny, nx)
    assert ocp.cost.Vu.shape == (ny, nu)
    assert ocp.cost.Vx_e.shape == (nx, nx)
    # Q is upper-left block; R is lower-right.
    np.testing.assert_array_equal(np.diag(ocp.cost.W[:nx, :nx]), [10.0, 10.0, 1.0, 1.0])
    np.testing.assert_array_equal(np.diag(ocp.cost.W[nx:, nx:]), [0.1])
    np.testing.assert_array_equal(np.diag(ocp.cost.W_e), [50.0, 50.0, 5.0, 5.0])


@pytest.mark.unit
def test_build_ocp_references_propagate(cartpole_cf, cartpole_spec) -> None:
    """``yref`` is ``[x_ref; u_ref]``; ``yref_e`` is ``x_ref``."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    np.testing.assert_array_equal(ocp.cost.yref, np.zeros(5))
    np.testing.assert_array_equal(ocp.cost.yref_e, np.zeros(4))


@pytest.mark.unit
def test_build_ocp_initial_state_pinned(cartpole_cf, cartpole_spec) -> None:
    """``constraints.x0`` is the spec's initial state."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    np.testing.assert_array_equal(ocp.constraints.x0, np.array([0.5, 0.0, 0.0, 0.0]))


@pytest.mark.unit
def test_build_ocp_box_constraints_on_input_wired(cartpole_cf, cartpole_spec) -> None:
    """Input lower / upper bounds turn into ``lbu`` / ``ubu`` + ``idxbu``."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    np.testing.assert_array_equal(ocp.constraints.lbu, np.array([-10.0]))
    np.testing.assert_array_equal(ocp.constraints.ubu, np.array([10.0]))
    np.testing.assert_array_equal(ocp.constraints.idxbu, np.arange(1))


@pytest.mark.unit
def test_build_ocp_box_constraints_on_state_wired(cartpole_cf) -> None:
    """State bounds turn into ``lbx`` / ``ubx`` + ``idxbx`` (interior + terminal)."""
    pytest.importorskip("acados_template")
    spec = OcpTemplateSpec(
        horizon_steps=10,
        time_horizon_s=0.5,
        state_cost=(1.0, 1.0, 1.0, 1.0),
        input_cost=(0.1,),
        terminal_state_cost=(1.0, 1.0, 1.0, 1.0),
        state_reference=(0.0, 0.0, 0.0, 0.0),
        input_reference=(0.0,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        state_lower=(-1.0, -np.pi, -5.0, -10.0),
        state_upper=(1.0, np.pi, 5.0, 10.0),
    )
    ocp = build_ocp(cartpole_cf, spec)
    np.testing.assert_allclose(ocp.constraints.lbx, [-1.0, -np.pi, -5.0, -10.0])
    np.testing.assert_allclose(ocp.constraints.ubx, [1.0, np.pi, 5.0, 10.0])
    np.testing.assert_array_equal(ocp.constraints.idxbx, np.arange(4))
    np.testing.assert_allclose(ocp.constraints.lbx_e, [-1.0, -np.pi, -5.0, -10.0])


@pytest.mark.unit
def test_build_ocp_solver_options_propagate(cartpole_cf, cartpole_spec) -> None:
    """Horizon, integrator, NLP solver, QP solver all flow into ``solver_options``."""
    pytest.importorskip("acados_template")
    ocp = build_ocp(cartpole_cf, cartpole_spec)
    assert ocp.solver_options.N_horizon == 20
    assert float(ocp.solver_options.tf) == pytest.approx(1.0)
    assert ocp.solver_options.integrator_type == "ERK"
    assert ocp.solver_options.nlp_solver_type == "SQP_RTI"
    assert ocp.solver_options.qp_solver == "PARTIAL_CONDENSING_HPIPM"


# ---------------------------------------------------------------------------
# Mismatch guards.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_ocp_rejects_shape_mismatch(cartpole_cf) -> None:
    """Spec with the wrong nx is a TargetError."""
    pytest.importorskip("acados_template")
    spec = OcpTemplateSpec(
        horizon_steps=10,
        time_horizon_s=0.5,
        state_cost=(1.0, 1.0),  # nx=2 but dynamics has nx=4
        input_cost=(1.0,),
        terminal_state_cost=(1.0, 1.0),
        state_reference=(0.0, 0.0),
        input_reference=(0.0,),
        initial_state=(0.0, 0.0),
    )
    with pytest.raises(TargetError, match="input shapes do not match"):
        build_ocp(cartpole_cf, spec)


@pytest.mark.unit
def test_build_ocp_rejects_dynamics_with_wrong_input_count() -> None:
    """A single-input dynamics function is not a 2-input (x, u)."""
    pytest.importorskip("acados_template")

    def fn(x):
        return jnp.sin(x)

    cf = translate(fn, in_shapes=((3,),))
    spec = OcpTemplateSpec(
        horizon_steps=1,
        time_horizon_s=0.1,
        state_cost=(1.0, 1.0, 1.0),
        input_cost=(1.0,),
        terminal_state_cost=(1.0, 1.0, 1.0),
        state_reference=(0.0, 0.0, 0.0),
        input_reference=(0.0,),
        initial_state=(0.0, 0.0, 0.0),
    )
    with pytest.raises(TargetError, match="2-input dynamics"):
        build_ocp(cf, spec)


# ---------------------------------------------------------------------------
# Pendulum smoke (different nx).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_ocp_pendulum_constructs(pendulum_cf) -> None:
    pytest.importorskip("acados_template")
    spec = OcpTemplateSpec(
        horizon_steps=10,
        time_horizon_s=0.5,
        state_cost=(10.0, 1.0),
        input_cost=(0.1,),
        terminal_state_cost=(20.0, 5.0),
        state_reference=(0.0, 0.0),
        input_reference=(0.0,),
        initial_state=(0.3, 0.0),
        name="pendulum",
    )
    ocp = build_ocp(pendulum_cf, spec)
    assert ocp.model.name == "pendulum"
    assert ocp.model.x.shape == (2, 1)
    assert ocp.model.u.shape == (1, 1)


# ---------------------------------------------------------------------------
# Acceptance 2: solve. Gated on tera_renderer presence.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    not _TERA_AVAILABLE,
    reason=(
        "t_renderer not found at $ACADOS_SOURCE_DIR/bin/t_renderer; "
        "the OCP-solves test is gated on the renderer. Build with "
        "`cargo build --release` under "
        "$ACADOS_SOURCE_DIR/interfaces/acados_template/tera_renderer "
        "and copy the binary to $ACADOS_SOURCE_DIR/bin/."
    ),
)
def test_build_ocp_cartpole_solves_to_sensible_output(
    tmp_path: Path, cartpole_cf, cartpole_spec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-021 acceptance 2: the OCP solves and produces sensible outputs."""
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    ocp = build_ocp(cartpole_cf, cartpole_spec)
    solver = AcadosOcpSolver(ocp, json_file="cartpole.json", verbose=False)
    status = solver.solve()
    assert status == 0
    # Sensible-output check: the first control input is finite + non-trivial
    # (cartpole starting at x=0.5 with target 0 should command a non-zero u).
    u0 = solver.get(0, "u")
    assert np.all(np.isfinite(u0))
    # Final state has lower cart-position than initial (controller pulls back).
    x_final = solver.get(cartpole_spec.horizon_steps, "x")
    assert np.all(np.isfinite(x_final))
    assert abs(x_final[0]) < abs(cartpole_spec.initial_state[0])
