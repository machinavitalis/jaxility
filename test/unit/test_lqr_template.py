# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the LQR template (T-022)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.errors import TargetError
from jaxility.lowering import OcpTemplateSpec, build_ocp, translate
from jaxility.templates import lqr


def _cartpole(state, control):
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


@pytest.fixture(scope="module")
def cartpole_cf():
    return translate(_cartpole, in_shapes=((4,), (1,)), name="cartpole")


def _tera_renderer_available() -> bool:
    import os
    import shutil

    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if acados_root and (Path(acados_root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA_AVAILABLE = _tera_renderer_available()


# ---------------------------------------------------------------------------
# Shape sanity and defaults.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lqr_returns_ocp_template_spec(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
    )
    assert isinstance(spec, OcpTemplateSpec)


@pytest.mark.unit
def test_lqr_default_state_reference_is_zeros(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
    )
    assert spec.state_reference == (0.0, 0.0, 0.0, 0.0)
    assert spec.input_reference == (0.0,)


@pytest.mark.unit
def test_lqr_default_q_terminal_is_factor_times_q(cartpole_cf) -> None:
    """Default ``Q_terminal = 10 * Q`` so end-of-horizon drift is penalised."""
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 2.0, 3.0, 4.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
    )
    assert spec.terminal_state_cost == (10.0, 20.0, 30.0, 40.0)


@pytest.mark.unit
def test_lqr_custom_q_terminal_overrides_default(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        Q_terminal=(50.0, 50.0, 5.0, 5.0),
    )
    assert spec.terminal_state_cost == (50.0, 50.0, 5.0, 5.0)


@pytest.mark.unit
def test_lqr_custom_terminal_factor(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        Q_terminal_factor=20.0,
    )
    assert spec.terminal_state_cost == (20.0, 20.0, 20.0, 20.0)


@pytest.mark.unit
def test_lqr_input_bounds_propagate(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
    )
    assert spec.input_lower == (-10.0,)
    assert spec.input_upper == (10.0,)


@pytest.mark.unit
def test_lqr_state_bounds_propagate(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        state_bounds=((-1.0, -np.pi, -5.0, -10.0), (1.0, np.pi, 5.0, 10.0)),
    )
    assert spec.state_lower == (-1.0, -np.pi, -5.0, -10.0)
    assert spec.state_upper == (1.0, np.pi, 5.0, 10.0)


@pytest.mark.unit
def test_lqr_solver_options_propagate(cartpole_cf) -> None:
    spec = lqr(
        cartpole_cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        integrator_type="IRK",
        nlp_solver_type="SQP",
        qp_solver="FULL_CONDENSING_HPIPM",
        horizon_steps=10,
        time_horizon_s=0.5,
        name="cartpole-lqr",
    )
    assert spec.integrator_type == "IRK"
    assert spec.nlp_solver_type == "SQP"
    assert spec.qp_solver == "FULL_CONDENSING_HPIPM"
    assert spec.horizon_steps == 10
    assert spec.time_horizon_s == 0.5
    assert spec.name == "cartpole-lqr"


# ---------------------------------------------------------------------------
# Shape-mismatch guards.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lqr_rejects_q_length_mismatch(cartpole_cf) -> None:
    with pytest.raises(TargetError, match="Q length"):
        lqr(
            cartpole_cf,
            Q=(1.0, 1.0),  # nx=4 expected
            R=(0.1,),
            initial_state=(0.0, 0.0, 0.0, 0.0),
        )


@pytest.mark.unit
def test_lqr_rejects_r_length_mismatch(cartpole_cf) -> None:
    with pytest.raises(TargetError, match="R length"):
        lqr(
            cartpole_cf,
            Q=(1.0, 1.0, 1.0, 1.0),
            R=(0.1, 0.2),  # nu=1 expected
            initial_state=(0.0, 0.0, 0.0, 0.0),
        )


@pytest.mark.unit
def test_lqr_rejects_initial_state_length_mismatch(cartpole_cf) -> None:
    with pytest.raises(TargetError, match="initial_state length"):
        lqr(
            cartpole_cf,
            Q=(1.0, 1.0, 1.0, 1.0),
            R=(0.1,),
            initial_state=(0.0, 0.0),  # nx=4 expected
        )


@pytest.mark.unit
def test_lqr_rejects_input_bounds_length_mismatch(cartpole_cf) -> None:
    with pytest.raises(TargetError, match="input_bounds"):
        lqr(
            cartpole_cf,
            Q=(1.0, 1.0, 1.0, 1.0),
            R=(0.1,),
            initial_state=(0.0, 0.0, 0.0, 0.0),
            input_bounds=((-10.0, -10.0), (10.0,)),  # mismatched lengths
        )


@pytest.mark.unit
def test_lqr_rejects_dynamics_with_wrong_input_count() -> None:
    def fn(x):
        return jnp.sin(x)

    cf = translate(fn, in_shapes=((3,),))
    with pytest.raises(TargetError, match="2-input dynamics"):
        lqr(cf, Q=(1.0, 1.0, 1.0), R=(0.1,), initial_state=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# End-to-end: lqr → build_ocp → solve on Cartpole.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lqr_composes_with_build_ocp_to_valid_acados_ocp(cartpole_cf) -> None:
    """LQR spec + build_ocp produces a valid AcadosOcp."""
    pytest.importorskip("acados_template")
    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="cartpole-lqr",
    )
    ocp = build_ocp(cartpole_cf, spec)
    from acados_template import AcadosOcp

    assert isinstance(ocp, AcadosOcp)
    assert ocp.model.name == "cartpole-lqr"


@pytest.mark.unit
@pytest.mark.skipif(
    not _TERA_AVAILABLE,
    reason=(
        "t_renderer not found at $ACADOS_SOURCE_DIR/bin/t_renderer; "
        "the LQR-solves test is gated on the renderer."
    ),
)
def test_lqr_cartpole_solves_to_sensible_output(
    tmp_path: Path, cartpole_cf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: lqr template → build_ocp → AcadosOcpSolver → status 0."""
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    spec = lqr(
        cartpole_cf,
        Q=(10.0, 10.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.5, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="cartpole_lqr_e2e",
    )
    ocp = build_ocp(cartpole_cf, spec)
    solver = AcadosOcpSolver(ocp, json_file="cartpole_lqr.json", verbose=False)
    status = solver.solve()
    assert status == 0
    u0 = solver.get(0, "u")
    assert np.all(np.isfinite(u0))
    x_final = solver.get(spec.horizon_steps, "x")
    assert np.all(np.isfinite(x_final))
    # Controller should drive the cart back from x=0.5 toward 0.
    assert abs(x_final[0]) < abs(spec.initial_state[0])
