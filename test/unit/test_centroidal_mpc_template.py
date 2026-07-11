# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the Centroidal MPC template (T-025)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.lowering import OcpTemplateSpec, build_ocp, translate
from jaxility.templates import centroidal_mpc


def _centroidal_linear(state, wrench):
    """3D double-integrator centroidal dynamics with gravity.

    State: ``[c_x, c_y, c_z, c_dot_x, c_dot_y, c_dot_z]``.
    Wrench: ``[F_x, F_y, F_z]`` (net force on CoM).
    Dynamics: ``c_ddot = F / m + g`` with ``m = 1``, ``g_z = -9.81``.
    """
    m = 1.0
    return jnp.array(
        [
            state[3],
            state[4],
            state[5],
            wrench[0] / m,
            wrench[1] / m,
            wrench[2] / m - 9.81,
        ]
    )


@pytest.fixture(scope="module")
def centroidal_cf():
    return translate(
        _centroidal_linear, in_shapes=((6,), (3,)), name="centroidal_linear"
    )


def _tera_renderer_available() -> bool:
    import os
    import shutil

    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if acados_root and (Path(acados_root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA_AVAILABLE = _tera_renderer_available()


# ---------------------------------------------------------------------------
# Spec shape.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_centroidal_mpc_returns_ocp_template_spec(centroidal_cf) -> None:
    spec = centroidal_mpc(
        centroidal_cf,
        Q=(10.0,) * 6,
        R=(0.01,) * 3,
        initial_com_state=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    assert isinstance(spec, OcpTemplateSpec)


@pytest.mark.unit
def test_centroidal_mpc_default_target_is_zeros(centroidal_cf) -> None:
    spec = centroidal_mpc(
        centroidal_cf,
        Q=(1.0,) * 6,
        R=(0.1,) * 3,
        initial_com_state=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    assert spec.state_reference == (0.0,) * 6
    assert spec.input_reference == (0.0,) * 3


@pytest.mark.unit
def test_centroidal_mpc_custom_target(centroidal_cf) -> None:
    spec = centroidal_mpc(
        centroidal_cf,
        Q=(1.0,) * 6,
        R=(0.1,) * 3,
        initial_com_state=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        target_com_state=(0.5, 0.0, 1.0, 0.0, 0.0, 0.0),
        wrench_reference=(0.0, 0.0, 9.81),  # holding-gravity reference
    )
    assert spec.state_reference == (0.5, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert spec.input_reference == (0.0, 0.0, 9.81)


@pytest.mark.unit
def test_centroidal_mpc_default_terminal_factor(centroidal_cf) -> None:
    spec = centroidal_mpc(
        centroidal_cf,
        Q=(2.0,) * 6,
        R=(0.1,) * 3,
        initial_com_state=(0.0,) * 6,
    )
    assert spec.terminal_state_cost == (20.0,) * 6


@pytest.mark.unit
def test_centroidal_mpc_wrench_bounds_propagate(centroidal_cf) -> None:
    spec = centroidal_mpc(
        centroidal_cf,
        Q=(1.0,) * 6,
        R=(0.1,) * 3,
        initial_com_state=(0.0,) * 6,
        wrench_bounds=((-50.0, -50.0, 0.0), (50.0, 50.0, 100.0)),
    )
    assert spec.input_lower == (-50.0, -50.0, 0.0)
    assert spec.input_upper == (50.0, 50.0, 100.0)


# ---------------------------------------------------------------------------
# End-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not _TERA_AVAILABLE, reason="t_renderer required.")
def test_centroidal_mpc_solves_to_hold_position(
    tmp_path: Path, centroidal_cf, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold CoM at z=1.0 against gravity — wrench should be ~ (0, 0, 9.81).

    The controller solves and the first wrench has a positive z component
    close to gravity, with small x/y components.
    """
    monkeypatch.chdir(tmp_path)
    from acados_template import AcadosOcpSolver

    spec = centroidal_mpc(
        centroidal_cf,
        Q=(100.0, 100.0, 100.0, 10.0, 10.0, 10.0),
        R=(0.001, 0.001, 0.001),
        initial_com_state=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        target_com_state=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        wrench_reference=(0.0, 0.0, 9.81),
        wrench_bounds=((-50.0, -50.0, 0.0), (50.0, 50.0, 100.0)),
        horizon_steps=15,
        time_horizon_s=0.5,
        name="hold_pos",
    )
    ocp = build_ocp(centroidal_cf, spec)
    solver = AcadosOcpSolver(ocp, json_file="hold_pos.json", verbose=False)
    status = solver.solve()
    assert status == 0
    u0 = solver.get(0, "u")
    assert np.all(np.isfinite(u0))
    # x and y forces should be near zero; z force near 9.81 (compensating gravity).
    assert abs(u0[0]) < 5.0
    assert abs(u0[1]) < 5.0
    assert u0[2] > 5.0  # at least lifting against gravity
