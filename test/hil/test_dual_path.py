# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Cartpole dual-path demo: composition + fallback, on host and on the Pi (T-044).

The integration capstone. A generated C binary closes the loop with acados MPC +
an embedded MLP policy combined by the T-043 arbiter; the T-033 harness validates
it against the host reference, *including the fallback to MPC*. Host parity is
gated on the acados toolchain; the on-Pi tier additionally needs
`JAXILITY_HIL_SSH_HOST` + a target acados install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cartpole_lqr_reference import (
    MODEL_NAME,
    PLANT_DT,
    _rk4_step,
    build_cartpole_lqr,
)

from jaxility.compose import CompositionPlan, SafetyEnvelope, arbitrate
from jaxility.compose.codegen import MLPPolicy, generate_dual_path_hil_source
from jaxility.hil import (
    CARTPOLE_LQR_SCHEMA,
    LocalRunner,
    build_controller_hil_binary,
    parse_trace,
    run_hil,
)

_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")


def _tera() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    return bool(root and (Path(root) / "bin" / "t_renderer").exists()) or (
        shutil.which("t_renderer") is not None
    )


_NEEDS_BUILD = pytest.mark.skipif(
    not _tera() or shutil.which("cc") is None,
    reason="acados t_renderer + host cc required for the dual-path build",
)


def _make_policy() -> MLPPolicy:
    flax = pytest.importorskip("flax")
    nn = flax.linen

    class Pol(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(nn.tanh(nn.Dense(8)(x)))

    m = Pol()
    params = m.init(jax.random.PRNGKey(3), jnp.zeros((4,)))
    return MLPPolicy.from_flax(params, input_dim=4)


def _plan() -> CompositionPlan:
    env = SafetyEnvelope(
        name="cartpole",
        state_lower=(-5.0, -3.0, -20.0, -20.0),
        state_upper=(5.0, 3.0, 20.0, 20.0),
        input_lower=(-20.0,),
        input_upper=(20.0,),
    )
    return CompositionPlan(
        name="cartpole-dual",
        mpc_period_ns=1_000_000,
        policy_period_ns=20_000_000,
        safety_envelope=env,
    )


def _host_ref(solver, policy, plan, n, seed, fb_from):
    """Host dual-path closed loop (acados control + JAX plant + T-043 arbiter)."""
    solver.reset()  # fresh warm-start, matching the C binary's fresh solver
    x = np.array([0.3, 0.0, 0.0, 0.0])
    x[0] += 0.01 * seed
    pos = np.empty((n, 2))
    vel = np.empty((n, 2))
    tau = np.empty((n, 1))
    for i in range(n):
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        solver.solve()
        u_mpc = np.asarray(solver.get(0, "u"))
        u_pol = policy.forward(x)
        res = arbitrate(
            plan,
            mpc_control=u_mpc,
            policy_action=u_pol,
            state=x,
            policy_timed_out=(fb_from >= 0 and i >= fb_from),
        )
        cmd = np.array(res.command)
        pos[i] = x[:2]
        vel[i] = x[2:]
        tau[i] = cmd
        x = _rk4_step(x, cmd, PLANT_DT)
    return {"joint_position": pos, "joint_velocity": vel, "actuator_torque": tau}


# ---------------------------------------------------------------------------
# Pure-Python: the embedded MLP matches its flax source
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mlp_policy_matches_flax() -> None:
    flax = pytest.importorskip("flax")
    nn = flax.linen

    class Pol(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(nn.tanh(nn.Dense(8)(x)))

    m = Pol()
    params = m.init(jax.random.PRNGKey(7), jnp.zeros((4,)))
    policy = MLPPolicy.from_flax(params, input_dim=4)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.standard_normal(4)
        ours = policy.forward(x)  # float64
        flx = np.asarray(m.apply(params, jnp.asarray(x)))  # flax float32
        # float64 (ours) vs float32 (flax) — agree to single-precision.
        assert np.abs(ours - flx).max() < 1e-5


# ---------------------------------------------------------------------------
# Host: composition parity + fallback
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_built(tmp_path_factory: pytest.TempPathFactory):
    if not _tera() or shutil.which("cc") is None:
        pytest.skip("acados t_renderer + cc required")
    policy = _make_policy()
    plan = _plan()
    bundle = build_cartpole_lqr(tmp_path_factory.mktemp("dual") / "build")
    gen = bundle.shared_library_path.parent
    src = generate_dual_path_hil_source(
        model_name=MODEL_NAME,
        nx=4,
        nu=1,
        policy=policy,
        plan=plan,
        initial_state=(0.3, 0.0, 0.0, 0.0),
    )
    exe = build_controller_hil_binary(
        generated_code_dir=gen,
        model_name=MODEL_NAME,
        source=src,
        out_path=gen / f"{MODEL_NAME}_dual",
    )
    return bundle, policy, plan, exe


@_NEEDS_BUILD
def test_dual_path_composition_parity(dual_built) -> None:
    """The on-host dual-path binary matches the host reference (acados+MLP+arbiter)."""
    bundle, policy, plan, exe = dual_built
    n = 40
    reference = _host_ref(bundle.solver, policy, plan, n, 0, -1)
    report = run_hil(
        reference,
        LocalRunner(exe),
        target_family="cortex-a76",
        dtype="float64",
        schema=CARTPOLE_LQR_SCHEMA,
        n_steps=n,
        seed=0,
    )
    report.assert_passed()
    assert report.passed


@_NEEDS_BUILD
def test_dual_path_fallback_drops_to_mpc(dual_built) -> None:
    """Forcing the policy timeout makes the command drop to the (clamped) MPC."""
    bundle, policy, plan, exe = dual_built
    n = 40
    # C binary, always-fallback (fb_from=0): command should be the clamped MPC.
    out = subprocess.run(
        [str(exe), str(n), "0", "0"], capture_output=True, text=True
    ).stdout
    fb_traj = parse_trace(out, CARTPOLE_LQR_SCHEMA, n_steps=n)

    ref_fallback = _host_ref(bundle.solver, policy, plan, n, 0, 0)  # arbiter falls back
    ref_dual = _host_ref(bundle.solver, policy, plan, n, 0, -1)  # policy active

    # The fallback binary matches the fallback reference (both = clamped MPC).
    assert (
        np.abs(fb_traj["actuator_torque"] - ref_fallback["actuator_torque"]).max()
        < 1e-8
    )
    # ...and the policy genuinely changes the command when it is NOT falling back.
    assert (
        np.abs(ref_dual["actuator_torque"] - ref_fallback["actuator_torque"]).max()
        > 1e-3
    )


# ---------------------------------------------------------------------------
# On-Pi: the dual-path composition runs + matches on real silicon
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_on_pi(tmp_path_factory: pytest.TempPathFactory):
    from jaxility.hil import build_controller_on_target

    if not _tera():
        pytest.skip("host acados t_renderer required")
    host = os.environ.get("JAXILITY_HIL_SSH_HOST")
    if not host:
        pytest.skip("JAXILITY_HIL_SSH_HOST not set; on-Pi dual-path tier skipped")
    if (
        subprocess.run(
            ["ssh", *_SSH_OPTS, host, "true"], capture_output=True
        ).returncode
        != 0
    ):
        pytest.skip(f"HIL ssh host {host!r} not reachable")
    acados = os.environ.get("JAXILITY_HIL_ACADOS")
    if not acados:
        probe = subprocess.run(
            ["ssh", *_SSH_OPTS, host, "echo $HOME/acados"],
            capture_output=True,
            text=True,
        )
        acados = probe.stdout.strip() if probe.returncode == 0 else ""
    if (
        not acados
        or subprocess.run(
            ["ssh", *_SSH_OPTS, host, f"test -d {acados}/lib"], capture_output=True
        ).returncode
        != 0
    ):
        pytest.skip(f"target acados install not found on {host}")

    policy = _make_policy()
    plan = _plan()
    bundle = build_cartpole_lqr(tmp_path_factory.mktemp("dual_pi") / "build")
    gen = bundle.shared_library_path.parent
    src = generate_dual_path_hil_source(
        model_name=MODEL_NAME,
        nx=4,
        nu=1,
        policy=policy,
        plan=plan,
        initial_state=(0.3, 0.0, 0.0, 0.0),
    )
    runner = build_controller_on_target(
        host=host,
        generated_code_dir=gen,
        model_name=MODEL_NAME,
        source=src,
        source_name=f"{MODEL_NAME}_dual_main.c",
        remote_dir="/tmp/jaxility-dual",
        remote_acados=acados,
    )
    return bundle, policy, plan, runner


@pytest.mark.hil
def test_dual_path_composition_on_pi(dual_on_pi) -> None:
    """T-044: acados MPC + MLP policy + arbiter, composed on real Cortex-A76."""
    bundle, policy, plan, runner = dual_on_pi
    n = 40
    reference = _host_ref(bundle.solver, policy, plan, n, 0, -1)
    report = run_hil(
        reference,
        runner,
        target_family="cortex-a76",
        dtype="float64",
        schema=CARTPOLE_LQR_SCHEMA,
        n_steps=n,
        seed=0,
    )
    report.assert_passed()
    assert report.runner_label.startswith("ssh:")
