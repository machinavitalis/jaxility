# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Generator tests for the dual-path timing bench (T-044 follow-up).

Pure Python — asserts the emitted C times the *composed* control computation
(acados solve + MLP policy + arbiter + clamp) and excludes the plant advance, and
emits the same timing record the single-path bench does so
``jaxility.bench.run_controller_bench`` consumes it unchanged. No acados / no
flax: the policy is built directly from ``DenseLayer``.
"""

from __future__ import annotations

import pytest

from jaxility.compose import (
    CompositionPlan,
    DenseLayer,
    MLPPolicy,
    SafetyEnvelope,
    generate_dual_path_bench_source,
)

_INITIAL_STATE = (0.3, 0.0, 0.0, 0.0)


def _policy() -> MLPPolicy:
    # 4 -> 2 -> 1 tanh MLP, weights as literals (no flax needed).
    return MLPPolicy(
        layers=(
            DenseLayer(
                weight=((0.1, 0.2), (0.0, -0.1), (0.3, 0.0), (-0.2, 0.1)),
                bias=(0.0, 0.0),
                activation="tanh",
            ),
            DenseLayer(weight=((0.5,), (-0.3,)), bias=(0.0,), activation="identity"),
        )
    )


def _plan() -> CompositionPlan:
    env = SafetyEnvelope(
        name="cartpole",
        state_lower=(-2.0, -1.0, -10.0, -10.0),
        state_upper=(2.0, 1.0, 10.0, 10.0),
        input_lower=(-20.0,),
        input_upper=(20.0,),
    )
    return CompositionPlan(
        name="cartpole-dual",
        mpc_period_ns=1_000_000,
        policy_period_ns=1_000_000,
        safety_envelope=env,
    )


@pytest.mark.unit
def test_bench_source_times_the_composed_cycle() -> None:
    src = generate_dual_path_bench_source(
        model_name="demo_model",
        nx=4,
        nu=1,
        policy=_policy(),
        plan=_plan(),
        initial_state=_INITIAL_STATE,
    )
    # the timed region brackets the composed control computation...
    assert "long t0 = now_ns();" in src
    assert "demo_model_acados_solve(ocp)" in src
    assert "jx_policy(x, u_pol)" in src  # learned-policy eval
    assert "u_mpc[j] + u_pol[j]" in src  # arbitration (residual combine)
    assert "clamp_input(cmd)" in src
    assert "long t1 = now_ns();" in src
    # ...and the plant advance is OUTSIDE it (it's the harness, not the controller)
    timed = src[src.index("long t0 = now_ns();") : src.index("long t1 = now_ns();")]
    assert "acados_sim_solve" not in timed
    # same timing-record contract as the single-path bench
    assert '\\"solve_ns\\": [' in src and '\\"max_rss_kib\\":' in src


@pytest.mark.unit
def test_bench_argv_is_the_bench_runner_form() -> None:
    src = generate_dual_path_bench_source(
        model_name="m",
        nx=4,
        nu=1,
        policy=_policy(),
        plan=_plan(),
        initial_state=_INITIAL_STATE,
        n_warmup=50,
    )
    assert "usage: %s <n_cycles> <seed>" in src
    assert "#define JX_N_WARMUP 50L" in src
