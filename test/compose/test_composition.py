# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Dual-path composition tests (T-043).

Pure Python — the arbiter + plan are numpy/pydantic, no toolchains. Exercises
the safety-critical paths: residual combine, envelope clamping, and fallback to
the MPC on policy timeout / state-envelope breach (invariant 8).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxility.compose import (
    CompositionPlan,
    SafetyEnvelope,
    arbitrate,
)
from jaxility.errors import CompositionError


def _envelope():
    # Cartpole-ish: nx=4, nu=1; actuator box [-20, 20].
    return SafetyEnvelope(
        name="cartpole",
        state_lower=(-2.0, -1.0, -10.0, -10.0),
        state_upper=(2.0, 1.0, 10.0, 10.0),
        input_lower=(-20.0,),
        input_upper=(20.0,),
    )


def _plan(**kw):
    base = dict(
        name="cartpole-dual",
        mpc_period_ns=1_000_000,  # 1 kHz
        policy_period_ns=20_000_000,  # 50 Hz
        safety_envelope=_envelope(),
    )
    base.update(kw)
    return CompositionPlan(**base)


@pytest.mark.unit
def test_rate_decimation_and_due() -> None:
    plan = _plan()
    assert plan.policy_decimation == 20  # 50 Hz policy inside 1 kHz MPC
    assert plan.policy_due(0)
    assert not plan.policy_due(1)
    assert plan.policy_due(20)


@pytest.mark.unit
def test_residual_combine_clamps_into_envelope() -> None:
    plan = _plan(mode="residual")
    # mpc + policy = 15 + 10 = 25 -> clamped to 20.
    res = arbitrate(
        plan,
        mpc_control=np.array([15.0]),
        policy_action=np.array([10.0]),
        state=np.zeros(4),
    )
    assert res.path == "dual"
    assert res.clamped is True
    assert res.command == (20.0,)


@pytest.mark.unit
def test_residual_within_box_not_clamped() -> None:
    plan = _plan(mode="residual")
    res = arbitrate(
        plan,
        mpc_control=np.array([3.0]),
        policy_action=np.array([-1.0]),
        state=np.zeros(4),
    )
    assert res.path == "dual" and res.clamped is False
    assert res.command == (2.0,)


@pytest.mark.unit
def test_policy_timeout_falls_back_to_mpc() -> None:
    plan = _plan()
    res = arbitrate(
        plan,
        mpc_control=np.array([5.0]),
        policy_action=np.array([99.0]),  # ignored on fallback
        state=np.zeros(4),
        policy_timed_out=True,
    )
    assert res.path == "mpc_fallback"
    assert res.fallback_reason == "timeout"
    assert res.command == (5.0,)  # MPC control, not the policy's


@pytest.mark.unit
def test_envelope_breach_falls_back_to_mpc() -> None:
    plan = _plan()
    # cart position 3.0 is outside the [-2, 2] state box.
    res = arbitrate(
        plan,
        mpc_control=np.array([-4.0]),
        policy_action=np.array([10.0]),
        state=np.array([3.0, 0.0, 0.0, 0.0]),
    )
    assert res.path == "mpc_fallback"
    assert res.fallback_reason == "envelope"
    assert res.command == (-4.0,)


@pytest.mark.unit
def test_fallback_disabled_keeps_dual_but_still_clamps() -> None:
    plan = _plan(fallback_on_envelope_breach=False)
    res = arbitrate(
        plan,
        mpc_control=np.array([5.0]),
        policy_action=np.array([100.0]),
        state=np.array([3.0, 0.0, 0.0, 0.0]),  # outside box, but fallback off
    )
    # No fallback, but the combined 105 is still clamped into [-20, 20].
    assert res.path == "dual"
    assert res.command == (20.0,)


@pytest.mark.unit
def test_dimension_mismatch_raises() -> None:
    plan = _plan()
    with pytest.raises(CompositionError, match="state dimension mismatch"):
        arbitrate(
            plan,
            mpc_control=np.array([0.0]),
            policy_action=np.array([0.0]),
            state=np.zeros(3),  # nx should be 4
        )
    with pytest.raises(CompositionError, match="control dimension mismatch"):
        arbitrate(
            plan,
            mpc_control=np.array([0.0, 0.0]),  # nu should be 1
            policy_action=np.array([0.0, 0.0]),
            state=np.zeros(4),
        )


@pytest.mark.unit
def test_plan_rejects_policy_faster_than_mpc() -> None:
    with pytest.raises(ValueError, match="lower-or-equal rate"):
        _plan(mpc_period_ns=1_000_000, policy_period_ns=500_000)


@pytest.mark.unit
def test_plan_rejects_non_integer_decimation() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        _plan(mpc_period_ns=1_000_000, policy_period_ns=1_500_000)


@pytest.mark.unit
def test_envelope_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="state_lower must be <="):
        SafetyEnvelope(
            name="bad",
            state_lower=(1.0,),
            state_upper=(-1.0,),
            input_lower=(-1.0,),
            input_upper=(1.0,),
        )
