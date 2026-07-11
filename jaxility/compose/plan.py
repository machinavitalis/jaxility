# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Dual-path runtime composition: acados MPC + learned-policy arbitration (T-043).

Per ADR-002 the deployment runs two paths: the acados MPC/LQR controller at the
high rate (1 kHz typical, hard real-time + the safety envelope) and the learned
policy at a lower rate (10–100 Hz), consuming the envelope as constraints. This
module makes the contract between them a **declarative, named, parameterized**
record — the :class:`CompositionPlan` — rather than burying the arbitration in
runtime code, so the safety envelope is a first-class artifact a regulator (or a
test) can read (invariant 8, ADR-002 consequence).

The plan declares the two rates, how the paths combine (:class:`ArbitrationMode`),
the :class:`SafetyEnvelope`, and the fallback policy. :func:`arbitrate` is the
reference implementation of one control step's decision — the same logic the
on-target C runtime mirrors (T-044) — and it **always** clamps the final command
into the envelope and falls back to the (constraint-respecting) MPC control on a
policy timeout or a state-envelope breach. There is no path by which the learned
policy drives the actuator outside the envelope.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import CompositionError

ArbitrationMode = Literal["residual", "policy_primary"]
"""How the two paths combine when both are live.

* ``"residual"`` — the learned policy is a correction *on top of* the MPC
  stabiliser: ``u = u_mpc + u_policy`` (the Cartpole-on-LQR demo shape, T-044).
* ``"policy_primary"`` — the learned policy is the command; the MPC is the
  fallback/safety floor. ``u = u_policy``.
"""


class SafetyEnvelope(BaseModel):
    """Named, parameterized limits the MPC enforces and the arbiter clamps to.

    Per invariant 8 the envelope is explicit (not buried in template code): the
    state box the deployment must stay within and the actuator box every command
    is clamped into. The acados layer enforces these as OCP constraints; the
    arbiter enforces them again at the composition boundary so the learned path
    can never exceed them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    state_lower: tuple[float, ...]
    state_upper: tuple[float, ...]
    input_lower: tuple[float, ...]
    input_upper: tuple[float, ...]

    @model_validator(mode="after")
    def _check(self) -> SafetyEnvelope:
        if len(self.state_lower) != len(self.state_upper):
            raise ValueError("state_lower / state_upper length mismatch")
        if len(self.input_lower) != len(self.input_upper):
            raise ValueError("input_lower / input_upper length mismatch")
        if any(lo > hi for lo, hi in zip(self.state_lower, self.state_upper)):
            raise ValueError("state_lower must be <= state_upper elementwise")
        if any(lo > hi for lo, hi in zip(self.input_lower, self.input_upper)):
            raise ValueError("input_lower must be <= input_upper elementwise")
        return self

    @property
    def nx(self) -> int:
        return len(self.state_lower)

    @property
    def nu(self) -> int:
        return len(self.input_lower)

    def state_within(self, state: np.ndarray) -> bool:
        """Whether ``state`` is inside the state box (inclusive)."""
        s = np.asarray(state, dtype=np.float64)
        return bool(
            np.all(s >= np.array(self.state_lower))
            and np.all(s <= np.array(self.state_upper))
        )

    def clamp_input(self, u: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp a control into the actuator box; report whether it was clamped."""
        u = np.asarray(u, dtype=np.float64)
        clamped = np.clip(u, np.array(self.input_lower), np.array(self.input_upper))
        return clamped, bool(np.any(clamped != u))


class CompositionPlan(BaseModel):
    """Declarative dual-path contract (ADR-002): rates, mode, envelope, fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    mpc_period_ns: int = Field(gt=0)
    """The acados MPC cycle period (e.g. 1e6 for 1 kHz)."""
    policy_period_ns: int = Field(gt=0)
    """The learned-policy cycle period; a multiple of the MPC period (lower rate)."""
    mode: ArbitrationMode = "residual"
    safety_envelope: SafetyEnvelope
    fallback_on_timeout: bool = True
    """Fall back to MPC-only if the policy misses its deadline."""
    fallback_on_envelope_breach: bool = True
    """Fall back to MPC-only if the state leaves the safety envelope."""

    @model_validator(mode="after")
    def _check(self) -> CompositionPlan:
        if self.policy_period_ns < self.mpc_period_ns:
            raise ValueError(
                "policy_period_ns must be >= mpc_period_ns (the learned policy "
                "runs at a lower-or-equal rate than the MPC)"
            )
        if self.policy_period_ns % self.mpc_period_ns != 0:
            raise ValueError(
                "policy_period_ns must be an integer multiple of mpc_period_ns "
                "for a clean decimation between the two rates"
            )
        return self

    @property
    def policy_decimation(self) -> int:
        """How many MPC cycles elapse per learned-policy cycle (>= 1)."""
        return self.policy_period_ns // self.mpc_period_ns

    def policy_due(self, cycle_index: int) -> bool:
        """Whether the learned policy runs on MPC ``cycle_index`` (0-based)."""
        return cycle_index % self.policy_decimation == 0


class ArbitrationResult(BaseModel):
    """The outcome of one composed control step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: tuple[float, ...]
    """The final actuator command — always inside the safety envelope."""
    path: Literal["dual", "mpc_fallback"]
    fallback_reason: Literal["timeout", "envelope"] | None = None
    clamped: bool = False
    """Whether the combined command had to be clamped into the actuator box."""


def arbitrate(
    plan: CompositionPlan,
    *,
    mpc_control: np.ndarray,
    policy_action: np.ndarray,
    state: np.ndarray,
    policy_timed_out: bool = False,
) -> ArbitrationResult:
    """Decide one composed control command under ``plan``.

    The acados ``mpc_control`` is the constraint-respecting baseline;
    ``policy_action`` is the learned path's contribution this cycle. Falls back
    to the MPC control on a policy timeout or a state-envelope breach (per the
    plan's flags), otherwise combines per :attr:`CompositionPlan.mode`. The
    returned command is **always** clamped into the envelope's actuator box.

    Raises
    ------
    CompositionError
        If the control / state dimensions disagree with the safety envelope.
    """
    env = plan.safety_envelope
    mpc = np.asarray(mpc_control, dtype=np.float64)
    pol = np.asarray(policy_action, dtype=np.float64)
    st = np.asarray(state, dtype=np.float64)
    if mpc.shape != (env.nu,) or pol.shape != (env.nu,):
        raise CompositionError(
            f"control dimension mismatch: mpc {mpc.shape} / policy {pol.shape} vs "
            f"envelope nu={env.nu}."
        )
    if st.shape != (env.nx,):
        raise CompositionError(
            f"state dimension mismatch: {st.shape} vs envelope nx={env.nx}."
        )

    if policy_timed_out and plan.fallback_on_timeout:
        cmd, _ = env.clamp_input(mpc)
        return ArbitrationResult(
            command=tuple(float(c) for c in cmd),
            path="mpc_fallback",
            fallback_reason="timeout",
        )
    if not env.state_within(st) and plan.fallback_on_envelope_breach:
        cmd, _ = env.clamp_input(mpc)
        return ArbitrationResult(
            command=tuple(float(c) for c in cmd),
            path="mpc_fallback",
            fallback_reason="envelope",
        )

    combined = mpc + pol if plan.mode == "residual" else pol
    cmd, clamped = env.clamp_input(combined)
    return ArbitrationResult(
        command=tuple(float(c) for c in cmd),
        path="dual",
        clamped=clamped,
    )
