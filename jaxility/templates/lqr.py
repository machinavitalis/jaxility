# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""LQR template (T-022).

A *Linear-Quadratic Regulator* configured as a short-horizon
quadratic-tracking OCP. For an actual LTI system, the acados solution
at every cycle is the LQR action; for nonlinear systems, the MPC
reformulation tracks a constant reference with the same quadratic
cost shape — the Cartpole-stabilisation pattern Jaxility's launch
demo relies on.

The template is a factory: given the translated dynamics
(:class:`~jaxility.lowering.CasadiFunction`) plus the LQR knobs
(``Q``, ``R``, references, horizon, optional bounds), it returns an
:class:`~jaxility.lowering.OcpTemplateSpec` that the T-021 builder
consumes. Sensible defaults fill in zeros for references and a
configurable factor (default 10×) for the terminal cost so the
solver penalises end-of-horizon drift more strongly than the stage
cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ..errors import TargetError
from ..lowering import CasadiFunction, OcpTemplateSpec


def lqr(
    dynamics: CasadiFunction,
    *,
    Q: Sequence[float],
    R: Sequence[float],
    initial_state: Sequence[float],
    state_reference: Sequence[float] | None = None,
    input_reference: Sequence[float] | None = None,
    Q_terminal: Sequence[float] | None = None,
    Q_terminal_factor: float = 10.0,
    horizon_steps: int = 20,
    time_horizon_s: float = 1.0,
    input_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    state_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    name: str = "lqr",
    integrator_type: Literal["ERK", "IRK", "GNSF"] = "ERK",
    nlp_solver_type: Literal["SQP", "SQP_RTI"] = "SQP_RTI",
    qp_solver: Literal[
        "PARTIAL_CONDENSING_HPIPM",
        "FULL_CONDENSING_HPIPM",
        "FULL_CONDENSING_QPOASES",
    ] = "PARTIAL_CONDENSING_HPIPM",
) -> OcpTemplateSpec:
    """Build an LQR-style :class:`OcpTemplateSpec` for ``dynamics``.

    Args
    ----
    dynamics : CasadiFunction
        Translated dynamics ``f(x, u) -> dx`` from
        :func:`jaxility.lowering.translate`. ``nx`` and ``nu`` are
        derived from its input shapes.
    Q : Sequence[float]
        Diagonal of the stage state-cost matrix; length ``nx``.
    R : Sequence[float]
        Diagonal of the stage input-cost matrix; length ``nu``.
    initial_state : Sequence[float]
        Pinned at ``constraints.x0``. The runtime updates this each
        cycle.
    state_reference : Sequence[float] | None
        Constant state reference; defaults to ``(0.0,) * nx``.
    input_reference : Sequence[float] | None
        Constant input reference; defaults to ``(0.0,) * nu``.
    Q_terminal : Sequence[float] | None
        Diagonal of the terminal state-cost matrix. When ``None``,
        defaults to ``Q_terminal_factor * Q`` componentwise.
    Q_terminal_factor : float
        Factor used to derive the terminal cost from ``Q`` when
        ``Q_terminal`` is omitted. Default ``10.0``.
    horizon_steps : int
        Discretisation horizon length ``N``. Default 20.
    time_horizon_s : float
        Total prediction horizon in seconds. Default 1.0.
    input_bounds : tuple[lower, upper] | None
        Optional box constraints on the input; each entry is a
        length-``nu`` sequence.
    state_bounds : tuple[lower, upper] | None
        Optional box constraints on the (interior + terminal) states;
        each entry is a length-``nx`` sequence.
    name : str
        Spec / model name used in code-generation artifacts.
    integrator_type, nlp_solver_type, qp_solver : str
        Passed through to ``OcpTemplateSpec``.

    Returns
    -------
    OcpTemplateSpec
        Ready for :func:`jaxility.lowering.build_ocp`.

    Raises
    ------
    TargetError
        On any shape mismatch between ``Q`` / ``R`` /
        ``initial_state`` / references / bounds and the dynamics
        ``input_shapes``.
    """
    if dynamics.input_shapes == () or len(dynamics.input_shapes) != 2:
        raise TargetError(
            "lqr expects a 2-input dynamics function f(x, u); got "
            f"input_shapes={dynamics.input_shapes!r}."
        )
    nx_shape, nu_shape = dynamics.input_shapes
    if len(nx_shape) != 1 or len(nu_shape) != 1:
        raise TargetError(
            "lqr expects rank-1 state and input vectors; got "
            f"state shape {nx_shape} and input shape {nu_shape}."
        )
    nx = nx_shape[0]
    nu = nu_shape[0]

    def _check(label: str, vec: Sequence[float] | None, expected: int) -> None:
        if vec is None:
            return
        if len(vec) != expected:
            raise TargetError(
                f"{label} length {len(vec)} does not match dynamics "
                f"(expected {expected})."
            )

    _check("Q", Q, nx)
    _check("R", R, nu)
    _check("initial_state", initial_state, nx)
    _check("state_reference", state_reference, nx)
    _check("input_reference", input_reference, nu)
    _check("Q_terminal", Q_terminal, nx)
    if input_bounds is not None:
        _check("input_bounds[0]", input_bounds[0], nu)
        _check("input_bounds[1]", input_bounds[1], nu)
    if state_bounds is not None:
        _check("state_bounds[0]", state_bounds[0], nx)
        _check("state_bounds[1]", state_bounds[1], nx)

    state_reference_t = (
        tuple(state_reference) if state_reference is not None else tuple([0.0] * nx)
    )
    input_reference_t = (
        tuple(input_reference) if input_reference is not None else tuple([0.0] * nu)
    )
    Q_t = tuple(Q)
    R_t = tuple(R)
    if Q_terminal is not None:
        Q_terminal_t = tuple(Q_terminal)
    else:
        Q_terminal_t = tuple(Q_terminal_factor * q for q in Q_t)

    input_lower = tuple(input_bounds[0]) if input_bounds is not None else None
    input_upper = tuple(input_bounds[1]) if input_bounds is not None else None
    state_lower = tuple(state_bounds[0]) if state_bounds is not None else None
    state_upper = tuple(state_bounds[1]) if state_bounds is not None else None

    return OcpTemplateSpec(
        horizon_steps=horizon_steps,
        time_horizon_s=time_horizon_s,
        state_cost=Q_t,
        input_cost=R_t,
        terminal_state_cost=Q_terminal_t,
        state_reference=state_reference_t,
        input_reference=input_reference_t,
        initial_state=tuple(initial_state),
        input_lower=input_lower,
        input_upper=input_upper,
        state_lower=state_lower,
        state_upper=state_upper,
        integrator_type=integrator_type,
        nlp_solver_type=nlp_solver_type,
        qp_solver=qp_solver,
        name=name,
    )
