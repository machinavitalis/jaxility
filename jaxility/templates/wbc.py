# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Whole-body control (WBC) template (T-024).

A *whole-body controller* expresses control intent as a set of named
tasks — "track end-effector position", "respect joint limits", "keep
torso upright" — each with its own importance. The canonical
formulation is a **hierarchical QP** in which lower-priority tasks
live in the null space of higher-priority ones, so a low-priority
tracking task never compromises a high-priority feasibility
constraint (Khatib's original formulation).

The template ships a **weighted** WBC, *not* strictly hierarchical:

* Each :class:`WBCTask` carries a positive ``priority`` weight.
* The combined cost is
  ``sum_i priority_i * ((x - x_ref_i)^T diag(state_weight_i) (x - x_ref_i)
                       + (u - u_ref_i)^T diag(input_weight_i) (u - u_ref_i))``.
* The result collapses into the ``LINEAR_LS`` cost shape
  :func:`jaxility.lowering.build_ocp` already supports. The effective
  ``Q`` / ``R`` are priority-weighted sums of the per-task weights;
  ``yref`` is the priority-and-weight-weighted average of the per-task
  references.

Strict hierarchical WBC (null-space projection, slack variables,
inequality task priorities) is a later enhancement when
acados gains the supporting cost / constraint structure or when we
add a Jaxility-side null-space projector.

Upstream gap (see also the PR body):
    Jaxterity does **not** yet expose a ``Task`` DSL. The
    Jaxility documentation set names one; the spec lands when
    Jaxterity does. The current :class:`WBCTask` is the Jaxility-side
    placeholder — callers construct tasks directly from numeric
    weights and references. Once Jaxterity ships its DSL, the
    integration point is a small adapter ``WBCTask.from_jaxterity_task``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import TargetError
from ..lowering import CasadiFunction, OcpTemplateSpec

WBC_SCHEMA_V0 = 0
"""Schema version of the ``WBCTask`` payload."""


class WBCTask(BaseModel):
    """A single weighted task in a WBC formulation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=WBC_SCHEMA_V0, ge=0)

    name: str = Field(description="Human-readable task identifier.")
    priority: float = Field(
        gt=0.0,
        description=(
            "Per-task importance weight. Higher numbers translate to a "
            "larger contribution to the combined cost. The WBC is "
            "weighted (not strictly hierarchical) so priorities trade "
            "off smoothly rather than enforcing dominance."
        ),
    )
    state_weight: tuple[float, ...] = Field(
        description=(
            "Diagonal of the per-task state cost matrix; length ``nx``. "
            "Entries that are zero mean 'this task does not care about "
            "that state component'."
        ),
    )
    input_weight: tuple[float, ...] = Field(
        description=("Diagonal of the per-task input cost matrix; length ``nu``."),
    )
    state_reference: tuple[float, ...] = Field(
        description="State reference the task tracks; length ``nx``.",
    )
    input_reference: tuple[float, ...] = Field(
        description="Input reference the task tracks; length ``nu``.",
    )

    @model_validator(mode="after")
    def _consistent_lengths(self) -> WBCTask:
        if len(self.state_weight) != len(self.state_reference):
            raise ValueError(
                f"task {self.name!r}: state_weight length "
                f"{len(self.state_weight)} != state_reference length "
                f"{len(self.state_reference)}."
            )
        if len(self.input_weight) != len(self.input_reference):
            raise ValueError(
                f"task {self.name!r}: input_weight length "
                f"{len(self.input_weight)} != input_reference length "
                f"{len(self.input_reference)}."
            )
        return self


def _combine_tasks(
    tasks: Sequence[WBCTask], nx: int, nu: int
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Collapse weighted tasks into ``(Q, R, x_ref, u_ref)`` diagonals.

    For each component ``j``:

    * ``Q_eff[j] = sum_i priority_i * state_weight_i[j]``
    * ``x_ref_eff[j] = sum_i priority_i * state_weight_i[j] * x_ref_i[j]
        / Q_eff[j]`` (when ``Q_eff[j] > 0``; otherwise ``0``)

    and analogously for ``R`` / ``u_ref``. Components with zero
    effective weight collapse to ``Q[j] = 1e-9`` and ``x_ref[j] = 0``
    so the spec validator's positive-weight requirement is satisfied
    while the cost contribution stays vanishing.
    """
    Q_eff = np.zeros(nx, dtype=np.float64)
    R_eff = np.zeros(nu, dtype=np.float64)
    xref_num = np.zeros(nx, dtype=np.float64)
    uref_num = np.zeros(nu, dtype=np.float64)

    for task in tasks:
        if len(task.state_weight) != nx:
            raise TargetError(
                f"task {task.name!r}: state_weight length "
                f"{len(task.state_weight)} != nx={nx}."
            )
        if len(task.input_weight) != nu:
            raise TargetError(
                f"task {task.name!r}: input_weight length "
                f"{len(task.input_weight)} != nu={nu}."
            )
        sw = np.asarray(task.state_weight, dtype=np.float64)
        iw = np.asarray(task.input_weight, dtype=np.float64)
        xr = np.asarray(task.state_reference, dtype=np.float64)
        ur = np.asarray(task.input_reference, dtype=np.float64)
        Q_eff += task.priority * sw
        R_eff += task.priority * iw
        xref_num += task.priority * sw * xr
        uref_num += task.priority * iw * ur

    # Avoid divide-by-zero on components no task weighs.
    Q_safe = np.where(Q_eff > 0.0, Q_eff, 1.0)
    R_safe = np.where(R_eff > 0.0, R_eff, 1.0)
    xref_eff = np.where(Q_eff > 0.0, xref_num / Q_safe, 0.0)
    uref_eff = np.where(R_eff > 0.0, uref_num / R_safe, 0.0)

    # Replace zero-weight components with a tiny positive value so the
    # OcpTemplateSpec validator (Q diagonal > 0 effectively required by
    # numerical stability) accepts them.
    Q_out = np.where(Q_eff > 0.0, Q_eff, 1e-9)
    R_out = np.where(R_eff > 0.0, R_eff, 1e-9)

    return (tuple(Q_out), tuple(R_out), tuple(xref_eff), tuple(uref_eff))


def wbc(
    dynamics: CasadiFunction,
    *,
    tasks: Sequence[WBCTask],
    initial_state: Sequence[float],
    Q_terminal: Sequence[float] | None = None,
    Q_terminal_factor: float = 10.0,
    horizon_steps: int = 10,
    time_horizon_s: float = 0.5,
    input_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    state_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    name: str = "wbc",
    integrator_type: Literal["ERK", "IRK", "GNSF"] = "ERK",
    nlp_solver_type: Literal["SQP", "SQP_RTI"] = "SQP_RTI",
    qp_solver: Literal[
        "PARTIAL_CONDENSING_HPIPM",
        "FULL_CONDENSING_HPIPM",
        "FULL_CONDENSING_QPOASES",
    ] = "PARTIAL_CONDENSING_HPIPM",
) -> OcpTemplateSpec:
    """Build a weighted whole-body-control spec.

    Args
    ----
    dynamics : CasadiFunction
        Translated dynamics ``f(x, u) -> dx``.
    tasks : Sequence[WBCTask]
        Tasks to combine. Must be non-empty.
    initial_state : Sequence[float]
        Pinned at ``constraints.x0``.
    Q_terminal, Q_terminal_factor, horizon_steps, time_horizon_s,
    input_bounds, state_bounds, name, integrator_type,
    nlp_solver_type, qp_solver : see :func:`jaxility.templates.lqr`.

    Returns
    -------
    OcpTemplateSpec
        Combined-cost LINEAR_LS spec ready for :func:`build_ocp`.

    Raises
    ------
    TargetError
        On empty task list or any per-task shape mismatch against the
        dynamics ``input_shapes``.
    """
    if not tasks:
        raise TargetError("wbc requires at least one WBCTask; got an empty task list.")
    if dynamics.input_shapes == () or len(dynamics.input_shapes) != 2:
        raise TargetError(
            "wbc expects a 2-input dynamics function f(x, u); got "
            f"input_shapes={dynamics.input_shapes!r}."
        )
    nx_shape, nu_shape = dynamics.input_shapes
    if len(nx_shape) != 1 or len(nu_shape) != 1:
        raise TargetError(
            "wbc expects rank-1 state and input vectors; got "
            f"state shape {nx_shape} and input shape {nu_shape}."
        )
    nx = nx_shape[0]
    nu = nu_shape[0]

    Q, R, x_ref, u_ref = _combine_tasks(tasks, nx, nu)

    if Q_terminal is not None:
        Q_terminal_t = tuple(Q_terminal)
        if len(Q_terminal_t) != nx:
            raise TargetError(f"Q_terminal length {len(Q_terminal_t)} != nx={nx}.")
    else:
        Q_terminal_t = tuple(Q_terminal_factor * q for q in Q)

    if len(initial_state) != nx:
        raise TargetError(f"initial_state length {len(initial_state)} != nx={nx}.")

    input_lower = tuple(input_bounds[0]) if input_bounds is not None else None
    input_upper = tuple(input_bounds[1]) if input_bounds is not None else None
    state_lower = tuple(state_bounds[0]) if state_bounds is not None else None
    state_upper = tuple(state_bounds[1]) if state_bounds is not None else None

    return OcpTemplateSpec(
        horizon_steps=horizon_steps,
        time_horizon_s=time_horizon_s,
        state_cost=Q,
        input_cost=R,
        terminal_state_cost=Q_terminal_t,
        state_reference=x_ref,
        input_reference=u_ref,
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
