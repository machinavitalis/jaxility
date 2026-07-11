# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Trajectory-tracking MPC template (T-023).

A *Trajectory-tracking Model Predictive Controller* is structurally
identical to LQR at build time — same quadratic cost, same dynamics,
same horizon — but its reference is **time-varying**, updated by the
runtime before each cycle's solve. The Crazyflie-class controllers in
Jaxility's launch matrix use this template.

The template produces:

* an :class:`~jaxility.lowering.OcpTemplateSpec` configured with a
  build-time *seed* reference (the first stage of the trajectory, or
  zeros), suitable for :func:`jaxility.lowering.build_ocp`, and
* a helper :func:`set_reference_trajectory` that the runtime calls
  on the constructed :class:`AcadosOcpSolver` to push per-stage
  ``yref`` values before each ``solve()``.

The split mirrors the acados API: ``ocp.cost.yref`` is a single
build-time default; per-stage references are pushed at runtime via
``solver.cost_set(k, "yref", ...)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ..errors import TargetError
from ..lowering import CasadiFunction, OcpTemplateSpec
from .lqr import lqr


def tracking_mpc(
    dynamics: CasadiFunction,
    *,
    Q: Sequence[float],
    R: Sequence[float],
    initial_state: Sequence[float],
    reference_trajectory: Sequence[Sequence[float]] | None = None,
    input_reference: Sequence[float] | None = None,
    Q_terminal: Sequence[float] | None = None,
    Q_terminal_factor: float = 10.0,
    horizon_steps: int = 30,
    time_horizon_s: float = 1.5,
    input_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    state_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    name: str = "tracking_mpc",
    integrator_type: Literal["ERK", "IRK", "GNSF"] = "ERK",
    nlp_solver_type: Literal["SQP", "SQP_RTI"] = "SQP_RTI",
    qp_solver: Literal[
        "PARTIAL_CONDENSING_HPIPM",
        "FULL_CONDENSING_HPIPM",
        "FULL_CONDENSING_QPOASES",
    ] = "PARTIAL_CONDENSING_HPIPM",
) -> OcpTemplateSpec:
    """Build a trajectory-tracking MPC spec.

    Same shape as :func:`lqr` but documents the time-varying-reference
    pattern. If ``reference_trajectory`` is supplied, its first stage
    seeds the build-time ``yref`` so a solve before the runtime first
    updates produces a sensible answer; subsequent stages are pushed
    by :func:`set_reference_trajectory` at runtime.

    Args
    ----
    reference_trajectory : Sequence[Sequence[float]] | None
        Optional preview of the desired state trajectory; each entry is
        a length-``nx`` reference. Length should equal ``horizon_steps``
        (one entry per stage 0..N-1) or ``horizon_steps + 1`` (including
        terminal). Length mismatches do not raise here — the runtime
        updater silently truncates / pads — but build-time uses the
        *first* entry as the seed yref.
    *other args*
        See :func:`lqr` — every option except the reference behaviour
        is the same.
    """
    if reference_trajectory is not None and len(reference_trajectory) == 0:
        raise TargetError(
            "reference_trajectory is empty; supply at least one stage "
            "or pass ``reference_trajectory=None`` to default to zeros."
        )

    state_reference = (
        tuple(reference_trajectory[0]) if reference_trajectory is not None else None
    )

    return lqr(
        dynamics,
        Q=Q,
        R=R,
        initial_state=initial_state,
        state_reference=state_reference,
        input_reference=input_reference,
        Q_terminal=Q_terminal,
        Q_terminal_factor=Q_terminal_factor,
        horizon_steps=horizon_steps,
        time_horizon_s=time_horizon_s,
        input_bounds=input_bounds,
        state_bounds=state_bounds,
        name=name,
        integrator_type=integrator_type,
        nlp_solver_type=nlp_solver_type,
        qp_solver=qp_solver,
    )


def set_reference_trajectory(
    solver: Any,
    reference_trajectory: Sequence[Sequence[float]],
    *,
    input_reference: Sequence[float] | None = None,
) -> None:
    """Push a per-stage state trajectory into an :class:`AcadosOcpSolver`.

    Calls ``solver.cost_set(k, "yref", yref_k)`` for ``k=0..N-1`` with
    ``yref_k = concat(state_ref[k], input_ref)`` and
    ``solver.cost_set(N, "yref", yref_e)`` with the terminal reference
    (which is state-only).

    Args
    ----
    solver : AcadosOcpSolver
        The constructed solver. The caller owns its lifecycle; this
        function only updates the cost references.
    reference_trajectory : Sequence[Sequence[float]]
        Per-stage state references. Length must be at least
        ``solver.acados_ocp.solver_options.N_horizon + 1`` so every
        stage including the terminal has a reference.
    input_reference : Sequence[float] | None
        Constant input reference appended to each interior-stage yref.
        Defaults to zeros of the right shape.
    """
    import numpy as np

    # acados 0.5.5+ exposes the OCP as ``solver.ocp``; ``solver.acados_ocp``
    # is deprecated. Prefer the new attribute when it exists.
    ocp = getattr(solver, "ocp", None) or solver.acados_ocp
    n = ocp.solver_options.N_horizon
    nx = ocp.model.x.shape[0]
    nu = ocp.model.u.shape[0]

    if len(reference_trajectory) < n + 1:
        raise TargetError(
            f"reference_trajectory length {len(reference_trajectory)} is "
            f"too short for horizon_steps={n}; need at least {n + 1} "
            "entries (one per stage including terminal)."
        )

    if input_reference is None:
        input_ref_np = np.zeros(nu, dtype=np.float64)
    else:
        if len(input_reference) != nu:
            raise TargetError(
                f"input_reference length {len(input_reference)} does not match nu={nu}."
            )
        input_ref_np = np.asarray(input_reference, dtype=np.float64)

    for k in range(n):
        state_ref_k = np.asarray(reference_trajectory[k], dtype=np.float64)
        if state_ref_k.shape != (nx,):
            raise TargetError(
                f"reference_trajectory[{k}] has shape {state_ref_k.shape}; "
                f"expected ({nx},)."
            )
        yref_k = np.concatenate([state_ref_k, input_ref_np])
        solver.cost_set(k, "yref", yref_k)

    state_ref_e = np.asarray(reference_trajectory[n], dtype=np.float64)
    if state_ref_e.shape != (nx,):
        raise TargetError(
            f"reference_trajectory[{n}] (terminal) has shape "
            f"{state_ref_e.shape}; expected ({nx},)."
        )
    solver.cost_set(n, "yref", state_ref_e)
