# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Centroidal MPC template (T-025).

A *centroidal MPC* tracks the centre-of-mass dynamics of a floating-base
robot rather than the full multibody dynamics. For a biped /
quadruped / humanoid this is the natural decomposition: the centroidal
layer plans CoM trajectories at the fast control loop; a lower-level
WBC (T-024) maps those into joint torques.

The template ships a **simplified single-contact-effort** form:

* The state is the centroidal CoM kinematics
  ``[c_x, c_y, c_z, c_dot_x, c_dot_y, c_dot_z]`` (6-state default; the
  template accepts any state size the caller has translated).
* The control is the net force / wrench acting on the CoM. The caller
  supplies dynamics ``f([CoM, CoM_dot], wrench) -> dx``; the template
  configures the cost without assuming a specific dynamics shape.

Multi-contact formulations with explicit per-foot forces, contact
schedules, and angular-momentum bookkeeping are a later
enhancement (T-080 in the post-launch backlog). The current surface
exists so the humanoid zoo entry can run end-to-end against a stub
centroidal model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ..lowering import CasadiFunction, OcpTemplateSpec
from .lqr import lqr


def centroidal_mpc(
    dynamics: CasadiFunction,
    *,
    Q: Sequence[float],
    R: Sequence[float],
    initial_com_state: Sequence[float],
    target_com_state: Sequence[float] | None = None,
    wrench_reference: Sequence[float] | None = None,
    Q_terminal: Sequence[float] | None = None,
    Q_terminal_factor: float = 10.0,
    horizon_steps: int = 20,
    time_horizon_s: float = 1.0,
    wrench_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    state_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    name: str = "centroidal_mpc",
    integrator_type: Literal["ERK", "IRK", "GNSF"] = "ERK",
    nlp_solver_type: Literal["SQP", "SQP_RTI"] = "SQP_RTI",
    qp_solver: Literal[
        "PARTIAL_CONDENSING_HPIPM",
        "FULL_CONDENSING_HPIPM",
        "FULL_CONDENSING_QPOASES",
    ] = "PARTIAL_CONDENSING_HPIPM",
) -> OcpTemplateSpec:
    """Build a centroidal-dynamics MPC spec.

    The template is structurally identical to :func:`lqr` but
    uses centroidal-state-and-wrench naming so callers reach for the
    right factory for the right intent. The dynamics are user-supplied
    via the translated :class:`CasadiFunction`; common forms are:

    * **Linear:** ``c_ddot = wrench / m + g`` (3D double integrator).
    * **Nonlinear with angular momentum:** ``L_dot = sum((p_i - c) × F_i)``
      coupled with the linear part.

    The template makes no assumption about which form the caller passes;
    it only assembles the OCP.

    Args
    ----
    dynamics : CasadiFunction
        ``f([CoM_state], wrench) -> dx``. ``nx`` and ``nu`` come from
        ``dynamics.input_shapes``.
    Q : Sequence[float]
        Diagonal of the stage state cost; length ``nx``.
    R : Sequence[float]
        Diagonal of the stage wrench cost; length ``nu``.
    initial_com_state : Sequence[float]
        Starting CoM state. Pinned at ``constraints.x0``.
    target_com_state : Sequence[float] | None
        Desired CoM state. Defaults to zeros (CoM at origin, at rest).
    wrench_reference : Sequence[float] | None
        Defaults to zeros — appropriate when only the dynamics gravity
        offset is non-zero.
    Q_terminal, Q_terminal_factor, horizon_steps, time_horizon_s,
    wrench_bounds, state_bounds, name, solver options :
        See :func:`lqr`.
    """
    return lqr(
        dynamics,
        Q=Q,
        R=R,
        initial_state=initial_com_state,
        state_reference=target_com_state,
        input_reference=wrench_reference,
        Q_terminal=Q_terminal,
        Q_terminal_factor=Q_terminal_factor,
        horizon_steps=horizon_steps,
        time_horizon_s=time_horizon_s,
        input_bounds=wrench_bounds,
        state_bounds=state_bounds,
        name=name,
        integrator_type=integrator_type,
        nlp_solver_type=nlp_solver_type,
        qp_solver=qp_solver,
    )


def friction_box_bounds(
    contact_active: Sequence[bool],
    *,
    fz_max: float = 200.0,
    mu: float = 0.7,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-foot box bounds approximating a friction cone, gated by contact.

    For each foot in ``contact_active``: a **stance** foot gets a pyramidal
    friction-cone box ``f_z ∈ [0, fz_max]``, ``|f_x|, |f_y| ≤ μ·fz_max`` (the
    standard linearised-cone surrogate); a **swing** foot gets ``[0, 0, 0]`` on
    both sides, forcing zero force. Stacked in foot order, the result is a
    ``(lower, upper)`` pair of length ``3·n_contacts`` ready for
    :func:`multi_contact_centroidal_mpc`'s ``force_bounds``.
    """
    lower: list[float] = []
    upper: list[float] = []
    for active in contact_active:
        if active:
            lower += [-mu * fz_max, -mu * fz_max, 0.0]
            upper += [mu * fz_max, mu * fz_max, fz_max]
        else:
            lower += [0.0, 0.0, 0.0]
            upper += [0.0, 0.0, 0.0]
    return tuple(lower), tuple(upper)


def multi_contact_centroidal_mpc(
    dynamics: CasadiFunction,
    *,
    Q: Sequence[float],
    R: Sequence[float],
    initial_com_state: Sequence[float],
    contact_active: Sequence[bool],
    fz_max: float = 200.0,
    mu: float = 0.7,
    target_com_state: Sequence[float] | None = None,
    force_reference: Sequence[float] | None = None,
    Q_terminal: Sequence[float] | None = None,
    Q_terminal_factor: float = 10.0,
    horizon_steps: int = 20,
    time_horizon_s: float = 1.0,
    name: str = "multi_contact_centroidal_mpc",
    integrator_type: Literal["ERK", "IRK", "GNSF"] = "ERK",
    nlp_solver_type: Literal["SQP", "SQP_RTI"] = "SQP_RTI",
    qp_solver: Literal[
        "PARTIAL_CONDENSING_HPIPM",
        "FULL_CONDENSING_HPIPM",
        "FULL_CONDENSING_QPOASES",
    ] = "PARTIAL_CONDENSING_HPIPM",
) -> OcpTemplateSpec:
    """Multi-contact centroidal MPC over the single-rigid-body model.

    Where :func:`centroidal_mpc` plans a single net wrench, this plans the
    **per-foot contact forces** of a multi-contact stance. The state is the full
    9-D SRBD state ``[c, ċ, L]`` (CoM position, velocity, *and angular momentum*
    about the CoM — the term that matters for balance); the control is the
    stacked contact forces ``[f₁ … f_{n_c}]`` (``nu = 3·n_c``). Per-foot
    :func:`friction_box_bounds` impose a linearised friction cone on stance feet
    and pin swing feet to zero force, so the same template serves any stance by
    flipping ``contact_active``.

    The matching plant is
    :func:`jaxterity.locomotion.reduced.srbd_dynamics` with the contact points
    baked in as constants — lower it with :func:`jaxility.lowering.translate`
    (``in_shapes=((9,), (3·n_c,))``) and pass the result here.

    Args
    ----
    dynamics : CasadiFunction
        SRBD dynamics ``f([c, ċ, L], forces_flat) -> dx`` (nx = 9, nu = 3·n_c).
    Q, R : Sequence[float]
        Stage state (len 9) and force (len ``3·n_c``) cost diagonals.
    initial_com_state : Sequence[float]
        Starting 9-D centroidal state.
    contact_active : Sequence[bool]
        Per-foot stance/swing flags (``len == n_contacts``).
    fz_max, mu : float
        Friction-box ceiling and coefficient (see :func:`friction_box_bounds`).
    target_com_state, force_reference, Q_terminal, Q_terminal_factor,
    horizon_steps, time_horizon_s, name, solver options :
        As in :func:`centroidal_mpc`.

    Returns
    -------
    OcpTemplateSpec

    Notes
    -----
    The contact set is held constant over the horizon (one stance). A
    time-varying contact *schedule* across the horizon (per-stage bounds) is the
    natural next extension and is gated on per-stage-constraint support in the
    builder.
    """
    force_bounds = friction_box_bounds(contact_active, fz_max=fz_max, mu=mu)
    return centroidal_mpc(
        dynamics,
        Q=Q,
        R=R,
        initial_com_state=initial_com_state,
        target_com_state=target_com_state,
        wrench_reference=force_reference,
        Q_terminal=Q_terminal,
        Q_terminal_factor=Q_terminal_factor,
        horizon_steps=horizon_steps,
        time_horizon_s=time_horizon_s,
        wrench_bounds=force_bounds,
        name=name,
        integrator_type=integrator_type,
        nlp_solver_type=nlp_solver_type,
        qp_solver=qp_solver,
    )
