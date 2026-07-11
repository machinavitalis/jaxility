# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""CasADi → acados OCP builder (T-021 / ADR-001).

Second of the three lowering passes from ADR-001 (JAX → CasADi → acados).
Takes the :class:`~jaxility.lowering.CasadiFunction` produced by T-020
(a dynamics function ``f(x, u) -> xdot``) plus an :class:`OcpTemplateSpec`
(horizon, sample time, cost weights, references, solver options) and
assembles an :class:`acados_template.AcadosOcp` configured with a
``LINEAR_LS`` stage cost — the canonical formulation that handles LQR,
trajectory-tracking MPC, and reference-tracking WBC all from one
template.

Surface notes:

* The builder only supports continuous-time dynamics through
  :attr:`AcadosModel.f_expl_expr`. Discretisation is the integrator
  layer's job (``integrator_type=ERK`` by default).
* The cost is ``y = [x; u]`` with ``W = blkdiag(Q, R)`` —
  the standard quadratic-tracking shape. Non-quadratic costs land
  in a later schema (NONLINEAR_LS or EXTERNAL).
* Initial-state constraints are pinned to a caller-supplied
  ``initial_state``; the OCP will track to ``state_reference`` over
  the horizon.
* Box constraints on inputs (``input_lower`` / ``input_upper``) are
  optional. Box constraints on states (``state_lower`` / ``state_upper``)
  are also optional and apply on the *interior* stages plus the
  terminal stage.

acados is imported lazily so the rest of ``jaxility.lowering`` stays
importable without the ``[acados]`` extra (PATTERNS §1.2).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import TargetError
from .jax_to_casadi import CasadiFunction

CASADI_TO_ACADOS_SCHEMA_V0 = 0
"""Schema version of the ``OcpTemplateSpec`` payload."""


class OcpTemplateSpec(BaseModel):
    """Declarative spec for an LQR / MPC OCP template.

    Captures everything ``build_ocp`` needs *beyond* the dynamics:
    horizon, sample time, cost weights, references, optional box
    constraints, and solver options. The spec is frozen and
    ``extra=\"forbid\"`` so tightening the schema is an ADR-grade
    decision (PATTERNS §3.4).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: int = Field(default=CASADI_TO_ACADOS_SCHEMA_V0, ge=0)

    horizon_steps: int = Field(
        gt=0, description="Discretisation horizon length, ``N``."
    )
    time_horizon_s: float = Field(
        gt=0.0,
        description=(
            "Total prediction horizon in seconds; per-stage step is "
            "``time_horizon_s / horizon_steps``."
        ),
    )

    state_cost: tuple[float, ...] = Field(
        description=(
            "Diagonal of the stage state-cost matrix ``Q``; length must match ``nx``."
        ),
    )
    input_cost: tuple[float, ...] = Field(
        description=(
            "Diagonal of the stage input-cost matrix ``R``; length must match ``nu``."
        ),
    )
    terminal_state_cost: tuple[float, ...] = Field(
        description=(
            "Diagonal of the terminal state-cost matrix ``Q_e``; length "
            "must match ``nx``. Use a larger weight than ``state_cost`` "
            "for stronger end-of-horizon convergence."
        ),
    )

    state_reference: tuple[float, ...] = Field(
        description="Constant state reference, length ``nx``.",
    )
    input_reference: tuple[float, ...] = Field(
        description="Constant input reference, length ``nu``.",
    )
    initial_state: tuple[float, ...] = Field(
        description=(
            "Initial state used to pin ``constraints.x0``. ``build_ocp`` "
            "writes it into the OCP; the runtime updates it before each "
            "solve."
        ),
    )

    input_lower: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Optional box-constraint lower bound on the input; "
            "length ``nu`` when supplied."
        ),
    )
    input_upper: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Optional box-constraint upper bound on the input; "
            "length ``nu`` when supplied."
        ),
    )
    state_lower: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Optional box-constraint lower bound on the (interior) "
            "states; length ``nx`` when supplied."
        ),
    )
    state_upper: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Optional box-constraint upper bound on the (interior) "
            "states; length ``nx`` when supplied."
        ),
    )

    integrator_type: Literal["ERK", "IRK", "GNSF"] = "ERK"
    """Integrator the acados solver uses. ``ERK`` is the smooth-op default."""

    nlp_solver_type: Literal["SQP", "SQP_RTI"] = "SQP_RTI"
    """``SQP_RTI`` is real-time iteration (single QP per step); ``SQP`` is full."""

    qp_solver: Literal[
        "PARTIAL_CONDENSING_HPIPM",
        "FULL_CONDENSING_HPIPM",
        "FULL_CONDENSING_QPOASES",
    ] = "PARTIAL_CONDENSING_HPIPM"
    """HPIPM is acados' default; QPOASES is the dense-active-set alternative."""

    name: str = Field(
        default="ocp",
        description="Name acados uses for code-generation artifacts.",
    )

    @model_validator(mode="after")
    def _consistent_lengths(self) -> OcpTemplateSpec:
        nx = len(self.state_cost)
        nu = len(self.input_cost)
        if len(self.terminal_state_cost) != nx:
            raise ValueError(
                f"terminal_state_cost length {len(self.terminal_state_cost)} "
                f"must equal state_cost length {nx}."
            )
        if len(self.state_reference) != nx:
            raise ValueError(
                f"state_reference length {len(self.state_reference)} must "
                f"equal state_cost length {nx}."
            )
        if len(self.input_reference) != nu:
            raise ValueError(
                f"input_reference length {len(self.input_reference)} must "
                f"equal input_cost length {nu}."
            )
        if len(self.initial_state) != nx:
            raise ValueError(
                f"initial_state length {len(self.initial_state)} must "
                f"equal state_cost length {nx}."
            )
        for label, vec in (
            ("input_lower", self.input_lower),
            ("input_upper", self.input_upper),
        ):
            if vec is not None and len(vec) != nu:
                raise ValueError(
                    f"{label} length {len(vec)} must equal input_cost length {nu}."
                )
        for label, vec in (
            ("state_lower", self.state_lower),
            ("state_upper", self.state_upper),
        ):
            if vec is not None and len(vec) != nx:
                raise ValueError(
                    f"{label} length {len(vec)} must equal state_cost length {nx}."
                )
        return self


def _extract_symbolic(dynamics: CasadiFunction) -> tuple[Any, Any, Any]:
    """Pull (x_sym, u_sym, f_expr) back out of a translated CasadiFunction.

    Uses the *preserved* symbolic inputs / outputs that T-020 stashes
    on :class:`CasadiFunction.sx_inputs` / :attr:`sx_outputs`.
    ``Function.sx_in()`` would return fresh SX symbols disjoint from
    the ones inside the expression tree, which acados rejects with
    a "free variable" error when it tries to compile the explicit
    ODE function from the model.
    """
    if len(dynamics.sx_inputs) != 2:
        raise TargetError(
            f"build_ocp expects a 2-input dynamics function f(x, u); "
            f"the CasadiFunction has {len(dynamics.sx_inputs)} inputs. "
            "Restructure the source so the dynamics take exactly state "
            "and control."
        )
    if len(dynamics.sx_outputs) != 1:
        raise TargetError(
            f"build_ocp expects a 1-output dynamics function f(x, u) -> dx; "
            f"the CasadiFunction has {len(dynamics.sx_outputs)} outputs."
        )
    x_sym = dynamics.sx_inputs[0]
    u_sym = dynamics.sx_inputs[1]
    f_expr = dynamics.sx_outputs[0]
    return x_sym, u_sym, f_expr


def _diagonal_matrix(diag: tuple[float, ...]) -> np.ndarray:
    return np.diag(np.asarray(diag, dtype=np.float64))


def build_ocp(
    dynamics: CasadiFunction,
    spec: OcpTemplateSpec,
) -> Any:
    """Construct an :class:`acados_template.AcadosOcp` from CasADi dynamics.

    The dynamics are continuous-time ``f(x, u) -> dx``. The OCP is
    configured with a ``LINEAR_LS`` stage cost and a quadratic
    terminal cost (the standard MPC / LQR shape; non-quadratic costs
    land in a later schema). Initial state is pinned via
    ``constraints.x0``; the runtime updates it per cycle.

    Args
    ----
    dynamics : CasadiFunction
        From :func:`jaxility.lowering.translate`. Signature must be
        ``f(x: nx-vector, u: nu-vector) -> dx: nx-vector``.
    spec : OcpTemplateSpec
        Horizon, sample time, costs, references, optional bounds,
        solver options.

    Returns
    -------
    acados_template.AcadosOcp
        The configured OCP. Ready to hand to
        :class:`AcadosOcpSolver` once the ``t_renderer`` binary is on
        ``$ACADOS_SOURCE_DIR/bin``.

    Raises
    ------
    TargetError
        On a dynamics signature mismatch (wrong number of in / out
        ports, or shape mismatch against the spec).
    ImportError
        If ``acados_template`` is not installed (PATTERNS §1.2 — the
        ``[acados]`` extra guards the import).
    """
    from acados_template import AcadosModel, AcadosOcp  # extras guard

    # Signature check first — a wrong port count produces a more
    # actionable error than a downstream shape mismatch.
    x_sym, u_sym, f_expr = _extract_symbolic(dynamics)

    nx_spec = len(spec.state_cost)
    nu_spec = len(spec.input_cost)
    if dynamics.input_shapes != ((nx_spec,), (nu_spec,)):
        raise TargetError(
            "CasadiFunction input shapes do not match the spec: dynamics "
            f"declares {dynamics.input_shapes}, spec declares "
            f"state_cost(len={nx_spec}) + input_cost(len={nu_spec})."
        )
    if dynamics.output_shapes != ((nx_spec,),):
        raise TargetError(
            "CasadiFunction output shape does not match the spec: dynamics "
            f"declares {dynamics.output_shapes}, expected ((nx,),) with "
            f"nx={nx_spec}."
        )

    model = AcadosModel()
    model.name = spec.name
    model.x = x_sym
    model.u = u_sym
    model.f_expl_expr = f_expr

    ocp = AcadosOcp()
    ocp.model = model

    # Cost: LINEAR_LS with y = [x; u], y_e = x. ``W = blkdiag(Q, R)``.
    Q = _diagonal_matrix(spec.state_cost)
    R = _diagonal_matrix(spec.input_cost)
    Q_e = _diagonal_matrix(spec.terminal_state_cost)

    ny = nx_spec + nu_spec

    Vx = np.zeros((ny, nx_spec))
    Vx[:nx_spec, :] = np.eye(nx_spec)
    Vu = np.zeros((ny, nu_spec))
    Vu[nx_spec:, :] = np.eye(nu_spec)
    Vx_e = np.eye(nx_spec)

    W = np.zeros((ny, ny))
    W[:nx_spec, :nx_spec] = Q
    W[nx_spec:, nx_spec:] = R

    yref = np.concatenate(
        [
            np.asarray(spec.state_reference, dtype=np.float64),
            np.asarray(spec.input_reference, dtype=np.float64),
        ]
    )
    yref_e = np.asarray(spec.state_reference, dtype=np.float64)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = W
    ocp.cost.W_e = Q_e
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    # Constraints — pin initial state, plus optional box constraints.
    ocp.constraints.x0 = np.asarray(spec.initial_state, dtype=np.float64)
    if spec.input_lower is not None and spec.input_upper is not None:
        ocp.constraints.lbu = np.asarray(spec.input_lower, dtype=np.float64)
        ocp.constraints.ubu = np.asarray(spec.input_upper, dtype=np.float64)
        ocp.constraints.idxbu = np.arange(nu_spec)
    if spec.state_lower is not None and spec.state_upper is not None:
        ocp.constraints.lbx = np.asarray(spec.state_lower, dtype=np.float64)
        ocp.constraints.ubx = np.asarray(spec.state_upper, dtype=np.float64)
        ocp.constraints.idxbx = np.arange(nx_spec)
        ocp.constraints.lbx_e = np.asarray(spec.state_lower, dtype=np.float64)
        ocp.constraints.ubx_e = np.asarray(spec.state_upper, dtype=np.float64)
        ocp.constraints.idxbx_e = np.arange(nx_spec)

    # Solver options.
    ocp.solver_options.N_horizon = spec.horizon_steps
    ocp.solver_options.tf = spec.time_horizon_s
    ocp.solver_options.integrator_type = spec.integrator_type
    ocp.solver_options.nlp_solver_type = spec.nlp_solver_type
    ocp.solver_options.qp_solver = spec.qp_solver

    return ocp
