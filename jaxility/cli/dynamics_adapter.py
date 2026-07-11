# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Jaxterity Robot → JAX dynamics adapter.

Closes the CLI gap: zoo entries store a ``source_factory``
that returns a :class:`Source` (with ``simulate(n_steps)``), but the
host build path needs a ``f(state, control) -> dx`` JAX function the
T-020 translator can hand to CasADi. This module bridges the two by
calling :meth:`jaxterity.robot.Robot.build_system` and wrapping the
returned Jaxonomy ``_RobotSystem.ode`` callable in a plain
``(state, control) -> dx`` function.

The wrapper:

* Builds the Robot's :class:`_RobotSystem` via ``build_system(q0, qd0,
  actuation=True)`` so the ``ode`` accepts a control input.
* Constructs a fresh :class:`LeafContext` for each invocation (the
  parameters dict is captured at adapter-construction time).
* Uses ``ctx.with_continuous_state(state)`` to set the dynamics state,
  then calls ``sys.ode(0.0, ctx, control, **params)`` and returns the
  resulting derivative array.

The adapter is JAX-traceable (``jax.jit`` works on it directly), which
is the property the T-020 ``translate`` function requires.

Why a separate module: keeps the import-time cost of touching
Jaxterity off the ``jaxility.cli`` boot path. The CLI only imports
this module when ``jaxility build <real-robot-entry>`` fires.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp


def jax_dynamics_from_robot(
    robot: Any,
    *,
    q0: list[float] | tuple[float, ...],
    qd0: list[float] | tuple[float, ...],
    nu: int,
) -> tuple[Callable[[Any, Any], Any], tuple[int, ...], tuple[int, ...]]:
    """Extract a JAX dynamics function from a Jaxterity Robot.

    Args
    ----
    robot : jaxterity.robot.Robot
        The upstream robot. The adapter does *not* require any
        calibration state — callers wanting the production gate use
        the :class:`jaxility.testing.JaxteritySource.from_robot`
        adapter's ``require_calibration_state`` argument upstream.
    q0, qd0 : Sequence[float]
        Initial joint positions and velocities. These pin the
        :class:`_RobotSystem`'s build (Jaxonomy needs an initial state
        to construct the context); they do *not* constrain the
        dynamics function's domain — the returned callable accepts
        any ``state`` of the right shape.
    nu : int
        Control vector size. Jaxterity's :class:`_RobotSystem` carries
        a named ``tau`` input port whose dimension the caller knows
        per-robot (cartpole: 1, SO-100: 6). Passing it explicitly keeps
        the adapter from having to introspect Jaxonomy's port internals.

    Returns
    -------
    (jax_fn, state_shape, control_shape) : tuple
        ``jax_fn(state, control) -> dx`` is a JAX-traceable callable.
        ``state_shape`` is ``(len(q0) + len(qd0),)``; ``control_shape``
        is ``(nu,)``.
    """
    sys = robot.build_system(q0=list(q0), qd0=list(qd0), actuation=True)
    ctx = sys.create_context()
    params = dict(ctx.parameters) if ctx.parameters else {}

    state_dim = len(q0) + len(qd0)
    state_shape: tuple[int, ...] = (state_dim,)
    control_shape: tuple[int, ...] = (nu,)

    def jax_fn(state: Any, control: Any) -> Any:
        state_j = jnp.asarray(state)
        control_j = jnp.asarray(control)
        ctx_with_x = ctx.with_continuous_state(state_j)
        return sys.ode(0.0, ctx_with_x, control_j, **params)

    return jax_fn, state_shape, control_shape


__all__ = ["jax_dynamics_from_robot"]
