# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Reference robot zoo: per-robot Jaxility deployment configurations.

Mirrors Jaxterity's zoo (cartpole, so100) with two additional
deployment configurations for robots Jaxility documents but Jaxterity
has not yet promoted to its zoo (crazyflie, berkeley_humanoid_lite).
Each entry pins:

* a source factory — what the lowering pipeline consumes,
* the canonical target the entry exercises,
* the controller template the eventual real deployment will use,
* the dtype + trajectory length the mock pipeline runs at,
* a documented upstream status and the remaining work to land the
  real deployment.

The mock pipeline runs end-to-end on every config in CI — this is the
regression suite for the full contract surface (T-017
acceptance criterion). Later work swaps the synthetic simulate
for real Jaxonomy-driven dynamics, and promotes Crazyflie /
Berkeley Humanoid Lite from stubs to real Robots once Jaxterity ships
their zoo entries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..targets import Target
from ..testing import MockArtifactBundle, Source, mock_lower

ControllerTemplate = Literal["LQR", "TrackingMPC", "WBC", "CentroidalMPC"]
"""Controller templates the zoo entries route to."""

UpstreamStatus = Literal["real-robot", "stub-pending-jaxterity"]
"""Where the source comes from: a real Jaxterity robot, or a stub."""

DynamicsFactory = Callable[
    [], tuple[Callable[[Any, Any], Any], tuple[int, ...], tuple[int, ...]]
]
"""Dynamics surface: a zero-arg factory returning the JAX dynamics function.

The factory yields ``(jax_fn, state_shape, control_shape)`` where
``jax_fn(state, control) -> dx`` is a JAX-traceable callable suitable
for :func:`jaxility.lowering.translate`. The factory is lazy because
realising the dynamics for a real robot triggers Jaxterity loading
which we keep off the import path.

Stub zoo entries (Crazyflie, Berkeley Humanoid Lite) have ``None`` for
:attr:`ZooDeploymentConfig.jax_dynamics_factory`; the CLI surfaces
that as a structured "stub upstream" error.
"""


@dataclass(frozen=True)
class ZooDeploymentConfig:
    """One row in the Jaxility deployment-zoo regression suite."""

    name: str
    source_factory: Callable[[], Source]
    target: Target
    template: ControllerTemplate
    dtype: Literal["float32", "float64"]
    n_steps: int
    description: str
    license: str
    upstream_status: UpstreamStatus
    remaining_work: tuple[str, ...]
    jax_dynamics_factory: DynamicsFactory | None = None
    """Dynamics hook: lazily extract a JAX dynamics function from the source.

    ``None`` for stub entries (Crazyflie, Berkeley Humanoid Lite). Real
    entries (Cartpole, SO-100) supply a factory that calls
    :func:`jaxility.cli.dynamics_adapter.jax_dynamics_from_robot` against
    the upstream Jaxterity Robot.
    """

    template_options: dict[str, Any] = field(default_factory=dict)
    """Template hook: template-specific knobs the CLI passes through.

    Shape depends on ``template``: ``LQR`` expects ``Q``, ``R``,
    ``initial_state``, optional ``input_bounds``; ``TrackingMPC`` adds
    ``reference_trajectory``; ``WBC`` expects ``tasks``;
    ``CentroidalMPC`` expects ``initial_com_state``. Missing keys fall
    back to the template's own defaults.
    """


def mock_build(config: ZooDeploymentConfig) -> MockArtifactBundle:
    """Run the mock pipeline on a zoo deployment config.

    The bundle's manifest verifies; the equivalence check passes
    trivially; the artifact's hash is determined by the source's
    attestation handle + the target + the dtype + the n_steps
    (invariant 5 holds across the whole zoo, not just one entry).
    """
    source = config.source_factory()
    return mock_lower(source, config.target, dtype=config.dtype, n_steps=config.n_steps)


# Lazy registry — entries import only when ``CONFIGS`` is touched, so
# importing :mod:`jaxility.zoo` does not load Jaxterity unless the caller
# actually walks the registry.
def _load_configs() -> dict[str, ZooDeploymentConfig]:
    from .berkeley_humanoid_lite import config as berkeley_config
    from .cartpole import config as cartpole_config
    from .crazyflie import config as crazyflie_config
    from .so100 import config as so100_config

    entries = (
        cartpole_config(),
        so100_config(),
        crazyflie_config(),
        berkeley_config(),
    )
    return {entry.name: entry for entry in entries}


_CONFIGS_CACHE: dict[str, ZooDeploymentConfig] | None = None


def CONFIGS() -> dict[str, ZooDeploymentConfig]:  # noqa: N802 (lookup-table API)
    """Return the registry of zoo deployment configs (cached)."""
    global _CONFIGS_CACHE
    if _CONFIGS_CACHE is None:
        _CONFIGS_CACHE = _load_configs()
    return _CONFIGS_CACHE


def available() -> list[str]:
    """List the registered zoo entries by name."""
    return sorted(CONFIGS())


def load(name: str) -> ZooDeploymentConfig:
    """Return a registered zoo entry by name."""
    configs = CONFIGS()
    if name not in configs:
        raise KeyError(f"unknown zoo entry {name!r}; available: {sorted(configs)}")
    return configs[name]


__all__ = [
    "CONFIGS",
    "ControllerTemplate",
    "UpstreamStatus",
    "ZooDeploymentConfig",
    "available",
    "load",
    "mock_build",
]
