# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Pinocchio rigid-body-dynamics generator (T-124).

``generate_dynamics`` lowers any fixed-base URDF/MJCF to a
:class:`~jaxility.lowering.CasadiFunction` — the general path that replaces
hand-writing ``zoo/*/_dynamics.py`` per topology. Two backends:

* ``"jaxility-aba"`` — Pinocchio parses, jaxility emits the CasADi ABA (pip-only,
  no CasADi-enabled Pinocchio needed). Exercised here directly.
* ``"pinocchio-casadi"`` — wraps Pinocchio's own CasADi ABA; needs the CasADi
  bindings the PyPI ``pin`` wheel omits, so that one test ``importorskip``s them.

Tiers: **unit** (Pinocchio only) — shapes/eval, the ``pin.aba`` oracle check on a
*branched* tree, damping, guards, ``build_ocp`` threading; **fidelity**
(+ Jaxterity) — the T-124 spike hardened: generated SO-100 dynamics match MJX.

Self-skips where a dep is absent (``pin`` is the optional ``[rbd]`` extra;
Jaxterity is editable-installed in the bootstrap env).
"""

from __future__ import annotations

import numpy as np
import pytest

pinocchio = pytest.importorskip("pinocchio")

from jaxility.errors import ToolchainError  # noqa: E402
from jaxility.lowering import CasadiFunction, generate_dynamics  # noqa: E402

# --------------------------------------------------------------------------- #
# Minimal inline descriptions (no external assets) for the Pinocchio-only tier. #
# --------------------------------------------------------------------------- #


def _link(name: str, com: str, mass: float, i: float) -> str:
    return (
        f'<link name="{name}"><inertial>'
        f'<origin xyz="{com}"/><mass value="{mass}"/>'
        f'<inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/>'
        "</inertial></link>"
    )


def _rev(name: str, parent: str, child: str, xyz: str, axis: str) -> str:
    return (
        f'<joint name="{name}" type="revolute">'
        f'<parent link="{parent}"/><child link="{child}"/>'
        f'<origin xyz="{xyz}"/><axis xyz="{axis}"/>'
        '<limit lower="-2" upper="2" effort="5" velocity="5"/></joint>'
    )


_TWO_LINK_URDF = f"""<?xml version="1.0"?>
<robot name="twolink">
  {_link("base", "0 0 0", 1.0, 0.01)}
  {_link("l1", "0.1 0 0", 0.5, 0.005)}
  {_link("l2", "0.1 0 0", 0.3, 0.003)}
  {_rev("j1", "base", "l1", "0 0 0.1", "0 0 1")}
  {_rev("j2", "l1", "l2", "0.2 0 0", "0 1 0")}
</robot>
"""

# A branched tree: l1 carries TWO children — a revolute and a prismatic — so this
# exercises multi-child articulated-inertia accumulation and mixed joint types.
_PRISMATIC_J3 = (
    '<joint name="j3" type="prismatic"><parent link="l1"/><child link="l3"/>'
    '<origin xyz="0.2 0.1 0"/><axis xyz="1 0 0"/>'
    '<limit lower="-0.5" upper="0.5" effort="5" velocity="5"/></joint>'
)
_BRANCHED_URDF = f"""<?xml version="1.0"?>
<robot name="branch">
  {_link("base", "0 0 0", 1.0, 0.01)}
  {_link("l1", "0.1 0 0", 0.5, 0.005)}
  {_link("l2", "0.1 0 0", 0.3, 0.003)}
  {_link("l3", "0.05 0 0", 0.2, 0.002)}
  {_rev("j1", "base", "l1", "0 0 0.1", "0 0 1")}
  {_rev("j2", "l1", "l2", "0.2 0 0", "0 1 0")}
  {_PRISMATIC_J3}
</robot>
"""

# A floating root joint makes nq (7, quaternion) != nv (6) -> scope guard.
_FLOATING_URDF = f"""<?xml version="1.0"?>
<robot name="floater">
  {_link("base", "0 0 0", 1.0, 0.01)}
  {_link("body", "0 0 0", 0.5, 0.005)}
  <joint name="free" type="floating"><parent link="base"/><child link="body"/></joint>
</robot>
"""


# --------------------------------------------------------------------------- #
# Unit tier — Pinocchio only, jaxility-aba backend (pip-only).                   #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_generate_dynamics_shapes_and_eval() -> None:
    """A 2-DoF URDF yields f(x=[q,v], u=tau) -> dx=[v, qdd] with dx[:n]==v."""
    dyn = generate_dynamics(_TWO_LINK_URDF, backend="jaxility-aba", name="twolink")
    assert isinstance(dyn, CasadiFunction)
    assert dyn.input_shapes == ((4,), (2,))
    assert dyn.output_shapes == ((4,),)
    assert dyn.primitives_used == frozenset()  # no jaxpr -> no coverage trail

    x = np.array([0.2, -0.3, 0.5, -0.1])  # [q(2), v(2)]
    u = np.array([0.1, -0.05])
    (dx,) = dyn(x, u)
    assert dx.shape == (4,)
    assert np.allclose(dx[:2], x[2:])  # d/dt q == v exactly
    assert np.all(np.isfinite(dx))


@pytest.mark.unit
def test_jaxility_aba_matches_pinocchio_oracle_branched() -> None:
    """The emitted tree-ABA matches Pinocchio's own numeric ABA to ~machine eps.

    On a *branched* robot with mixed revolute/prismatic joints — the strongest
    correctness check, and topology-general (``pin.aba`` is the reference).
    """
    model = pinocchio.buildModelFromXML(_BRANCHED_URDF)
    model.gravity.linear = np.array([0.0, 0.0, -9.81])
    data = model.createData()
    nv = model.nv
    assert nv == 3
    dyn = generate_dynamics(_BRANCHED_URDF, backend="jaxility-aba", name="branch")

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-0.5, 0.5, nv)
        v = rng.uniform(-0.3, 0.3, nv)
        tau = rng.uniform(-0.1, 0.1, nv)
        qdd_ref = np.asarray(pinocchio.aba(model, data, q, v, tau))
        (dx,) = dyn(np.concatenate([q, v]), tau)
        rel = np.linalg.norm(dx[nv:] - qdd_ref) / (np.linalg.norm(qdd_ref) + 1e-12)
        worst = max(worst, rel)
    assert worst < 1e-10, f"tree-ABA vs pin.aba worst rel {worst:.2e}"


@pytest.mark.unit
def test_generate_dynamics_damping_folds_into_torque() -> None:
    """Viscous damping enters as a passive -d*v force folded into the torque."""
    d = np.array([0.3, 0.7])
    f_damped = generate_dynamics(_TWO_LINK_URDF, backend="jaxility-aba", damping=d)
    f_plain = generate_dynamics(_TWO_LINK_URDF, backend="jaxility-aba")

    q, v, tau = np.array([0.1, 0.2]), np.array([0.4, -0.6]), np.array([0.05, 0.02])
    x = np.concatenate([q, v])
    (dx_damped,) = f_damped(x, tau)
    (dx_plain,) = f_plain(x, tau - d * v)  # feed the reduced torque explicitly
    assert np.allclose(dx_damped, dx_plain, atol=1e-12)


@pytest.mark.unit
def test_generate_dynamics_rejects_floating_base() -> None:
    """nq != nv (a free-flyer's quaternion) is out of scope and fails loudly."""
    with pytest.raises(ToolchainError, match="nq == nv|floating"):
        generate_dynamics(_FLOATING_URDF, backend="jaxility-aba")


@pytest.mark.unit
def test_generate_dynamics_input_guards() -> None:
    """Unknown format/backend and mis-sized armature/damping raise ToolchainError."""
    with pytest.raises(ToolchainError, match="source_format"):
        generate_dynamics(_TWO_LINK_URDF, source_format="sdf")  # type: ignore[arg-type]
    with pytest.raises(ToolchainError, match="backend"):
        generate_dynamics(_TWO_LINK_URDF, backend="magic")  # type: ignore[arg-type]
    with pytest.raises(ToolchainError, match="armature"):
        generate_dynamics(
            _TWO_LINK_URDF, backend="jaxility-aba", armature=[1.0, 2.0, 3.0]
        )
    with pytest.raises(ToolchainError, match="damping"):
        generate_dynamics(_TWO_LINK_URDF, backend="jaxility-aba", damping=[1.0])


@pytest.mark.unit
def test_generated_dynamics_builds_an_ocp() -> None:
    """The emitted SX threads into build_ocp unchanged (source-agnostic pipeline)."""
    pytest.importorskip("acados_template")
    from jaxility.lowering import OcpTemplateSpec, build_ocp

    dyn = generate_dynamics(_TWO_LINK_URDF, backend="jaxility-aba", name="twolink_ocp")
    nx, nu = 4, 2
    spec = OcpTemplateSpec(
        name="twolink_ocp",
        horizon_steps=10,
        time_horizon_s=0.2,
        state_cost=tuple([1.0] * nx),
        input_cost=tuple([0.1] * nu),
        terminal_state_cost=tuple([10.0] * nx),
        state_reference=tuple([0.0] * nx),
        input_reference=tuple([0.0] * nu),
        initial_state=(0.1, -0.1, 0.0, 0.0),
    )
    ocp = build_ocp(dyn, spec)  # must not raise a CasADi "free variable" error
    assert ocp.model.f_expl_expr is not None


@pytest.mark.unit
def test_pinocchio_casadi_backend_when_available() -> None:
    """The native backend, where the CasADi-enabled Pinocchio bindings exist."""
    pytest.importorskip("pinocchio.casadi")
    dyn = generate_dynamics(
        _TWO_LINK_URDF, backend="pinocchio-casadi", name="twolink_native"
    )
    assert dyn.input_shapes == ((4,), (2,))
    (dx,) = dyn(np.array([0.1, 0.2, 0.3, -0.1]), np.array([0.05, -0.02]))
    assert dx.shape == (4,) and np.all(np.isfinite(dx))


# --------------------------------------------------------------------------- #
# Fidelity tier — Pinocchio + Jaxterity (the T-124 spike as a regression).      #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_generated_so100_matches_mjx_reference() -> None:
    """Generated SO-100 dynamics match the robot's MJX functional_dynamics.

    The T-124 build-vs-buy evidence, hardened: over contact-free, joint-limit-
    respecting states the generated ABA agrees with MJX to manipulator grade
    (the jaxility-aba backend measures ~1e-9; the bound here is the 1e-4 gate).
    """
    pytest.importorskip("jaxterity")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxterity.zoo import load

    robot = load("so100")
    f_mjx = robot.functional_dynamics()
    dyn = generate_dynamics(
        robot.to_mjcf(), source_format="mjcf", backend="jaxility-aba", name="so100"
    )
    nv = 6
    assert dyn.input_shapes == ((12,), (6,))

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(32):
        # within all SO-100 joint limits (gripper lower is ~-0.17) -> contact-free
        q = rng.uniform(-0.15, 0.15, nv)
        v = rng.uniform(-0.1, 0.1, nv)
        tau = rng.uniform(-0.05, 0.05, nv)
        x = np.concatenate([q, v])

        (dx_gen,) = dyn(x, tau)
        qdd_mjx = np.asarray(f_mjx(jnp.asarray(x), jnp.asarray(tau)))[nv:]
        rel = np.linalg.norm(dx_gen[nv:] - qdd_mjx) / (np.linalg.norm(qdd_mjx) + 1e-12)
        worst = max(worst, rel)

    assert worst < 1e-4, f"generated SO-100 vs MJX worst rel err {worst:.2e}"
