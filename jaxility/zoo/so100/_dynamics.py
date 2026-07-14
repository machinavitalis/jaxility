# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Closed-form articulated-body dynamics for a serial revolute manipulator.

The zoo's flyers (Cartpole, Crazyflie) deploy a hand-written *explicit* closed
form because their forward dynamics have no matrix inverse. A manipulator does:
``q̈ = M(q)⁻¹(τ − h(q, q̇))``, and the lowering coverage table has no linear
solve. The lowerable route is **Featherstone's Articulated Body Algorithm
(ABA)** — O(n) forward dynamics that never forms ``M⁻¹``; for 1-DoF revolute
joints it needs only spatial (6-vector) matmuls and *scalar* reciprocals
``1/D_i``, all inside the smooth-op subset, so it translates to CasADi.

Spatial quantities use Featherstone's ``[angular; linear]`` 6-vector convention
(``Rigid Body Dynamics Algorithms``, Table 7.1). The tree — fixed
parent→child transforms, joint axes, and per-link spatial inertias — is read
from a Jaxterity :class:`~jaxterity.robot.Robot` so that calibrating a mass or
inertia propagates into the lowered dynamics.

Fidelity: the ABA is validated against the robot's MJX ``functional_dynamics``
in a contact-free regime to manipulator grade (see the so100 tests). It is *not*
ULP-exact against MJX — an independent recursive algorithm and MuJoCo's internal
``cinert`` CRB diverge by ~1e-6 on this featherweight arm, and that is a genuine
representational floor, not a bug (see AGENTS/DECISIONS.md ADR-016).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

# Fixed 3×3 blocks reused when assembling 6×6 spatial matrices.
_Z3 = jnp.zeros((3, 3))
_I3 = jnp.eye(3)


def spatial_tree(robot: Any) -> list[dict[str, Any]]:
    """Extract the serial revolute chain (base→tip) as per-joint spatial data.

    Each entry carries the fixed parent→child rotation ``E`` and translation
    ``r`` (from the joint's ``origin`` transform), the joint ``axis``, and the
    child link's ``mass`` / ``com`` / full ``inertia`` (about the com, in the
    body frame). Sourced from the live Robot, so calibration flows through.
    """
    revolute = {
        j.child_link: j for j in robot.joints if str(j.joint_type).endswith("REVOLUTE")
    }
    children = set(revolute)
    roots = [j for j in revolute.values() if j.parent_link not in children]
    if len(roots) != 1:
        raise ValueError(
            f"expected a single serial revolute chain; found roots {roots!r}"
        )

    order = []
    cur = roots[0]
    while True:
        order.append(cur)
        nxt = [j for j in revolute.values() if j.parent_link == cur.child_link]
        if not nxt:
            break
        cur = nxt[0]

    tree = []
    for j in order:
        link = robot.link_map[j.child_link]
        tree.append(
            {
                # ``origin.rotation`` is child-in-parent; the motion transform
                # parent→child uses its transpose.
                "E": np.asarray(j.origin.rotation).T,
                "r": np.asarray(j.origin.translation),
                "axis": np.asarray(j.axis.value.xyz),
                "m": float(link.mass),
                "com": np.asarray(link.com),
                "inertia": np.asarray(link.inertia),
            }
        )
    return tree


def _skew(v: Any) -> Any:
    z = v[0] * 0.0
    return jnp.stack(
        [
            jnp.stack([z, -v[2], v[1]]),
            jnp.stack([v[2], z, -v[0]]),
            jnp.stack([-v[1], v[0], z]),
        ]
    )


def _blk(a: Any, b: Any, c: Any, d: Any) -> Any:
    """Assemble a 6×6 from four 3×3 blocks ``[[a, b], [c, d]]``."""
    return jnp.concatenate(
        [jnp.concatenate([a, b], axis=1), jnp.concatenate([c, d], axis=1)], axis=0
    )


def _xrot(E: Any) -> Any:
    return _blk(E, _Z3, _Z3, E)


def _xtrans(r: Any) -> Any:
    return _blk(_I3, _Z3, -_skew(r), _I3)


def _crm(v: Any) -> Any:
    """Spatial motion cross-product matrix (Featherstone ``crm``)."""
    return _blk(_skew(v[:3]), _Z3, _skew(v[3:]), _skew(v[:3]))


def _mci(m: Any, c: Any, ic: Any) -> Any:
    """Spatial inertia about the body origin from mass ``m``, com ``c``, and
    rotational inertia ``ic`` (about the com, body frame)."""
    C = _skew(c)
    return _blk(ic + m * C @ C.T, m * C, m * C.T, m * _I3)


def _rot_axis(axis: Any, angle: Any) -> Any:
    c, s = jnp.cos(angle), jnp.sin(angle)
    x, y, z = axis[0], axis[1], axis[2]
    one_c = 1.0 - c
    return jnp.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def manipulator_ode(tree: list[dict[str, Any]], gravity: float):
    """Build the ABA forward dynamics ``f(state, tau) -> dstate`` for the chain.

    ``state = concatenate([q, q̇])`` (each length ``n``), ``tau`` the ``n`` joint
    torques, and ``dstate = concatenate([q̇, q̈])``. Returns ``(f, n)``.
    """
    n = len(tree)
    E = [jnp.asarray(t["E"]) for t in tree]
    r = [jnp.asarray(t["r"]) for t in tree]
    axis = [jnp.asarray(t["axis"]) for t in tree]
    inertia = [
        _mci(t["m"], jnp.asarray(t["com"]), jnp.asarray(t["inertia"])) for t in tree
    ]
    S = [jnp.concatenate([axis[i], jnp.zeros(3)]) for i in range(n)]
    # Base spatial acceleration = −gravity (Featherstone's gravity trick).
    a0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, gravity])

    def f(state: Any, tau: Any) -> Any:
        q, qd = state[:n], state[n:]
        xup = [
            _xrot(_rot_axis(axis[i], q[i]).T) @ (_xrot(E[i]) @ _xtrans(r[i]))
            for i in range(n)
        ]

        # Pass 1 (base→tip): velocities, velocity-product accel, bias force.
        v: list[Any] = [None] * n
        c: list[Any] = [None] * n
        ia = [inertia[i] for i in range(n)]
        pa: list[Any] = [None] * n
        for i in range(n):
            vj = S[i] * qd[i]
            vp = jnp.zeros(6) if i == 0 else v[i - 1]
            v[i] = xup[i] @ vp + vj
            c[i] = _crm(v[i]) @ vj
            pa[i] = (-_crm(v[i]).T) @ ia[i] @ v[i]

        # Pass 2 (tip→base): articulated inertia + bias.
        u_arr: list[Any] = [None] * n
        d_arr: list[Any] = [None] * n
        uu: list[Any] = [None] * n
        for i in reversed(range(n)):
            uu[i] = ia[i] @ S[i]
            d_arr[i] = S[i] @ uu[i]
            u_arr[i] = tau[i] - S[i] @ pa[i]
            if i > 0:
                small = ia[i] - jnp.outer(uu[i], uu[i]) / d_arr[i]
                bias = pa[i] + small @ c[i] + uu[i] * (u_arr[i] / d_arr[i])
                ia[i - 1] = ia[i - 1] + xup[i].T @ small @ xup[i]
                pa[i - 1] = pa[i - 1] + xup[i].T @ bias

        # Pass 3 (base→tip): accelerations and joint accelerations.
        a: list[Any] = [None] * n
        qdd: list[Any] = [None] * n
        for i in range(n):
            ap = a0 if i == 0 else a[i - 1]
            at = xup[i] @ ap + c[i]
            qdd[i] = (u_arr[i] - uu[i] @ at) / d_arr[i]
            a[i] = at + S[i] * qdd[i]

        return jnp.concatenate([qd, jnp.stack(qdd)])

    return f, n
