# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Generate lowerable rigid-body dynamics from a robot description (T-124).

The hand-written zoo dynamics (``zoo/*/_dynamics.py``) are authored *per
topology*: SO-100's Featherstone ABA is specialised to a single serial revolute
chain with a fixed base. This module is the general path — it reads any
fixed-base URDF/MJCF and emits the forward dynamics as a
:class:`~jaxility.lowering.CasadiFunction`, ready for :func:`build_ocp`.

**Two backends, one API** (``generate_dynamics(..., backend=...)``):

* ``"pinocchio-casadi"`` — wrap Pinocchio's own CasADi codegen of the ABA. Needs
  Pinocchio built ``WITH_CASADI_SUPPORT`` (conda-forge, or a source build); the
  PyPI ``pin`` wheel omits it. Machine-precision fidelity (spike: ``~1e-9`` vs
  MJX on SO-100).
* ``"jaxility-aba"`` — use Pinocchio purely as the **parser** (the pip wheel is
  enough) and emit the ABA in jaxility's *own* CasADi, generalizing
  ``zoo/so100/_dynamics`` from a serial chain to Pinocchio's parent-index tree.
  No exotic dependency; manipulator-grade fidelity (the ``~1e-5`` independent-
  recursion floor, same class as the hand-written zoo). Verified against the
  numeric ``pin.aba`` oracle for serial *and* branched topologies.
* ``"auto"`` (default) — ``"pinocchio-casadi"`` when the bindings are importable,
  else ``"jaxility-aba"``.

**Why this shape (T-124 spike).** ``build_ocp`` is source-agnostic — it needs
only the original SX symbols, so either backend threads in unchanged (no "free
variable" error). The manifest never attested JAX primitives (coverage is a
translation-time gate a generated graph never enters), so provenance travels via
``Manifest.toolchain_versions['pinocchio']`` (see
:func:`jaxility.manifest.detect_pinocchio_version`) rather than a coverage trail.

**Scope of this first cut.** Fixed-base robots whose joints are all 1-DoF
(revolute / prismatic), i.e. ``nq == nv``. Floating base (``nq != nv``) is
rejected loudly — its state derivative is not the plain ``[v, a]`` this emits,
and legged robots additionally need a deploy-friendly *contact* model (T-122),
out of scope here. ``frictionloss`` is non-smooth and unsupported; ``armature``
(reflected rotor inertia) and joint viscous ``damping`` are carried across when
supplied, because Pinocchio's parser does not populate them from the source and
for light distal links armature *dominates* the effective inertia (spike
finding).
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal as PyLiteral

from ..errors import ToolchainError
from .jax_to_casadi import CasadiFunction

__all__ = ["generate_dynamics"]

_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)


# --------------------------------------------------------------------------- #
# Dependency guards                                                            #
# --------------------------------------------------------------------------- #


def _require_pinocchio() -> Any:
    """Import Pinocchio or raise a loud, install-directing error (PATTERNS §1.2)."""
    try:
        import pinocchio  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised via the [rbd] guard
        raise ToolchainError(
            "pinocchio is not installed. The rigid-body-dynamics generator "
            "(T-124) parses a URDF/MJCF via Pinocchio. Install the optional "
            "extra: `pip install 'jaxility[rbd]'` (or `pip install pin`)."
        ) from exc
    return pinocchio


def _has_pinocchio_casadi() -> bool:
    """True iff Pinocchio's CasADi bindings are importable."""
    try:
        import pinocchio.casadi  # noqa: F401,PLC0415
    except Exception:
        return False
    return True


def _require_pinocchio_casadi() -> Any:
    """Import Pinocchio's CasADi bindings or raise a loud, actionable error.

    The ``"pinocchio-casadi"`` backend needs Pinocchio built
    ``WITH_CASADI_SUPPORT``. The PyPI ``pin`` wheel does **not** ship
    ``pinocchio.casadi`` (T-124 finding), so this names the two fixes rather than
    surfacing a raw ``ModuleNotFoundError`` — and the pip-only ``"jaxility-aba"``
    backend needs none of this.
    """
    _require_pinocchio()
    try:
        import pinocchio.casadi as cpin  # noqa: PLC0415
    except Exception as exc:
        raise ToolchainError(
            "the 'pinocchio-casadi' backend needs Pinocchio's CasADi bindings "
            "(`pinocchio.casadi`), but they are not present — the PyPI `pin` wheel "
            "is built without CasADi support. Either install a CasADi-enabled "
            "Pinocchio (`conda install -c conda-forge pinocchio`, or build from "
            "source with `-DBUILD_WITH_CASADI_SUPPORT=ON`), or use "
            "`backend='jaxility-aba'` (the pip-only path, no extra deps)."
        ) from exc
    return cpin


def _looks_like_xml(source: str | Path) -> bool:
    return isinstance(source, str) and source.lstrip().startswith("<")


def _build_model(pinocchio: Any, source: str | Path, source_format: str) -> Any:
    """Parse ``source`` into a Pinocchio ``Model``.

    ``source`` is a path to a description file, or (for convenience) the XML
    text itself. MJCF is parsed through a temp file because Pinocchio's MJCF
    entry point takes a filename only.
    """
    if source_format == "urdf":
        if _looks_like_xml(source):
            return pinocchio.buildModelFromXML(str(source))
        return pinocchio.buildModelFromUrdf(str(source))
    if source_format == "mjcf":
        if _looks_like_xml(source):
            with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
                fh.write(str(source))
                tmp = fh.name
            try:
                return pinocchio.buildModelFromMJCF(tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
        return pinocchio.buildModelFromMJCF(str(source))
    raise ToolchainError(
        f"unknown source_format {source_format!r}; expected 'urdf' or 'mjcf'."
    )


# --------------------------------------------------------------------------- #
# jaxility-aba backend: spatial algebra (Featherstone [angular; linear], the    #
# same convention as zoo/so100/_dynamics, generalized to a parent-index tree).  #
# Numeric helpers build the fixed per-joint data; CasADi helpers build the      #
# q-dependent expressions.                                                      #
# --------------------------------------------------------------------------- #


def _np_skew(np: Any, v: Any) -> Any:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _np_xrot(np: Any, E: Any) -> Any:
    Z = np.zeros((3, 3))
    return np.block([[E, Z], [Z, E]])


def _np_xtrans(np: Any, r: Any) -> Any:
    I3, Z = np.eye(3), np.zeros((3, 3))
    return np.block([[I3, Z], [-_np_skew(np, r), I3]])


def _np_spatial_inertia(np: Any, m: float, c: Any, ic: Any) -> Any:
    """6x6 spatial inertia about the joint origin from mass, com ``c``, and the
    rotational inertia ``ic`` about the com (matches ``so100._dynamics._mci``)."""
    C = _np_skew(np, c)
    return np.block([[ic + m * C @ C.T, m * C], [m * C.T, m * np.eye(3)]])


def _ca_skew(ca: Any, v: Any) -> Any:
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def _ca_xrot(ca: Any, R: Any) -> Any:
    Z = ca.SX.zeros(3, 3)
    return ca.vertcat(ca.horzcat(R, Z), ca.horzcat(Z, R))


def _ca_xtrans(ca: Any, r: Any) -> Any:
    I3, Z = ca.SX.eye(3), ca.SX.zeros(3, 3)
    return ca.vertcat(ca.horzcat(I3, Z), ca.horzcat(-_ca_skew(ca, r), I3))


def _ca_rot_axis(ca: Any, axis: Any, angle: Any) -> Any:
    """Rodrigues rotation about a *numeric* unit ``axis`` by a *symbolic* angle."""
    c, s = ca.cos(angle), ca.sin(angle)
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    oc = 1.0 - c
    return ca.vertcat(
        ca.horzcat(c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s),
        ca.horzcat(y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s),
        ca.horzcat(z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc),
    )


def _ca_crm(ca: Any, m: Any) -> Any:
    """Spatial motion cross-product matrix (Featherstone ``crm``)."""
    w, vl = m[0:3], m[3:6]
    Sw, Sv, Z = _ca_skew(ca, w), _ca_skew(ca, vl), ca.SX.zeros(3, 3)
    return ca.vertcat(ca.horzcat(Sw, Z), ca.horzcat(Sv, Sw))


def _joint_type_axis(jm: Any, np: Any) -> tuple[str, Any]:
    """Return ``("R"|"P", unit-axis)`` for a 1-DoF Pinocchio joint model."""
    name = jm.shortname()
    aligned = {
        "JointModelRX": ("R", (1, 0, 0)),
        "JointModelRY": ("R", (0, 1, 0)),
        "JointModelRZ": ("R", (0, 0, 1)),
        "JointModelPX": ("P", (1, 0, 0)),
        "JointModelPY": ("P", (0, 1, 0)),
        "JointModelPZ": ("P", (0, 0, 1)),
    }
    if name in aligned:
        kind, ax = aligned[name]
        return kind, np.asarray(ax, dtype=float)
    kind = "R" if "Revolute" in name else "P" if "Prismatic" in name else ""
    axis = getattr(jm, "axis", None)
    if kind and axis is not None:
        return kind, np.asarray(axis, dtype=float).reshape(3)
    raise ToolchainError(
        f"unsupported joint type {name!r}; the 'jaxility-aba' backend supports "
        "1-DoF revolute / prismatic joints only (nq == nv)."
    )


def _extract_tree(model: Any, np: Any) -> list[dict[str, Any]]:
    """Per-joint tree data from a parsed Pinocchio model (base -> tip order).

    Pinocchio numbers joints so a parent's index is smaller than its children's,
    so the returned list is already a valid base->tip topological order.
    """
    tree: list[dict[str, Any]] = []
    for i in range(1, model.njoints):
        parent = model.parents[i]
        M = model.jointPlacements[i]
        E = np.asarray(M.rotation).T  # motion transform parent->child uses R^T
        r = np.asarray(M.translation).reshape(3)
        kind, axis = _joint_type_axis(model.joints[i], np)
        Y = model.inertias[i]
        spatial_I = _np_spatial_inertia(
            np, float(Y.mass), np.asarray(Y.lever).reshape(3), np.asarray(Y.inertia)
        )
        if kind == "R":
            subspace = np.concatenate([axis, np.zeros(3)])
        else:
            subspace = np.concatenate([np.zeros(3), axis])
        tree.append(
            {
                "parent": -1 if parent == 0 else parent - 1,
                "XT": _np_xrot(np, E) @ _np_xtrans(np, r),  # fixed parent->joint
                "kind": kind,
                "axis": axis,
                "S": subspace,
                "I": spatial_I,
            }
        )
    return tree


def _emit_jaxility_aba(
    ca: Any,
    np: Any,
    tree: list[dict[str, Any]],
    gravity: tuple[float, float, float],
    armature: Any,
    damping: Any,
) -> tuple[Any, Any, Any]:
    """Emit tree-ABA forward dynamics as SX ``(x, u, dx)`` with x=[q,v], u=tau."""
    n = len(tree)
    XT = [ca.DM(t["XT"]) for t in tree]
    S = [ca.DM(t["S"]) for t in tree]
    Ispat = [ca.DM(t["I"]) for t in tree]
    parent = [t["parent"] for t in tree]
    arm = np.zeros(n) if armature is None else armature
    # base spatial acceleration = -gravity (Featherstone's gravity trick).
    a0 = ca.DM(np.concatenate([np.zeros(3), -np.asarray(gravity, dtype=float)]))

    nx = 2 * n
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", n)
    q, qd = x[:n], x[n:nx]
    tau = u if damping is None else u - ca.DM(np.asarray(damping)) * qd

    xup: list[Any] = []
    for j, t in enumerate(tree):
        if t["kind"] == "R":
            xj = _ca_xrot(ca, _ca_rot_axis(ca, t["axis"], q[j]).T)
        else:
            xj = _ca_xtrans(ca, q[j] * ca.DM(t["axis"]))
        xup.append(xj @ XT[j])

    # Pass 1 (base -> tip): velocities, velocity-product accel, bias force.
    v: list[Any] = [None] * n
    c: list[Any] = [None] * n
    IA: list[Any] = [Ispat[j] for j in range(n)]
    pA: list[Any] = [None] * n
    for j in range(n):
        vJ = S[j] * qd[j]
        vp = ca.SX.zeros(6) if parent[j] < 0 else v[parent[j]]
        v[j] = xup[j] @ vp + vJ
        c[j] = _ca_crm(ca, v[j]) @ vJ
        pA[j] = (-_ca_crm(ca, v[j]).T) @ Ispat[j] @ v[j]

    # Pass 2 (tip -> base): articulated inertia + bias, propagated to parents.
    U: list[Any] = [None] * n
    D: list[Any] = [None] * n
    uacc: list[Any] = [None] * n
    for j in reversed(range(n)):
        U[j] = IA[j] @ S[j]
        D[j] = S[j].T @ U[j] + arm[j]
        uacc[j] = tau[j] - (S[j].T @ pA[j])
        p = parent[j]
        if p >= 0:
            Ia = IA[j] - (U[j] @ U[j].T) / D[j]
            pa = pA[j] + Ia @ c[j] + U[j] * (uacc[j] / D[j])
            IA[p] = IA[p] + xup[j].T @ Ia @ xup[j]
            pA[p] = pA[p] + xup[j].T @ pa

    # Pass 3 (base -> tip): accelerations and joint accelerations.
    a: list[Any] = [None] * n
    qdd_terms: list[Any] = []
    for j in range(n):
        ap = a0 if parent[j] < 0 else a[parent[j]]
        at = xup[j] @ ap + c[j]
        qdd_j = (uacc[j] - (U[j].T @ at)) / D[j]
        a[j] = at + S[j] * qdd_j
        qdd_terms.append(qdd_j)

    qdd = ca.vertcat(*qdd_terms)
    dx = ca.vertcat(qd, qdd)
    return x, u, dx


def _emit_pinocchio_casadi(
    cpin: Any,
    ca: Any,
    np: Any,
    model: Any,
    gravity: tuple[float, float, float],
    armature: Any,
    damping: Any,
) -> tuple[Any, Any, Any]:
    """Emit forward dynamics via Pinocchio's own CasADi ABA codegen."""
    model.gravity.linear = np.asarray(gravity, dtype=float)
    if armature is not None:
        model.armature = np.asarray(armature, dtype=float)
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    nq, nv = model.nq, model.nv
    nx = nq + nv
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nv)
    qc, vc = x[:nq], x[nq:nx]
    tau = u if damping is None else u - ca.DM(np.asarray(damping)) * vc
    qdd = cpin.aba(cmodel, cdata, qc, vc, tau)
    dx = ca.vertcat(vc, qdd)
    return x, u, dx


def generate_dynamics(
    source: str | Path,
    *,
    source_format: PyLiteral["urdf", "mjcf"] = "urdf",
    backend: PyLiteral["auto", "pinocchio-casadi", "jaxility-aba"] = "auto",
    gravity: tuple[float, float, float] = _DEFAULT_GRAVITY,
    armature: Sequence[float] | None = None,
    damping: Sequence[float] | None = None,
    name: str = "f",
    dtype: PyLiteral["float32", "float64"] = "float64",
) -> CasadiFunction:
    """Emit fixed-base forward dynamics as a lowerable :class:`CasadiFunction`.

    The returned function has the zoo's continuous-time signature
    ``f(x, u) -> dx`` with ``x = [q, v]`` (length ``2n``), ``u = tau`` (length
    ``n``), and ``dx = [v, qdd]`` — the same shape a hand-written
    ``_dynamics_factory`` produces, so it drops straight into :func:`build_ocp`.

    Args
    ----
    source:
        Path to a URDF/MJCF file, or the XML text itself.
    source_format:
        ``"urdf"`` or ``"mjcf"``. Selects the Pinocchio parser.
    backend:
        ``"pinocchio-casadi"`` wraps Pinocchio's own CasADi ABA (needs a CasADi-
        enabled Pinocchio; ``~1e-9`` fidelity). ``"jaxility-aba"`` emits the ABA
        in jaxility's own CasADi from the parsed model (pip-only, no extra deps;
        manipulator-grade ``~1e-5``). ``"auto"`` picks the former when the
        bindings are importable, else the latter.
    gravity:
        World-frame gravity vector. Defaults to ``(0, 0, -9.81)``; Pinocchio's
        parsers do not reliably apply the description's own gravity option, so it
        is set explicitly (spike finding).
    armature:
        Per-joint reflected rotor inertia (length ``n``), added to the joint-
        space inertia like MuJoCo's ``armature``. Supply it when the source has
        non-zero armature — Pinocchio's parser leaves it zeroed, and for light
        distal links it dominates the effective inertia.
    damping:
        Per-joint viscous damping (length ``n``); applied as a passive force
        ``-damping * v`` folded into the input torque, matching MuJoCo.
    name:
        Name for the CasADi ``Function``.
    dtype:
        Retained for API symmetry with :func:`translate`; CasADi SX is symbolic
        so this does not change the graph (acados evaluates in float64).

    Returns
    -------
    CasadiFunction
        ``sx_inputs`` / ``sx_outputs`` preserve the original SX symbols so
        :func:`build_ocp` accepts the model directly. ``primitives_used`` is
        empty: there is no jaxpr, so the coverage audit trail does not apply —
        provenance travels via ``Manifest.toolchain_versions['pinocchio']``.

    Raises
    ------
    ToolchainError
        Pinocchio missing, an unknown ``source_format`` / ``backend``, a
        floating-base / multi-DoF-joint model (``nq != nv``), a bad ``armature``
        / ``damping`` length, or (``"pinocchio-casadi"``) the CasADi bindings
        being absent.
    """
    import numpy as np  # noqa: PLC0415
    import casadi as ca  # noqa: PLC0415

    pinocchio = _require_pinocchio()
    model = _build_model(pinocchio, source, source_format)
    nq, nv = model.nq, model.nv
    if nq != nv:
        raise ToolchainError(
            f"generate_dynamics supports fixed-base 1-DoF-joint robots only "
            f"(nq == nv); got nq={nq}, nv={nv}. A floating base or a "
            "spherical/free joint is out of scope for this first cut (its state "
            "derivative is not [v, qdd]); legged robots additionally need a "
            "contact model (T-122)."
        )

    arm = _checked_vector(np, armature, nv, "armature")
    damp = _checked_vector(np, damping, nv, "damping")

    if backend == "auto":
        backend = "pinocchio-casadi" if _has_pinocchio_casadi() else "jaxility-aba"

    if backend == "pinocchio-casadi":
        cpin = _require_pinocchio_casadi()
        x, u, dx = _emit_pinocchio_casadi(cpin, ca, np, model, gravity, arm, damp)
    elif backend == "jaxility-aba":
        tree = _extract_tree(model, np)
        x, u, dx = _emit_jaxility_aba(ca, np, tree, gravity, arm, damp)
    else:
        raise ToolchainError(
            f"unknown backend {backend!r}; expected 'auto', 'pinocchio-casadi', "
            "or 'jaxility-aba'."
        )

    nx = nq + nv
    fn = ca.Function(name, [x, u], [dx])
    return CasadiFunction(
        name=name,
        fn=fn,
        input_shapes=((nx,), (nv,)),
        output_shapes=((nx,),),
        primitives_used=frozenset(),
        sx_inputs=(x, u),
        sx_outputs=(dx,),
    )


def _checked_vector(np: Any, value: Any, n: int, label: str) -> Any:
    """Validate an optional per-joint vector, returning a length-``n`` array or None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (n,):
        raise ToolchainError(f"{label} must have length nv={n}; got shape {arr.shape}.")
    return arr
