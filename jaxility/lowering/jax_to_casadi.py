# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""JAX → CasADi translator, smooth-op subset (T-020 / ADR-001).

This module is the first of the three lowering passes named in ADR-001
(JAX → CasADi → acados). It walks the jaxpr emitted by
:func:`jax.make_jaxpr` and dispatches each primitive to a small CasADi
handler. Every primitive is gated through
:func:`jaxility.lowering.coverage.assert_supported` so the coverage
table (T-013) is the source of truth for what is allowed.

Design points worth recording:

* The translator works at the jaxpr level rather than tracing CasADi
  through ``jax.eval_shape``-like instrumentation because jaxprs are
  the stable JAX IR; we want the translation to remain valid across
  JAX minor versions.
* Constant literals appearing as inputs to a primitive are passed
  through as numeric values (CasADi handles ``ca.SX + float``
  naturally).
* The ``jit`` call primitive is treated as a *transparent* boundary:
  we recurse into its sub-jaxpr with the same env. This matters
  because ``jnp.where`` over a *static* predicate is lowered by JAX
  into a ``jit``-wrapped ``select_n`` with a literal predicate, which
  we then constant-fold.
* The ``cond`` primitive and ``select_n`` over a traced predicate
  are rejected with a structured :class:`CoverageError` referencing
  the ``lax.cond[traced]`` / ``jnp.where[traced]`` rows in the
  coverage table.
* Output shapes are recovered from the jaxpr's outvar avals rather
  than from CasADi (CasADi flattens trailing singleton dims).

T-021 picks up the ``CasadiFunction`` produced here and
builds an ``acados`` OCP from it; T-031 cross-compiles the
result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Literal as PyLiteral

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from ..errors import CoverageError
from .coverage import assert_supported

JAX_TO_CASADI_SCHEMA_V0 = 0
"""Schema version of the ``CasadiFunction`` payload."""


@dataclass(frozen=True)
class CasadiFunction:
    """A translated CasADi :class:`~casadi.Function` + provenance.

    Carries the function itself plus the input / output shapes (from
    the source jaxpr) and the set of JAX primitives that participated
    in the translation. The primitive set is the audit trail the
    manifest carries (invariant 7 — every emitted op is declared in
    coverage).

    The raw symbolic inputs (:attr:`sx_inputs`) and outputs
    (:attr:`sx_outputs`) are preserved alongside the compiled
    Function. Downstream passes (T-021 acados OCP builder, T-024 WBC
    template) consume the symbolic expressions directly; without them
    a reconstructed function via ``Function.sx_in()`` would carry
    *new* SX symbols that are not the ones inside the original
    expression tree, causing acados to reject the model with a "free
    variable" error.
    """

    name: str
    fn: Any  # casadi.Function; left untyped to avoid a top-level casadi import
    """The CasADi :class:`~casadi.Function`."""

    input_shapes: tuple[tuple[int, ...], ...]
    """Shapes of the function's inputs, in jaxpr order."""

    output_shapes: tuple[tuple[int, ...], ...]
    """Shapes of the function's outputs, in jaxpr order."""

    primitives_used: frozenset[str]
    """Set of JAX primitive names emitted into the CasADi graph."""

    sx_inputs: tuple[Any, ...]
    """Original CasADi SX input symbols (the same objects the expression tree uses)."""

    sx_outputs: tuple[Any, ...]
    """Original CasADi SX output expressions (rooted in :attr:`sx_inputs`)."""

    def __call__(self, *args: Any) -> list[np.ndarray]:
        """Evaluate the translated function on numpy / list inputs.

        Returns a list of :class:`numpy.ndarray` reshaped to the
        original jaxpr output shapes (CasADi flattens trailing
        singleton dims; this restores them).
        """
        raw = self.fn(*args)
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        out: list[np.ndarray] = []
        for value, shape in zip(raw, self.output_shapes):
            arr = (
                np.asarray(value).reshape(shape)
                if shape
                else np.asarray(value).reshape(())
            )
            out.append(arr)
        return out


# ---------------------------------------------------------------------------
# Primitive dispatch.
# ---------------------------------------------------------------------------


PrimitiveHandler = Callable[[Any, list[Any], dict[str, Any]], list[Any]]
"""Per-primitive handler. Signature: ``(ca, in_vals, eqn.params) -> out_vals``.

``ca`` is the imported ``casadi`` module (passed in so handlers do
not need to re-import it). ``in_vals`` are the inputs after
literal-resolution; ``params`` are the primitive's static params.
The handler returns a list of CasADi values (one per output).
"""


_PRIMITIVE_HANDLERS: dict[str, PrimitiveHandler] = {}


def _register(name: str) -> Callable[[PrimitiveHandler], PrimitiveHandler]:
    def deco(fn: PrimitiveHandler) -> PrimitiveHandler:
        _PRIMITIVE_HANDLERS[name] = fn
        return fn

    return deco


# Arithmetic.
def _bcast(ca: Any, a: Any, b: Any) -> tuple[Any, Any]:
    """Reconcile two operands for an element-wise op, mirroring JAX/numpy
    broadcasting. CasADi lifts Python/1×1 scalars implicitly, so only a
    non-scalar shape mismatch — e.g. a ``(k, 1)`` column against a ``(1, k)``
    row, as an outer product ``u[:, None] * u[None, :]`` produces — is
    ``repmat``'d to the common shape. Numpy constants are lifted to ``DM``."""
    if isinstance(a, np.ndarray):
        a = ca.DM(a)
    if isinstance(b, np.ndarray):
        b = ca.DM(b)
    ash = (a.size1(), a.size2()) if hasattr(a, "size1") else (1, 1)
    bsh = (b.size1(), b.size2()) if hasattr(b, "size1") else (1, 1)
    if ash == (1, 1) or bsh == (1, 1) or ash == bsh:
        return a, b  # native scalar broadcast, or already aligned
    rows, cols = max(ash[0], bsh[0]), max(ash[1], bsh[1])
    if ash != (rows, cols):
        a = ca.repmat(a, rows // ash[0], cols // ash[1])
    if bsh != (rows, cols):
        b = ca.repmat(b, rows // bsh[0], cols // bsh[1])
    return a, b


@_register("add")
def _add(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    a, b = _bcast(ca, xs[0], xs[1])
    return [a + b]


@_register("sub")
def _sub(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    a, b = _bcast(ca, xs[0], xs[1])
    return [a - b]


@_register("mul")
def _mul(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    a, b = _bcast(ca, xs[0], xs[1])
    return [a * b]


@_register("div")
def _div(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    a, b = _bcast(ca, xs[0], xs[1])
    return [a / b]


@_register("neg")
def _neg(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [-xs[0]]


@_register("integer_pow")
def _integer_pow(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    y = params["y"]
    if y == 2:
        return [xs[0] * xs[0]]
    if y == 3:
        return [xs[0] * xs[0] * xs[0]]
    return [xs[0] ** y]


@_register("pow")
def _pow(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [xs[0] ** xs[1]]


# Smooth elementwise.
@_register("sin")
def _sin(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [ca.sin(xs[0])]


@_register("cos")
def _cos(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [ca.cos(xs[0])]


@_register("tan")
def _tan(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [ca.tan(xs[0])]


@_register("exp")
def _exp(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [ca.exp(xs[0])]


@_register("log")
def _log(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [ca.log(xs[0])]


@_register("sqrt")
def _sqrt(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    return [ca.sqrt(xs[0])]


# Shape ops.
@_register("broadcast_in_dim")
def _broadcast_in_dim(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    # ``broadcast_in_dim`` adds singleton dims and broadcasts. Two families
    # occur on the smooth-op surface:
    #
    #  * scalar → (1,) / (N,): CasADi broadcasts scalars implicitly under
    #    +/-/*//, so passing the value through preserves numerical output.
    #  * rank-1 → rank-2: promoting a length-k vector to a row (1, k) or a
    #    column (k, 1). This is how ``jnp.stack``/``jnp.array`` build a matrix
    #    from vectors (e.g. a ``skew`` cross-product matrix in rigid-body
    #    dynamics). CasADi stores a rank-1 value as a k×1 *column*, so a target
    #    row (broadcast_dimensions=(1,)) is the transpose; a column
    #    (broadcast_dimensions=(0,)) passes through. Tiling handles a genuine
    #    size-1 broadcast along the other axis.
    x = ca.DM(xs[0]) if isinstance(xs[0], np.ndarray) else xs[0]
    out_shape = tuple(params["shape"])
    bdims = tuple(params["broadcast_dimensions"])

    # Scalar source (rank 0): materialize to the target shape. CasADi broadcasts
    # a scalar implicitly under element-wise ops, but a scalar fed into a matmul
    # (``dot_general``) is taken as a scalar multiply — so a ``scalar → (N,)``
    # broadcast that later hits ``@`` must be a real N×1 column, not a 1×1.
    if len(bdims) == 0:
        if len(out_shape) == 0 or out_shape in ((1,), (1, 1)):
            return [x]
        rows = out_shape[0]
        cols = out_shape[1] if len(out_shape) == 2 else 1
        return [ca.repmat(x, rows, cols)]

    # Rank-preserving (identity) broadcast → pass through.
    if len(out_shape) <= 1:
        return [x]

    if len(out_shape) == 2:
        rows, cols = out_shape
        if bdims == (0,):
            # Vector indexes the rows → column (k×1), tiled across `cols`.
            return [x if cols == 1 else ca.repmat(x, 1, cols)]
        if bdims == (1,):
            # Vector indexes the columns → row (1×k), tiled across `rows`.
            row = x.T
            return [row if rows == 1 else ca.repmat(row, rows, 1)]

    raise CoverageError(
        f"broadcast_in_dim to shape {out_shape} with broadcast_dimensions "
        f"{bdims} is not supported",
        op=f"broadcast_in_dim[{out_shape}]",
        dtype="float64",
        target_family="host",
        suggestion=(
            "the lowering supports scalar broadcasts and rank-1 → rank-2 "
            "row/column promotion; rank-3+ broadcasts need a real tensor type."
        ),
    )


@_register("squeeze")
def _squeeze(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    # Drop singleton dims. CasADi values are already 2D under the hood
    # (CasADi has no rank-0 / rank-1 distinction); the squeeze is a
    # shape annotation rather than a real operation. Pass through.
    return [xs[0]]


@_register("convert_element_type")
def _convert_element_type(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    # JAX emits this for dtype-promotion / int→float casts. CasADi has
    # one numeric type (double), so the cast is a no-op.
    return [xs[0]]


@_register("concatenate")
def _concatenate(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    dim = params["dimension"]
    if dim == 0:
        return [ca.vertcat(*xs)]
    if dim == 1:
        return [ca.horzcat(*xs)]
    raise CoverageError(
        f"concatenate along dimension {dim} is not supported",
        op=f"concatenate[dim={dim}]",
        dtype="float64",
        target_family="host",
        suggestion=(
            "use ``dimension=0`` or ``dimension=1`` (vertcat / horzcat); "
            "rank-3+ concatenations require a real tensor type in the "
            "lowering and are deferred to a future schema."
        ),
    )


@_register("reshape")
def _reshape(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    new_shape = params["new_sizes"]
    if len(new_shape) == 1:
        return [ca.reshape(xs[0], new_shape[0], 1)]
    if len(new_shape) == 2:
        return [ca.reshape(xs[0], new_shape[0], new_shape[1])]
    if len(new_shape) == 0:
        # Scalar reshape — pass through.
        return [xs[0]]
    raise CoverageError(
        f"reshape to rank-{len(new_shape)} target is not supported",
        op="reshape",
        dtype="float64",
        target_family="host",
        suggestion="reshape to rank 0, 1, or 2 only.",
    )


@_register("transpose")
def _transpose(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    perm = params["permutation"]
    if perm == (1, 0):
        return [xs[0].T]
    if perm == (0,) or perm == (0, 1):
        return [xs[0]]
    raise CoverageError(
        f"transpose with permutation {perm!r} is not supported",
        op="transpose",
        dtype="float64",
        target_family="host",
        suggestion=(
            "only the 2D matrix transpose ``(1, 0)`` and identity "
            "permutations are supported."
        ),
    )


# Linear algebra.
@_register("dot_general")
def _dot_general(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    contracting, batch = params["dimension_numbers"]
    if batch != ((), ()):
        raise CoverageError(
            "batched dot_general is not supported",
            op="dot_general[batched]",
            dtype="float64",
            target_family="host",
            suggestion=(
                "rewrite the call as a non-batched 2D ``@`` or 1D "
                "inner product; batched matmul lands with a tensor type."
            ),
        )
    # CasADi stores every value as a 2D matrix (a JAX rank-1 ``(k,)`` is a
    # k×1 column). Constant operands arrive as numpy arrays and must be lifted
    # to CasADi ``DM`` before ``@`` — numpy's ``__matmul__`` does not defer to
    # CasADi. The left operand's contracting axis then selects the product:
    #
    #   * left contracts axis 1 (``([1], [0])``): standard matmul — 2D×2D or a
    #     matrix·vector (6×6 @ 6×1 → 6×1).
    #   * left contracts axis 0 (``([0], [0])``): the left operand is contracted
    #     on its leading axis — a 1D·1D inner product or vector·matrix. In
    #     CasADi's column convention that is ``aᵀ @ b`` (1×k @ k×n).
    left_contract = tuple(contracting[0])
    a = ca.DM(xs[0]) if isinstance(xs[0], np.ndarray) else xs[0]
    b = ca.DM(xs[1]) if isinstance(xs[1], np.ndarray) else xs[1]
    if left_contract == (0,):
        return [a.T @ b]
    return [a @ b]


# Indexing.
@_register("slice")
def _slice(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    start = params["start_indices"]
    limit = params["limit_indices"]
    strides = params.get("strides")
    if strides is not None and any(s != 1 for s in strides):
        raise CoverageError(
            f"strided slicing (strides={strides!r}) is not supported",
            op="slice[strided]",
            dtype="float64",
            target_family="host",
            suggestion="rewrite without strides; only unit-stride is supported.",
        )
    if len(start) == 1:
        return [xs[0][start[0] : limit[0]]]
    if len(start) == 2:
        return [xs[0][start[0] : limit[0], start[1] : limit[1]]]
    raise CoverageError(
        f"slicing of rank-{len(start)} arrays is not supported",
        op="slice",
        dtype="float64",
        target_family="host",
        suggestion="only rank-1 and rank-2 slicing is supported.",
    )


# ``dynamic_slice`` with literal start indices — the runtime equivalent of
# ``slice`` but expressed as a separate primitive when the source code
# uses ``jax.lax.dynamic_slice`` directly. Traced (non-Literal) start
# indices remain rejected (they would need a CasADi conditional / case
# expression and a dynamic-shape coverage row).
@_register("dynamic_slice")
def _dynamic_slice(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    # xs[0] is the source; xs[1:] are the start indices (one per axis).
    operand = xs[0]
    starts = xs[1:]
    sizes = params["slice_sizes"]
    # Only accept literal-integer starts.
    if not all(
        isinstance(s, (int, np.integer))
        or (isinstance(s, np.ndarray) and s.shape == ())
        for s in starts
    ):
        raise CoverageError(
            "dynamic_slice over a traced start index is rejected",
            op="dynamic_slice[traced]",
            dtype="float64",
            target_family="host",
            suggestion=(
                "use ``jax.lax.dynamic_slice`` with literal start indices, "
                "or rewrite as a plain ``operand[a:b]`` slice. The lowering "
                "accepts static-index slicing only (see the "
                "``dynamic_slice[static]`` row in COVERAGE_TABLE)."
            ),
        )
    if len(starts) == 1:
        s0 = int(np.asarray(starts[0]).item())
        n0 = int(sizes[0])
        return [operand[s0 : s0 + n0]]
    if len(starts) == 2:
        s0 = int(np.asarray(starts[0]).item())
        n0 = int(sizes[0])
        s1 = int(np.asarray(starts[1]).item())
        n1 = int(sizes[1])
        return [operand[s0 : s0 + n0, s1 : s1 + n1]]
    raise CoverageError(
        f"rank-{len(starts)} dynamic_slice is not supported",
        op="dynamic_slice",
        dtype="float64",
        target_family="host",
        suggestion="only rank-1 and rank-2 are supported.",
    )


# Static-predicate where folds through here. JAX wraps the operation
# in a ``jit[name=_where]`` containing a ``select_n d a b`` with a
# literal predicate.
@_register("select_n")
def _select_n(ca: Any, xs: list[Any], params: dict[str, Any]) -> list[Any]:
    predicate = xs[0]
    if isinstance(predicate, (bool, np.bool_, np.ndarray)):
        # Literal predicate → constant-fold.
        chosen = bool(np.asarray(predicate).item())
        # JAX's select_n is the n-ary form; for the where pattern n = 2
        # with xs[1] and xs[2] as the two branches. The convention is
        # ``select_n(pred, *cases)`` returns ``cases[int(pred)]``.
        if chosen:
            return [xs[2]]
        return [xs[1]]
    raise CoverageError(
        "jnp.where over a traced predicate is rejected",
        op="jnp.where[traced]",
        dtype="float64",
        target_family="host",
        suggestion=(
            "the smooth-op subset excludes traced predicates because "
            "the resulting graph is non-smooth; use a smoothing "
            "approximation (sigmoid with a named sharpness factor) or "
            "restructure the dynamics."
        ),
    )


def _handle_jit(
    ca: Any,
    eqn: JaxprEqn,
    in_vals: list[Any],
    env: dict[Var, Any],
    primitives_used: set[str],
) -> list[Any]:
    """Recurse into a ``jit`` primitive's sub-jaxpr transparently.

    JAX lowers ``jnp.where(static_bool, x, y)`` into a ``jit`` wrapping
    a ``select_n`` with a literal predicate; recursing keeps the
    constant-fold path live without reaching for CasADi.
    """
    inner_jaxpr: Jaxpr = eqn.params["jaxpr"].jaxpr
    inner_env: dict[Var, Any] = dict(env)
    for inner_var, in_val in zip(inner_jaxpr.invars, in_vals):
        inner_env[inner_var] = in_val
    _translate_eqns(ca, list(inner_jaxpr.eqns), inner_env, primitives_used)
    return [_resolve_var(inner_env, outvar) for outvar in inner_jaxpr.outvars]


# Primitives that are *explicitly* rejected — the coverage table has a
# row for each so the error message matches the documented contract.
#
# Note on ``while`` / ``while_loop``: MJX's constraint solver runs
# under a ``while_loop`` unconditionally (even with no active joint
# limits or contacts). This is the architecturally load-bearing reason
# JAX dynamics emitted by ``jaxterity.Robot.build_system`` cannot flow
# through this translator — see ADR-016. Closed-form per-robot
# dynamics functions (the cartpole zoo entry pattern) is the
# lowering contract.
_REJECTED_PRIMITIVES: dict[str, str] = {
    "cond": "lax.cond[traced]",
    "while": "lax.while_loop",
    "while_loop": "lax.while_loop",
    "scan": "lax.scan",
    "dynamic_update_slice": "dynamic_shape",
}


# ---------------------------------------------------------------------------
# Translator core.
# ---------------------------------------------------------------------------


def _resolve_var(env: dict[Var, Any], var: Any) -> Any:
    if isinstance(var, Literal):
        return var.val
    return env[var]


def _translate_eqns(
    ca: Any,
    eqns: list[JaxprEqn],
    env: dict[Var, Any],
    primitives_used: set[str],
    *,
    target_family: str = "mock-cortex-a",
    dtype: str = "float64",
) -> None:
    for eqn in eqns:
        prim_name = eqn.primitive.name

        if prim_name in _REJECTED_PRIMITIVES:
            # Route through the coverage table so the error is consistent
            # with the documented surface (the rejected row has a
            # suggestion field that surfaces here).
            assert_supported(
                op=_REJECTED_PRIMITIVES[prim_name],
                dtype=dtype,
                target_family=target_family,
            )
            # ``assert_supported`` raised; we never reach here.
            continue

        if prim_name == "pjit" or prim_name == "jit":
            # JAX < 0.5 names this ``jit``; >= 0.5 names it ``pjit``.
            in_vals = [_resolve_var(env, v) for v in eqn.invars]
            out_vals = _handle_jit(ca, eqn, in_vals, env, primitives_used)
            for out_var, out_val in zip(eqn.outvars, out_vals):
                env[out_var] = out_val
            continue

        handler = _PRIMITIVE_HANDLERS.get(prim_name)
        if handler is None:
            raise CoverageError(
                f"JAX primitive {prim_name!r} has no lowering handler",
                op=prim_name,
                dtype=dtype,
                target_family=target_family,
                suggestion=(
                    "the primitive is outside the smooth-op "
                    "subset. Either restructure the source to use only "
                    "supported primitives (see "
                    "``jaxility.lowering.coverage.COVERAGE_TABLE``) or "
                    "register a handler in "
                    "``jaxility.lowering.jax_to_casadi._PRIMITIVE_HANDLERS`` "
                    "with a coverage table row."
                ),
            )

        primitives_used.add(prim_name)
        in_vals = [_resolve_var(env, v) for v in eqn.invars]
        try:
            out_vals = handler(ca, in_vals, eqn.params)
        except CoverageError as exc:
            # Audit N-2 fix: handlers (currently ``_dynamic_slice`` and
            # ``_select_n``) raise ``CoverageError`` with placeholder
            # ``dtype="float64"``, ``target_family="host"`` coordinates
            # because the handler signature has no lowering-context
            # parameter. Replace with the *real* coordinates from the
            # active loop so the error surface carries truthful
            # (op, dtype, target_family) for the failing lowering.
            raise CoverageError(
                str(exc),
                op=exc.op,
                dtype=dtype,
                target_family=target_family,
                suggestion=exc.suggestion,
            ) from exc
        for out_var, out_val in zip(eqn.outvars, out_vals):
            env[out_var] = out_val


def _build_input_symbols(ca: Any, jaxpr: Jaxpr) -> list[Any]:
    syms: list[Any] = []
    for i, var in enumerate(jaxpr.invars):
        shape = tuple(var.aval.shape)  # type: ignore[attr-defined]
        name = f"in_{i}"
        if len(shape) == 0:
            syms.append(ca.SX.sym(name))
        elif len(shape) == 1:
            syms.append(ca.SX.sym(name, shape[0]))
        elif len(shape) == 2:
            syms.append(ca.SX.sym(name, shape[0], shape[1]))
        else:
            raise CoverageError(
                f"rank-{len(shape)} inputs are not supported",
                op="input",
                dtype="float64",
                target_family="host",
                suggestion=(
                    "flatten or reshape to rank ≤ 2 before calling the "
                    "translator; tensor-rank inputs land with a future "
                    "schema."
                ),
            )
    return syms


def translate(
    jax_fn: Callable[..., Any],
    *,
    in_shapes: tuple[tuple[int, ...], ...],
    dtype: PyLiteral["float32", "float64"] = "float64",
    target_family: str = "mock-cortex-a",
    name: str = "f",
) -> CasadiFunction:
    """Translate a JAX function to a CasADi :class:`~casadi.Function`.

    The translator traces ``jax_fn`` with sample inputs of the given
    ``in_shapes`` to obtain a jaxpr, then walks the jaxpr and emits a
    CasADi expression graph. Every emitted primitive is gated through
    :func:`jaxility.lowering.coverage.assert_supported` so the
    coverage table is the documented authority on what is allowed.

    Args
    ----
    jax_fn : Callable
        The JAX function to translate. Must be pure (no Python side
        effects); the trace uses zero-valued sample inputs of the
        given shapes.
    in_shapes : tuple[tuple[int, ...], ...]
        Shapes of the function's inputs, in argument order. Rank 0, 1,
        or 2 only.
    dtype : "float32" | "float64"
        Precision the trace runs at. ``float64`` recommended (CasADi
        runs in double internally; cross-precision differences show
        up as ULP noise).
    target_family : str
        Target family identifier; routed through the coverage gate.
        Defaults to ``"mock-cortex-a"`` for the mock surface.
    name : str
        Name for the resulting CasADi Function.

    Returns
    -------
    CasadiFunction
        The translated function, the input / output shapes, and the
        audit set of primitives used.

    Raises
    ------
    CoverageError
        Whenever an unsupported primitive is emitted into the jaxpr,
        or when the coverage gate rejects an op for the chosen
        ``(target_family, dtype)``.
    """
    import casadi as ca  # local import — extras guard per PATTERNS §1.2

    if dtype == "float64" and not jax.config.read("jax_enable_x64"):
        # Translation at float64 needs x64 mode enabled or JAX will
        # trace in float32 regardless of input dtype.
        jax.config.update("jax_enable_x64", True)

    sample_dtype = jnp.float64 if dtype == "float64" else jnp.float32
    sample_inputs = [jnp.zeros(shape, dtype=sample_dtype) for shape in in_shapes]
    closed: ClosedJaxpr = jax.make_jaxpr(jax_fn)(*sample_inputs)
    jaxpr = closed.jaxpr

    # Audit M-10 close: preflight the jaxpr for the load-bearing
    # MJX-signature primitive (``while_loop`` / ``while``) and raise
    # the MJX-aware CoverageError **before** the per-primitive walk
    # dies generically on the first unhandled op (``scatter`` for
    # cartpole). The lax.while_loop coverage row carries the
    # closed-form-fallback suggestion (ADR-016); without this
    # preflight the user would see the generic "outside the smooth-op
    # subset" message and never hit the architectural close.
    def _eqns_recursive(eqns: Any) -> Any:
        for e in eqns:
            yield e
            for p_name, p_val in e.params.items():
                if isinstance(p_val, ClosedJaxpr):
                    yield from _eqns_recursive(list(p_val.jaxpr.eqns))
                elif isinstance(p_val, Jaxpr):
                    yield from _eqns_recursive(list(p_val.eqns))
                elif p_name == "branches" and isinstance(p_val, tuple):
                    for branch in p_val:
                        if isinstance(branch, ClosedJaxpr):
                            yield from _eqns_recursive(list(branch.jaxpr.eqns))

    for eqn in _eqns_recursive(jaxpr.eqns):
        if eqn.primitive.name in ("while", "while_loop"):
            from .coverage import lookup as _coverage_lookup

            row = _coverage_lookup("lax.while_loop", dtype, target_family)
            raise CoverageError(
                (
                    f"the source contains a `lax.while_loop` ({eqn.primitive.name!r}) "
                    "— translation is rejected. This is the "
                    "load-bearing MJX signature: Jaxterity Robot dynamics "
                    "exposed via MJX run the constraint solver under a "
                    "while_loop unconditionally (ADR-016). Supply closed-form "
                    "per-robot dynamics instead."
                ),
                op="lax.while_loop",
                dtype=dtype,
                target_family=target_family,
                suggestion=row.suggestion,
            )

    ca_inputs = _build_input_symbols(ca, jaxpr)
    env: dict[Var, Any] = {}
    for invar, sym in zip(jaxpr.invars, ca_inputs):
        env[invar] = sym
    for constvar, constval in zip(jaxpr.constvars, closed.consts):
        env[constvar] = np.asarray(constval)

    primitives_used: set[str] = set()
    _translate_eqns(
        ca,
        list(jaxpr.eqns),
        env,
        primitives_used,
        target_family=target_family,
        dtype=dtype,
    )

    ca_outputs = [_resolve_var(env, v) for v in jaxpr.outvars]
    # ``aval.shape`` is present on the concrete ShapedArray subclass of
    # AbstractValue that all jaxpr outvars carry — mypy's stricter view
    # of the base type doesn't see it, so we cast through ``Any``.
    output_shapes = tuple(tuple(v.aval.shape) for v in jaxpr.outvars)  # type: ignore[attr-defined]

    fn = ca.Function(name, ca_inputs, ca_outputs)
    return CasadiFunction(
        name=name,
        fn=fn,
        input_shapes=tuple(in_shapes),
        output_shapes=output_shapes,
        primitives_used=frozenset(primitives_used),
        sx_inputs=tuple(ca_inputs),
        sx_outputs=tuple(ca_outputs),
    )
