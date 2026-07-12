# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""JAX policy → ONNX export, robotics-policy subset (T-040).

The learned-policy counterpart to the JAX → CasADi translator
(:mod:`jaxility.lowering.jax_to_casadi`): it walks the jaxpr emitted by
:func:`jax.make_jaxpr` and dispatches each primitive to a small ONNX handler,
gated through a coverage set so an unsupported op fails loudly with a structured
:class:`~jaxility.errors.CoverageError` (invariant 7) rather than emitting a
silently wrong graph. The output is a self-contained ONNX `ModelProto` that
runs under ONNX Runtime and feeds the downstream LiteRT / ExecuTorch lowering
(T-041) and quantization (T-042).

Scope (T-040): the **smooth MLP** subset that robotics policies use —
fully-connected layers (`dot_general` → `MatMul`), biases (`add`), and smooth
activations (`tanh`, `logistic`/sigmoid, plus `relu` via `max`). Policies are
exported for **single-observation** (unbatched) inference, which is how a
control policy runs on the target — one observation in, one action out. Basic
CNN/RNN ops are a documented next increment; transformers are deferred to v0.2
(see `KNOWN_GAPS.md`). The supported op set is declared in
:data:`SUPPORTED_PRIMITIVES`; everything else raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from ..errors import CoverageError, ToolchainError

try:  # ``onnx`` ships in the optional ``[policy]`` extra; import must not need it
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # pragma: no cover - exercised in the extra-less CI env
    onnx = None  # type: ignore[assignment]
    # ``TensorProto`` is a class, so None-assigning it is [misc], not [assignment].
    TensorProto = helper = numpy_helper = None  # type: ignore[assignment, misc]


def _require_onnx() -> None:
    """Raise a structured :class:`ToolchainError` if the optional ``[policy]``
    extra (``onnx``) is not installed — loud failure at call time, invariant 7
    (mirrors :func:`jaxility.policy.litert._require_converter`)."""
    if onnx is None:
        raise ToolchainError(
            "ONNX export needs the optional 'onnx' dependency; it is not "
            "installed. Install the policy extra: pip install 'jaxility[policy]'."
        )


ONNX_OPSET = 18
"""ONNX opset the exporter targets (well within ONNX Runtime 1.25 support)."""

ONNX_EXPORT_SCHEMA_V0 = 0
"""Schema version of the export metadata payload."""


@dataclass(frozen=True)
class PolicyOnnxModel:
    """An exported policy: the ONNX bytes plus the provenance of the export."""

    model_bytes: bytes
    """Serialised ONNX ``ModelProto`` (``onnx.load_from_string`` round-trips)."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]
    primitives_used: frozenset[str]
    """JAX primitives that participated in the export — the audit trail."""

    onnx_ops_used: frozenset[str]
    """ONNX op types emitted into the graph."""

    def model(self) -> onnx.ModelProto:
        """Deserialise back to a ``ModelProto``."""
        _require_onnx()
        return onnx.load_from_string(self.model_bytes)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class _OnnxBuilder:
    """Accumulates ONNX nodes + initializers while the jaxpr is walked."""

    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.op_types: set[str] = set()
        self._counter = 0

    def fresh(self, stem: str) -> str:
        self._counter += 1
        return f"{stem}_{self._counter}"

    def const(self, array: np.ndarray, *, stem: str = "const") -> str:
        name = self.fresh(stem)
        tensor = numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)
        self.initializers.append(tensor)
        return name

    def add_node(
        self, op_type: str, inputs: list[str], *, stem: str | None = None, **attrs: Any
    ) -> str:
        out = self.fresh(stem or op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        self.op_types.add(op_type)
        return out


# ---------------------------------------------------------------------------
# Per-primitive handlers
# ---------------------------------------------------------------------------

# Signature: ``(b, in_names, params, out_avals) -> out_names``.
PrimitiveHandler = Callable[
    ["_OnnxBuilder", list[str], dict[str, Any], list[Any]], list[str]
]

_HANDLERS: dict[str, PrimitiveHandler] = {}


def _register(*names: str) -> Callable[[PrimitiveHandler], PrimitiveHandler]:
    def deco(fn: PrimitiveHandler) -> PrimitiveHandler:
        for n in names:
            _HANDLERS[n] = fn
        return fn

    return deco


def _unary(op_type: str) -> PrimitiveHandler:
    def handler(
        b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
    ) -> list[str]:
        return [b.add_node(op_type, [xs[0]])]

    return handler


def _binary(op_type: str) -> PrimitiveHandler:
    def handler(
        b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
    ) -> list[str]:
        return [b.add_node(op_type, [xs[0], xs[1]])]

    return handler


_register("add")(_binary("Add"))
_register("sub")(_binary("Sub"))
_register("mul")(_binary("Mul"))
_register("div")(_binary("Div"))
_register("tanh")(_unary("Tanh"))
_register("logistic")(_unary("Sigmoid"))
_register("exp")(_unary("Exp"))
_register("log")(_unary("Log"))
_register("sqrt")(_unary("Sqrt"))


@_register("neg")
def _neg(
    b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
) -> list[str]:
    return [b.add_node("Neg", [xs[0]])]


@_register("max")
def _max(
    b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
) -> list[str]:
    # ``relu`` lowers to ``max(x, 0)``; ONNX Max is variadic.
    return [b.add_node("Max", [xs[0], xs[1]])]


@_register("integer_pow")
def _integer_pow(
    b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
) -> list[str]:
    y = int(params["y"])
    exp_name = b.const(np.array(float(y), dtype=np.float32), stem="pow_exp")
    return [b.add_node("Pow", [xs[0], exp_name])]


@_register("dot_general")
def _dot_general(
    b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
) -> list[str]:
    (lhs_c, rhs_c), (lhs_b, rhs_b) = params["dimension_numbers"]
    # Supported: the dense/MLP contraction — no batch dims, lhs contracts its
    # last axis against rhs's first axis (``x @ W``). ONNX MatMul matches.
    if lhs_b or rhs_b or rhs_c != (0,):
        raise CoverageError(
            f"dot_general with dimension_numbers {params['dimension_numbers']} is "
            f"outside the supported policy subset (dense layers / MatMul).",
            op=f"dot_general[{params['dimension_numbers']}]",
            dtype="float32",
            target_family="policy-onnx",
            suggestion=(
                "use a standard dense contraction (no batch dims, rhs contracting "
                "axis 0 — ``x @ W``); reshape the layer accordingly."
            ),
        )
    return [b.add_node("MatMul", [xs[0], xs[1]])]


@_register("convert_element_type")
def _convert_element_type(
    b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
) -> list[str]:
    # Float-to-float conversions are transparent at the float32 export layer.
    return [xs[0]]


@_register("reshape")
def _reshape(
    b: _OnnxBuilder, xs: list[str], params: dict[str, Any], out_avals: list[Any]
) -> list[str]:
    shape = np.array(params["new_sizes"], dtype=np.int64)
    shape_name = b.fresh("reshape_shape")
    b.initializers.append(numpy_helper.from_array(shape, shape_name))
    return [b.add_node("Reshape", [xs[0], shape_name])]


SUPPORTED_PRIMITIVES: frozenset[str] = frozenset(_HANDLERS) | {
    "pjit",
    "jit",
    "closed_call",
    "custom_jvp_call",
    "custom_vjp_call",
}
"""JAX primitives the exporter accepts. The call-like ones (``pjit``,
``custom_jvp_call``, ...) are transparent boundaries recursed into."""


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def _np_dtype_to_onnx(dtype: Any) -> int:
    if np.dtype(dtype) == np.float64:
        return TensorProto.DOUBLE
    return TensorProto.FLOAT


def _resolve(env: dict[Any, str], b: _OnnxBuilder, var: Any) -> str:
    from jax.extend.core import Literal  # jaxpr literal node

    if isinstance(var, Literal):
        return b.const(np.asarray(var.val, dtype=np.float32), stem="lit")
    return env[var]


def _walk(
    jaxpr: Any,
    consts: list[Any],
    b: _OnnxBuilder,
    env: dict[Any, str],
    primitives: set[str],
) -> None:
    for cvar, cval in zip(jaxpr.constvars, consts):
        env[cvar] = b.const(np.asarray(cval), stem="param")
    for eqn in jaxpr.eqns:
        name = str(eqn.primitive)
        primitives.add(name)
        # Transparent call boundaries — recurse into the sub-jaxpr. ``pjit``
        # wraps jit'd sub-functions; ``custom_jvp_call`` / ``custom_vjp_call``
        # wrap ops with hand-written derivatives (e.g. ``jax.nn.relu`` =
        # ``custom_jvp(max(x, 0))``) — for the forward export only the primal
        # ``call_jaxpr`` matters.
        sub_key = {
            "pjit": "jaxpr",
            "jit": "jaxpr",
            "closed_call": "call_jaxpr",
            "custom_jvp_call": "call_jaxpr",
            "custom_vjp_call": "call_jaxpr",
        }.get(name)
        if sub_key is not None:
            sub = eqn.params[sub_key]
            sub_env = dict(env)
            for outer, inner in zip(eqn.invars, sub.jaxpr.invars):
                sub_env[inner] = _resolve(env, b, outer)
            _walk(sub.jaxpr, list(sub.consts), b, sub_env, primitives)
            for outer, inner in zip(eqn.outvars, sub.jaxpr.outvars):
                env[outer] = sub_env[inner]
            continue
        handler = _HANDLERS.get(name)
        if handler is None:
            raise CoverageError(
                f"JAX primitive {name!r} is not in the policy ONNX-export subset "
                f"(T-040: smooth MLPs).",
                op=name,
                dtype="float32",
                target_family="policy-onnx",
                suggestion=(
                    f"supported primitives: {', '.join(sorted(SUPPORTED_PRIMITIVES))}. "
                    f"Restructure the policy to the smooth-MLP subset, or extend "
                    f"jaxility.policy.onnx_export._HANDLERS with a reviewed handler."
                ),
            )
        in_names = [_resolve(env, b, v) for v in eqn.invars]
        out_names = handler(b, in_names, eqn.params, [v.aval for v in eqn.outvars])
        for var, oname in zip(eqn.outvars, out_names):
            env[var] = oname


def export_policy_to_onnx(
    fn: Callable[..., Any],
    *,
    in_shapes: tuple[tuple[int, ...], ...],
    name: str = "policy",
) -> PolicyOnnxModel:
    """Export a JAX policy ``fn`` to ONNX over the smooth-MLP subset.

    Args
    ----
    fn : callable
        A JAX function mapping the declared inputs to the policy output
        (e.g. ``lambda obs: model.apply(params, obs)``). Built from the
        smooth-MLP subset; unsupported primitives raise ``CoverageError``.
    in_shapes : tuple of shape tuples
        The shapes of ``fn``'s positional inputs (unbatched — a single
        observation per call).
    name : str
        Graph name; also the ONNX input/output base name.

    Returns
    -------
    PolicyOnnxModel
        The serialised ONNX model + the export provenance.

    Raises
    ------
    CoverageError
        If ``fn`` uses a primitive outside :data:`SUPPORTED_PRIMITIVES`.
    ToolchainError
        If the optional ``[policy]`` extra (``onnx``) is not installed.
    """
    _require_onnx()
    sample = [np.zeros(s, dtype=np.float32) for s in in_shapes]
    closed = jax.make_jaxpr(fn)(*sample)
    jaxpr = closed.jaxpr

    b = _OnnxBuilder()
    env: dict[Any, str] = {}
    input_names: list[str] = []
    inputs_vi: list[Any] = []
    for i, invar in enumerate(jaxpr.invars):
        iname = f"{name}_in_{i}"
        env[invar] = iname
        input_names.append(iname)
        inputs_vi.append(
            helper.make_tensor_value_info(
                iname, _np_dtype_to_onnx(invar.aval.dtype), list(invar.aval.shape)
            )
        )

    primitives: set[str] = set()
    _walk(jaxpr, list(closed.consts), b, env, primitives)

    output_names: list[str] = []
    outputs_vi: list[Any] = []
    out_shapes: list[tuple[int, ...]] = []
    for i, outvar in enumerate(jaxpr.outvars):
        src = _resolve(env, b, outvar)
        oname = f"{name}_out_{i}"
        # Identity-tie the graph output name to the producing tensor.
        b.nodes.append(helper.make_node("Identity", [src], [oname]))
        output_names.append(oname)
        outputs_vi.append(
            helper.make_tensor_value_info(
                oname, _np_dtype_to_onnx(outvar.aval.dtype), list(outvar.aval.shape)
            )
        )
        out_shapes.append(tuple(int(d) for d in outvar.aval.shape))

    graph = helper.make_graph(
        b.nodes, name, inputs_vi, outputs_vi, initializer=b.initializers
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)]
    )
    model.ir_version = 9  # ONNX Runtime 1.25 compatible
    onnx.checker.check_model(model)

    return PolicyOnnxModel(
        model_bytes=model.SerializeToString(),
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        in_shapes=tuple(tuple(int(d) for d in s) for s in in_shapes),
        out_shapes=tuple(out_shapes),
        primitives_used=frozenset(primitives),
        onnx_ops_used=frozenset(b.op_types),
    )
