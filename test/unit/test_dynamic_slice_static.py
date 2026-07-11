# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for ``dynamic_slice[static]`` support (A1 / ADR-016)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxility.errors import CoverageError
from jaxility.lowering import translate
from jaxility.lowering.coverage import lookup

# ---------------------------------------------------------------------------
# Coverage-table additions.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dynamic_slice_static_is_supported_in_coverage() -> None:
    entry = lookup("dynamic_slice[static]", "float64", "mock-cortex-a")
    assert entry.supported is True


@pytest.mark.unit
def test_dynamic_slice_traced_is_documented_as_unsupported() -> None:
    entry = lookup("dynamic_slice[traced]", "float64", "mock-cortex-a")
    assert entry.supported is False
    assert "literal" in entry.suggestion.lower()


@pytest.mark.unit
def test_lax_while_loop_suggestion_mentions_mjx() -> None:
    """ADR-016: the while_loop rejection names MJX explicitly."""
    entry = lookup("lax.while_loop", "float64", "mock-cortex-a")
    assert entry.supported is False
    assert "mjx" in entry.suggestion.lower() or "MJX" in entry.suggestion


@pytest.mark.unit
def test_dynamic_slice_handler_rejects_rank_3_directly() -> None:
    """Audit N-1: the rank-3+ branch in ``_dynamic_slice`` is a
    defensive guard inside the handler.

    Reaching it from a normal :func:`translate` call is impossible
    today because :func:`_build_input_symbols` rejects rank-3+
    inputs at the input-allocation gate (CasADi's SX/MX symbolic
    types are 2-D, so rank-3+ arrays don't map cleanly to them
    end-to-end). The defensive guard exists in case a future
    callsite invokes the handler directly with a rank-3+ start
    tuple — this test pins the contract.
    """
    import casadi as ca  # noqa: PLC0415

    from jaxility.lowering.jax_to_casadi import _PRIMITIVE_HANDLERS

    handler = _PRIMITIVE_HANDLERS["dynamic_slice"]
    # Build a synthetic invocation: rank-3 operand + 3 literal starts.
    operand = ca.SX.sym("x", 2, 2)  # CasADi SX is 2-D; the operand
    # shape doesn't reach the rank-3 branch — only ``len(starts)``
    # does, since the dispatcher computes rank from the start tuple.
    starts = [np.int32(0), np.int32(0), np.int32(0)]
    params = {"slice_sizes": (1, 1, 1)}

    with pytest.raises(CoverageError) as exc_info:
        handler(ca, [operand, *starts], params)
    err = exc_info.value
    assert "dynamic_slice" in str(err) or err.op.startswith("dynamic_slice")
    # The error must name the rejected rank so the user sees WHY.
    assert "rank-3" in str(err) or "rank-3" in err.suggestion


@pytest.mark.unit
def test_translate_rejects_rank_3_jax_inputs_at_input_gate() -> None:
    """Companion to the test above. From a normal ``translate`` call,
    rank-3+ jax inputs hit the input-allocation gate before any
    handler runs; the error names that constraint clearly."""

    def f(x):
        return jax.lax.dynamic_slice(x, (0, 0, 0), (1, 1, 1))

    with pytest.raises(CoverageError) as exc_info:
        translate(f, in_shapes=((2, 2, 2),), name="rank3_dyn_slice_input")
    assert "rank" in str(exc_info.value).lower()


@pytest.mark.unit
def test_handler_coverage_error_carries_real_lowering_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit N-2: when a handler raises ``CoverageError`` with
    placeholder coordinates, the dispatcher must rewrap it with the
    real ``(dtype, target_family)`` from the active ``translate``
    call. Tests against ``mock-cortex-m`` + ``float32`` to prove
    both coordinates are actually threaded, not hardcoded.

    The simplest construction that reaches a handler-raised
    CoverageError without going through intermediate primitives the
    translator dies on first: monkeypatch ``_PRIMITIVE_HANDLERS["add"]``
    to raise the same placeholder-shaped CoverageError that
    ``_dynamic_slice`` does, run a trivial ``add`` translation, and
    assert the dispatcher rewrap stamped the right coordinates.
    """
    from jaxility.lowering import jax_to_casadi

    def fake_add_handler(ca, xs, params):
        raise CoverageError(
            "synthetic handler error (audit N-2 test fixture)",
            op="add[synthetic]",
            dtype="float64",  # placeholder; dispatcher should replace
            target_family="host",  # placeholder; dispatcher should replace
            suggestion="this is the handler's suggestion; must survive rewrap",
        )

    original = jax_to_casadi._PRIMITIVE_HANDLERS["add"]
    monkeypatch.setitem(jax_to_casadi._PRIMITIVE_HANDLERS, "add", fake_add_handler)
    try:

        def f(x):
            return x + 1.0

        with pytest.raises(CoverageError) as exc_info:
            translate(
                f,
                in_shapes=((2,),),
                name="n2_dispatcher_rewrap",
                dtype="float32",
                target_family="mock-cortex-m",
            )
    finally:
        # Defensive: monkeypatch will undo via fixture teardown, but
        # belt-and-braces against future test isolation regressions.
        jax_to_casadi._PRIMITIVE_HANDLERS["add"] = original

    err = exc_info.value
    # Before N-2, these would have been "float64" / "host" placeholders.
    assert err.dtype == "float32", (
        f"CoverageError carries dtype {err.dtype!r}; dispatcher should "
        "have rewrapped with the active translate dtype."
    )
    assert err.target_family == "mock-cortex-m", (
        f"CoverageError carries target_family {err.target_family!r}; "
        "dispatcher should have rewrapped with the active target_family."
    )
    # The op + suggestion + message from the handler must survive
    # the rewrap (only the coordinates change).
    assert err.op == "add[synthetic]"
    assert "this is the handler's suggestion" in err.suggestion
    assert "audit N-2 test fixture" in str(err)


# ---------------------------------------------------------------------------
# Translator: dynamic_slice[static] flows through.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dynamic_slice_with_literal_start_translates() -> None:
    def f(x):
        return jax.lax.dynamic_slice(x, (1,), (2,))

    cf = translate(f, in_shapes=((4,),), name="dyn_slice_static")
    result = np.asarray(cf.fn(jnp.array([10.0, 20.0, 30.0, 40.0])))
    np.testing.assert_allclose(result.flatten(), [20.0, 30.0])


@pytest.mark.unit
def test_dynamic_slice_2d_with_literal_starts_translates() -> None:
    def f(x):
        return jax.lax.dynamic_slice(x, (0, 1), (2, 1))

    cf = translate(f, in_shapes=((2, 3),), name="dyn_slice_2d")
    # CasADi's Function.call takes DM / numpy arrays, not JAX arrays.
    result = np.asarray(cf.fn(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])))
    np.testing.assert_allclose(result.flatten(), [2.0, 5.0])


# ---------------------------------------------------------------------------
# MJX integration check: ADR-016 — translation of MJX-driven Robot.ode
# fails with CoverageError. The *first* unhandled primitive determines the
# exact text; what matters at the test level is that it fails, not which
# primitive trips first. The COVERAGE_TABLE row for ``lax.while_loop``
# carries the MJX-specific suggestion (asserted above).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mjx_robot_ode_rejected_by_translator() -> None:
    """ADR-016 — MJX-driven Robot.ode is not translatable.

    The cartpole MJX ode emits 34 distinct primitives across 2,375 jaxpr
    equations, including ``while_loop`` (the constraint solver) plus
    several primitives that have no lowering handler. Translation raises
    CoverageError on the first unhandled one; the ``lax.while_loop``
    row of COVERAGE_TABLE documents the architectural reason. See
    ADR-016.
    """
    pytest.importorskip("jaxterity")
    from jaxterity.zoo import load

    robot = load("cartpole")
    sys = robot.build_system(q0=[0.0, 0.0], qd0=[0.0, 0.0], actuation=True)
    ctx = sys.create_context()
    params = dict(ctx.parameters)

    def mjx_ode(state, control):
        c = ctx.with_continuous_state(state)
        return sys.ode(0.0, c, control, **params)

    # Audit M-10 close: the test must verify the architectural
    # close is *visible to the user*, not merely that *some*
    # CoverageError fires. The preflight in jax_to_casadi.translate
    # catches ``while_loop`` before the generic per-primitive walk
    # dies on ``scatter``.
    with pytest.raises(CoverageError) as exc_info:
        translate(mjx_ode, in_shapes=((4,), (1,)), name="mjx_cartpole")
    msg = str(exc_info.value)
    suggestion = exc_info.value.suggestion
    assert "while_loop" in msg or "MJX" in msg, msg
    # The lax.while_loop coverage row's suggestion names MJX + the
    # closed-form fallback.
    assert "MJX" in suggestion or "closed-form" in suggestion, suggestion
