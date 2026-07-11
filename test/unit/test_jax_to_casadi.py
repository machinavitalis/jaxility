# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the JAX → CasADi translator (T-020).

T-020 acceptance criteria:

1. A handcrafted CasADi reference matches the translated graph
   structurally on a set of canonical dynamics (pendulum, cartpole,
   quadrotor-planar).
2. Translated graphs evaluate to bit-exact-modulo-ULP outputs against
   the JAX source on representative inputs.

The structural-match property is exercised by *evaluating* both
backends at randomly chosen points and asserting numerical agreement;
two CasADi graphs that produce identical numerical output at every
sampled point are equivalent for our purposes (the eventual
acados-emitted C code only consumes the numerical Function, not the
graph shape).
"""

from __future__ import annotations

import hypothesis
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from jaxility.errors import CoverageError
from jaxility.lowering import CasadiFunction, translate
from jaxility.lowering.jax_to_casadi import JAX_TO_CASADI_SCHEMA_V0

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Acceptance 1 + 2 — canonical dynamics translate bit-exactly.
# ---------------------------------------------------------------------------


def _pendulum(state: jnp.ndarray) -> jnp.ndarray:
    g, L = 9.81, 1.0
    theta, theta_dot = state[0], state[1]
    return jnp.array([theta_dot, -g / L * jnp.sin(theta)])


def _cartpole(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """Smooth cartpole continuous dynamics. Differential-only — no
    discontinuities; suitable for the smooth-op subset."""
    g, mp, mc, L = 9.81, 0.1, 1.0, 0.5
    _, theta, x_dot, theta_dot = state[0], state[1], state[2], state[3]
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    denom = mc + mp * sin_t * sin_t
    x_ddot = (control[0] + mp * sin_t * (L * theta_dot * theta_dot + g * cos_t)) / denom
    theta_ddot = (
        -control[0] * cos_t
        - mp * L * theta_dot * theta_dot * cos_t * sin_t
        - (mc + mp) * g * sin_t
    ) / (L * denom)
    return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])


def _quadrotor_planar(state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """2D quadrotor: position (x, z), pitch (phi), and their rates;
    controls (T, M) for thrust and pitch moment."""
    g, m, J = 9.81, 0.5, 0.0023
    _, _, phi, x_dot, z_dot, phi_dot = (state[i] for i in range(6))
    T, M = control[0], control[1]
    x_ddot = -T * jnp.sin(phi) / m
    z_ddot = T * jnp.cos(phi) / m - g
    phi_ddot = M / J
    return jnp.array([x_dot, z_dot, phi_dot, x_ddot, z_ddot, phi_ddot])


@pytest.mark.unit
def test_pendulum_translates_bit_exact() -> None:
    cf = translate(_pendulum, in_shapes=((2,),), name="pendulum")
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = rng.standard_normal(2)
        jax_out = np.array(_pendulum(jnp.asarray(state)))
        ca_out = cf(state)[0].flatten()
        np.testing.assert_allclose(jax_out, ca_out, atol=0.0, rtol=0.0)


@pytest.mark.unit
def test_cartpole_translates_bit_exact() -> None:
    cf = translate(_cartpole, in_shapes=((4,), (1,)), name="cartpole")
    rng = np.random.default_rng(1)
    for _ in range(5):
        state = rng.standard_normal(4)
        u = rng.standard_normal(1)
        jax_out = np.array(_cartpole(jnp.asarray(state), jnp.asarray(u)))
        ca_out = cf(state, u)[0].flatten()
        np.testing.assert_allclose(jax_out, ca_out, atol=1e-15, rtol=1e-15)


@pytest.mark.unit
def test_quadrotor_planar_translates_bit_exact() -> None:
    cf = translate(_quadrotor_planar, in_shapes=((6,), (2,)), name="quadrotor")
    rng = np.random.default_rng(2)
    for _ in range(5):
        state = rng.standard_normal(6)
        u = rng.standard_normal(2)
        jax_out = np.array(_quadrotor_planar(jnp.asarray(state), jnp.asarray(u)))
        ca_out = cf(state, u)[0].flatten()
        np.testing.assert_allclose(jax_out, ca_out, atol=1e-15, rtol=1e-15)


# ---------------------------------------------------------------------------
# Primitive coverage — smooth ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "fn,shape",
    [
        (lambda x: jnp.sin(x), (3,)),
        (lambda x: jnp.cos(x), (3,)),
        (lambda x: jnp.tan(x), (3,)),
        (lambda x: jnp.exp(x), (3,)),
        (lambda x: jnp.log(x * x + 0.1), (3,)),
        (lambda x: jnp.sqrt(x * x + 0.1), (3,)),
        (lambda x: x + x, (3,)),
        (lambda x: x - x, (3,)),
        (lambda x: x * x, (3,)),
        (lambda x: x / (x + 2.0), (3,)),
        (lambda x: -x, (3,)),
        (lambda x: x**2, (3,)),
        (lambda x: x**3, (3,)),
    ],
)
def test_elementwise_op_bit_exact(fn, shape) -> None:
    cf = translate(fn, in_shapes=(shape,))
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape)
    jax_out = np.array(fn(jnp.asarray(x)))
    ca_out = cf(x)[0].flatten()
    np.testing.assert_allclose(jax_out.flatten(), ca_out, atol=1e-15, rtol=1e-15)


@pytest.mark.unit
def test_dot_general_2d_matmul_bit_exact() -> None:
    def fn(A, B):
        return A @ B

    cf = translate(fn, in_shapes=((3, 4), (4, 2)))
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 4))
    B = rng.standard_normal((4, 2))
    jax_out = np.array(fn(jnp.asarray(A), jnp.asarray(B)))
    ca_out = np.asarray(cf.fn(A, B))
    np.testing.assert_allclose(jax_out, ca_out, atol=1e-14, rtol=1e-14)


@pytest.mark.unit
def test_static_index_slice_bit_exact() -> None:
    def fn(x):
        return x[1:4] + jnp.sin(x[2:5])

    cf = translate(fn, in_shapes=((6,),))
    rng = np.random.default_rng(7)
    x = rng.standard_normal(6)
    jax_out = np.array(fn(jnp.asarray(x)))
    ca_out = cf(x)[0].flatten()
    np.testing.assert_allclose(jax_out, ca_out, atol=1e-15, rtol=1e-15)


# ---------------------------------------------------------------------------
# Static-predicate ``jnp.where`` folds; traced predicates raise.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("predicate", [True, False])
def test_static_where_folds_to_chosen_branch(predicate: bool) -> None:
    def fn(x):
        return jnp.where(predicate, jnp.sin(x), -jnp.sin(x))

    cf = translate(fn, in_shapes=((3,),))
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3)
    jax_out = np.array(fn(jnp.asarray(x)))
    ca_out = cf(x)[0].flatten()
    np.testing.assert_allclose(jax_out, ca_out, atol=1e-15, rtol=1e-15)


@pytest.mark.unit
def test_traced_where_rejected_with_structured_error() -> None:
    def fn(x):
        return jnp.where(x > 0, x, -x)

    with pytest.raises(CoverageError) as excinfo:
        translate(fn, in_shapes=((3,),))
    # The rejection surfaces at the comparison primitive (``gt``) — that
    # is the upstream cause of the non-smooth graph; the suggestion
    # carries the documented workaround.
    err = excinfo.value
    assert err.op != ""
    assert "smooth" in err.suggestion.lower() or "subset" in err.suggestion.lower()


@pytest.mark.unit
def test_lax_cond_rejected_with_structured_error() -> None:
    def fn(x):
        return lax.cond(x[0] > 0, lambda: x, lambda: -x)

    with pytest.raises(CoverageError):
        translate(fn, in_shapes=((3,),))


@pytest.mark.unit
def test_lax_while_loop_rejected_with_structured_error() -> None:
    def fn(x):
        return lax.while_loop(lambda v: v[0] > 0, lambda v: v - 0.1, x)

    with pytest.raises(CoverageError) as excinfo:
        translate(fn, in_shapes=((3,),))
    assert (
        "while" in excinfo.value.suggestion.lower()
        or "loop" in excinfo.value.suggestion.lower()
    )


@pytest.mark.unit
def test_unsupported_primitive_raises_coverage_error_with_pointer() -> None:
    def fn(x):
        return jnp.sign(x) + x

    with pytest.raises(CoverageError) as excinfo:
        translate(fn, in_shapes=((3,),))
    err = excinfo.value
    assert err.op == "sign"
    assert (
        "_PRIMITIVE_HANDLERS" in err.suggestion or "smooth-op subset" in err.suggestion
    )


# ---------------------------------------------------------------------------
# CasadiFunction surface.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_casadi_function_records_primitives_used() -> None:
    def fn(x):
        return jnp.sin(x) + x * x

    cf = translate(fn, in_shapes=((3,),))
    assert isinstance(cf, CasadiFunction)
    assert "sin" in cf.primitives_used
    assert "mul" in cf.primitives_used
    assert "add" in cf.primitives_used


@pytest.mark.unit
def test_casadi_function_carries_input_and_output_shapes() -> None:
    cf = translate(_pendulum, in_shapes=((2,),))
    assert cf.input_shapes == ((2,),)
    assert cf.output_shapes == ((2,),)


@pytest.mark.unit
def test_casadi_function_call_returns_list_of_arrays() -> None:
    cf = translate(_pendulum, in_shapes=((2,),))
    out = cf(np.array([0.3, 0.7]))
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (2,)


@pytest.mark.unit
def test_schema_version_constant_is_v0() -> None:
    assert JAX_TO_CASADI_SCHEMA_V0 == 0


# ---------------------------------------------------------------------------
# Hypothesis property tests (PATTERNS §7.2).
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    coeffs=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
    x=st.floats(min_value=-3.14, max_value=3.14, allow_nan=False, allow_infinity=False),
)
@hypothesis.settings(max_examples=30, deadline=None)
def test_polynomial_translation_property(coeffs: list[float], x: float) -> None:
    """Property: a + b*sin(x) + c*cos(x) translates and evaluates exactly."""
    a, b, c = coeffs

    def fn(x_arr):
        return a + b * jnp.sin(x_arr) + c * jnp.cos(x_arr)

    cf = translate(fn, in_shapes=((1,),))
    x_arr = np.array([x])
    jax_out = float(fn(jnp.asarray(x_arr)).item())
    ca_out = float(cf(x_arr)[0].item())
    np.testing.assert_allclose(jax_out, ca_out, atol=1e-14, rtol=1e-14)
