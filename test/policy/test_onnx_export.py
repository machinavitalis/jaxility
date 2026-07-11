# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""JAX policy → ONNX export tests (T-040).

Gated on `onnx` / `onnxruntime` (the export + runtime) and `flax` (a realistic
policy module); all three are dev deps but the tests self-skip if absent so a
minimal install still collects.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from jaxility.errors import CoverageError  # noqa: E402
from jaxility.policy import (  # noqa: E402
    PolicyOnnxModel,
    export_policy_to_onnx,
)


def _mlp_policy(hidden=(32, 32), act_dim=1, activation=None):
    flax = pytest.importorskip("flax")
    nn = flax.linen
    act = activation or nn.tanh

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x):
            for h in hidden:
                x = act(nn.Dense(h)(x))
            return nn.Dense(act_dim)(x)

    return Policy()


def _export_and_session(fn, in_shape, name="policy"):
    exported = export_policy_to_onnx(fn, in_shapes=(in_shape,), name=name)
    sess = ort.InferenceSession(exported.model_bytes)
    return exported, sess


def _assert_parity(fn, exported, sess, in_shape, *, n=25, tol=1e-5):
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n):
        obs = rng.standard_normal(in_shape).astype(np.float32)
        jax_out = np.asarray(fn(jnp.asarray(obs)))
        onnx_out = sess.run(None, {exported.input_names[0]: obs})[0]
        max_err = max(max_err, float(np.abs(jax_out - onnx_out).max()))
    assert max_err < tol, f"JAX vs ONNX max error {max_err:.3e} exceeds {tol:.0e}"
    return max_err


@pytest.mark.unit
def test_mlp_tanh_export_parity() -> None:
    obs_dim = 4
    m = _mlp_policy(hidden=(32, 32), act_dim=1)
    params = m.init(jax.random.PRNGKey(0), jnp.zeros((obs_dim,)))
    fn = lambda obs: m.apply(params, obs)  # noqa: E731

    exported, sess = _export_and_session(fn, (obs_dim,), name="cartpole_policy")
    assert isinstance(exported, PolicyOnnxModel)
    assert exported.primitives_used == {"add", "dot_general", "tanh"}
    assert exported.onnx_ops_used >= {"MatMul", "Add", "Tanh"}
    assert exported.out_shapes == ((1,),)
    # Smooth MLP exports bit-exactly (no smoothing / quantisation yet).
    err = _assert_parity(fn, exported, sess, (obs_dim,))
    assert err == 0.0


@pytest.mark.unit
def test_relu_and_sigmoid_activations_export() -> None:
    flax = pytest.importorskip("flax")
    nn = flax.linen
    obs_dim = 6
    for activation, op in ((nn.relu, "Max"), (nn.sigmoid, "Sigmoid")):
        m = _mlp_policy(hidden=(16,), act_dim=2, activation=activation)
        params = m.init(jax.random.PRNGKey(1), jnp.zeros((obs_dim,)))
        fn = lambda obs, p=params: m.apply(p, obs)  # noqa: E731
        exported, sess = _export_and_session(fn, (obs_dim,))
        assert op in exported.onnx_ops_used
        _assert_parity(fn, exported, sess, (obs_dim,))


@pytest.mark.unit
def test_model_bytes_round_trip() -> None:
    obs_dim = 3
    m = _mlp_policy(hidden=(8,), act_dim=1)
    params = m.init(jax.random.PRNGKey(2), jnp.zeros((obs_dim,)))
    fn = lambda obs: m.apply(params, obs)  # noqa: E731
    exported = export_policy_to_onnx(fn, in_shapes=((obs_dim,),), name="p")
    model = exported.model()  # deserialises + is a valid ModelProto
    onnx.checker.check_model(model)
    assert model.graph.name == "p"


@pytest.mark.unit
def test_unsupported_primitive_raises_coverage_error() -> None:
    # A traced boolean select (jnp.where over a traced predicate) is outside
    # the smooth subset — it must fail loudly, not silently mis-export.
    def policy(obs):
        return jnp.where(obs > 0.0, obs, -obs)  # abs via select_n → unsupported

    with pytest.raises(CoverageError, match="not in the policy ONNX-export subset"):
        export_policy_to_onnx(policy, in_shapes=((4,),), name="bad")


@pytest.mark.unit
def test_unsupported_dot_general_raises() -> None:
    # A batched / contracted-otherwise dot_general is outside the dense subset.
    def policy(x):
        # Valid contraction, but rhs contracts axis 1 (not axis 0) → outside
        # the supported ``x @ W`` dense pattern. x[2,3] · w[5,3] over axis 1.
        w = jnp.ones((5, 3))
        return jax.lax.dot_general(x, w, (((1,), (1,)), ((), ())))

    with pytest.raises(CoverageError, match="dot_general"):
        export_policy_to_onnx(policy, in_shapes=((2, 3),), name="bad_dot")
