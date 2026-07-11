# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Quantization-recipe tests (T-042).

Argument validation (unknown recipe, static_int8 missing calibration data) is
tested everywhere; the actual quantization + parity is gated on the `[litert]`
tooling (onnx2tf + ai-edge-litert) and self-skips without it. Verified in a
`[litert]` env: float16 ≈ 3.4e-4 and dynamic_int8 ≈ 7.6e-3 vs the float32 policy,
both inside their documented budgets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("onnx")

from jaxility.errors import ToolchainError  # noqa: E402
from jaxility.policy import export_policy_to_onnx  # noqa: E402
from jaxility.policy.quantize import (  # noqa: E402
    QUANT_TOLERANCE,
    quantization_parity,
    quantize_onnx_to_litert,
)


def _mlp_onnx(obs_dim=4):
    flax = pytest.importorskip("flax")
    nn = flax.linen

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(nn.tanh(nn.Dense(16)(x)))

    m = Policy()
    params = m.init(jax.random.PRNGKey(0), jnp.zeros((obs_dim,)))
    fn = lambda obs: m.apply(params, obs)  # noqa: E731
    return export_policy_to_onnx(fn, in_shapes=((obs_dim,),), name="policy")


@pytest.mark.unit
def test_tolerance_table_covers_recipes() -> None:
    assert set(QUANT_TOLERANCE) == {"float16", "dynamic_int8", "static_int8"}
    # Budgets are ordered: float16 tightest, static_int8 loosest.
    assert (
        QUANT_TOLERANCE["float16"]
        < QUANT_TOLERANCE["dynamic_int8"]
        < QUANT_TOLERANCE["static_int8"]
    )


@pytest.mark.unit
def test_unknown_recipe_raises() -> None:
    with pytest.raises(ToolchainError, match="unknown quantization recipe"):
        quantize_onnx_to_litert(_mlp_onnx(), "int4_magic")  # type: ignore[arg-type]


@pytest.mark.unit
def test_static_int8_requires_representative_data() -> None:
    with pytest.raises(ToolchainError, match="representative_data"):
        quantize_onnx_to_litert(_mlp_onnx(), "static_int8")


@pytest.mark.unit
def test_quantize_parity_within_budget() -> None:
    pytest.importorskip("onnx2tf")
    pytest.importorskip("ai_edge_litert")
    onnx_model = _mlp_onnx(obs_dim=4)
    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal(4).astype(np.float32) for _ in range(8)]
    for recipe in ("float16", "dynamic_int8"):
        q = quantize_onnx_to_litert(onnx_model, recipe)  # type: ignore[arg-type]
        assert q.recipe == recipe
        for obs in inputs:
            report = quantization_parity(q, onnx_model.model_bytes, [obs])
            assert report.passed, (
                f"{recipe} degraded beyond budget: "
                f"{report.max_abs_error:.3e} > {report.tol:.0e}"
            )
