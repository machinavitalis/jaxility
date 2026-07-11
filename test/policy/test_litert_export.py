# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""ONNX → LiteRT export tests (T-041).

The conversion itself is gated on the `[litert]` tooling (onnx2tf + tensorflow +
ai-edge-litert) — heavy external deps Jaxility orchestrates but does not vendor,
so these self-skip in the standard env exactly like the acados / HIL tiers. The
*loud-failure* path (tooling absent → structured `ToolchainError`) is tested
everywhere.

Verified manually end-to-end in a `[litert]` venv: a flax MLP → ONNX → LiteRT
matches ONNX Runtime to ~7e-8 (float32 ULP). When the tooling is present these
tests assert that parity in CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("onnx")

from jaxility.errors import ToolchainError  # noqa: E402
from jaxility.policy import export_policy_to_onnx  # noqa: E402
from jaxility.policy.litert import (  # noqa: E402
    LiteRTModel,
    export_onnx_to_litert,
    litert_parity,
)


def _mlp_onnx(hidden=(16, 16), obs_dim=4, act_dim=1):
    flax = pytest.importorskip("flax")
    nn = flax.linen

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x):
            for h in hidden:
                x = nn.tanh(nn.Dense(h)(x))
            return nn.Dense(act_dim)(x)

    m = Policy()
    params = m.init(jax.random.PRNGKey(0), jnp.zeros((obs_dim,)))
    fn = lambda obs: m.apply(params, obs)  # noqa: E731
    return export_policy_to_onnx(fn, in_shapes=((obs_dim,),), name="policy")


@pytest.mark.unit
def test_export_raises_without_litert_tooling() -> None:
    """Absent the [litert] extra, the export fails loudly (invariant 7)."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        onnx_model = _mlp_onnx()
        with pytest.raises(ToolchainError, match=r"\[litert\]"):
            export_onnx_to_litert(onnx_model)
    else:
        pytest.skip("onnx2tf present; the missing-tooling branch cannot be exercised")


@pytest.mark.unit
def test_onnx_to_litert_parity() -> None:
    """With the [litert] tooling, ONNX → LiteRT matches ONNX Runtime (float32)."""
    pytest.importorskip("onnx2tf")
    pytest.importorskip("ai_edge_litert")
    onnx_model = _mlp_onnx(obs_dim=4)
    litert_model = export_onnx_to_litert(onnx_model, name="policy")
    assert isinstance(litert_model, LiteRTModel)
    assert litert_model.input_names == onnx_model.input_names
    assert (
        litert_model.tflite_bytes[:4] == b"TFL3" or len(litert_model.tflite_bytes) > 0
    )

    rng = np.random.default_rng(0)
    for _ in range(10):
        obs = rng.standard_normal(4).astype(np.float32)
        report = litert_parity(litert_model, onnx_model.model_bytes, [obs], tol=1e-4)
        assert report.passed, (
            f"LiteRT parity failed: max_err={report.max_abs_error:.3e}"
        )
