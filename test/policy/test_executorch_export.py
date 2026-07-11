# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""ONNX → ExecuTorch export tests (T-041, parallel path).

The conversion is gated on the `[executorch]` tooling (torch + onnx2torch +
executorch); the loud-failure path (tooling absent → `ToolchainError`) is tested
everywhere. The actual conversion + parity run where the (heavy) tooling is
installed, the same gated-tier posture as the LiteRT path and the acados
cross-build. LiteRT is the verified priority path (`test_litert_export.py`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("onnx")

from jaxility.errors import ToolchainError  # noqa: E402
from jaxility.policy import export_policy_to_onnx  # noqa: E402
from jaxility.policy.executorch import export_onnx_to_executorch  # noqa: E402


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
def test_executorch_export_raises_without_tooling() -> None:
    """Absent the [executorch] extra, the export fails loudly (invariant 7)."""
    try:
        import executorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ToolchainError, match=r"\[executorch\]"):
            export_onnx_to_executorch(_mlp_onnx())
    else:
        pytest.skip("executorch tooling present; missing-tooling branch unreachable")
