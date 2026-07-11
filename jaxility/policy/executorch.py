# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""ONNX → ExecuTorch lowering for the learned-policy path (T-041, parallel).

The parallel on-device inference target to LiteRT (CONTEXT §"Learned-policy
deployment path" — "dual path: ... LiteRT/ExecuTorch ... whichever has the
cleanest embedded story per target"). The lowering path is ONNX → torch
(``onnx2torch``) → ExecuTorch (`torch.export` + `to_edge`). As with LiteRT,
Jaxility **orchestrates** the external converter rather than vendoring it; the
toolchain (`torch`, `executorch`, `onnx2torch`) ships behind the
``[executorch]`` extra and the export raises a structured
:class:`~jaxility.errors.ToolchainError` if it is absent (invariant 7).

LiteRT is the priority path and is parity-verified; this ExecuTorch path shares
the same export interface so a caller can target either runtime. Its parity
verification runs where the (heavier) ExecuTorch toolchain is installed — the
same gated-tier posture as the acados cross-build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ToolchainError
from .onnx_export import PolicyOnnxModel

EXECUTORCH_EXPORT_SCHEMA_V0 = 0
"""Schema version of the ExecuTorch export metadata."""


@dataclass(frozen=True)
class ExecuTorchModel:
    """An exported ExecuTorch (.pte) policy plus the provenance of the export."""

    pte_bytes: bytes
    """The serialised ExecuTorch program (``.pte``)."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]
    source_onnx_ops: frozenset[str]


def _require_toolchain() -> tuple[Any, Any, Any]:
    """Import torch + onnx2torch + executorch, or raise a structured error."""
    try:
        import onnx2torch  # noqa: F401
        import torch  # noqa: F401
        from executorch.exir import to_edge  # noqa: F401
    except ImportError as exc:
        raise ToolchainError(
            "the ONNX → ExecuTorch toolchain (torch + onnx2torch + executorch) is "
            "not installed. Install `pip install 'jaxility[executorch]'`. "
            "Jaxility orchestrates the converter (it does not vendor it); the "
            "export is gated on the extra (the acados pattern). LiteRT is the "
            "priority path and is the verified default — see jaxility.policy.litert."
        ) from exc
    import onnx2torch
    import torch
    from executorch.exir import to_edge

    return torch, onnx2torch, to_edge


def export_onnx_to_executorch(
    policy_onnx: PolicyOnnxModel,
    *,
    name: str = "policy",
) -> ExecuTorchModel:
    """Convert a :class:`PolicyOnnxModel` to an ExecuTorch program (.pte).

    Path: ONNX → torch (``onnx2torch``) → ``torch.export`` → ``to_edge`` →
    ``to_executorch``. Raises :class:`ToolchainError` if the ``[executorch]``
    tooling is absent.
    """
    import io

    import numpy as np
    import onnx

    torch, onnx2torch, to_edge = _require_toolchain()

    model = onnx.load_from_string(policy_onnx.model_bytes)
    torch_model = onnx2torch.convert(model).eval()
    sample = tuple(
        torch.zeros(tuple(s), dtype=torch.float32) for s in policy_onnx.in_shapes
    )
    exported = torch.export.export(torch_model, sample)
    edge = to_edge(exported)
    program = edge.to_executorch()
    buf = io.BytesIO()
    program.write_to_file(buf)  # type: ignore[attr-defined]
    pte_bytes = buf.getvalue()

    # Output shape inference: run the torch model once on the sample.
    with torch.no_grad():
        out = torch_model(*sample)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    out_shapes = tuple(tuple(int(d) for d in np.asarray(o).shape) for o in outs)

    return ExecuTorchModel(
        pte_bytes=pte_bytes,
        input_names=policy_onnx.input_names,
        output_names=policy_onnx.output_names,
        in_shapes=policy_onnx.in_shapes,
        out_shapes=out_shapes,
        source_onnx_ops=policy_onnx.onnx_ops_used,
    )
