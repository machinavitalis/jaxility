# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""Learned-policy deployment path: JAX checkpoint → ONNX → LiteRT / ExecuTorch.

Quantisation recipes and parity tests live here. The runtime composition
with the acados control layer is in :mod:`jaxility.compose`.

Public surface (T-040):

* :func:`export_policy_to_onnx` / :class:`PolicyOnnxModel` — export a JAX
  policy (smooth-MLP subset) to a self-contained ONNX model.
* :data:`SUPPORTED_PRIMITIVES` — the JAX primitives the exporter accepts;
  anything else raises a structured ``CoverageError`` (invariant 7).
"""

from __future__ import annotations

from .executorch import ExecuTorchModel, export_onnx_to_executorch
from .litert import (
    LiteRTModel,
    LiteRTParityReport,
    export_onnx_to_litert,
    litert_parity,
)
from .onnx_export import (
    ONNX_OPSET,
    SUPPORTED_PRIMITIVES,
    PolicyOnnxModel,
    export_policy_to_onnx,
)
from .quantize import (
    QUANT_TOLERANCE,
    QuantizedLiteRTModel,
    QuantRecipe,
    quantization_parity,
    quantize_onnx_to_litert,
)

__all__ = [
    "ONNX_OPSET",
    "QUANT_TOLERANCE",
    "SUPPORTED_PRIMITIVES",
    "ExecuTorchModel",
    "LiteRTModel",
    "LiteRTParityReport",
    "PolicyOnnxModel",
    "QuantRecipe",
    "QuantizedLiteRTModel",
    "export_onnx_to_executorch",
    "export_onnx_to_litert",
    "export_policy_to_onnx",
    "litert_parity",
    "quantization_parity",
    "quantize_onnx_to_litert",
]
