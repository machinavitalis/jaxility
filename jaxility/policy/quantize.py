# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Quantization recipes for the LiteRT policy path (T-042).

Post-training quantization of an exported policy, driven through the same
`onnx2tf` converter as the float32 LiteRT path (T-041) — Jaxility orchestrates
it, gated behind the ``[litert]`` extra. Three recipes:

* ``"float16"`` — half-precision weights + activations. Near-lossless; the
  cheapest accuracy/size trade.
* ``"dynamic_int8"`` — int8 **per-channel** weights, float activations
  (dynamic-range quantization). No calibration data needed.
* ``"static_int8"`` — full-integer (int8 weights *and* activations) using a
  representative dataset for activation ranges. The embedded-default; smallest
  + fastest, largest accuracy cost.

Every recipe carries a **documented degradation budget** (:data:`QUANT_TOLERANCE`)
measured against the float32 policy, and :func:`quantization_parity` checks the
quantized model against it (mirrors the source-to-artifact equivalence
discipline — a quantization that degrades beyond its budget fails loudly).
Measured on smooth MLPs: float16 ≈ 3e-4, dynamic_int8 ≈ 8e-3 vs the float32
ONNX.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..errors import ToolchainError
from .litert import LiteRTModel, litert_parity
from .onnx_export import PolicyOnnxModel

QuantRecipe = Literal["float16", "dynamic_int8", "static_int8"]

QUANT_TOLERANCE: dict[str, float] = {
    # Per-step max-abs degradation budget vs the float32 policy, with margin
    # over the measured values (smooth MLP). Documented in CLAIMS / KNOWN_GAPS.
    "float16": 1e-3,  # measured ~3.4e-4
    "dynamic_int8": 3e-2,  # measured ~7.6e-3 (int8 per-channel weights)
    "static_int8": 1.5e-1,  # full integer; range/representative-data dependent
}
"""Documented accuracy-degradation budget per recipe (vs float32)."""

# onnx2tf output filename suffix per recipe.
_RECIPE_SUFFIX: dict[str, tuple[str, ...]] = {
    "float16": ("_float16.tflite",),
    "dynamic_int8": ("_dynamic_range_quant.tflite",),
    "static_int8": ("_integer_quant.tflite", "_full_integer_quant.tflite"),
}


@dataclass(frozen=True)
class QuantizedLiteRTModel:
    """A quantized LiteRT (.tflite) policy + the recipe + provenance."""

    tflite_bytes: bytes
    recipe: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]
    source_onnx_ops: frozenset[str]

    def as_litert_model(self) -> LiteRTModel:
        """View as a :class:`LiteRTModel` (for running / parity-checking)."""
        return LiteRTModel(
            tflite_bytes=self.tflite_bytes,
            input_names=self.input_names,
            output_names=self.output_names,
            in_shapes=self.in_shapes,
            out_shapes=self.out_shapes,
            source_onnx_ops=self.source_onnx_ops,
        )


def quantize_onnx_to_litert(
    policy_onnx: PolicyOnnxModel,
    recipe: QuantRecipe,
    *,
    representative_data: np.ndarray | None = None,
    work_dir: Path | None = None,
    name: str = "policy",
) -> QuantizedLiteRTModel:
    """Quantize a :class:`PolicyOnnxModel` to LiteRT under ``recipe``.

    ``static_int8`` requires ``representative_data`` (shape ``(N, *in_shape)``)
    to calibrate activation ranges. Raises :class:`ToolchainError` if the
    ``[litert]`` tooling is absent or the converter produces no model for the
    recipe (loud failure, invariant 7).
    """
    # Argument validation first (testable without the heavy tooling).
    if recipe not in _RECIPE_SUFFIX:
        raise ToolchainError(
            f"unknown quantization recipe {recipe!r}; supported: "
            f"{', '.join(sorted(_RECIPE_SUFFIX))}."
        )
    if recipe == "static_int8" and representative_data is None:
        raise ToolchainError(
            "static_int8 (full-integer) quantization needs `representative_data` "
            "(shape (N, *input_shape)) to calibrate activation ranges. Supply a "
            "representative batch of observations, or use 'dynamic_int8' (no "
            "calibration) / 'float16'."
        )
    try:
        import onnx2tf
    except ImportError as exc:
        raise ToolchainError(
            "the ONNX → LiteRT converter (onnx2tf + tensorflow) is not installed. "
            "Install `pip install 'jaxility[litert]'`."
        ) from exc

    ctx: tempfile.TemporaryDirectory[str] | None
    if work_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix="jaxility-quant-")
        base = Path(ctx.name)
    else:
        ctx = None
        base = Path(work_dir)
    try:
        base.mkdir(parents=True, exist_ok=True)
        onnx_path = base / f"{name}.onnx"
        onnx_path.write_bytes(policy_onnx.model_bytes)
        out_dir = base / "tflite"

        kwargs: dict[str, object] = {
            "input_onnx_file_path": str(onnx_path),
            "output_folder_path": str(out_dir),
            "non_verbose": True,
        }
        if recipe == "dynamic_int8":
            kwargs["output_dynamic_range_quantized_tflite"] = True
        elif recipe == "static_int8":
            kwargs["output_integer_quantized_tflite"] = True
            calib = base / "calib.npy"
            np.save(calib, np.asarray(representative_data, dtype=np.float32))
            kwargs["custom_input_op_name_np_data_path"] = [
                [policy_onnx.input_names[0], str(calib), 0.0, 1.0]
            ]
        onnx2tf.convert(**kwargs)

        tflite_path: Path | None = None
        for suffix in _RECIPE_SUFFIX[recipe]:
            hits = sorted(out_dir.glob(f"*{suffix}"))
            if hits:
                tflite_path = hits[0]
                break
        if tflite_path is None:
            raise ToolchainError(
                f"the converter produced no {recipe!r} .tflite under {out_dir}. "
                f"For static_int8 the model may not be fully quantizable or the "
                f"representative dataset is insufficient; try dynamic_int8."
            )
        tflite_bytes = tflite_path.read_bytes()
    finally:
        if ctx is not None:
            ctx.cleanup()

    return QuantizedLiteRTModel(
        tflite_bytes=tflite_bytes,
        recipe=recipe,
        input_names=policy_onnx.input_names,
        output_names=policy_onnx.output_names,
        in_shapes=policy_onnx.in_shapes,
        out_shapes=policy_onnx.out_shapes,
        source_onnx_ops=policy_onnx.onnx_ops_used,
    )


def quantization_parity(
    quantized: QuantizedLiteRTModel,
    onnx_model_bytes: bytes,
    inputs: list[np.ndarray],
    *,
    tol: float | None = None,
):
    """Check a quantized model against the float32 policy under its budget.

    ``tol`` defaults to :data:`QUANT_TOLERANCE` for the recipe — the documented
    degradation budget. Returns the underlying
    :class:`jaxility.policy.litert.LiteRTParityReport`.
    """
    budget = tol if tol is not None else QUANT_TOLERANCE[quantized.recipe]
    return litert_parity(
        quantized.as_litert_model(), onnx_model_bytes, inputs, tol=budget
    )
