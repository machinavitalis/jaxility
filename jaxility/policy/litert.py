# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""ONNX → LiteRT (.tflite) lowering for the learned-policy path (T-041).

The priority on-device inference target (CONTEXT §"Learned-policy deployment
path"). Jaxility **orchestrates** the external converter (`onnx2tf`: ONNX →
TensorFlow → TFLite) rather than vendoring it — the same posture as acados and
the Arm toolchains. The converter + the LiteRT runtime ship behind the
``[litert]`` extra; if they are absent, the export raises a structured
:class:`~jaxility.errors.ToolchainError` naming the extra (loud failure,
invariant 7), exactly like ``verify_toolchain_installed`` for the cross
toolchains.

The export consumes a :class:`jaxility.policy.PolicyOnnxModel` (T-040) and
produces a :class:`LiteRTModel` — the ``.tflite`` bytes plus provenance. A
parity harness (:func:`litert_parity`) runs the converted model under the LiteRT
interpreter and compares it to ONNX Runtime so a conversion that silently
changes numerics fails the gate (mirrors the source-to-artifact equivalence
invariant on the controller side).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ToolchainError
from .onnx_export import PolicyOnnxModel

LITERT_EXPORT_SCHEMA_V0 = 0
"""Schema version of the LiteRT export metadata."""


@dataclass(frozen=True)
class LiteRTModel:
    """An exported LiteRT (.tflite) policy plus the provenance of the export."""

    tflite_bytes: bytes
    """The serialised TFLite flatbuffer (LiteRT interpreter loads it directly)."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]
    source_onnx_ops: frozenset[str]
    """ONNX op types of the source model — the conversion-coverage audit trail."""


def _require_converter() -> Any:
    """Import the onnx2tf converter, or raise a structured ToolchainError."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError as exc:
        raise ToolchainError(
            "the ONNX → LiteRT converter (onnx2tf + tensorflow) is not installed. "
            "Install the LiteRT tooling extra: `pip install 'jaxility[litert]'`. "
            "Jaxility orchestrates the converter (it does not vendor it), so the "
            "export is gated on the extra being present (the acados pattern)."
        ) from exc
    return onnx2tf


def _require_runtime() -> Any:
    """Import the LiteRT interpreter, or raise a structured ToolchainError."""
    try:
        from ai_edge_litert import interpreter as litert_interp
    except ImportError as exc:
        raise ToolchainError(
            "the LiteRT runtime (ai-edge-litert) is not installed. "
            "Install `pip install 'jaxility[litert]'` to run / parity-check the "
            "converted .tflite model."
        ) from exc
    return litert_interp


def export_onnx_to_litert(
    policy_onnx: PolicyOnnxModel,
    *,
    work_dir: Path | None = None,
    name: str = "policy",
) -> LiteRTModel:
    """Convert a :class:`PolicyOnnxModel` to LiteRT (.tflite).

    Orchestrates ``onnx2tf`` (ONNX → TF → TFLite). The float32 model is taken
    (quantisation recipes are T-042). Raises :class:`ToolchainError` if the
    ``[litert]`` tooling is absent.
    """
    onnx2tf = _require_converter()

    ctx: tempfile.TemporaryDirectory[str] | None
    if work_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix="jaxility-litert-")
        base = Path(ctx.name)
    else:
        ctx = None
        base = Path(work_dir)
    try:
        base.mkdir(parents=True, exist_ok=True)
        onnx_path = base / f"{name}.onnx"
        onnx_path.write_bytes(policy_onnx.model_bytes)
        out_dir = base / "tflite"
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(out_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )
        # onnx2tf emits ``<model>_float32.tflite`` (+ other dtypes we ignore).
        candidates = sorted(out_dir.glob("*_float32.tflite")) or sorted(
            out_dir.glob("*.tflite")
        )
        if not candidates:
            raise ToolchainError(
                f"onnx2tf produced no .tflite under {out_dir}; conversion failed "
                f"for model {name!r}."
            )
        tflite_bytes = candidates[0].read_bytes()
    finally:
        if ctx is not None:
            ctx.cleanup()

    return LiteRTModel(
        tflite_bytes=tflite_bytes,
        input_names=policy_onnx.input_names,
        output_names=policy_onnx.output_names,
        in_shapes=policy_onnx.in_shapes,
        out_shapes=policy_onnx.out_shapes,
        source_onnx_ops=policy_onnx.onnx_ops_used,
    )


def run_litert(model: LiteRTModel, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Run a LiteRT model under the interpreter; return the outputs in order."""
    litert_interp = _require_runtime()
    interp = litert_interp.Interpreter(model_content=model.tflite_bytes)
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    for det, arr in zip(in_details, inputs):
        interp.set_tensor(det["index"], np.asarray(arr, dtype=det["dtype"]))
    interp.invoke()
    return [interp.get_tensor(det["index"]) for det in out_details]


@dataclass(frozen=True)
class LiteRTParityReport:
    """Numerical parity of the LiteRT conversion vs the source ONNX model."""

    max_abs_error: float
    passed: bool
    tol: float


def litert_parity(
    litert_model: LiteRTModel,
    onnx_model_bytes: bytes,
    inputs: list[np.ndarray],
    *,
    tol: float = 1e-4,
) -> LiteRTParityReport:
    """Compare LiteRT vs ONNX Runtime on the same inputs.

    A conversion that changes numerics beyond ``tol`` fails the gate. ``tol``
    defaults to a tight float32 bound — the float32 LiteRT path should match
    ONNX closely (quantisation, with its looser tolerances, is T-042).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_model_bytes)
    onnx_out = sess.run(
        None, {n: np.asarray(a) for n, a in zip(litert_model.input_names, inputs)}
    )
    litert_out = run_litert(litert_model, inputs)
    max_err = 0.0
    for onnx_o, litert_o in zip(onnx_out, litert_out):
        max_err = max(
            max_err, float(np.abs(np.asarray(onnx_o) - np.asarray(litert_o)).max())
        )
    return LiteRTParityReport(max_abs_error=max_err, passed=max_err <= tol, tol=tol)
