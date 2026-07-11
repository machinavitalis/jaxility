# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""T-034 — closed-loop HIL parity for the real acados Cartpole controller.

Two layers:

* **generator** — pure-Python tests for the emitted C source (no acados, no
  compiler needed): the source names the controller + sim ABI and emits the
  trace for the requested quantity layout.
* **end-to-end host parity** — build the real Cartpole LQR controller, generate
  + compile the closed-loop HIL binary, run it through the T-033 harness, and
  assert step-locked parity against the host reference (acados control + JAX
  ERK4 plant) under the `cortex-a76` x float64 tolerances. Gated on the acados
  toolchain (t_renderer) and a host `cc`; this is the host stand-in for the
  on-Pi gate (T-036).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from cartpole_lqr_reference import (
    MODEL_NAME,
    build_cartpole_lqr,
    closed_loop_reference,
)

from jaxility.hil import (
    CARTPOLE_LQR_SCHEMA,
    CARTPOLE_LQR_TRACE,
    LocalRunner,
    build_controller_hil_binary,
    generate_controller_hil_source,
    run_hil,
)


def _tera_available() -> bool:
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if root and (Path(root) / "bin" / "t_renderer").exists():
        return True
    return shutil.which("t_renderer") is not None


_TERA = _tera_available()
_NEEDS_BUILD = pytest.mark.skipif(
    not _TERA or shutil.which("cc") is None,
    reason="acados t_renderer + host cc required for the controller HIL build",
)


# ---------------------------------------------------------------------------
# Generator — pure, no acados / compiler
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_source_names_controller_and_sim_abi() -> None:
    src = generate_controller_hil_source(
        model_name="demo_model", nx=4, nu=1, initial_state=(0.3, 0.0, 0.0, 0.0)
    )
    # OCP control path
    assert "demo_model_acados_create_capsule" in src
    assert 'ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "lbx", x)' in src
    assert 'ocp_nlp_out_get(cfg, dims, out, 0, "u", u)' in src
    # sim (plant) path
    assert "demo_model_acados_sim_create" in src
    assert 'sim_out_get(scfg, sdims, sout, "xn", x)' in src
    # trace contract for the cartpole layout
    assert '\\"joint_position\\": [%.17g, %.17g]' in src
    assert '\\"actuator_torque\\": [%.17g]' in src
    assert "#define JX_NX 4" in src and "#define JX_NU 1" in src


@pytest.mark.unit
def test_generate_source_rejects_initial_state_mismatch() -> None:
    from jaxility.errors import HILError

    with pytest.raises(HILError, match="nx=4"):
        generate_controller_hil_source(
            model_name="m", nx=4, nu=1, initial_state=(0.0, 0.0)
        )


# ---------------------------------------------------------------------------
# End-to-end host parity — build + generate + run + compare
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cartpole_bundle(tmp_path_factory: pytest.TempPathFactory):
    work = tmp_path_factory.mktemp("t034_build")
    return build_cartpole_lqr(work / "build")


@_NEEDS_BUILD
def test_controller_hil_parity_on_host(cartpole_bundle) -> None:
    n_steps, seed = 40, 0
    gen_dir = cartpole_bundle.shared_library_path.parent

    source = generate_controller_hil_source(
        model_name=MODEL_NAME,
        nx=4,
        nu=1,
        trace=CARTPOLE_LQR_TRACE,
        initial_state=(0.3, 0.0, 0.0, 0.0),
    )
    exe = build_controller_hil_binary(
        generated_code_dir=gen_dir,
        model_name=MODEL_NAME,
        source=source,
        out_path=gen_dir / f"{MODEL_NAME}_hil",
    )

    reference = closed_loop_reference(cartpole_bundle.solver, n_steps, seed)
    runner = LocalRunner(exe)
    report = run_hil(
        reference,
        runner,
        target_family="cortex-a76",
        dtype="float64",
        schema=CARTPOLE_LQR_SCHEMA,
        n_steps=n_steps,
        seed=seed,
    )
    report.assert_passed()
    assert report.passed
