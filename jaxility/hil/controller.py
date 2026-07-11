# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Generate + build a closed-loop HIL binary around an acados controller (T-034).

This is the controller-side counterpart to the T-033 harness. `build_for_target`
produces an acados OCP controller and, alongside it, an acados *sim*
solver (an ERK integrator over the same CasADi-from-JAX dynamics). This module
emits a small C `main` that closes the loop between them — each cycle it pins the
OCP's initial state to the current plant state, solves for the control, advances
the plant with the sim solver, and emits one line of the JSON-Lines HIL trace
contract (:mod:`jaxility.hil.trace`). The resulting binary is a
:class:`jaxility.hil.TargetRunner` target: run it, parse its trace, compare it to
the host reference under the documented tolerances.

Why a *generated* closed loop rather than the deploy launcher
(``jaxility.runtime.deploy``): the deploy ABI (`jx_controller_step`) is
fire-and-forget with no per-cycle state I/O — correct for driving real hardware,
but a HIL parity check needs to *observe* the trajectory. So the HIL binary
embeds the plant (the acados sim solver, the user-chosen closed-loop semantics)
and emits the state each cycle. The deployment path and this HIL path share the
same generated controller; they differ only in what closes the loop (real
actuators/sensors vs. the sim-solver plant) and whether a trace is emitted.

Scope (T-034 slice 1): the binary is generated and built/run on the host here.
The on-Pi build reuses the same source via the Pi's native toolchain + an acados
install on the target (build-on-Pi; the attested cross-toolchain build is T-036).
The acados C call sequence is the one proven during bring-up — see the
``acados_solver_<model>`` / ``acados_sim_solver_<model>`` generated headers.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..errors import HILError, ToolchainError

DEFAULT_HIL_BUILD_TIMEOUT_S = 180.0
"""Wall-clock budget for compiling the HIL binary."""


@dataclass(frozen=True)
class TraceQuantity:
    """Maps a slice of a C state/control buffer to a named trace quantity.

    ``buffer`` is ``"x"`` (the plant state) or ``"u"`` (the control); the
    emitted line carries ``buffer[start : start + length]`` under ``name``.
    ``name`` and ``(length,)`` must match the :class:`jaxility.hil.StateSchema`
    the parser is given, and ``name`` must be a tolerance-table key.
    """

    name: str
    buffer: str
    start: int
    length: int


# Cartpole LQR layout: state is [x_cart, theta, x_dot, theta_dot] (nx=4),
# control is [force] (nu=1). Mirrors test/unit/test_host_equivalence.py's
# _trajectory_to_quantities so the two reference paths agree.
CARTPOLE_LQR_TRACE = (
    TraceQuantity("joint_position", "x", 0, 2),
    TraceQuantity("joint_velocity", "x", 2, 2),
    TraceQuantity("actuator_torque", "u", 0, 1),
)


def _acados_root() -> Path:
    """Resolve the acados install (headers + libs). Loud if absent."""
    root = os.environ.get("ACADOS_SOURCE_DIR")
    if not root:
        raise ToolchainError(
            "ACADOS_SOURCE_DIR is not set; cannot locate the acados headers/libs "
            "to build the HIL controller binary. Export it to your acados install."
        )
    p = Path(root)
    if not (p / "include").is_dir() or not (p / "lib").is_dir():
        raise ToolchainError(
            f"ACADOS_SOURCE_DIR={root!r} does not look like an acados install "
            f"(missing include/ or lib/)."
        )
    return p


def _printf_statement(trace: tuple[TraceQuantity, ...]) -> str:
    """Build the per-cycle ``printf`` that emits one JSON-Lines trace record."""
    frags = ['\\"step\\": %ld']
    args = ["i"]
    for q in trace:
        slots = ", ".join(["%.17g"] * q.length)
        frags.append(f'\\"{q.name}\\": [{slots}]')
        args.extend(f"{q.buffer}[{q.start + k}]" for k in range(q.length))
    fmt = "{" + ", ".join(frags) + "}\\n"
    return f'printf("{fmt}", {", ".join(args)});'


def generate_controller_hil_source(
    *,
    model_name: str,
    nx: int,
    nu: int,
    trace: tuple[TraceQuantity, ...] = CARTPOLE_LQR_TRACE,
    initial_state: tuple[float, ...],
    seed_index: int = 0,
    seed_step: float = 0.01,
) -> str:
    """Emit the closed-loop HIL ``main`` C source for ``model_name``.

    The generated program takes ``<n_steps> <seed>`` argv, runs the
    OCP-controller / sim-plant closed loop for ``n_steps`` cycles from
    ``initial_state`` (perturbed at index ``seed_index`` by ``seed * seed_step``,
    matching the host reference), and writes the JSON-Lines trace to stdout.
    """
    if len(initial_state) != nx:
        raise HILError(
            f"initial_state has {len(initial_state)} entries; model {model_name!r} "
            f"has nx={nx}."
        )
    x0_init = ", ".join(repr(float(v)) for v in initial_state)
    return f"""\
/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * GENERATED by jaxility.hil.controller (T-034) — do not edit by hand.
 *
 * Closed-loop HIL binary for the acados controller {model_name!r}. Each cycle:
 * pin the OCP initial state to the plant state, solve for the control, advance
 * the plant with the acados sim solver, emit one JSON-Lines trace record.
 */
#include "acados_solver_{model_name}.h"
#include "acados_sim_solver_{model_name}.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/sim_interface.h"
#include <stdio.h>
#include <stdlib.h>

#define JX_NX {nx}
#define JX_NU {nu}

int main(int argc, char **argv) {{
    if (argc != 3) {{
        fprintf(stderr, "usage: %s <n_steps> <seed>\\n", argv[0]);
        return 2;
    }}
    long n_steps = strtol(argv[1], NULL, 10);
    long seed = strtol(argv[2], NULL, 10);
    if (n_steps < 1 || n_steps > 100000) {{
        fprintf(stderr, "n_steps out of range [1, 100000]: %ld\\n", n_steps);
        return 2;
    }}

    {model_name}_solver_capsule *ocp = {model_name}_acados_create_capsule();
    if (ocp == NULL || {model_name}_acados_create(ocp) != 0) {{
        fprintf(stderr, "acados OCP create failed\\n");
        return 1;
    }}
    ocp_nlp_config *cfg = {model_name}_acados_get_nlp_config(ocp);
    ocp_nlp_dims *dims = {model_name}_acados_get_nlp_dims(ocp);
    ocp_nlp_in *in = {model_name}_acados_get_nlp_in(ocp);
    ocp_nlp_out *out = {model_name}_acados_get_nlp_out(ocp);

    {model_name}_sim_solver_capsule *sim =
        {model_name}_acados_sim_solver_create_capsule();
    if (sim == NULL || {model_name}_acados_sim_create(sim) != 0) {{
        fprintf(stderr, "acados sim create failed\\n");
        return 1;
    }}
    sim_config *scfg = {model_name}_acados_get_sim_config(sim);
    void *sdims = {model_name}_acados_get_sim_dims(sim);
    sim_in *sin = {model_name}_acados_get_sim_in(sim);
    sim_out *sout = {model_name}_acados_get_sim_out(sim);

    double x[JX_NX] = {{{x0_init}}};
    double u[JX_NU];
    x[{seed_index}] += {seed_step!r} * (double)seed;

    for (long i = 0; i < n_steps; ++i) {{
        /* Pin the OCP initial state to the current plant state. */
        ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "lbx", x);
        ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "ubx", x);
        if ({model_name}_acados_solve(ocp) != 0) {{
            fprintf(stderr, "acados solve failed at step %ld\\n", i);
            return 1;
        }}
        ocp_nlp_out_get(cfg, dims, out, 0, "u", u);

        {_printf_statement(trace)}

        /* Advance the plant one cycle with the acados sim (ERK) solver. */
        sim_in_set(scfg, sdims, sin, "x", x);
        sim_in_set(scfg, sdims, sin, "u", u);
        if ({model_name}_acados_sim_solve(sim) != 0) {{
            fprintf(stderr, "acados sim solve failed at step %ld\\n", i);
            return 1;
        }}
        sim_out_get(scfg, sdims, sout, "xn", x);
    }}
    return 0;
}}
"""


def build_controller_hil_binary(
    *,
    generated_code_dir: Path,
    model_name: str,
    source: str,
    out_path: Path,
    cc: str = "cc",
    timeout_s: float = DEFAULT_HIL_BUILD_TIMEOUT_S,
) -> Path:
    """Compile the generated HIL ``source`` into an executable at ``out_path``.

    Links the acados-generated OCP + sim solver sources and the precompiled
    model objects against the acados / hpipm / blasfeo libraries from
    ``ACADOS_SOURCE_DIR``, embedding an rpath so the dynamic loader finds them
    at run time (the macOS-SIP-safe path proven during bring-up). Raises
    :class:`~jaxility.errors.HILError` on a compile failure.
    """
    gen = Path(generated_code_dir)
    if not gen.is_dir():
        raise HILError(f"generated_code_dir not found: {gen}")
    aca = _acados_root()
    lib = aca / "lib"

    main_c = gen / f"{model_name}_hil_main.c"
    main_c.write_text(source)

    model_objs = sorted((gen / f"{model_name}_model").glob("*.o"))
    if not model_objs:
        raise HILError(
            f"no compiled model objects under {gen / (model_name + '_model')}; "
            f"build_for_target must run before generating the HIL binary."
        )

    argv = [
        cc,
        "-std=c99",
        "-O2",
        f"-I{gen}",
        f"-I{aca / 'include'}",
        f"-I{aca / 'include' / 'acados'}",
        f"-I{aca / 'include' / 'blasfeo' / 'include'}",
        f"-I{aca / 'include' / 'hpipm' / 'include'}",
        str(main_c),
        str(gen / f"acados_solver_{model_name}.c"),
        str(gen / f"acados_sim_solver_{model_name}.c"),
        *[str(o) for o in model_objs],
        f"-L{lib}",
        "-lacados",
        "-lhpipm",
        "-lblasfeo",
        "-lm",
        f"-Wl,-rpath,{lib}",
        "-o",
        str(out_path),
    ]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise HILError(f"HIL binary compile timed out after {timeout_s}s") from exc
    if completed.returncode != 0:
        raise HILError(
            f"HIL binary compile failed ({shlex.join(argv[:1])} ...):\n"
            f"{completed.stderr.strip() or '(no stderr)'}"
        )
    return Path(out_path)
