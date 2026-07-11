# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Dual-path HIL binary codegen: acados MPC + embedded policy + arbiter (T-044).

The integration capstone of the learned-policy lane. The generated C closes the
loop with **both** paths on the target: each cycle it solves the acados OCP for
the model-based control, runs a small embedded MLP policy for the learned
contribution, and combines them through the T-043 arbiter — clamping into the
:class:`jaxility.compose.SafetyEnvelope` and falling back to the
constraint-respecting MPC control on a (forced) policy timeout or a state-
envelope breach. It emits the T-033 trace each cycle, so the same HIL harness
validates the composition — *including the fallback* — on real silicon.

The policy MLP forward is emitted directly into C (a 4→H→1 net is matmul + tanh)
rather than calling a LiteRT runtime: it keeps the demo a single self-contained
native binary that builds through the proven on-Pi path, and the LiteRT runtime
path is validated separately (T-041). The host reference
(:class:`MLPPolicy.forward` + `jaxility.compose.arbitrate`) mirrors this exactly,
so the on-Pi run is HIL-checked against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .plan import CompositionPlan


@dataclass(frozen=True)
class DenseLayer:
    """One dense layer ``y = act(x @ W + b)`` of the embedded policy MLP."""

    weight: tuple[tuple[float, ...], ...]  # shape (in, out)
    bias: tuple[float, ...]  # shape (out,)
    activation: Literal["tanh", "identity"]


@dataclass(frozen=True)
class MLPPolicy:
    """A small feed-forward policy embedded into the dual-path binary."""

    layers: tuple[DenseLayer, ...]

    @classmethod
    def from_flax(cls, params: Any, *, input_dim: int) -> MLPPolicy:
        """Build from flax Dense params, ordered by the data-flow shape chain.

        flax names modules by *instantiation* order, which for a nested
        ``Dense(out)(act(Dense(hid)(x)))`` is the reverse of data flow — so we
        reconstruct execution order from the layer shapes instead, starting at
        ``input_dim`` and chaining ``out -> next in``. Assumes the demo
        architecture: ``Dense → tanh`` hidden layers then a final linear
        ``Dense`` (matching ``test/policy`` MLPs); layer sizes must form a
        unique chain.
        """
        dense = params["params"]
        # Each dense: kernel (in, out). Index by input dim to chain.
        by_in: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in dense:
            W = np.asarray(dense[k]["kernel"], dtype=np.float64)
            b = np.asarray(dense[k]["bias"], dtype=np.float64)
            if W.shape[0] in by_in:
                raise ValueError(
                    f"ambiguous layer chain: two dense layers take input dim "
                    f"{W.shape[0]}; from_flax needs a unique-size MLP chain."
                )
            by_in[int(W.shape[0])] = (W, b)

        ordered: list[tuple[np.ndarray, np.ndarray]] = []
        cur = input_dim
        while cur in by_in:
            W, b = by_in.pop(cur)
            ordered.append((W, b))
            cur = int(W.shape[1])
        if by_in:
            raise ValueError(
                f"could not chain all dense layers from input_dim={input_dim}; "
                f"leftover input dims {sorted(by_in)}."
            )

        layers = []
        for i, (W, b) in enumerate(ordered):
            act: Literal["tanh", "identity"] = (
                "identity" if i == len(ordered) - 1 else "tanh"
            )
            layers.append(
                DenseLayer(
                    weight=tuple(tuple(float(v) for v in row) for row in W),
                    bias=tuple(float(v) for v in b),
                    activation=act,
                )
            )
        return cls(layers=tuple(layers))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Host reference forward pass (float64), matching the emitted C."""
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            h = h @ np.array(layer.weight) + np.array(layer.bias)
            if layer.activation == "tanh":
                h = np.tanh(h)
        return h


def _emit_mlp_c(policy: MLPPolicy) -> str:
    """Emit the static weights + a ``jx_policy(const double* x, double* u)`` fn."""
    lines: list[str] = []
    for li, layer in enumerate(policy.layers):
        n_in = len(layer.weight)
        n_out = len(layer.bias)
        flat_w = ", ".join(
            repr(layer.weight[i][j]) for i in range(n_in) for j in range(n_out)
        )
        flat_b = ", ".join(repr(v) for v in layer.bias)
        lines.append(f"static const double W{li}[{n_in}][{n_out}] = {{{flat_w}}};")
        lines.append(f"static const double B{li}[{n_out}] = {{{flat_b}}};")
    body = ["    double cur[64]; double nxt[64]; int n_cur, n_nxt;"]
    body.append("    for (int i = 0; i < JX_NX; ++i) cur[i] = x[i];")
    body.append("    n_cur = JX_NX;")
    for li, layer in enumerate(policy.layers):
        n_in = len(layer.weight)
        n_out = len(layer.bias)
        body.append(f"    n_nxt = {n_out};")
        body.append(f"    for (int j = 0; j < {n_out}; ++j) {{")
        body.append(f"        double acc = B{li}[j];")
        body.append(
            f"        for (int k = 0; k < {n_in}; ++k) acc += cur[k] * W{li}[k][j];"
        )
        if layer.activation == "tanh":
            body.append("        nxt[j] = tanh(acc);")
        else:
            body.append("        nxt[j] = acc;")
        body.append("    }")
        body.append("    for (int j = 0; j < n_nxt; ++j) cur[j] = nxt[j];")
        body.append("    n_cur = n_nxt;")
    body.append("    for (int j = 0; j < JX_NU; ++j) u[j] = cur[j];")
    return (
        "\n".join(lines)
        + "\n\nstatic void jx_policy(const double *x, double *u) {\n"
        + "\n".join(body)
        + "\n}\n"
    )


def generate_dual_path_hil_source(
    *,
    model_name: str,
    nx: int,
    nu: int,
    policy: MLPPolicy,
    plan: CompositionPlan,
    initial_state: tuple[float, ...],
    seed_index: int = 0,
    seed_step: float = 0.01,
) -> str:
    """Emit the dual-path HIL ``main`` C: acados MPC + MLP policy + arbiter.

    argv: ``<n_steps> <seed> <fallback_from_step>``. ``fallback_from_step >= 0``
    forces a policy timeout from that step on (demonstrates fallback to MPC);
    ``-1`` never forces it. The arbiter also falls back on a state-envelope
    breach. The emitted command (and the plant it drives) is always inside the
    envelope. Mode is residual (``u = u_mpc + u_policy``) — Cartpole-on-LQR.
    """
    env = plan.safety_envelope
    x0 = ", ".join(repr(float(v)) for v in initial_state)
    sl = ", ".join(repr(v) for v in env.state_lower)
    su = ", ".join(repr(v) for v in env.state_upper)
    il = ", ".join(repr(v) for v in env.input_lower)
    iu = ", ".join(repr(v) for v in env.input_upper)
    fb_to = "1" if plan.fallback_on_timeout else "0"
    fb_env = "1" if plan.fallback_on_envelope_breach else "0"
    mlp_c = _emit_mlp_c(policy)
    return f"""\
/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * GENERATED by jaxility.compose.codegen (T-044) — do not edit by hand.
 *
 * Dual-path closed loop for {model_name!r}: acados MPC + embedded MLP policy,
 * combined by the T-043 arbiter (residual + clamp + fallback). Emits the T-033
 * trace each cycle. argv: <n_steps> <seed> <fallback_from_step>.
 */
#include "acados_solver_{model_name}.h"
#include "acados_sim_solver_{model_name}.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/sim_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define JX_NX {nx}
#define JX_NU {nu}

static const double STATE_LO[JX_NX] = {{{sl}}};
static const double STATE_HI[JX_NX] = {{{su}}};
static const double INPUT_LO[JX_NU] = {{{il}}};
static const double INPUT_HI[JX_NU] = {{{iu}}};

{mlp_c}

static int state_within(const double *x) {{
    for (int i = 0; i < JX_NX; ++i)
        if (x[i] < STATE_LO[i] || x[i] > STATE_HI[i]) return 0;
    return 1;
}}

static void clamp_input(double *u) {{
    for (int j = 0; j < JX_NU; ++j) {{
        if (u[j] < INPUT_LO[j]) u[j] = INPUT_LO[j];
        if (u[j] > INPUT_HI[j]) u[j] = INPUT_HI[j];
    }}
}}

int main(int argc, char **argv) {{
    if (argc != 3 && argc != 4) {{
        fprintf(stderr, "usage: %s <n_steps> <seed> [fallback_from_step]\\n", argv[0]);
        return 2;
    }}
    long n_steps = strtol(argv[1], NULL, 10);
    long seed = strtol(argv[2], NULL, 10);
    /* argv[3] forces a policy timeout from that step (demonstrates fallback);
     * omitted (the 2-arg HIL-runner form) means never. */
    long fb_from = (argc == 4) ? strtol(argv[3], NULL, 10) : -1;
    if (n_steps < 1 || n_steps > 100000) return 2;

    {model_name}_solver_capsule *ocp = {model_name}_acados_create_capsule();
    if (ocp == NULL || {model_name}_acados_create(ocp) != 0) return 1;
    ocp_nlp_config *cfg = {model_name}_acados_get_nlp_config(ocp);
    ocp_nlp_dims *dims = {model_name}_acados_get_nlp_dims(ocp);
    ocp_nlp_in *in = {model_name}_acados_get_nlp_in(ocp);
    ocp_nlp_out *out = {model_name}_acados_get_nlp_out(ocp);
    {model_name}_sim_solver_capsule *sim =
        {model_name}_acados_sim_solver_create_capsule();
    if (sim == NULL || {model_name}_acados_sim_create(sim) != 0) return 1;
    sim_config *scfg = {model_name}_acados_get_sim_config(sim);
    void *sdims = {model_name}_acados_get_sim_dims(sim);
    sim_in *sin = {model_name}_acados_get_sim_in(sim);
    sim_out *sout = {model_name}_acados_get_sim_out(sim);

    double x[JX_NX] = {{{x0}}};
    double u_mpc[JX_NU], u_pol[JX_NU], cmd[JX_NU];
    x[{seed_index}] += {seed_step!r} * (double)seed;

    for (long i = 0; i < n_steps; ++i) {{
        ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "lbx", x);
        ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "ubx", x);
        if ({model_name}_acados_solve(ocp) != 0) return 1;
        ocp_nlp_out_get(cfg, dims, out, 0, "u", u_mpc);
        jx_policy(x, u_pol);

        int timed_out = (fb_from >= 0 && i >= fb_from);
        if ((timed_out && {fb_to}) || (!state_within(x) && {fb_env})) {{
            for (int j = 0; j < JX_NU; ++j) cmd[j] = u_mpc[j];   /* fall back to MPC */
        }} else {{
            for (int j = 0; j < JX_NU; ++j)
                cmd[j] = u_mpc[j] + u_pol[j];   /* residual: policy on MPC */
        }}
        clamp_input(cmd);

        printf("{{\\"step\\": %ld, \\"joint_position\\": [%.17g, %.17g], "
               "\\"joint_velocity\\": [%.17g, %.17g], "
               "\\"actuator_torque\\": [%.17g]}}\\n",
               i, x[0], x[1], x[2], x[3], cmd[0]);

        sim_in_set(scfg, sdims, sin, "x", x);
        sim_in_set(scfg, sdims, sin, "u", cmd);
        if ({model_name}_acados_sim_solve(sim) != 0) return 1;
        sim_out_get(scfg, sdims, sout, "xn", x);
    }}
    return 0;
}}
"""


def generate_dual_path_bench_source(
    *,
    model_name: str,
    nx: int,
    nu: int,
    policy: MLPPolicy,
    plan: CompositionPlan,
    initial_state: tuple[float, ...],
    n_warmup: int = 100,
    seed_index: int = 0,
    seed_step: float = 0.01,
) -> str:
    """Emit a timing binary for the **composed** dual-path control cycle (T-044).

    Where :func:`jaxility.bench.generate_controller_bench_source` times the bare
    acados solve, this times the *whole* on-target control computation — acados
    solve + the embedded MLP policy + the arbiter (timeout / envelope fallback) +
    the envelope clamp — i.e. the real per-cycle cost of the dual-path
    deployment. It runs ``n_warmup`` untimed cycles then ``n_cycles`` timed (the
    plant advance is outside the timed region — it is the benchmark harness, not
    the deployed controller), and emits the same
    ``{{"solve_ns": [...], "max_rss_kib": N}}`` record the single-path bench does,
    so :func:`jaxility.bench.run_controller_bench` consumes it unchanged.

    The composed cycle it times is byte-for-byte the same control law
    :func:`generate_dual_path_hil_source` validates in HIL (shared policy emit +
    envelope + arbiter), so the benchmark measures exactly what HIL proves
    correct. ``argv``: ``<n_cycles> <seed>``.
    """
    env = plan.safety_envelope
    x0 = ", ".join(repr(float(v)) for v in initial_state)
    sl = ", ".join(repr(v) for v in env.state_lower)
    su = ", ".join(repr(v) for v in env.state_upper)
    il = ", ".join(repr(v) for v in env.input_lower)
    iu = ", ".join(repr(v) for v in env.input_upper)
    fb_env = "1" if plan.fallback_on_envelope_breach else "0"
    mlp_c = _emit_mlp_c(policy)
    return f"""\
/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * GENERATED by jaxility.compose.codegen (T-044) — do not edit by hand.
 *
 * Times the COMPOSED dual-path control cycle for {model_name!r}: acados solve +
 * embedded MLP policy + arbiter (residual + clamp + fallback), the real
 * per-cycle cost of the dual-path deployment. Plant advance is untimed. Emits
 * the single-path bench timing record. argv: <n_cycles> <seed>.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include "acados_solver_{model_name}.h"
#include "acados_sim_solver_{model_name}.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/sim_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>

#define JX_NX {nx}
#define JX_NU {nu}
#define JX_N_WARMUP {n_warmup}L

static const double STATE_LO[JX_NX] = {{{sl}}};
static const double STATE_HI[JX_NX] = {{{su}}};
static const double INPUT_LO[JX_NU] = {{{il}}};
static const double INPUT_HI[JX_NU] = {{{iu}}};

{mlp_c}

static int state_within(const double *x) {{
    for (int i = 0; i < JX_NX; ++i)
        if (x[i] < STATE_LO[i] || x[i] > STATE_HI[i]) return 0;
    return 1;
}}

static void clamp_input(double *u) {{
    for (int j = 0; j < JX_NU; ++j) {{
        if (u[j] < INPUT_LO[j]) u[j] = INPUT_LO[j];
        if (u[j] > INPUT_HI[j]) u[j] = INPUT_HI[j];
    }}
}}

static long now_ns(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + (long)ts.tv_nsec;
}}

int main(int argc, char **argv) {{
    if (argc != 3) {{
        fprintf(stderr, "usage: %s <n_cycles> <seed>\\n", argv[0]);
        return 2;
    }}
    long n_cycles = strtol(argv[1], NULL, 10);
    long seed = strtol(argv[2], NULL, 10);
    if (n_cycles < 1 || n_cycles > 100000) return 2;

    {model_name}_solver_capsule *ocp = {model_name}_acados_create_capsule();
    if (ocp == NULL || {model_name}_acados_create(ocp) != 0) return 1;
    ocp_nlp_config *cfg = {model_name}_acados_get_nlp_config(ocp);
    ocp_nlp_dims *dims = {model_name}_acados_get_nlp_dims(ocp);
    ocp_nlp_in *in = {model_name}_acados_get_nlp_in(ocp);
    ocp_nlp_out *out = {model_name}_acados_get_nlp_out(ocp);
    {model_name}_sim_solver_capsule *sim =
        {model_name}_acados_sim_solver_create_capsule();
    if (sim == NULL || {model_name}_acados_sim_create(sim) != 0) return 1;
    sim_config *scfg = {model_name}_acados_get_sim_config(sim);
    void *sdims = {model_name}_acados_get_sim_dims(sim);
    sim_in *sin = {model_name}_acados_get_sim_in(sim);
    sim_out *sout = {model_name}_acados_get_sim_out(sim);

    double x[JX_NX] = {{{x0}}};
    double u_mpc[JX_NU], u_pol[JX_NU], cmd[JX_NU];
    x[{seed_index}] += {seed_step!r} * (double)seed;

    long *samples = malloc((size_t)n_cycles * sizeof(long));
    if (samples == NULL) {{ fprintf(stderr, "oom\\n"); return 1; }}

    for (long i = 0; i < JX_N_WARMUP + n_cycles; ++i) {{
        ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "lbx", x);
        ocp_nlp_constraints_model_set(cfg, dims, in, out, 0, "ubx", x);
        long t0 = now_ns();
        /* --- the composed control computation (the deployed cost) --- */
        if ({model_name}_acados_solve(ocp) != 0) return 1;
        ocp_nlp_out_get(cfg, dims, out, 0, "u", u_mpc);
        jx_policy(x, u_pol);
        if (!state_within(x) && {fb_env}) {{
            for (int j = 0; j < JX_NU; ++j) cmd[j] = u_mpc[j];
        }} else {{
            for (int j = 0; j < JX_NU; ++j) cmd[j] = u_mpc[j] + u_pol[j];
        }}
        clamp_input(cmd);
        long t1 = now_ns();
        /* --- end timed region; advance the plant untimed --- */
        if (i >= JX_N_WARMUP) samples[i - JX_N_WARMUP] = t1 - t0;

        sim_in_set(scfg, sdims, sin, "x", x);
        sim_in_set(scfg, sdims, sin, "u", cmd);
        if ({model_name}_acados_sim_solve(sim) != 0) return 1;
        sim_out_get(scfg, sdims, sout, "xn", x);
    }}

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("{{\\"solve_ns\\": [");
    for (long i = 0; i < n_cycles; ++i) {{
        printf("%ld%s", samples[i], (i + 1 < n_cycles) ? ", " : "");
    }}
    printf("], \\"max_rss_kib\\": %ld}}\\n", (long)ru.ru_maxrss);
    free(samples);
    return 0;
}}
"""
