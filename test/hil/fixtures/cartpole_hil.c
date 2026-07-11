/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * cartpole_hil.c — deterministic HIL test fixture (T-033).
 *
 * This is NOT a generated acados controller (that is T-034). It is a
 * self-contained, fully deterministic stand-in that exercises the HIL
 * harness end to end *before* real codegen exists — the same way the
 * mock Source / mock_lower layers exercise the build pipeline before
 * the real lowering. It deliberately depends on nothing but the C
 * standard library so it builds with the host `cc` (CI / dev box) and
 * natively on the Pi 5 (`gcc`, Cortex-A76).
 *
 * Plant + controller: a damped linear pendulum stabilised by a fixed
 * state-feedback gain, integrated with explicit Euler at the 1 kHz
 * deployment cadence. Stable closed loop, so the trajectory decays and
 * stays bounded — which keeps the float32 (this fixture) vs float64
 * (the host reference) divergence inside the documented `cortex-a76`
 * tolerance rows.
 *
 * Emits the HIL trace contract (jaxility.hil.trace): one JSON object
 * per control cycle on stdout, in step order, e.g.
 *
 *   {"step": 0, "joint_position": [0.200000], "joint_velocity": [0.0], "actuator_torque": [-1.6]}
 *
 * Usage: cartpole_hil <n_steps> <seed>
 *   n_steps : number of control cycles to emit (1..100000)
 *   seed    : integer; perturbs the initial angle deterministically so
 *             the harness can vary the run without changing the binary.
 *
 * The arithmetic is intentionally `float` (single precision): float32 is
 * the embedded-default precision (CONTEXT.md design philosophy 3), so
 * the fixture models what a real Cortex-A76 deployment would carry. The
 * host reference (test/hil/cartpole_reference.py) replicates this exact
 * recurrence in float64.
 */
#include <stdio.h>
#include <stdlib.h>

/* Plant + controller constants. Kept in the C and mirrored byte-for-byte
 * in cartpole_reference.py — change one, change the other. */
#define JX_HIL_DT 0.001f        /* 1 kHz control period */
#define JX_HIL_G_OVER_L 10.0f   /* linearised gravity / length term */
#define JX_HIL_DAMPING 0.5f     /* viscous damping */
#define JX_HIL_GAIN_THETA 8.0f  /* feedback gain on angle */
#define JX_HIL_GAIN_RATE 2.0f   /* feedback gain on angular rate */
#define JX_HIL_THETA0 0.2f      /* nominal initial angle (rad) */
#define JX_HIL_SEED_STEP 0.01f  /* per-seed initial-angle perturbation */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <n_steps> <seed>\n", argv[0]);
        return 2;
    }
    long n_steps = strtol(argv[1], NULL, 10);
    long seed = strtol(argv[2], NULL, 10);
    if (n_steps < 1 || n_steps > 100000) {
        fprintf(stderr, "n_steps out of range [1, 100000]: %ld\n", n_steps);
        return 2;
    }

    /* Deterministic initial condition: nominal angle nudged by the seed. */
    float theta = JX_HIL_THETA0 + JX_HIL_SEED_STEP * (float)seed;
    float theta_dot = 0.0f;

    for (long i = 0; i < n_steps; ++i) {
        /* Controller: fixed state feedback. */
        float u = -JX_HIL_GAIN_THETA * theta - JX_HIL_GAIN_RATE * theta_dot;

        /* Emit the state at the *start* of the cycle plus the control
         * applied this cycle. printf("%.9g") round-trips a float32 value
         * losslessly into the JSON the harness parses back. */
        printf(
            "{\"step\": %ld, \"joint_position\": [%.9g], "
            "\"joint_velocity\": [%.9g], \"actuator_torque\": [%.9g]}\n",
            i, (double)theta, (double)theta_dot, (double)u);

        /* Plant: explicit Euler. theta uses the *old* rate (so the two
         * updates are independent within the step); the reference mirrors
         * this exactly. */
        float accel = -JX_HIL_G_OVER_L * theta - JX_HIL_DAMPING * theta_dot + u;
        float theta_next = theta + JX_HIL_DT * theta_dot;
        float theta_dot_next = theta_dot + JX_HIL_DT * accel;
        theta = theta_next;
        theta_dot = theta_dot_next;
    }
    return 0;
}
