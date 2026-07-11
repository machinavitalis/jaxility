/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * jaxility_runtime/cycle.h — deterministic cycle scheduler interface.
 *
 * The control-loop scheduler the deployment runtime uses to drive
 * the acados solver at a fixed cadence (1 kHz on PI5; 2 kHz on
 * Cortex-M). The interface is platform-independent; per-target
 * implementations live under runtime-c/src/cycle_*.c:
 *
 *   - cycle_posix.c     — Cortex-A / Pi 5: clock_gettime + nanosleep
 *                         under PREEMPT_RT. Runs on every POSIX host
 *                         (Linux, macOS, GitHub Linux runners).
 *   - cycle_systick.c   — Cortex-M baremetal: SysTick timer + ISR.
 *                         Not in this skeleton; lands per-MCU in the
 *                         T-052 follow-up alongside the linker scripts
 *                         and startup files for STM32F4 / nRF52.
 *
 * Contract:
 *   1. Caller initialises the scheduler with the target period (ns).
 *   2. Caller registers a single ``tick`` callback (the control loop
 *      body — for the acados deployment, this is one solver step).
 *   3. Caller starts the loop; the scheduler invokes ``tick`` at the
 *      configured period, sleeping the remainder of each cycle.
 *   4. The loop exits when the tick callback returns non-zero, or
 *      when ``jx_cycle_stop`` is called from another thread / signal.
 *
 * Jitter is measured and accumulated into ``jx_cycle_stats`` so the
 * HIL parity tier (T-033) can assert worst-case-jitter bounds without
 * spawning a separate measurement pass.
 */
#ifndef JAXILITY_RUNTIME_CYCLE_H
#define JAXILITY_RUNTIME_CYCLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*jx_cycle_tick_fn)(void *user_ctx);

typedef struct {
    /* Total number of ticks invoked since start. */
    uint64_t ticks;
    /* Number of ticks whose body ran longer than the period — the
     * load-bearing soft-real-time failure signal. */
    uint64_t overruns;
    /* Cumulative *slack* (sum of |target - tick_end| in ns) — the
     * time the scheduler spent sleeping each cycle. Audit M-9
     * close: this is **not** scheduling jitter; the previous field
     * names ``jitter_sum_ns`` / ``jitter_max_ns`` were misleading.
     * Slack is still useful — a slack→0 trend signals overload. */
    uint64_t slack_sum_ns;
    /* Worst-case observed slack in ns since start. */
    uint64_t slack_max_ns;
    /* Cumulative *scheduling jitter* (sum of |tick_start - target|
     * in ns) — measured AFTER the sleep returns, comparing when the
     * next tick actually started against when it should have. This
     * is the load-bearing HIL-parity signal. tick 0 has no prior
     * deadline, so jitter starts accumulating from tick 1. */
    uint64_t jitter_sum_ns;
    /* Worst-case observed scheduling jitter in ns since start. */
    uint64_t jitter_max_ns;
} jx_cycle_stats_t;

typedef struct jx_cycle jx_cycle_t;

/* Allocate-out: caller passes the scheduler struct backing memory
 * (typically from a jx_arena_t). Returns 0 on success, -1 on
 * configuration error (period_ns == 0). */
int jx_cycle_init(jx_cycle_t *cycle, uint64_t period_ns,
                  jx_cycle_tick_fn tick, void *user_ctx);

/* Run the loop in the calling thread. Returns the tick callback's
 * non-zero exit code or 0 if ``jx_cycle_stop`` was called. Returns
 * -1 if ``cycle`` is NULL or not initialised. */
int jx_cycle_run(jx_cycle_t *cycle);

/* Request termination. Safe to call from a signal handler or from
 * another thread; the running loop exits at the next cycle boundary. */
void jx_cycle_stop(jx_cycle_t *cycle);

/* Snapshot the accumulated stats. Safe to call mid-run; the snapshot
 * is *not* atomic per-field but each individual field is wider than
 * the host's native word size only on 32-bit Cortex-M, where the
 * caller serialises against the tick. */
void jx_cycle_stats(const jx_cycle_t *cycle, jx_cycle_stats_t *out);

/* Size of the opaque scheduler struct in bytes. Useful for arena
 * sizing without exposing struct internals. */
size_t jx_cycle_struct_size(void);

#ifdef __cplusplus
}
#endif

#endif /* JAXILITY_RUNTIME_CYCLE_H */
