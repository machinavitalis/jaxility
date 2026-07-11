/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * cycle_posix.c — POSIX clock_gettime + nanosleep scheduler (T-032).
 *
 * Used by:
 *   - Pi 5 deployment (Cortex-A76 + PREEMPT_RT Linux).
 *   - CI Linux x86_64 runners (where the Tier-B real-cross-compile
 *     test verifies this file compiles for aarch64-linux-gnu).
 *   - Host equivalence tier (T-027) when a deployment artifact's
 *     loop is exercised on the dev host before crossing to the Pi.
 *
 * Drift correction: each tick computes its deadline as
 *   deadline = start_time + (ticks_executed + 1) * period
 * so missed cycles do not accumulate phase error — the next tick
 * still fires at its absolute target time, registering an overrun
 * rather than letting drift compound.
 */
#if defined(__linux__) || defined(__APPLE__)

#define _POSIX_C_SOURCE 200809L
#include "jaxility_runtime/cycle.h"

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifndef NSEC_PER_SEC
#define NSEC_PER_SEC 1000000000ULL
#endif

struct jx_cycle {
    uint64_t          period_ns;
    jx_cycle_tick_fn  tick;
    void             *user_ctx;
    jx_cycle_stats_t  stats;
    volatile int      stop_requested;
    int               initialised;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * NSEC_PER_SEC + (uint64_t)ts.tv_nsec;
}

static void sleep_until_ns(uint64_t target_ns) {
    /* Portable absolute-sleep. On Linux glibc this could use
     * ``clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ...)`` but
     * macOS does not implement that POSIX-2008 entry point, so we
     * loop on relative ``nanosleep`` and recompute the remainder
     * against ``CLOCK_MONOTONIC`` each iteration. EINTR returns
     * loop back through the same path. */
    for (;;) {
        uint64_t now = now_ns();
        if (now >= target_ns) return;
        uint64_t delta = target_ns - now;
        struct timespec rel;
        rel.tv_sec  = (time_t)(delta / NSEC_PER_SEC);
        rel.tv_nsec = (long)(delta % NSEC_PER_SEC);
        if (nanosleep(&rel, NULL) == 0) return;
        if (errno != EINTR) return;
    }
}

int jx_cycle_init(jx_cycle_t *cycle, uint64_t period_ns,
                  jx_cycle_tick_fn tick, void *user_ctx) {
    if (cycle == NULL || period_ns == 0U || tick == NULL) {
        return -1;
    }
    cycle->period_ns       = period_ns;
    cycle->tick            = tick;
    cycle->user_ctx        = user_ctx;
    cycle->stats.ticks         = 0U;
    cycle->stats.overruns      = 0U;
    cycle->stats.slack_sum_ns  = 0U;
    cycle->stats.slack_max_ns  = 0U;
    cycle->stats.jitter_sum_ns = 0U;
    cycle->stats.jitter_max_ns = 0U;
    cycle->stop_requested  = 0;
    cycle->initialised     = 1;
    return 0;
}

int jx_cycle_run(jx_cycle_t *cycle) {
    if (cycle == NULL || !cycle->initialised) {
        return -1;
    }
    uint64_t start = now_ns();
    /* tick 0's deadline is start; the first scheduled tick starts at
     * start + period. We measure jitter for ticks 1..N comparing the
     * actual tick-start time to its deadline. */
    while (!cycle->stop_requested) {
        uint64_t tick_target =
            start + (cycle->stats.ticks + 1U) * cycle->period_ns;
        uint64_t tick_start = now_ns();
        /* Jitter (M-9): how late this tick STARTED relative to its
         * deadline. tick 0 has no prior deadline; subsequent ticks
         * are compared against the previous cycle's target. */
        if (cycle->stats.ticks > 0U) {
            uint64_t prev_target =
                start + cycle->stats.ticks * cycle->period_ns;
            uint64_t jitter = (tick_start > prev_target)
                                  ? (tick_start - prev_target)
                                  : (prev_target - tick_start);
            cycle->stats.jitter_sum_ns += jitter;
            if (jitter > cycle->stats.jitter_max_ns) {
                cycle->stats.jitter_max_ns = jitter;
            }
        }
        int rc = cycle->tick(cycle->user_ctx);
        cycle->stats.ticks += 1U;
        uint64_t after = now_ns();
        /* Slack (formerly mis-named "jitter"): how much of the period
         * we have left after the tick body. */
        uint64_t slack = (after < tick_target) ? (tick_target - after) : 0U;
        cycle->stats.slack_sum_ns += slack;
        if (slack > cycle->stats.slack_max_ns) {
            cycle->stats.slack_max_ns = slack;
        }
        if (after > tick_target) {
            cycle->stats.overruns += 1U;
        }
        if (rc != 0) {
            return rc;
        }
        if (after < tick_target) {
            sleep_until_ns(tick_target);
        }
    }
    return 0;
}

void jx_cycle_stop(jx_cycle_t *cycle) {
    if (cycle != NULL) {
        cycle->stop_requested = 1;
    }
}

void jx_cycle_stats(const jx_cycle_t *cycle, jx_cycle_stats_t *out) {
    if (cycle == NULL || out == NULL) {
        return;
    }
    *out = cycle->stats;
}

size_t jx_cycle_struct_size(void) {
    return sizeof(struct jx_cycle);
}

#endif /* __linux__ || __APPLE__ */
