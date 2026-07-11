/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * test_cycle_posix.c — host-runnable cycle scheduler tests (M-8).
 *
 * Closes audit finding M-8: the POSIX cycle scheduler had ZERO
 * functional tests before this file. The 1-tick / N-tick / stop /
 * overrun / slack-vs-jitter cases exercised here run on every POSIX
 * host (Linux + macOS), reusing the same arena.c contract for the
 * scheduler's backing memory.
 *
 * Compiled by the Python test driver with the host ``cc``, not the
 * cross toolchain — the POSIX implementation is what runs on Pi 5
 * PREEMPT_RT in production and on x86_64 Linux runners in CI, so the
 * host test exercises the same code path.
 */
#define _POSIX_C_SOURCE 200809L
#include "jaxility_runtime/arena.h"
#include "jaxility_runtime/cycle.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define RUN(test_fn) do { \
    printf("  " #test_fn " ... "); \
    fflush(stdout); \
    test_fn(); \
    printf("ok\n"); \
} while (0)

static uint8_t arena_slab[256];

static jx_cycle_t *new_scheduler(void) {
    static jx_arena_t arena;
    jx_arena_init(&arena, arena_slab, sizeof(arena_slab));
    return (jx_cycle_t *)jx_arena_alloc(&arena, jx_cycle_struct_size());
}

typedef struct {
    int count;
    int stop_at;
    jx_cycle_t *cycle;
    int return_code;
} tick_ctx_t;

static int counting_tick(void *user_ctx) {
    tick_ctx_t *ctx = (tick_ctx_t *)user_ctx;
    ctx->count += 1;
    if (ctx->stop_at > 0 && ctx->count >= ctx->stop_at) {
        jx_cycle_stop(ctx->cycle);
    }
    return ctx->return_code;
}

static int overrunning_tick(void *user_ctx) {
    tick_ctx_t *ctx = (tick_ctx_t *)user_ctx;
    ctx->count += 1;
    /* Sleep longer than the cycle period to force an overrun. */
    struct timespec ts = {0, 5 * 1000 * 1000}; /* 5 ms */
    nanosleep(&ts, NULL);
    if (ctx->stop_at > 0 && ctx->count >= ctx->stop_at) {
        jx_cycle_stop(ctx->cycle);
    }
    return 0;
}

static void test_init_rejects_null_cycle(void) {
    assert(jx_cycle_init(NULL, 1000U, counting_tick, NULL) == -1);
}

static void test_init_rejects_zero_period(void) {
    jx_cycle_t *c = new_scheduler();
    assert(jx_cycle_init(c, 0U, counting_tick, NULL) == -1);
}

static void test_init_rejects_null_tick(void) {
    jx_cycle_t *c = new_scheduler();
    assert(jx_cycle_init(c, 1000U, NULL, NULL) == -1);
}

static void test_run_uninitialised_returns_minus_one(void) {
    /* Arena memory is zeroed by jx_arena_init implicitly via the
     * slab; ``initialised`` defaults to 0 -> run returns -1. */
    static uint8_t slab[256];
    static jx_arena_t a;
    jx_arena_init(&a, slab, sizeof(slab));
    /* Zero the slab so cycle->initialised is 0. */
    for (size_t i = 0; i < sizeof(slab); i++) slab[i] = 0;
    jx_cycle_t *c = (jx_cycle_t *)jx_arena_alloc(&a, jx_cycle_struct_size());
    assert(jx_cycle_run(c) == -1);
}

static void test_run_invokes_tick_n_times(void) {
    jx_cycle_t *c = new_scheduler();
    tick_ctx_t ctx = {0, 5, c, 0};
    assert(jx_cycle_init(c, 1000U, counting_tick, &ctx) == 0);
    assert(jx_cycle_run(c) == 0);
    assert(ctx.count == 5);
    jx_cycle_stats_t stats;
    jx_cycle_stats(c, &stats);
    assert(stats.ticks == 5U);
}

static void test_tick_nonzero_return_terminates_loop(void) {
    jx_cycle_t *c = new_scheduler();
    tick_ctx_t ctx = {0, 0, c, 42};
    assert(jx_cycle_init(c, 1000U, counting_tick, &ctx) == 0);
    assert(jx_cycle_run(c) == 42);
    /* Tick fired once, returned 42, loop exited. */
    assert(ctx.count == 1);
}

static void test_stop_from_tick_exits_after_current_cycle(void) {
    jx_cycle_t *c = new_scheduler();
    tick_ctx_t ctx = {0, 3, c, 0};
    assert(jx_cycle_init(c, 1000U, counting_tick, &ctx) == 0);
    assert(jx_cycle_run(c) == 0);
    assert(ctx.count == 3);
}

static void test_overrun_is_counted_and_doesnt_compound(void) {
    /* 1 ms period, tick sleeps 5 ms → every tick overruns. With drift
     * correction the loop schedules tick N+1 at start + (N+1)*period,
     * not at end_of_tick_N + period, so the overrun count == ticks. */
    jx_cycle_t *c = new_scheduler();
    tick_ctx_t ctx = {0, 3, c, 0};
    assert(jx_cycle_init(c, 1000U * 1000U, overrunning_tick, &ctx) == 0);
    assert(jx_cycle_run(c) == 0);
    jx_cycle_stats_t stats;
    jx_cycle_stats(c, &stats);
    assert(stats.ticks == 3U);
    /* Every tick overran (5 ms body, 1 ms period). */
    assert(stats.overruns == 3U);
}

static void test_slack_is_nonzero_for_fast_ticks(void) {
    /* 10 ms period, ~no-op tick → slack ≈ period. */
    jx_cycle_t *c = new_scheduler();
    tick_ctx_t ctx = {0, 3, c, 0};
    assert(jx_cycle_init(c, 10U * 1000U * 1000U, counting_tick, &ctx) == 0);
    assert(jx_cycle_run(c) == 0);
    jx_cycle_stats_t stats;
    jx_cycle_stats(c, &stats);
    assert(stats.ticks == 3U);
    /* Slack accumulated; M-9 semantics: slack > 0 for fast ticks. */
    assert(stats.slack_sum_ns > 0U);
    assert(stats.slack_max_ns > 0U);
}

static void test_jitter_is_bounded_under_fast_ticks(void) {
    /* For ~no-op ticks at 5 ms period, scheduling jitter (measured
     * after the sleep returns, M-9) should be far below the period. */
    jx_cycle_t *c = new_scheduler();
    tick_ctx_t ctx = {0, 4, c, 0};
    uint64_t period = 5U * 1000U * 1000U; /* 5 ms */
    assert(jx_cycle_init(c, period, counting_tick, &ctx) == 0);
    assert(jx_cycle_run(c) == 0);
    jx_cycle_stats_t stats;
    jx_cycle_stats(c, &stats);
    /* Allow generous bound (CI hosts can be slow); the structural
     * point is that real jitter is far less than the period under
     * normal load. */
    assert(stats.jitter_max_ns < period);
}

static void test_struct_size_is_nonzero(void) {
    /* Defensive: arena alloc downstream relies on this. */
    assert(jx_cycle_struct_size() > 0U);
}

static void test_null_stats_call_is_safe(void) {
    jx_cycle_t *c = new_scheduler();
    /* Should not segfault. */
    jx_cycle_stats(c, NULL);
    jx_cycle_stats(NULL, NULL);
}

int main(void) {
    printf("cycle_posix tests:\n");
    RUN(test_struct_size_is_nonzero);
    RUN(test_init_rejects_null_cycle);
    RUN(test_init_rejects_zero_period);
    RUN(test_init_rejects_null_tick);
    RUN(test_run_uninitialised_returns_minus_one);
    RUN(test_run_invokes_tick_n_times);
    RUN(test_tick_nonzero_return_terminates_loop);
    RUN(test_stop_from_tick_exits_after_current_cycle);
    RUN(test_overrun_is_counted_and_doesnt_compound);
    RUN(test_slack_is_nonzero_for_fast_ticks);
    RUN(test_jitter_is_bounded_under_fast_ticks);
    RUN(test_null_stats_call_is_safe);
    printf("all 12 tests passed.\n");
    return 0;
}
