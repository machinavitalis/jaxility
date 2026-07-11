/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * test_rt.c — host-runnable RT placement tests (T-032).
 *
 * Compiled by the Python test driver with the host ``cc`` (link with
 * -pthread), not the cross toolchain. The behaviour is platform-split:
 *
 *   - On Linux (CI x86_64 runners; the same code path that runs on Pi 5
 *     PREEMPT_RT): the real sched_setaffinity / SCHED_FIFO calls run.
 *     Unprivileged, raising SCHED_FIFO returns JX_RT_ERR_PRIVILEGE; the
 *     test accepts {OK, PRIVILEGE} so it passes whether or not the
 *     runner grants CAP_SYS_NICE.
 *   - On non-Linux (macOS dev host): every facility returns
 *     JX_RT_ERR_UNSUPPORTED. Argument validation still rejects bad
 *     input first.
 *
 * The #ifdef branches keep the assertions meaningful on both.
 */
#include "jaxility_runtime/rt.h"

#include <assert.h>
#include <stdio.h>

#define RUN(test_fn)            \
    do {                        \
        printf("  " #test_fn " ... "); \
        fflush(stdout);         \
        test_fn();              \
        printf("ok\n");         \
    } while (0)

/* Argument validation is platform-independent: a negative core is
 * rejected before any syscall. */
static void test_pin_rejects_negative_core(void) {
    assert(jx_rt_pin_to_core(-1) == JX_RT_ERR_ARG);
}

#if defined(__linux__)

static void test_priority_bounds_are_ordered(void) {
    int lo = jx_rt_priority_min();
    int hi = jx_rt_priority_max();
    assert(lo > 0);
    assert(hi > lo);
}

static void test_priority_out_of_range_is_arg_error(void) {
    int hi = jx_rt_priority_max();
    assert(jx_rt_set_realtime_priority(hi + 1) == JX_RT_ERR_ARG);
}

static void test_pin_to_core_zero_ok_or_privilege(void) {
    /* Core 0 always exists. Unprivileged sched_setaffinity to an
     * allowed CPU usually succeeds; treat a privilege denial as an
     * acceptable environment-dependent outcome. */
    int rc = jx_rt_pin_to_core(0);
    assert(rc == JX_RT_OK || rc == JX_RT_ERR_PRIVILEGE);
}

static void test_set_realtime_priority_ok_or_privilege(void) {
    int rc = jx_rt_set_realtime_priority(jx_rt_priority_min());
    assert(rc == JX_RT_OK || rc == JX_RT_ERR_PRIVILEGE);
}

static void test_configure_ok_or_privilege(void) {
    int rc = jx_rt_configure(0, jx_rt_priority_min());
    assert(rc == JX_RT_OK || rc == JX_RT_ERR_PRIVILEGE);
}

static void test_current_core_is_valid(void) {
    int core = jx_rt_current_core();
    assert(core >= 0);
}

#else /* non-Linux */

static void test_priority_min_unsupported(void) {
    assert(jx_rt_priority_min() == JX_RT_ERR_UNSUPPORTED);
}

static void test_pin_unsupported_off_linux(void) {
    /* Valid argument, but the platform lacks the facility. */
    assert(jx_rt_pin_to_core(0) == JX_RT_ERR_UNSUPPORTED);
}

static void test_set_priority_unsupported_off_linux(void) {
    assert(jx_rt_set_realtime_priority(1) == JX_RT_ERR_UNSUPPORTED);
}

static void test_configure_unsupported_off_linux(void) {
    assert(jx_rt_configure(0, 1) == JX_RT_ERR_UNSUPPORTED);
}

static void test_current_core_unsupported_off_linux(void) {
    assert(jx_rt_current_core() == JX_RT_ERR_UNSUPPORTED);
}

#endif /* __linux__ */

int main(void) {
    RUN(test_pin_rejects_negative_core);
#if defined(__linux__)
    RUN(test_priority_bounds_are_ordered);
    RUN(test_priority_out_of_range_is_arg_error);
    RUN(test_pin_to_core_zero_ok_or_privilege);
    RUN(test_set_realtime_priority_ok_or_privilege);
    RUN(test_configure_ok_or_privilege);
    RUN(test_current_core_is_valid);
    printf("all 7 tests passed.\n");
#else
    RUN(test_priority_min_unsupported);
    RUN(test_pin_unsupported_off_linux);
    RUN(test_set_priority_unsupported_off_linux);
    RUN(test_configure_unsupported_off_linux);
    RUN(test_current_core_unsupported_off_linux);
    printf("all 6 tests passed.\n");
#endif
    return 0;
}
