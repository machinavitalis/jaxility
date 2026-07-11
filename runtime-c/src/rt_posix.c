/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * rt_posix.c — PREEMPT_RT thread placement + priority (T-032).
 *
 * Implements jaxility_runtime/rt.h. Used by:
 *   - Pi 5 deployment (Cortex-A76 + PREEMPT_RT Linux): the control
 *     thread pins to an isolated core and raises to SCHED_FIFO before
 *     entering jx_cycle_run.
 *   - CI Linux x86_64 runners (the host test exercises the real Linux
 *     path; the Tier-B cross-compile verifies it builds for
 *     aarch64-linux-gnu).
 *   - macOS dev host: compiles, but the syscalls are Linux-only, so
 *     the functions return JX_RT_ERR_UNSUPPORTED there.
 *
 * _GNU_SOURCE is required for sched_setaffinity / cpu_set_t /
 * sched_getcpu (glibc). Defined here (not via a build flag) so the
 * file is self-contained like cycle_posix.c.
 */
#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "jaxility_runtime/rt.h"

#if defined(__linux__)

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

int jx_rt_pin_to_core(int core) {
    if (core < 0) {
        return JX_RT_ERR_ARG;
    }
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0 && core >= (int)online) {
        return JX_RT_ERR_ARG;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        return (errno == EPERM) ? JX_RT_ERR_PRIVILEGE : JX_RT_ERR_SYSTEM;
    }
    return JX_RT_OK;
}

int jx_rt_priority_min(void) { return sched_get_priority_min(SCHED_FIFO); }

int jx_rt_priority_max(void) { return sched_get_priority_max(SCHED_FIFO); }

int jx_rt_set_realtime_priority(int priority) {
    int lo = sched_get_priority_min(SCHED_FIFO);
    int hi = sched_get_priority_max(SCHED_FIFO);
    if (priority < lo || priority > hi) {
        return JX_RT_ERR_ARG;
    }

    struct sched_param param;
    param.sched_priority = priority;
    /* pthread_setschedparam returns an errno value (0 on success), not
     * -1/errno like the raw syscalls. */
    int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    if (rc != 0) {
        return (rc == EPERM) ? JX_RT_ERR_PRIVILEGE : JX_RT_ERR_SYSTEM;
    }
    return JX_RT_OK;
}

int jx_rt_current_core(void) {
    int core = sched_getcpu();
    return (core < 0) ? JX_RT_ERR_SYSTEM : core;
}

#else /* non-Linux: compile, but the facility is absent. */

int jx_rt_pin_to_core(int core) {
    /* Argument validation still applies everywhere. */
    if (core < 0) {
        return JX_RT_ERR_ARG;
    }
    return JX_RT_ERR_UNSUPPORTED;
}

int jx_rt_priority_min(void) { return JX_RT_ERR_UNSUPPORTED; }

int jx_rt_priority_max(void) { return JX_RT_ERR_UNSUPPORTED; }

int jx_rt_set_realtime_priority(int priority) {
    (void)priority;
    return JX_RT_ERR_UNSUPPORTED;
}

int jx_rt_current_core(void) { return JX_RT_ERR_UNSUPPORTED; }

#endif /* __linux__ */

int jx_rt_configure(int core, int priority) {
    int rc = jx_rt_pin_to_core(core);
    if (rc != JX_RT_OK) {
        return rc;
    }
    return jx_rt_set_realtime_priority(priority);
}
