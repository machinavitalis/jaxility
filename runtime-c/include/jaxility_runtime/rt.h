/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * jaxility_runtime/rt.h — PREEMPT_RT thread placement + priority (T-032).
 *
 * The Pi 5 control loop (the cycle scheduler in cycle.h) must run on a
 * dedicated core at real-time priority for the 1 kHz cadence to hold
 * its jitter budget under PREEMPT_RT. This header exposes the two
 * placement primitives the deployment runtime calls once, at start, on
 * the control thread before entering jx_cycle_run:
 *
 *   1. jx_rt_pin_to_core            — sched_setaffinity to a single core
 *                                     (pair with kernel `isolcpus=` so
 *                                     nothing else is scheduled there).
 *   2. jx_rt_set_realtime_priority  — SCHED_FIFO at a fixed priority.
 *
 * jx_rt_configure does both in the required order (affinity, then
 * priority). These are the helpers named in the PI5 quirk
 * `preempt-rt-soft-not-hard`: PREEMPT_RT is *soft* real-time, so
 * placement is necessary-but-not-sufficient — it removes the
 * scheduler-contention class of jitter, not the worst-case.
 *
 * Platform: the real implementation is Linux/glibc (sched_setaffinity,
 * cpu_set_t, SCHED_FIFO). On non-Linux hosts (the macOS dev box) the
 * functions compile but return JX_RT_ERR_UNSUPPORTED — Jaxility's RT
 * story is PREEMPT_RT Linux, mirroring how the rest of the runtime
 * verifies on CI Linux runners rather than the dev host. Argument
 * validation happens before the platform guard, so bad inputs are
 * rejected everywhere.
 *
 * Failure is loud (invariant 7): every function returns a negative
 * JX_RT_ERR_* code; none silently no-ops. Setting SCHED_FIFO requires
 * CAP_SYS_NICE / root — an unprivileged caller gets
 * JX_RT_ERR_PRIVILEGE, distinct from JX_RT_ERR_UNSUPPORTED, so the
 * deployer can tell "wrong OS" from "needs privilege".
 */
#ifndef JAXILITY_RUNTIME_RT_H
#define JAXILITY_RUNTIME_RT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. JX_RT_OK is 0; all errors are negative so a caller can
 * test ``rc < 0``. SCHED_FIFO priorities are 1..99 on Linux, so the
 * negative sentinels never collide with a real priority value returned
 * by jx_rt_priority_min / jx_rt_priority_max. */
#define JX_RT_OK 0
#define JX_RT_ERR_ARG (-1)         /* invalid core / priority argument */
#define JX_RT_ERR_UNSUPPORTED (-2) /* platform lacks the facility (non-Linux) */
#define JX_RT_ERR_PRIVILEGE (-3)   /* EPERM — needs CAP_SYS_NICE / root */
#define JX_RT_ERR_SYSTEM (-4)      /* other OS error from the syscall */

/* Pin the calling thread to a single CPU core via sched_setaffinity.
 * ``core`` is a 0-based logical CPU index. Returns JX_RT_OK on success,
 * JX_RT_ERR_ARG if ``core`` is negative or >= the online CPU count,
 * JX_RT_ERR_PRIVILEGE on EPERM, JX_RT_ERR_SYSTEM on any other error,
 * or JX_RT_ERR_UNSUPPORTED off Linux. */
int jx_rt_pin_to_core(int core);

/* Switch the calling thread to SCHED_FIFO at ``priority`` via
 * pthread_setschedparam. ``priority`` must lie in
 * [jx_rt_priority_min(), jx_rt_priority_max()]. Returns JX_RT_OK,
 * JX_RT_ERR_ARG (out of range), JX_RT_ERR_PRIVILEGE (EPERM),
 * JX_RT_ERR_SYSTEM, or JX_RT_ERR_UNSUPPORTED off Linux. */
int jx_rt_set_realtime_priority(int priority);

/* Pin to ``core`` then raise to SCHED_FIFO ``priority``. Order matters:
 * affinity is applied first so the priority change takes effect on the
 * destination core. On partial failure the first failing step's code
 * is returned; note that if affinity succeeded but priority failed,
 * the affinity is already applied (there is no rollback — the deployer
 * treats a non-OK return as fatal and aborts start-up). */
int jx_rt_configure(int core, int priority);

/* Minimum / maximum valid SCHED_FIFO priority on this platform
 * (sched_get_priority_min/max). Returns JX_RT_ERR_UNSUPPORTED off
 * Linux. On Linux these are 1 and 99 respectively. */
int jx_rt_priority_min(void);
int jx_rt_priority_max(void);

/* The logical CPU the calling thread is currently running on
 * (sched_getcpu), for verifying a pin took. Returns the 0-based core
 * index (>= 0) or JX_RT_ERR_UNSUPPORTED off Linux. */
int jx_rt_current_core(void);

#ifdef __cplusplus
}
#endif

#endif /* JAXILITY_RUNTIME_RT_H */
