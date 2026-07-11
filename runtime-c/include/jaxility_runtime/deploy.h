/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * jaxility_runtime/deploy.h — on-target deployment launcher (T-032).
 *
 * The deployment glue that ties the on-target runtime to a
 * cross-compiled controller. At deploy time the controller is a
 * shared object (produced by jaxility.builder_cross.cross_build_for_target,
 * linked against the cross-built acados / blasfeo / hpipm archives).
 * The launcher:
 *
 *   1. initialises a bump arena over a caller-owned slab (arena.h);
 *   2. optionally pins the calling thread to a core and raises it to
 *      SCHED_FIFO (rt.h) — best-effort, see below;
 *   3. dlopen()s the controller .so and resolves the deployment ABI
 *      symbols below;
 *   4. calls the controller's init with the arena (this is how the
 *      arena is "threaded into" the controller — the controller
 *      bump-allocates its acados working set from it, honouring
 *      invariant 4 / no-malloc-in-loop);
 *   5. drives the controller's step at the controller-declared period
 *      via the cycle scheduler (cycle.h) until the step returns
 *      non-zero or jx_cycle_stop fires.
 *
 * The controller .so MUST export these three symbols (resolved by
 * name via dlsym). They form the deployment ABI between Jaxility's
 * launcher and a generated controller; for the acados deployment the
 * thin shim wrapping acados_create / acados_solve / acados_get is
 * generated alongside the controller C (T-034 territory).
 */
#ifndef JAXILITY_RUNTIME_DEPLOY_H
#define JAXILITY_RUNTIME_DEPLOY_H

#include "jaxility_runtime/arena.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Controller-side ABI (the .so exports these). */

/* Initialise the controller. The launcher passes the arena so the
 * controller bump-allocates its working set from it (no malloc in the
 * loop). Returns 0 on success, non-zero to abort deployment. */
typedef int (*jx_controller_init_fn)(jx_arena_t *arena);

/* Execute one control step (one acados solve in the real deployment).
 * Returns 0 to continue, non-zero to stop the control loop cleanly. */
typedef int (*jx_controller_step_fn)(void);

/* The control period in nanoseconds the controller wants the loop to
 * run at (1e6 for the 1 kHz Pi 5 cadence). Must be non-zero. */
typedef uint64_t (*jx_controller_period_ns_fn)(void);

/* Canonical exported symbol names the controller .so must define. */
#define JX_CONTROLLER_INIT_SYMBOL "jx_controller_init"
#define JX_CONTROLLER_STEP_SYMBOL "jx_controller_step"
#define JX_CONTROLLER_PERIOD_SYMBOL "jx_controller_period_ns"

/* Launcher configuration. */
typedef struct {
    const char *so_path; /* path to the controller shared object */
    int core;            /* CPU to pin to; < 0 => do not pin */
    int priority;        /* SCHED_FIFO priority; < 0 => do not raise */
    void *arena_base;    /* caller-owned slab backing the arena */
    size_t arena_capacity;
} jx_deploy_config_t;

/* Return codes. 0 on a clean stop; negative on a setup failure, each
 * distinct so the deployer can tell where start-up broke (loud
 * failure, invariant 7). RT-placement failures are NOT fatal unless
 * they indicate a programmer error (a bad core/priority argument):
 * a privilege or unsupported-platform denial leaves the loop running
 * without the real-time guarantee (PREEMPT_RT is soft anyway), with a
 * diagnostic on stderr. */
#define JX_DEPLOY_OK 0
#define JX_DEPLOY_ERR_CONFIG (-1)   /* NULL cfg / so_path / arena */
#define JX_DEPLOY_ERR_RT_ARG (-2)   /* bad core / priority argument */
#define JX_DEPLOY_ERR_DLOPEN (-3)   /* controller .so failed to load */
#define JX_DEPLOY_ERR_DLSYM (-4)    /* a required ABI symbol is missing */
#define JX_DEPLOY_ERR_INIT (-5)     /* controller init returned non-zero */
#define JX_DEPLOY_ERR_PERIOD (-6)   /* controller declared a zero period */
#define JX_DEPLOY_ERR_CYCLE (-7)    /* scheduler init failed */
#define JX_DEPLOY_ERR_UNSUPPORTED (-8) /* dlopen unavailable on this build */

/* Load + run the controller per ``cfg``. Blocks in the control loop
 * until the controller's step returns non-zero or jx_cycle_stop is
 * called — both are clean stops and yield JX_DEPLOY_OK. A non-zero
 * return is always a negative JX_DEPLOY_ERR_* setup failure. */
int jx_deploy_run(const jx_deploy_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* JAXILITY_RUNTIME_DEPLOY_H */
