/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * deploy_posix.c — on-target deployment launcher (T-032).
 *
 * Implements jaxility_runtime/deploy.h: load the cross-compiled
 * controller .so, thread the arena into it, place the thread for
 * real-time, and drive the controller step from the cycle scheduler.
 *
 * Used by:
 *   - Pi 5 deployment (Cortex-A76 + PREEMPT_RT Linux). The launcher
 *     binary links this against libjaxility_runtime_cortex-a76.a and
 *     dlopen()s the controller at run time.
 *   - The host test driver (test_deploy.c via cc), which drives
 *     jx_deploy_run against a fake controller .so to verify the
 *     dlopen / arena-threading / cycle-loop wiring without hardware.
 *
 * dlfcn (dlopen/dlsym) is POSIX; present on Linux and macOS. On a
 * build without it the function returns JX_DEPLOY_ERR_UNSUPPORTED.
 */
#include "jaxility_runtime/deploy.h"

#include "jaxility_runtime/arena.h"
#include "jaxility_runtime/cycle.h"
#include "jaxility_runtime/rt.h"

#include <stddef.h>

#if defined(__linux__) || defined(__APPLE__)

#include <dlfcn.h>
#include <stdio.h>

/* Tick adapter: the cycle scheduler calls this with our context; we
 * forward to the controller's step. Wrapping the step fn in a struct
 * (rather than casting it to void*) keeps the function-pointer /
 * object-pointer separation clean. */
typedef struct {
    jx_controller_step_fn step;
} jx_deploy_tick_ctx_t;

static int jx_deploy_tick(void *user_ctx) {
    jx_deploy_tick_ctx_t *ctx = (jx_deploy_tick_ctx_t *)user_ctx;
    return ctx->step();
}

/* Best-effort RT placement. A bad argument is a programmer error and
 * is fatal; a privilege / unsupported denial is tolerated (the loop
 * still runs, just without the RT guarantee). */
static int jx_deploy_place_thread(int core, int priority) {
    if (core >= 0) {
        int rc = jx_rt_pin_to_core(core);
        if (rc == JX_RT_ERR_ARG) {
            return JX_DEPLOY_ERR_RT_ARG;
        }
        if (rc != JX_RT_OK) {
            fprintf(stderr,
                    "jx_deploy: core pin to %d not applied (rc=%d); "
                    "continuing without affinity.\n",
                    core, rc);
        }
    }
    if (priority >= 0) {
        int rc = jx_rt_set_realtime_priority(priority);
        if (rc == JX_RT_ERR_ARG) {
            return JX_DEPLOY_ERR_RT_ARG;
        }
        if (rc != JX_RT_OK) {
            fprintf(stderr,
                    "jx_deploy: SCHED_FIFO priority %d not applied (rc=%d); "
                    "continuing at default scheduling.\n",
                    priority, rc);
        }
    }
    return JX_DEPLOY_OK;
}

int jx_deploy_run(const jx_deploy_config_t *cfg) {
    if (cfg == NULL || cfg->so_path == NULL || cfg->arena_base == NULL ||
        cfg->arena_capacity == 0U) {
        return JX_DEPLOY_ERR_CONFIG;
    }

    jx_arena_t arena;
    if (jx_arena_init(&arena, cfg->arena_base, cfg->arena_capacity) != 0) {
        return JX_DEPLOY_ERR_CONFIG;
    }

    int place_rc = jx_deploy_place_thread(cfg->core, cfg->priority);
    if (place_rc != JX_DEPLOY_OK) {
        return place_rc;
    }

    /* RTLD_NOW so missing symbols surface here, not mid-loop;
     * RTLD_LOCAL so the controller's symbols don't leak globally. */
    void *handle = dlopen(cfg->so_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "jx_deploy: dlopen(%s) failed: %s\n", cfg->so_path,
                dlerror());
        return JX_DEPLOY_ERR_DLOPEN;
    }

    /* POSIX dlsym returns void*; the documented idiom for function
     * symbols is to copy through a void* and cast. */
    jx_controller_init_fn init_fn =
        (jx_controller_init_fn)dlsym(handle, JX_CONTROLLER_INIT_SYMBOL);
    jx_controller_step_fn step_fn =
        (jx_controller_step_fn)dlsym(handle, JX_CONTROLLER_STEP_SYMBOL);
    jx_controller_period_ns_fn period_fn =
        (jx_controller_period_ns_fn)dlsym(handle, JX_CONTROLLER_PERIOD_SYMBOL);
    if (init_fn == NULL || step_fn == NULL || period_fn == NULL) {
        fprintf(stderr,
                "jx_deploy: controller .so missing a required ABI symbol "
                "(init=%p step=%p period=%p)\n", (void *)init_fn,
                (void *)step_fn, (void *)period_fn);
        dlclose(handle);
        return JX_DEPLOY_ERR_DLSYM;
    }

    if (init_fn(&arena) != 0) {
        dlclose(handle);
        return JX_DEPLOY_ERR_INIT;
    }

    uint64_t period_ns = period_fn();
    if (period_ns == 0U) {
        dlclose(handle);
        return JX_DEPLOY_ERR_PERIOD;
    }

    jx_deploy_tick_ctx_t *tick_ctx =
        (jx_deploy_tick_ctx_t *)jx_arena_alloc(&arena, sizeof(*tick_ctx));
    jx_cycle_t *cycle = (jx_cycle_t *)jx_arena_alloc(&arena, jx_cycle_struct_size());
    if (tick_ctx == NULL || cycle == NULL) {
        dlclose(handle);
        return JX_DEPLOY_ERR_CYCLE;
    }
    tick_ctx->step = step_fn;

    if (jx_cycle_init(cycle, period_ns, jx_deploy_tick, tick_ctx) != 0) {
        dlclose(handle);
        return JX_DEPLOY_ERR_CYCLE;
    }

    /* jx_cycle_run returns the tick's non-zero code (the controller's
     * "I'm done" signal — a clean stop per the deploy ABI) or 0 if
     * jx_cycle_stop fired. Either is a clean stop; only a negative
     * return means the scheduler itself faulted (NULL / uninitialised,
     * which we've already guarded). Collapse to JX_DEPLOY_OK. */
    int run_rc = jx_cycle_run(cycle);
    dlclose(handle);
    return (run_rc < 0) ? JX_DEPLOY_ERR_CYCLE : JX_DEPLOY_OK;
}

#else /* no dlfcn on this build */

int jx_deploy_run(const jx_deploy_config_t *cfg) {
    (void)cfg;
    return JX_DEPLOY_ERR_UNSUPPORTED;
}

#endif /* dlfcn availability */
