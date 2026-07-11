/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * deploy_main.c — the on-target launcher entry point (T-032).
 *
 * This is the ``main`` compiled into the deployment launcher binary
 * (jx_deploy_<family>). It is intentionally NOT part of the runtime
 * archive (libjaxility_runtime_<family>.a) — an archive carrying a
 * ``main`` symbol would collide when linked into anything else.
 * jaxility.runtime.deploy.plan_deploy_launcher compiles this source
 * against the archive to produce the launcher.
 *
 * Usage on the Pi:
 *   jx_deploy_cortex-a76 <controller.so> [core] [priority]
 *
 * The controller .so is the cross-compiled acados controller. It is
 * dlopen()ed at run time, so the launcher binary itself does not link
 * against acados — only the runtime archive (arena + cycle + rt +
 * dlopen glue). The launcher is built with -rdynamic so the
 * controller can resolve runtime symbols (e.g. jx_arena_alloc) from
 * the launcher's exported symbol table.
 */
#include "jaxility_runtime/deploy.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Static slab backing the deployment arena. 4 MiB is generous for the
 * acados working set of the Pi 5 launch controllers; override at
 * compile time with -DJX_DEPLOY_ARENA_BYTES=<n>. */
#ifndef JX_DEPLOY_ARENA_BYTES
#define JX_DEPLOY_ARENA_BYTES (4U * 1024U * 1024U)
#endif

static unsigned char jx_deploy_arena_slab[JX_DEPLOY_ARENA_BYTES];

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <controller.so> [core] [priority]\n",
                argv[0]);
        return 2;
    }

    jx_deploy_config_t cfg;
    cfg.so_path = argv[1];
    cfg.core = (argc > 2) ? atoi(argv[2]) : -1;
    cfg.priority = (argc > 3) ? atoi(argv[3]) : -1;
    cfg.arena_base = jx_deploy_arena_slab;
    cfg.arena_capacity = sizeof(jx_deploy_arena_slab);

    int rc = jx_deploy_run(&cfg);
    if (rc != JX_DEPLOY_OK) {
        fprintf(stderr, "jx_deploy: exited with code %d\n", rc);
    }
    return rc;
}
