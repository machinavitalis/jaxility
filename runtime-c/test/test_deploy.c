/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * test_deploy.c — host harness for the deployment launcher (T-032).
 *
 * Drives jx_deploy_run against a fake controller .so (built by the
 * Python test driver) to verify the dlopen / dlsym / arena-threading /
 * cycle-loop wiring without hardware or acados. Takes the controller
 * .so path as argv[1]. Prints "deploy test passed." and exits 0 on a
 * clean stop.
 *
 * The fake controller's step returns non-zero after a few calls so the
 * cycle loop terminates; its init verifies it received a usable arena.
 */
#include "jaxility_runtime/deploy.h"

#include <stddef.h>
#include <stdio.h>

/* 1 MiB slab — comfortably covers the tick context + scheduler struct
 * plus the fake controller's allocation. */
static unsigned char slab[1U << 20];

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <controller.so>\n", argv[0]);
        return 2;
    }

    jx_deploy_config_t cfg;
    cfg.so_path = argv[1];
    cfg.core = -1;     /* skip RT placement — unprivileged host */
    cfg.priority = -1;
    cfg.arena_base = slab;
    cfg.arena_capacity = sizeof(slab);

    int rc = jx_deploy_run(&cfg);
    if (rc != JX_DEPLOY_OK) {
        fprintf(stderr, "deploy run returned %d\n", rc);
        return 1;
    }
    printf("deploy test passed.\n");
    return 0;
}
