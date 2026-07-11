/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * dcache_thumb.c — Cortex-M D-cache / I-cache coherence (T-052).
 *
 * Builds only for ``arm-none-eabi-gcc`` targeting Thumb / M-profile.
 *
 * Cortex-M3 / M4 / M4F have unified bus topology and no separate
 * D-cache / I-cache — the read-after-write coherence problem the
 * Cortex-A version solves does not arise. The call collapses to a
 * memory barrier sequence (``DSB`` + ``ISB``) so the contract is
 * preserved: after this returns, every write through ``addr`` is
 * visible to the next instruction fetch.
 *
 * Cortex-M7 / M33 / M55 *do* have optional L1 caches; the runtime
 * deployment on those parts must enable them via SCB->CCR.IC / .DC
 * at boot, and the cache-maintenance ops live in ``cmsis_armcc.h``.
 * The Ethos-U55/U65 targets inherit the M55 host; the cache
 * maintenance for those targets is a follow-up that lands with the
 * first M55-host runtime PR.
 */
#if defined(__ARM_ARCH) && !defined(__aarch64__)

#include "jaxility_runtime/dcache.h"

#include <stddef.h>

int jx_dcache_clean_for_exec(const void *addr, size_t len) {
    if (addr == NULL || len == 0U) {
        return -1;
    }
    /* DSB ensures all explicit memory accesses complete before the
     * next instruction; ISB flushes the pipeline so the next fetch
     * sees the post-DSB state. */
    __asm__ volatile ("dsb 0xF" ::: "memory");
    __asm__ volatile ("isb 0xF" ::: "memory");
    return 0;
}

#endif /* __ARM_ARCH && !__aarch64__ */
