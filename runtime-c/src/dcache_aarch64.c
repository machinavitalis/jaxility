/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * dcache_aarch64.c — Cortex-A D-cache / I-cache coherence (T-032).
 *
 * Builds only for ``aarch64-none-linux-gnu-gcc``. The cache-line
 * size is read at runtime from ``CTR_EL0`` so the impl is correct on
 * any aarch64 implementation (Cortex-A76's 64-byte L1 D-cache line
 * happens to match the baseline but the runtime-detect path is the
 * portable thing).
 *
 * Sequence per ARM ARM B2.2.6:
 *   1. ``DC CIVAC, addr`` — clean line to point of coherency.
 *   2. ``DSB ISH``         — wait for the clean to complete in the
 *                            inner shareable domain (covers SMP).
 *   3. ``IC IVAU, addr``   — invalidate the I-cache line.
 *   4. ``DSB ISH``         — wait for the invalidate.
 *   5. ``ISB``             — flush the prefetch pipeline.
 */
#if defined(__aarch64__)

#include "jaxility_runtime/dcache.h"

#include <stddef.h>
#include <stdint.h>

static size_t read_cache_line_bytes(void) {
    /* CTR_EL0.DminLine and IminLine encode log2(line size / 4). The
     * effective coherency line for cache-maintenance instructions is
     * max(D, I). The encoding is stable across A76 (4 → 64 bytes). */
    uint64_t ctr;
    __asm__ volatile ("mrs %0, ctr_el0" : "=r"(ctr));
    uint32_t d_log2 = (uint32_t)((ctr >> 16) & 0xFU);
    uint32_t i_log2 = (uint32_t)(ctr & 0xFU);
    uint32_t log2   = d_log2 > i_log2 ? d_log2 : i_log2;
    return (size_t)(4U << log2);
}

int jx_dcache_clean_for_exec(const void *addr, size_t len) {
    if (addr == NULL || len == 0U) {
        return -1;
    }
    size_t line  = read_cache_line_bytes();
    uintptr_t start = (uintptr_t)addr;
    uintptr_t end   = start + len;
    /* Align start down to the line boundary. */
    start &= ~((uintptr_t)line - 1U);

    for (uintptr_t p = start; p < end; p += line) {
        __asm__ volatile ("dc civac, %0" : : "r"(p) : "memory");
    }
    __asm__ volatile ("dsb ish" ::: "memory");

    for (uintptr_t p = start; p < end; p += line) {
        __asm__ volatile ("ic ivau, %0" : : "r"(p) : "memory");
    }
    __asm__ volatile ("dsb ish" ::: "memory");
    __asm__ volatile ("isb" ::: "memory");
    return 0;
}

#endif /* __aarch64__ */
