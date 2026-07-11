/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * arena.c — portable bump-pointer arena (T-032 / T-052).
 *
 * Builds on Cortex-A (aarch64-none-linux-gnu) and Cortex-M
 * (arm-none-eabi). No platform-specific code; all logic is plain
 * C99 + <stddef.h> / <stdint.h>. Verified by the unit-test program
 * in runtime-c/test/test_arena.c which exercises every reachable
 * branch.
 */
#include "jaxility_runtime/arena.h"

#include <stddef.h>
#include <stdint.h>

static int is_power_of_two(size_t x) {
    return x != 0U && (x & (x - 1U)) == 0U;
}

static uintptr_t align_up(uintptr_t value, size_t align) {
    /* Caller must ensure ``align`` is a power of two — the sole
     * callsite verifies this beforehand. Operates on an absolute
     * address so the rounding accounts for the base buffer's own
     * alignment, not just the offset within it. */
    return (value + ((uintptr_t)align - 1U)) & ~((uintptr_t)align - 1U);
}

int jx_arena_init(jx_arena_t *arena, void *base, size_t capacity) {
    if (arena == NULL || base == NULL || capacity == 0U) {
        return -1;
    }
    arena->base     = (uint8_t *)base;
    arena->capacity = capacity;
    arena->used     = 0U;
    return 0;
}

void *jx_arena_alloc(jx_arena_t *arena, size_t size) {
    return jx_arena_alloc_aligned(arena, size, JX_ARENA_DEFAULT_ALIGN);
}

void *jx_arena_alloc_aligned(jx_arena_t *arena, size_t size, size_t align) {
    if (arena == NULL || size == 0U || !is_power_of_two(align)) {
        return NULL;
    }
    /* Align the *absolute* address ``base + used`` rather than the
     * offset ``used`` alone: the returned pointer must satisfy
     * ``align`` no matter how the caller's base buffer is itself
     * aligned. Aligning only the offset silently under-aligns the
     * result whenever ``base`` is less aligned than the request —
     * which broke 64-byte (cache-line) allocations on Cortex-A76,
     * where the base buffer was only 16-aligned. */
    uintptr_t base_addr     = (uintptr_t)arena->base;
    size_t    aligned_used  = (size_t)(align_up(base_addr + arena->used, align) - base_addr);
    /* Overflow check: aligned_used + size must fit. ``aligned_used``
     * already includes any leading pad; check via subtraction against
     * capacity. */
    if (aligned_used > arena->capacity ||
        size > arena->capacity - aligned_used) {
        return NULL;
    }
    void *result   = (void *)(arena->base + aligned_used);
    arena->used    = aligned_used + size;
    return result;
}

void jx_arena_reset(jx_arena_t *arena) {
    if (arena != NULL) {
        arena->used = 0U;
    }
}

size_t jx_arena_available(const jx_arena_t *arena) {
    if (arena == NULL) {
        return 0U;
    }
    return arena->capacity - arena->used;
}
