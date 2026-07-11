/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * test_arena.c — host-runnable arena unit tests (T-032 / T-052).
 *
 * Compiled with the *host* compiler (gcc / clang), not the cross
 * toolchain — the arena is portable C and the contract is the same
 * on every target. The Python test driver runs this binary and
 * asserts its exit code is 0.
 */
#include "jaxility_runtime/arena.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SLAB_BYTES 1024U

static uint8_t slab[SLAB_BYTES];

#define RUN(test_fn) do { \
    printf("  " #test_fn " ... "); \
    test_fn(); \
    printf("ok\n"); \
} while (0)

static void test_init_rejects_null_arena(void) {
    int rc = jx_arena_init(NULL, slab, SLAB_BYTES);
    assert(rc == -1);
}

static void test_init_rejects_null_base(void) {
    jx_arena_t a;
    int rc = jx_arena_init(&a, NULL, SLAB_BYTES);
    assert(rc == -1);
}

static void test_init_rejects_zero_capacity(void) {
    jx_arena_t a;
    int rc = jx_arena_init(&a, slab, 0U);
    assert(rc == -1);
}

static void test_alloc_default_align_advances_pointer(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, SLAB_BYTES) == 0);
    void *p = jx_arena_alloc(&a, 32U);
    assert(p != NULL);
    /* Default alignment is 16; the first allocation is at offset 0. */
    assert(((uintptr_t)p & (JX_ARENA_DEFAULT_ALIGN - 1U)) == 0U);
    assert(a.used == 32U);
}

static void test_alloc_aligns_subsequent_calls(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, SLAB_BYTES) == 0);
    void *p1 = jx_arena_alloc(&a, 1U);
    void *p2 = jx_arena_alloc(&a, 1U);
    assert(p1 != NULL);
    assert(p2 != NULL);
    /* p2 must be aligned to 16 even though p1 was only 1 byte. */
    assert(((uintptr_t)p2 & (JX_ARENA_DEFAULT_ALIGN - 1U)) == 0U);
    assert(((uintptr_t)p2 - (uintptr_t)p1) == JX_ARENA_DEFAULT_ALIGN);
}

static void test_alloc_aligned_custom(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, SLAB_BYTES) == 0);
    void *p = jx_arena_alloc_aligned(&a, 8U, 64U);
    assert(p != NULL);
    assert(((uintptr_t)p & 63U) == 0U);
}

static void test_alloc_rejects_non_power_of_two_align(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, SLAB_BYTES) == 0);
    assert(jx_arena_alloc_aligned(&a, 8U, 3U) == NULL);
    assert(jx_arena_alloc_aligned(&a, 8U, 0U) == NULL);
}

static void test_alloc_rejects_zero_size(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, SLAB_BYTES) == 0);
    assert(jx_arena_alloc(&a, 0U) == NULL);
}

static void test_alloc_returns_null_on_exhaustion(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, 64U) == 0);
    assert(jx_arena_alloc(&a, 64U) != NULL);
    assert(jx_arena_alloc(&a, 1U) == NULL);
}

static void test_reset_rewinds_to_base(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, SLAB_BYTES) == 0);
    void *p1 = jx_arena_alloc(&a, 100U);
    jx_arena_reset(&a);
    void *p2 = jx_arena_alloc(&a, 100U);
    assert(p1 == p2);
    assert(a.used == 100U);
}

static void test_available_tracks_remaining(void) {
    jx_arena_t a;
    assert(jx_arena_init(&a, slab, 256U) == 0);
    assert(jx_arena_available(&a) == 256U);
    (void)jx_arena_alloc(&a, 16U);
    assert(jx_arena_available(&a) == 240U);
}

static void test_null_arena_returns_zero_available(void) {
    /* Defensive: ``jx_arena_available(NULL)`` returns 0 rather than
     * dereferencing. */
    assert(jx_arena_available(NULL) == 0U);
}

static void test_alloc_aligned_on_misaligned_base(void) {
    /* Regression — found on Cortex-A76 during Pi 5 bring-up. The earlier
     * implementation aligned the offset within the arena, not the
     * absolute address, so a 64-byte request on a base that is only
     * 16-aligned returned a 16-aligned pointer. The host (x86_64 / macOS)
     * happened to place the static slab on a 64-aligned address, hiding
     * the bug; the Pi's slab landed at ``addr % 64 == 16`` and exposed it.
     *
     * Construct that exact condition deliberately so the test fails on
     * every platform if the regression returns. */
    static uint8_t big[256];
    uintptr_t aligned64 = ((uintptr_t)big + 63U) & ~(uintptr_t)63U;
    uint8_t  *base      = (uint8_t *)(aligned64 + 16U); /* addr % 64 == 16 */
    assert(((uintptr_t)base & 63U) == 16U);             /* precondition holds */

    jx_arena_t a;
    assert(jx_arena_init(&a, base, 128U) == 0);
    void *p = jx_arena_alloc_aligned(&a, 8U, 64U);
    assert(p != NULL);
    assert(((uintptr_t)p & 63U) == 0U);                 /* the actual contract */
    /* base is 16 past a 64 boundary, so the aligned slot is +48. */
    assert((uintptr_t)p - (uintptr_t)base == 48U);
    assert(a.used == 48U + 8U);
}

int main(void) {
    printf("arena tests:\n");
    RUN(test_init_rejects_null_arena);
    RUN(test_init_rejects_null_base);
    RUN(test_init_rejects_zero_capacity);
    RUN(test_alloc_default_align_advances_pointer);
    RUN(test_alloc_aligns_subsequent_calls);
    RUN(test_alloc_aligned_custom);
    RUN(test_alloc_aligned_on_misaligned_base);
    RUN(test_alloc_rejects_non_power_of_two_align);
    RUN(test_alloc_rejects_zero_size);
    RUN(test_alloc_returns_null_on_exhaustion);
    RUN(test_reset_rewinds_to_base);
    RUN(test_available_tracks_remaining);
    RUN(test_null_arena_returns_zero_available);
    printf("all 13 tests passed.\n");
    return 0;
}
