/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * jaxility_runtime/arena.h — bump-pointer arena (T-032 / T-052).
 *
 * Enforces invariant 4 / PATTERNS §4.1: no dynamic allocation after
 * init. Embedded deployment targets (Cortex-M baremetal, PREEMPT_RT
 * Linux on Cortex-A) need a deterministic allocator with no malloc()
 * in the control loop. The arena gives the runtime a contiguous slab
 * of memory the user reserves at init, then bump-allocates from at
 * the deterministic phase.
 *
 * The arena is a *single-shot bump pointer*: alloc() never frees;
 * reset() rewinds the pointer to base; the only way to recover
 * memory is reset()-back-to-base. That is the contract; do not add
 * free().
 *
 * All operations are O(1). No system calls. No threads. The arena is
 * intended to be wrapped by a higher-level pool (per-OCP horizon, per
 * MPC step) that owns the lifetime contract.
 */
#ifndef JAXILITY_RUNTIME_ARENA_H
#define JAXILITY_RUNTIME_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default alignment for arena allocations. 16 bytes covers SIMD load
 * requirements on aarch64 (NEON) and the larger of double + pointer
 * on Cortex-M (FPv4-SP-D16 + 32-bit thumb). Per-call alignment can
 * be requested via ``jx_arena_alloc_aligned``. */
#define JX_ARENA_DEFAULT_ALIGN 16U

typedef struct {
    uint8_t *base;     /* start of the user-supplied slab */
    size_t   capacity; /* total slab size in bytes */
    size_t   used;     /* bytes consumed from base */
} jx_arena_t;

/* Initialise an arena over a caller-owned slab. Returns 0 on success,
 * -1 if ``base`` is NULL or ``capacity`` is 0. The caller retains
 * ownership of the slab; the arena does not allocate. */
int jx_arena_init(jx_arena_t *arena, void *base, size_t capacity);

/* Allocate ``size`` bytes aligned to ``JX_ARENA_DEFAULT_ALIGN``.
 * Returns NULL on exhaustion (the only allocation failure mode). */
void *jx_arena_alloc(jx_arena_t *arena, size_t size);

/* Allocate ``size`` bytes with a custom power-of-two alignment.
 * Returns NULL on exhaustion or if ``align`` is zero / not a power of
 * two. */
void *jx_arena_alloc_aligned(jx_arena_t *arena, size_t size, size_t align);

/* Reset the bump pointer to ``arena->base``. Does not zero the slab.
 * Equivalent to "rewind to checkpoint(0)"; callers needing
 * checkpoint-style undo can save ``arena->used`` and restore it. */
void jx_arena_reset(jx_arena_t *arena);

/* Bytes remaining (capacity - used). Useful for asserts. */
size_t jx_arena_available(const jx_arena_t *arena);

#ifdef __cplusplus
}
#endif

#endif /* JAXILITY_RUNTIME_ARENA_H */
