/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 *
 * jaxility_runtime/dcache.h — D-cache / I-cache coherence ops.
 *
 * Closes the PI5 quirk ``d-cache-clean-required-for-codegen-buffers``:
 * Cortex-A76 requires explicit D-cache clean (``DC CIVAC``) before
 * the I-cache invalidate (``IC IVAU``) that lets the CPU execute
 * newly-generated code. Without this dance, the just-written code
 * lives in the D-cache as data while the I-fetch path sees stale
 * bytes from main memory (or the I-cache).
 *
 * The contract is *barrier-shaped*: a single call after writing the
 * codegen buffer, before jumping into it. The implementation differs
 * by family:
 *   - Cortex-A (aarch64): ``DC CIVAC`` + ``DSB ISH`` + ``IC IVAU`` +
 *     ``DSB ISH`` + ``ISB`` on each cache-line-sized chunk.
 *   - Cortex-M baremetal: the M-profile architecture either has no
 *     separate I/D caches (M3/M4) or has unified-coherent caches
 *     (M7+) — the call is a memory barrier (``__DSB`` + ``__ISB``).
 *
 * Sources that need this on Cortex-A include ``acados``-generated
 * code buffers loaded at deploy time, JIT-style runtime patching
 * (the learned-policy lane), and self-modifying lookup tables.
 */
#ifndef JAXILITY_RUNTIME_DCACHE_H
#define JAXILITY_RUNTIME_DCACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Make ``len`` bytes starting at ``addr`` coherent between D-cache
 * and I-cache. Must be called after writing executable code to
 * memory and before jumping into it. ``addr`` need not be cache-line
 * aligned — the implementation walks aligned-down to the line.
 *
 * Returns 0 on success, -1 if ``addr`` is NULL or ``len`` is 0.
 * The call is a barrier; it never fails for any other reason. */
int jx_dcache_clean_for_exec(const void *addr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* JAXILITY_RUNTIME_DCACHE_H */
