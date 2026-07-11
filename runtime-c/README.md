# `runtime-c/` — Jaxility on-target C runtime

This directory holds the MISRA-aware C runtime that wires generated
artifacts to target I/O on real hardware (motor controllers, IMU
drivers, encoder readers, cycle-time schedulers).

It is **not** a Python subpackage. The build-time Python helpers
that *invoke* toolchains live in
[`jaxility/runtime/`](../jaxility/runtime/) (see
`jaxility.runtime.subprocess_runner`, PATTERNS §2.1). This directory
holds the C *source* that ends up linked into the final binary.

## Layout (planned)

Per ADR-010, Cortex-A and Cortex-M share the `Target` abstraction but
not the runtime code. The directory will be partitioned:

- `runtime-c/common/` — shared headers (manifest header, fixed-point
  primitives, the float-width typedefs from PATTERNS §4.4).
- `runtime-c/cortex-a/` — Linux + PREEMPT_RT runtime; Pi 5 first
  (T-032).
- `runtime-c/cortex-m/` — bare-metal / FreeRTOS runtime; STM32H7
  first (T-052; FreeRTOS-vs-bare-metal is OQ-2).

## Current status

This directory is intentionally a placeholder at v0.0.1-phase0. The Pi 5
runtime lands in T-032; the STM32H7 runtime in T-052. CI does not yet
cross-compile from this directory.

## Patterns

Emitted C and hand-written runtime C both follow PATTERNS §4:

- MISRA-C:2012 *awareness* (full compliance is enterprise/v0.2+); no
  `goto`, no recursion, explicit casts, bounded loops.
- Standard-library calls restricted to a whitelist (`memcpy`, `memset`,
  `memcmp`, FP intrinsics; debug-only `assert.h`, `fputc`).
- Generated files carry a provenance header binding them to the source
  manifest hash and the target profile.
- Float widths are explicit (`float32_t`, `float64_t`); the bare C
  `float` / `double` are forbidden.

Hand-written runtime C is reviewed by a human before each release.

## License

MIT, matching the rest of Jaxility. SPDX header on every C / H file:

```c
/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 */
```
