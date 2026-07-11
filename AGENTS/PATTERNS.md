# Jaxility — Coding Patterns

This file documents the coding conventions agents and humans follow when
writing Jaxility code. It is the reference Claude Code consults on every
diff.

Conventions inherited from Jaxonomy and Jaxterity still apply. This file
documents additions and Jaxility-specific specializations only. Compiler
work has different failure modes from library work; the additions here
are about catching those failure modes early.

If a pattern here disagrees with a Jaxterity or Jaxonomy pattern,
**this file wins for code under `jaxility/`.**

---

## 1. File and module conventions

### 1.1 SPDX header

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
```

C source files under `runtime-c/` use the equivalent comment style:

```c
/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 The Jaxility Authors
 */
```

### 1.2 Module docstrings, JAX imports, type hints

Same conventions as Jaxterity PATTERNS.md §1.2–§1.4. Type hints
mandatory on public surfaces; `jaxtyping` for arrays.

Jaxility-specific: imports of CasADi, acados, and vendor SDKs go in
the third-party-non-JAX group, but always inside a function or behind
an extras-guarded import. Top-level `import casadi` in
`jaxility.lowering.coverage` is fine; top-level `import casadi` in
`jaxility.targets.cortex_a76` is wrong (the target profile must
import-clean without the lowering extras installed).

```python
# At top of file
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import casadi as ca  # type-only

# Inside a function
def translate_dynamics(jax_fn, *, in_avals):
    import casadi as ca  # actual runtime import
    ...
```

The extras guard makes `pip install jaxility` (without `[acados]`)
install successfully and lets the user inspect targets, manifests,
and coverage without pulling 200MB of CasADi+acados.

---

## 2. Subprocess discipline

Jaxility shells out to external toolchains constantly: compilers,
linkers, acados, FVP binaries, vendor SDKs. The patterns here exist
because subprocess code is where reproducibility goes to die.

### 2.1 Every subprocess call goes through `jaxility.runtime.subprocess_runner`

There is exactly one wrapper around `subprocess.run` in the package:

```python
from jaxility.runtime.subprocess_runner import run

result = run(
    ["aarch64-linux-gnu-gcc", "-O2", "-c", "main.c"],
    cwd=build_dir,
    env=hermetic_env,
    timeout_s=120,
    expected_returncode=0,
)
```

The wrapper:

- Records the full command line, the working directory, the
  environment, and the toolchain version into the build log
  (which lands in the manifest).
- Captures stdout and stderr to structured files.
- Enforces a timeout. There is no untimed subprocess in Jaxility.
- Raises a structured `ToolchainError` on non-expected returncode,
  with the command and the captured logs attached.

Direct `subprocess.run`, `os.system`, `subprocess.Popen`, and
`subprocess.check_output` calls are forbidden in library code. CI
has a grep check.

### 2.2 Toolchain detection at startup, not on demand

Every target's toolchain is detected and version-pinned at startup
(when the `Target` profile is loaded), not lazily on first build. A
build that would shell out to a missing toolchain fails at target
load, with a clear "install Arm GCC 15.2.1+aarch64 from <vendor URL>"
message.

This is the Bazel philosophy: hermetic environments. We don't
achieve full hermeticity at v0.1 (vendor SDKs are too varied), but
we move in that direction.

### 2.3 Environment whitelisting

Subprocess calls receive a deliberately-minimized environment, not
`os.environ`. The whitelist is documented per-target in
`jaxility.targets.<target>.environment`. PATH, HOME, and the
target-specific compiler-config variables are passed through; the
rest are stripped.

This prevents the "works on my laptop because `LD_LIBRARY_PATH` was
set" class of bugs.

---

## 3. Manifest and serialization discipline

### 3.1 Canonical JSON for everything that gets hashed

All manifests, target profiles, coverage entries, and benchmark
records use a single canonical JSON serializer:
`jaxility.manifest.canonical_dumps`. The serializer enforces:

- UTF-8.
- Keys sorted lexicographically.
- No whitespace between separators (compact form).
- No floats; numbers are integers, strings, or scientific-notation
  strings.
- No `NaN`, `Infinity`, or `-Infinity` (representable cases use the
  string forms documented in the schema).

Hashes are taken over the output of `canonical_dumps`, never over
`json.dumps`. `json.dumps` is forbidden for content-hashed payloads
(CI grep check).

### 3.2 Timestamps live outside the content hash

Builds at different times of the same source produce byte-identical
artifacts — except for the timestamp field in the manifest, which is
explicitly excluded from the content hash. This is invariant 5
(deterministic builds).

### 3.3 Pydantic v2 everywhere

All structured data — `Target`, `Manifest`, `Artifact`, `Coverage`,
`BenchmarkRecord` — is a `pydantic.BaseModel` with strict mode.
`extra="forbid"` on every model; an unknown field is an error.

### 3.4 Schema versioning is explicit

Every schema carries a `schema_version: int` field. v0 schemas are
the OSS-minimum; v1 will be SLSA-aligned. Old artifacts must be
readable indefinitely; the verify tool dispatches on
`schema_version`.

Changing a schema is an ADR-grade decision.

---

## 4. Emitted C code patterns

The C code Jaxility emits (the runtime glue, manifest binding, safety
envelopes) is the most safety-critical code in the project. It runs
on real hardware, often without supervision, sometimes near humans.
The patterns here protect that surface.

### 4.1 MISRA C:2012 awareness, not full compliance at v0.1

Full MISRA compliance is a v0.2+ goal (enterprise/regulatory tier).
At v0.1, the emitted code follows the MISRA principles that are
zero-cost to apply:

- No `goto`.
- No recursion.
- Explicit casts.
- No implicit type conversions in arithmetic.
- No dynamic allocation after initialization.
- All variables initialized before use.
- One return per function (relaxed for early-exit error handling).
- Bounded loops; loop bounds visible at the call site.

Deviations are documented in the build log with a justification.
There is no "magic" emitted C; an engineer reading the output can
trace every line to a Jaxility template.

### 4.2 No emitted code calls into the standard library beyond a whitelist

Whitelist: `memcpy`, `memset`, `memcmp`, `assert.h` (only in debug
builds, never in release), the floating-point intrinsics that map to
single instructions. `printf`, `malloc`, `free`, `getenv`, anything
in `<stdio.h>` beyond `fputc` for explicit debug builds, anything
that calls into the OS — forbidden in release builds.

The whitelist is enforced by the linker (release builds reject
unresolved symbols outside the whitelist).

### 4.3 Generated C files carry their provenance in a header comment

Every emitted `.c` and `.h` file starts with:

```c
/* Generated by Jaxility v<version> at <UTC timestamp>
 * Source manifest hash: <hash>
 * Target profile: <target_name>
 * Coverage entries used: <list>
 *
 * DO NOT EDIT MANUALLY. Regeneration replaces this file.
 */
```

The hash in the header is the same one in the manifest; a downstream
audit can match a deployed `.c` to its build record.

### 4.4 Floats are explicit width

`float` (32-bit) and `double` (64-bit) are written as `float32_t`
and `float64_t` (from `<stdint.h>` extensions or a typedef header).
This catches the "but on this target a `float` is..." family of
silent-portability bugs.

---

## 5. Target dispatch patterns

### 5.1 Targets are data, not subclasses

A `Target` is a Pydantic model. Adding a new target is filling out the
model; it is not subclassing a base class with virtual methods.

If a target genuinely needs custom behavior (e.g., the Ethos-U55 NPU
requires graph partitioning that the others don't), the custom
behavior lives in a *strategy* registered against a target capability
flag, not in a target subclass.

```python
@register_strategy(when=lambda t: t.has_npu)
def lower_with_npu_partition(graph, target):
    ...
```

This keeps `Target` itself pure data and serializable.

### 5.2 Target-conditional code is centralized

Code that branches on target capability — vector width, NPU presence,
RT-OS vs. bare-metal — lives in `jaxility.targets.dispatch`. Other
modules ask "what does this target support" by querying the target,
not by importing target-specific modules.

A grep for `if target.name == "cortex_a76"` outside the dispatch
module is a CI failure. Use capability queries:
`if target.supports("neon")`.

### 5.3 Mock targets are first-class

`mock-cortex-a` and `mock-cortex-m` are not test fixtures bolted on
the side. They are full `Target` profiles. The mock toolchain is a
Python stub that produces a function-call wrapper around the source
JAX. Tests that exercise the lowering pipeline use mock targets;
tests that exercise toolchain integration use real targets.

This makes the mock pipeline testable without any cross-compilation
infrastructure.

---

## 6. Error patterns

### 6.1 Exception hierarchy

```
JaxilityError
├── CoverageError       # unsupported op for target
├── ToolchainError      # subprocess failure or version mismatch
├── EquivalenceError    # numerical equivalence check failed
├── ManifestError       # schema violation, signature mismatch
├── TargetError         # target profile invalid or toolchain missing
├── HILError            # hardware-in-loop test diverged
└── BenchmarkError      # benchmark harness failure
```

Library code raises one of these. Never `RuntimeError`,
`ValueError`, or `AssertionError`.

### 6.2 Errors carry the build log

Every error from a compilation pipeline carries the partial build
log up to the failure point, in a structured form. The agent
reading the error sees what the toolchain emitted before failing.

```python
raise ToolchainError(
    "Cross-compilation failed",
    command=cmd,
    returncode=result.returncode,
    stdout=result.stdout,
    stderr=result.stderr,
    target=target.name,
    suggestion="The error pattern suggests a missing CMSIS-DSP "
               "dependency. Install with: <command>.",
)
```

### 6.3 EquivalenceError points at the offending op

When numerical equivalence fails, the error names the op (or
template, or template parameter) most likely responsible, based on
sensitivity analysis. This is the most expensive error to debug
manually; the error message earns its keep.

---

## 7. Test patterns

### 7.1 Three test tiers

- `test/unit/` — pure Python, no toolchains, no hardware. Run on
  every PR. Fast.
- `test/host/` — host-only builds (Linux x86). Requires CasADi and
  acados. Run on every PR.
- `test/hil/` — cross-compiled builds run on FVP or hardware. Run
  nightly, on `pull_request` only when `runtime-c/` or a `Target`
  profile changes.

A test that needs hardware is `pytestmark = pytest.mark.hil`.

### 7.2 Equivalence tests are property tests

Numerical equivalence is verified across a sampled parameter space
(initial conditions, parameter perturbations), not on a single
hand-chosen trajectory. Hypothesis-based property tests are
preferred over example tests for equivalence.

### 7.3 Golden artifacts are tracked

Every supported (template, target) pair has a committed golden
artifact under `test/golden/<template>/<target>/`. The golden
includes the binary content hash and the full manifest. Tests
compare new builds against the golden; deliberate changes update the
golden in the same PR with a justification.

### 7.4 Tolerances come from `test/EQUIVALENCE.md`

No magic numbers. `jaxility.testing.tolerances` reads from the
documented table.

---

## 8. CLI patterns

Jaxility's primary surface is the CLI. The patterns here ensure the
CLI is agent-legible (readable and scriptable by coding agents).

### 8.1 Top-level commands

```
jaxility build <robot> --target <soc> [--template <name>]
jaxility verify <manifest>
jaxility bench <robot> --target <soc>
jaxility hil <robot> --target <soc>
jaxility targets [--detail]
jaxility coverage --target <soc>
jaxility skills
jaxility mcp serve
```

Subcommands follow `<verb> <object> [--qualifiers]`. No nested verbs
beyond two levels.

### 8.2 JSON-mode for every command

Every command supports `--json` and produces machine-readable
output to stdout. Human-readable output goes to stderr; the JSON
goes to stdout. This is the agent-consumption discipline: a
subprocess running `jaxility build --json | jq` should work.

### 8.3 Exit codes are documented

```
0   — success
1   — generic error
2   — invalid arguments
10  — coverage error (unsupported op)
20  — toolchain error
30  — equivalence error
40  — manifest error
50  — target error
60  — HIL error
70  — benchmark error
```

CI checks these. A passing test on a failed build is a CI bug.

---

## 9. Documentation patterns

### 9.1 Per-target documentation

Each supported target gets a documentation page describing: hardware
spec, toolchain version, known quirks, build performance, runtime
characteristics. Generated from the `Target` profile, augmented with
hand-written prose.

### 9.2 CLAIMS.md and KNOWN_GAPS.md

These are first-class files, not just sections in README:

- `docs/CLAIMS.md` — what Jaxility actually does, today. Every claim
  is measured; every measurement cites a benchmark record.
- `docs/KNOWN_GAPS.md` — what it does not do, with brief explanation.

Updating these when shipping or deprecating a feature is part of the
relevant task spec. A claim without a benchmark record is a CI
failure.

### 9.3 Build log → docs

The build log from `jaxility build` is a documented artifact format.
The schema is published; auditors and the enterprise tier consume
it.

---

## 10. Anti-patterns

Specific things to refuse, with reasons:

- **Top-level imports of CasADi / acados / vendor SDKs.** Breaks
  default `pip install jaxility`. Use extras guards.
- **`subprocess.run` outside `runtime.subprocess_runner`.** No
  exceptions.
- **`json.dumps` for hashed payloads.** Use `canonical_dumps`.
- **Floating-point timestamps.** Integer microseconds-since-epoch
  only.
- **Emitted C that calls outside the whitelist.** Linker enforces;
  do not work around.
- **Target-conditional code outside `targets.dispatch`.** Capability
  queries only.
- **`assert` in release-mode emitted C.** Compile flag strips them;
  releases without the strip are not shippable.
- **Bare `RuntimeError`, `ValueError`, `AssertionError` in library
  code.** Use the hierarchy.
- **Vendoring acados or CasADi.** Pin the versions; do not ship a
  fork.
- **Unsigned manifests in benchmark records.** Hash-chain at minimum.
- **HIL skips in main CI.** Either it runs (on hardware or FVP) or it
  is not yet supported.
