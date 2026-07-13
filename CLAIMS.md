# What Jaxility claims today

> **Maintainer note (the doc-drift rule).** This file, `KNOWN_GAPS.md`,
> and the `README.md` quick-reference table form a single
> three-document contract. Any PR that **closes a gap in code** must
> move text from `KNOWN_GAPS.md` into this file in the same PR; any PR
> that **opens a new capability** must add a row here. Pin bumps,
> target-profile changes, coverage-table additions, and family-flag
> additions all require corresponding edits in this file. The
> anti-drift tests in `test/unit/test_top_level_docs.py` enforce
> *some* of the rule (target enumeration, pin-string match, key
> phrases) but cannot catch a gap closing without an agent moving
> the text. See [`AGENTS/CONTEXT.md`](AGENTS/CONTEXT.md) §"Contract
> docs and the doc-drift rule" for the full governance.

Jaxility is the JAX→Arm-based robotics deployment compiler. This document is
the **load-bearing list of things the library promises to do** at the
current version. If you build on top of Jaxility, you can rely on
every claim here being under test. For the symmetric list — what
Jaxility explicitly **does not** do — see
[`KNOWN_GAPS.md`](KNOWN_GAPS.md).

Stage in the roadmap: the runtime, HIL, 1 kHz bench,
and the attested cross-compiled artifact are all green on a real Pi 5;
the learned-policy lane (JAX → ONNX → LiteRT/ExecuTorch, quantization,
dual-path composition + demo on a real Pi 5) has landed. Every claim
below is backed by a green test in `test/unit/` unless explicitly marked
otherwise.

## Architecture

- **Three-package stack.** Jaxility consumes calibrated robots from
  [Jaxterity](https://github.com/machinavitalis/jaxterity), which sits
  on [Jaxonomy](https://github.com/machinavitalis/jaxonomy). All three
  are versioned independently and pinned at the package boundary.
- **JAX → CasADi → acados → embedded C** (ADR-001). CasADi is the
  symbolic boundary between JAX traces and acados; ADR-001 documents
  why and what alternatives were rejected.
- **Data over subclasses for Targets** (ADR-003). A new SoC is a new
  `Target(...)` row, not a new class. Capability-keyed dispatch
  (PATTERNS §5.2) reads `target.supports(...)`; it never tests
  `target.name == ...`.

## What we translate from JAX

The translator (`jaxility.lowering.translate`) accepts JAX functions
built from the **smooth-op subset acados consumes**. The supported
op set is declared in `jaxility/lowering/coverage.py` and enforced
at trace time — there is no silent fallback (invariant 7).

**Supported at float64 on `mock-cortex-a` (and the mirroring mock
float32 / cortex-m rows):**

- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`.
- Transcendentals: `jnp.sin`, `jnp.cos`, `jnp.tan`, `jnp.exp`,
  `jnp.log`, `jnp.sqrt`.
- Linear algebra: `matmul`.
- Indexing: `slice[static]`, `dynamic_slice[static]`.
- Control: `jnp.where[static]` (predicate is a compile-time constant).

The lowering attaches an `implementation_hint`, a numerical-grade
label (`ulp-bounded`, `approximate`), and a per-op `suggestion`
string to every supported row. The same table records every op the
translator deliberately **does not** accept — see `KNOWN_GAPS.md` for
that side.

## What we generate

- **acados OCP**. `jaxility.lowering.build_ocp(dynamics, spec)` takes
  a translated `CasadiFunction` and an `OcpTemplateSpec` and produces
  an `AcadosOcp` ready for `AcadosOcpSolver`.
- **Four problem templates** wired end-to-end (T-022..T-025):
  - **LQR** — finite-horizon discrete LQR with linearised dynamics.
  - **TrackingMPC** — reference-following MPC with quadratic stage
    cost + terminal cost.
  - **WBC (weighted)** — weighted whole-body controller; task
    priorities expressed as relative cost weights.
  - **Centroidal MPC (single-contact)** — single-contact phase
    centroidal MPC.

## What we build

- **Host build path** (`jaxility.build_for_target`): runs the
  acados-generated C through the host's compiler and produces a
  shared library wrapped in a content-addressed `Artifact` with a
  chain-linked `Manifest`. Works on macOS (Darwin) and Linux x86_64
  hosts.
- **Structural cross-compile wrapper** (T-031,
  `jaxility.builder_cross`): composes a deterministic `compiler_argv`
  for the deployment target's toolchain. Verifies the binary against
  `Target.toolchain.version_regex`.
  - **Cortex-M lane (`arm-none-eabi-gcc 15.2.1`)** runs locally and
    in CI. End-to-end test compiles a hand-rolled C source to a
    Cortex-M4 ELF `.o`.
  - **Pi 5 lane (`aarch64-none-linux-gnu-gcc 15.2.1`)** runs in CI
    (Linux x86_64 runners install the archive; cached per release).
    End-to-end test compiles a hand-rolled C source to a Cortex-A76
    ELF `.so`. On Apple Silicon dev hosts the test skips locally —
    Arm does not ship a darwin-arm64 host build of this chain in
    15.2.Rel1.
  - `_FAMILY_CFLAGS` carries rows for `cortex-a76` (A-profile,
    `-shared -fPIC` → `.so`) and `cortex-m4`, `ethos-u55`, `ethos-u65`
    (M-profile, `-c` → relocatable `.o`).
- **Dependency cross-build orchestrator** (T-031 follow-up, ADR-018,
  `jaxility.builder_deps`): cross-builds the acados / blasfeo / hpipm
  static archives **from source** for an Arm target — never vendored
  binary blobs — via a CMake toolchain file
  (`cmake/toolchains/aarch64-none-linux-gnu.cmake`),
  `-DBUILD_SHARED_LIBS=OFF`, and a per-family blasfeo `TARGET`
  (`ARMV8A_ARM_CORTEX_A76` for the Pi 5). `cross_build_for_target(...,
  deps=...)` links the controller against the produced archives and
  records each archive's BLAKE3 hash in the manifest under
  `dep-archive:<name>` for provenance. The plan layer (configure/build
  argv, link-arg grouping, include dirs) is deterministic and verified
  on every host; the real cross-build + end-to-end link is a Tier-B
  test that runs where the aarch64 toolchain **and** an acados source
  tree are present (the `cross-deps` CI job; not the standing per-PR
  gate). Pi 5 / aarch64 only — bare-metal Cortex-M archives are T-052.
  See [`KNOWN_GAPS.md`](KNOWN_GAPS.md) "Linker gap" for the exact
  verification boundary.

## On-target C runtime (T-032 / T-052)

- **Runtime source tree** (`runtime-c/`):
  - `arena.c` — bump-pointer no-malloc arena enforcing invariant 4
    / PATTERNS §4.1 (no dynamic allocation after init). Portable C99;
    runs on every target.
  - `cycle_posix.c` — `clock_gettime(CLOCK_MONOTONIC)` + `clock_nanosleep`
    cycle scheduler with drift correction and jitter accumulation.
    Cortex-A targets (Pi 5 / PREEMPT_RT Linux).
  - `dcache_aarch64.c` — `DC CIVAC` + `DSB ISH` + `IC IVAU` + `ISB`
    sequence closing the PI5 quirk
    `d-cache-clean-required-for-codegen-buffers`. Cache-line size read
    from `CTR_EL0` at runtime.
  - `dcache_thumb.c` — `DSB` + `ISB` barrier sequence for Cortex-M
    (no separate I/D caches on M3/M4).
  - `rt_posix.c` (T-032) — PREEMPT_RT control-thread placement:
    `jx_rt_pin_to_core` (sched_setaffinity to one core),
    `jx_rt_set_realtime_priority` (SCHED_FIFO via pthread), and
    `jx_rt_configure` (both, in order). Cortex-A / Linux only; off
    Linux every call returns `JX_RT_ERR_UNSUPPORTED`, and EPERM is
    surfaced distinctly as `JX_RT_ERR_PRIVILEGE` (loud failure,
    invariant 7). This is the helper named in the PI5 quirk
    `preempt-rt-soft-not-hard`; placement removes the
    scheduler-contention jitter class but PREEMPT_RT remains *soft*
    real-time.
  - `deploy_posix.c` (T-032) — the deployment launcher
    (`jx_deploy_run`, ABI in `jaxility_runtime/deploy.h`): dlopen()s
    the cross-compiled controller `.so`, resolves the three-symbol
    controller ABI (`jx_controller_init/step/period_ns`), threads the
    arena into the controller's `init`, and drives its `step` from the
    cycle scheduler at the controller-declared period. RT placement is
    best-effort; setup failures return distinct negative
    `JX_DEPLOY_ERR_*` codes. Cortex-A / POSIX only. The host test
    (`test_deploy.c`) drives this against a fake controller `.so` end
    to end without hardware.
- **Host arena unit tests** (`runtime-c/test/test_arena.c`, 12 tests)
  run via the host `cc` against the same source. The Python test
  driver compiles and runs them on every dev host. The `cycle_posix`,
  `dcache`, and `rt_posix` (T-032) sources have companion host tests
  compiled + run the same way — `rt_posix` exercises the real Linux
  path on CI and the unsupported branch on macOS.
- **Build orchestrator** (`jaxility.runtime.c_runtime`):
  - `plan_runtime_build(target, work_dir)` composes deterministic
    compile + ar argv tuples per source.
  - `build_runtime_archive(target, work_dir)` runs the plan and
    returns a `RuntimeArchive` carrying the path plus a BLAKE3
    content hash. Output: `libjaxility_runtime_<family>.a`.
- **Real archives produced**:
  - **Cortex-M lane** (local + CI with `arm-none-eabi-gcc`):
    `libjaxility_runtime_cortex-m4.a` containing `arena.o` +
    `dcache_thumb.o` as ARM EABI5 relocatables.
  - **Pi 5 lane** (CI with `aarch64-none-linux-gnu-gcc`):
    `libjaxility_runtime_cortex-a76.a` containing `arena.o` +
    `cycle_posix.o` + `dcache_aarch64.o` + `rt_posix.o` +
    `deploy_posix.o` as aarch64 ELF relocatables.
- **Deployment launcher build** (`jaxility.runtime.deploy`,
  `plan_deploy_launcher` / `execute_deploy_launcher`): cross-links
  `deploy_main.c` against the runtime archive into `jx_deploy_<family>`
  (`-rdynamic -ldl -pthread`), the binary that dlopen()s the controller
  on the Pi. Plan layer is deterministic + tested everywhere; the real
  cross-link is Tier-B (aarch64 toolchain on CI). Cortex-A only —
  Cortex-M deployment is a different mechanism (T-052).

## Targets (14 profiles)

The full known set is round-tripped through canonical JSON and
verified for hash distinctness (invariant 5):

| Profile | Family | Toolchain | NPU | Realtime |
|---|---|---|---|---|
| `MOCK_CORTEX_A` | mock-cortex-a | mock | — | SOFT, PREEMPT_RT |
| `MOCK_CORTEX_M` | mock-cortex-m | mock | — | HARD, cyclic |
| `HOST_DARWIN` | host-darwin | clang | — | NONE |
| `HOST_LINUX` | host-linux | gcc | — | NONE |
| `PI5` | cortex-a76 | aarch64-none-linux-gnu-gcc 15.2.1 | — | SOFT, PREEMPT_RT |
| `CORTEX_A55` | cortex-a55 | aarch64-none-linux-gnu-gcc 15.2.1 | — | SOFT, PREEMPT_RT |
| `CORTEX_A78` | cortex-a78 | aarch64-none-linux-gnu-gcc 15.2.1 | — | SOFT, PREEMPT_RT |
| `CORTEX_A710` | cortex-a710 | aarch64-none-linux-gnu-gcc 15.2.1 | — | SOFT, PREEMPT_RT |
| `NEOVERSE_N1` | neoverse-n1 | aarch64-none-linux-gnu-gcc 15.2.1 | — | SOFT, PREEMPT_RT |
| `CORTEX_M4` | cortex-m4 | arm-none-eabi-gcc 15.2.1 | — | HARD, cyclic |
| `ETHOS_U55` | ethos-u55 | arm-none-eabi-gcc 15.2.1 | Ethos-U55 (0.5 TOPS @ INT8) | HARD, cyclic |
| `ETHOS_U65` | ethos-u65 | arm-none-eabi-gcc 15.2.1 | Ethos-U65 (1.0 TOPS @ INT8) | HARD, cyclic |
| `QUALCOMM_IQ10` | qualcomm-iq10 | aarch64-none-linux-gnu-gcc 15.2.1 | Hexagon (≈12 TOPS @ INT8) | SOFT, PREEMPT_RT |
| `APPLE_SILICON` | apple-silicon | clang 15.0.0 | ANE (pending OQ-4) | NONE |

Adding a target is filling out one `Target(...)` row. The `Target`
abstraction is `frozen=True` + `extra="forbid"`; mutation is impossible
and unknown fields are an error.

## Manifests

- **Schema v0** (`jaxility.manifest`). All payload models are Pydantic
  v2 `frozen=True` / `extra="forbid"`. PATTERNS §3.4 governs schema
  versioning.
- **Canonical JSON** (`canonical_dumps`). Stable across hosts: equal
  Python values produce byte-identical bytes — that property is what
  makes the manifest's content hash reproducible.
- **BLAKE3 hash chain** (ADR-005). Every Artifact records a
  `content_hash` over its payload + a `source_manifest_hash` linking
  it to the upstream manifest.
- **HexBytes serialisation** (review fix). Bytes fields
  round-trip through JSON as lowercase hex via
  `PlainSerializer(when_used="json")`; canonical and content-payload
  encodings agree.
- **Detected toolchain versions** (A6). `Manifest.toolchain_versions`
  carries `{target.toolchain.name, "acados-template",
  "acados-library", "casadi"}`. Detection in
  `jaxility.manifest.toolchain_detect` raises `ToolchainError` on
  missing packages and returns a self-explaining
  `library-unknown:<reason>` marker (never silent `"unknown"`) when
  the local acados source tree is absent. Pinned upstreams are
  importable constants (`JAXILITY_ACADOS_TEMPLATE_PIN`,
  `JAXILITY_ACADOS_LIBRARY_PIN`).
- **Cross-toolchain integrity status** (T-112). Every cross-build records
  `toolchain-integrity:<binary>` in `Manifest.toolchain_versions` via
  `resolve_toolchain_integrity`: `sha256:<hex>` when the pin carries a real
  hash and the installed binary matches, or `"unverified"` when the pin is
  the `UNVERIFIED_SHA256` sentinel. A pinned hash that does **not** match
  the installed binary aborts the build (`ToolchainError`) — the attestation
  never silently implies a toolchain check that did not happen. (No shipped
  Arm pin carries a real hash yet — see `KNOWN_GAPS.md`.)
- **Verification** (`jaxility verify`): cryptographically validates
  the chain end-to-end and surfaces the upstream Jaxterity attestation
  handle.

## Equivalence

- **Host equivalence harness** (T-027). Compares the source JAX
  dynamics against the compiled controller's solver step. Tolerances
  are calibrated to engineering-grade — acados sub-stepping precludes
  ULP-grade parity at the host level. Per-template tolerances live in
  `jaxility.testing.tolerances`.

## Hardware-in-the-loop on real silicon (T-033 / T-034 / T-035)

- **Step-locked HIL harness** (`jaxility.hil`). `run_hil` drives a
  deployed artifact through a `TargetRunner` (`LocalRunner` or
  `SshRunner`), parses its per-cycle JSON-Lines trace, and compares it
  to the host reference under the documented tolerances — the same
  equivalence surface as T-027 (invariant 6). Malformed traces fail
  loudly (`HILError`), never silently.
- **The acados Cartpole controller runs on a real Raspberry Pi 5 /
  Cortex-A76** and passes step-locked HIL parity against the host
  reference at `cortex-a76 × float64`, measured to ~1e-15 (ULP)
  cross-architecture. `jaxility.hil.controller` generates the closed-loop
  binary; `build_controller_on_target` builds it on the Pi.
- **The robot-rooted launch demo runs on the Pi 5** (Jaxterity T-102): a
  controller built **from a calibrated Robot** — dynamics via
  `jaxterity.zoo.cartpole.reduced_params`, manifest rooted on the robot's real
  `attestation_handle` — is shipped, built natively on the Cortex-A76, and
  verified on-Pi (HIL parity + ~103 µs mean solve, meets 1 kHz).
  `examples/cartpole_end_to_end.py` (with `JAXILITY_HIL_SSH_HOST`);
  `test/hil/test_robot_on_pi.py`.
- **Benchmark harness** (`jaxility.bench`, `jaxility bench cartpole
  --target pi5`). Times the bare acados solve on the target and emits a
  `BenchRecord` (min/mean/p50/p99/σ, jitter, peak RSS, `meets_1khz`)
  tied to the source manifest hash. Cartpole LQR on the Pi 5 meets the
  1 kHz budget with margin (max 265 µs at 2.4 GHz, native gcc build).
- **Attested cross-compiled artifact** (T-037). The controller is also built
  with the **pinned Arm GNU 15.2.Rel1** toolchain in a reproducible Docker
  container (`docker/cross-aarch64.Dockerfile`,
  `scripts/cross_build_attested_pi5.sh`), linking cross-built static
  acados/blasfeo/hpipm. Its manifest records `aarch64-none-linux-gnu-gcc
  15.2.1` + the dep-archive hashes (invariant 3) and verifies; on the Pi it
  passes the same HIL parity (~1e-15) and benchmarks faster + tighter than the
  native build (mean 87 µs / max 121 µs).

## Learned-policy export

- **JAX policy → ONNX** (T-040, `jaxility.policy.export_policy_to_onnx`).
  Walks the policy jaxpr and emits a self-contained ONNX `ModelProto` over the
  **smooth-MLP subset**: dense layers (`dot_general` → `MatMul`), biases
  (`add`), and smooth activations (`tanh`, `logistic`/sigmoid, `relu` via
  `max`). Call-like primitives (`pjit` / `jit` / `custom_jvp_call`) are
  transparent boundaries. The accepted op set is `SUPPORTED_PRIMITIVES`;
  anything else raises a structured `CoverageError` (invariant 7 — no silent
  mis-export). Exports are **single-observation** (unbatched), the on-target
  inference shape. Bit-exact parity against ONNX Runtime on flax MLPs is under
  test. The `[policy]` extra pulls `onnx` / `onnxruntime` / `flax`.
- **ONNX → LiteRT** (T-041, `jaxility.policy.litert`). Orchestrates the
  ONNX → TF → TFLite conversion (`onnx2tf`) — the priority on-device path — and
  `litert_parity` checks the converted `.tflite` against ONNX Runtime under the
  LiteRT interpreter. Verified end to end in the `[litert]` env: a flax MLP →
  ONNX → LiteRT matches ONNX to ~7e-8 (float32 ULP). **ONNX → ExecuTorch**
  (`jaxility.policy.executorch`) is the parallel path (ONNX → torch →
  ExecuTorch) with the same export interface. Both converters are **external
  toolchains Jaxility orchestrates, not vendors** — gated behind the
  `[litert]` / `[executorch]` extras; absent the extra the export raises a
  structured `ToolchainError` (invariant 7, the acados pattern), and the
  conversion tests self-skip while the loud-failure path is tested everywhere.
- **Quantization recipes** (T-042, `jaxility.policy.quantize`). Post-training
  quantization through `onnx2tf`: `float16`, `dynamic_int8` (int8 per-channel
  weights, float activations), and `static_int8` (full-integer, with a
  representative-dataset calibration). Each recipe carries a documented
  degradation budget (`QUANT_TOLERANCE`), and `quantization_parity` checks the
  quantized model against the float32 policy under it — a quantization that
  degrades beyond budget fails loudly. Measured on a smooth MLP: float16
  ≈ 3.4e-4, dynamic_int8 ≈ 7.6e-3. `static_int8` fails loudly if the converter
  produces no integer model (no silent float fallback).
- **Dual-path composition** (T-043, `jaxility.compose`). A declarative
  `CompositionPlan` (ADR-002) binds the high-rate acados MPC and the lower-rate
  learned policy: the two periods, the `ArbitrationMode` (`residual` /
  `policy_primary`), the named `SafetyEnvelope` (state + actuator boxes,
  invariant 8), and the fallback policy. `arbitrate(...)` is the reference
  one-step decision: it falls back to the constraint-respecting MPC control on a
  policy timeout or a state-envelope breach, and **always clamps the final
  command into the envelope** — there is no path by which the learned policy
  drives the actuator outside it. The safety surface is the plan, not buried
  code, so it is readable + testable.
- **Dual-path Cartpole demo on real silicon** (T-044,
  `jaxility.compose.codegen`). `generate_dual_path_hil_source` emits a C binary
  that closes the loop with acados MPC + an embedded MLP policy
  (`MLPPolicy.from_flax`) combined by the T-043 arbiter, and the T-033 harness
  validates it: the **composition runs on a real Raspberry Pi 5 / Cortex-A76**
  and matches the host reference to ~ULP, and the **fallback is exercised in
  HIL** — forcing a policy timeout drops the command to the clamped MPC, while
  the policy genuinely changes the command when not falling back. The MLP
  forward is embedded in C; the LiteRT runtime path is validated separately
  (T-041).
- **Dual-path attestation** (T-045, `jaxility.compose.attest_dual_path`). A
  `DualPathAttestation` binds the controller's `Manifest` hash, the policy
  artifact's BLAKE3, and the `CompositionPlan` hash into one verifiable
  `content_hash` — so the attestation covers **both** deployed artifacts plus
  the safety surface. Recalibrating the robot, retraining the policy, or
  changing the envelope each moves the attestation.

## Coverage and equivalence as first-class concepts

- The coverage table is the source of truth for what the lowering
  pipeline emits. `assert_supported(op, dtype, target_family)` raises
  a structured `CoverageError` with the documented suggestion when
  the row is unsupported — no silent default.
- The equivalence tier consumes the same `(dtype, target_family)` key
  the coverage tier produces, so the two stay in lockstep.

## Errors

- `jaxility.errors.JaxilityError` is the root. Every library error
  derives from it (PATTERNS §6.1). Never `RuntimeError`,
  `ValueError`, or `AssertionError`. Subclasses cover Coverage,
  Artifact, Manifest, Target, Toolchain, Bench, HIL, Equivalence.

## Reproducibility

- **Deterministic builds** (invariant 5). Same inputs produce
  byte-identical manifests *and* byte-identical artifact content
  hashes. The cross-compile plan's `compiler_argv` is sorted /
  composed deterministically for the same reason.

## What runs on every CI

GitHub Actions: `lint + types + tests` on Python 3.10 / 3.11 / 3.12
on Ubuntu, plus the macOS local dev path. ADR-015 documents the
two-step `--no-deps` install path that gets Jaxterity / Jaxonomy
through CI without their private upstreams.

---

For anything you would expect to find here and don't — see
[`KNOWN_GAPS.md`](KNOWN_GAPS.md). Gaps are explicit and pointed-at;
silence is a bug.
