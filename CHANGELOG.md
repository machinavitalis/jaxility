# Changelog

All notable changes to Jaxility are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

Entries describe user-visible changes, grouped by area. Pure internal refactors
live in commits, not here.

---

## [1.0.0] - 2026-07-11

The initial public release, grouped by theme.

### Lowering — JAX → CasADi → acados

- **JAX → CasADi translator.** Walks `jax.make_jaxpr` and dispatches each
  primitive to a CasADi handler over a **smooth-op subset**: arithmetic, smooth
  elementwise (`sin`/`cos`/`tan`/`exp`/`log`/`sqrt`), shape ops,
  `dot_general`/`matmul`, `slice[static]`, `dynamic_slice[static]`, and
  `jnp.where[static]` (folds via `select_n`-with-literal). `jit`/`pjit` recurse
  transparently. `cond`/`while_loop`/`scan`/`dynamic_slice[traced]`/traced
  `select_n` are rejected with a structured `CoverageError` (invariant 7 — no
  silent mis-lowering). Pendulum, cartpole, and planar-quadrotor all translate
  bit-exact-modulo-ULP.
- **CasADi → acados OCP builder.** `OcpTemplateSpec` (Pydantic v2: horizon,
  time, costs, references, optional box bounds, solver options).
  `build_ocp(dynamics, spec) -> AcadosOcp` assembles a `LINEAR_LS` OCP from the
  translator's preserved `sx_inputs`/`sx_outputs` (so acados compiles the
  explicit ODE function without "free variable" errors). Cartpole LQR solves to
  status 0 end-to-end.
- **Host build path.** `HOST_DARWIN` / `HOST_LINUX` `Target` profiles +
  `current_host_target()`. `build_for_target(...) -> BuildBundle` runs acados'
  generate + compile, locates the `.dylib`/`.so`, and packages it into a
  content-addressed `Artifact` with a chain-linked `Manifest`. CLI:
  `jaxility build <zoo_name> [--target host]` with structured JSON exit errors.
- **Host equivalence.** Cartpole LQR and Cartpole TrackingMPC pass the
  equivalence check between the acados-solved trajectory and a JAX RK4 forward
  integration, under engineering-tight bounds documented in
  `test/EQUIVALENCE.md`.

### Control & MPC templates

- **LQR** (`jaxility.templates.lqr`) — derives `nx`/`nu` from the translated
  dynamics, fills sensible defaults (`Q_terminal = 10 × Q`), threads box bounds
  and solver options.
- **TrackingMPC** — thin layer over `lqr` for the time-varying-reference
  pattern; `set_reference_trajectory(solver, traj)` pushes per-stage `yref`.
- **WBC** — `WBCTask` (Pydantic v2) + `wbc(dynamics, tasks, ...)` weighted-
  combination factory. Ships a **weighted** (not strictly hierarchical)
  formulation; the null-space-projection variant is a later enhancement.
  Upstream gap flagged: Jaxterity has no Task DSL yet, so `WBCTask` is the
  Jaxility-side placeholder.
- **Centroidal MPC** — `centroidal_mpc` over `lqr` with centroidal-state /
  wrench naming. Single-contact, 6-state simplification; multi-contact +
  angular-momentum bookkeeping is a later enhancement.

### Learned-policy deployment lane

- **JAX policy → ONNX export** (`jaxility.policy.export_policy_to_onnx`). Walks
  the policy jaxpr and emits a self-contained ONNX `ModelProto` over the
  **smooth-MLP subset**: dense (`dot_general` → `MatMul`), bias (`add`), smooth
  activations (`tanh`, `logistic`, `relu`). Coverage-gated via
  `SUPPORTED_PRIMITIVES`; anything else raises a structured `CoverageError`.
  `PolicyOnnxModel` carries the ONNX bytes + provenance. Bit-exact vs ONNX
  Runtime on flax MLPs. New `[policy]` extra (onnx / onnxruntime / flax).
  CNN/RNN handlers are the next increment; transformers deferred to v0.2.
- **ONNX → LiteRT / ExecuTorch.** `jaxility.policy.litert` orchestrates
  ONNX → LiteRT (`.tflite`) via `onnx2tf` (the priority on-device path);
  `litert_parity` checks it against ONNX Runtime — a flax MLP → ONNX → LiteRT
  matches ONNX to **~7e-8** (float32 ULP). `jaxility.policy.executorch` is the
  parallel ONNX → torch → ExecuTorch (`.pte`) path with the same interface. Both
  converters are **orchestrated, not vendored** — behind the `[litert]` /
  `[executorch]` extras; absent the extra, export raises a structured
  `ToolchainError`.
- **Quantization recipes** (`jaxility.policy.quantize`). Post-training
  quantization through `onnx2tf`: `float16`, `dynamic_int8` (int8 per-channel
  weights + float activations), and `static_int8` (full-integer, needs a
  `representative_data` calibration batch). Each recipe has a documented
  degradation budget (`QUANT_TOLERANCE`) checked by `quantization_parity`.
  Verified in `[litert]` on a smooth MLP: **float16 ≈ 3.4e-4, dynamic_int8 ≈
  7.6e-3** vs float32. `static_int8` fails loudly if the converter produces no
  integer model (no silent float fallback).
- **Dual-path runtime composition** (`jaxility.compose`). The declarative
  contract binding the high-rate acados MPC and the lower-rate learned policy.
  `CompositionPlan` carries the two periods, the `ArbitrationMode` (`residual` /
  `policy_primary`), the named `SafetyEnvelope` (state + actuator boxes,
  invariant 8), and the fallback policy. `arbitrate(...)` is the reference
  one-step decision the on-target C runtime mirrors: it falls back to the
  constraint-respecting MPC control on a policy timeout or a state-envelope
  breach, and **always clamps the final command into the envelope** — no path
  lets the learned policy drive the actuator outside it. Kept tooling-agnostic.
- **Dual-path Cartpole demo on real silicon.**
  `jaxility.compose.codegen.generate_dual_path_hil_source` emits a C binary that
  closes the loop with acados MPC + an **embedded MLP policy**
  (`MLPPolicy.from_flax`, the forward pass codegen'd to C) combined by the
  arbiter. The **dual-path composition runs on a real Raspberry Pi 5 /
  Cortex-A76** and matches the host reference to ~ULP, and the **fallback is
  exercised in HIL** (forcing the policy timeout drops the command to the
  clamped MPC; the policy genuinely changes the command otherwise).
- **Dual-path real-time cost benchmark.**
  `jaxility.compose.generate_dual_path_bench_source` times the *composed*
  control cycle (acados solve + embedded MLP + arbiter + clamp). On the Pi 5
  (2.4 GHz, n=5000): dual-path mean **97.9 µs** / p99 129 µs vs single-path
  99.7 / 135 — the learned-path + arbiter overhead is within run-to-run noise,
  so the composed loop meets 1 kHz with the same ~4.6× margin as single-path.
- **Dual-path attestation.** `jaxility.compose.attest_dual_path` →
  `DualPathAttestation` binds the acados controller's `Manifest` hash, the
  policy artifact's BLAKE3, and the `CompositionPlan` hash into one verifiable
  `content_hash`, covering **both** deployed artifacts plus the safety surface.
  Any of the three changing (recalibrate / retrain / re-envelope) moves the
  attestation.

### Embedded targets & on-device runtime

- **Raspberry Pi 5 / Cortex-A76 `Target` profile** — the first real target.
- **Cross-compilation with a pinned toolchain.** The **Arm GNU 15.2.Rel1**
  toolchain is pinned for the aarch64 lane. `jaxility.builder_deps` cross-builds
  the acados / blasfeo / hpipm static archives **from source** (ADR-018) via
  `cmake/toolchains/aarch64-none-linux-gnu.cmake` (`-DBUILD_SHARED_LIBS=OFF`,
  `-DBLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76`); `cross_build_for_target(...,
  deps=...)` links the controller against them and records each archive's
  BLAKE3 hash in the manifest (`dep-archive:<name>`). `docker/cross-aarch64.Dockerfile`
  + `scripts/cross_build_attested_pi5.sh` give a reproducible cross-build. The
  cross-compiled artifact records `aarch64-none-linux-gnu-gcc 15.2.1` + the dep
  hashes and verifies; on the Pi it passes the same HIL parity (~1e-15) and
  benchmarks **faster + tighter** than the native build (mean 87 µs / max 121 µs
  vs 98 / 265). glibc is a measured non-issue (artifact references only
  `GLIBC ≤ 2.34` ≤ the Pi's 2.36). Prebuilt binary archives are **never**
  vendored.
- **On-target C runtime** (`runtime-c/`). A byte-deterministic `.a` archive
  built per family. `rt_posix.c` provides PREEMPT_RT control-thread placement
  (`jx_rt_pin_to_core` / `jx_rt_set_realtime_priority` / `jx_rt_configure`;
  Cortex-A / Linux only — `JX_RT_ERR_UNSUPPORTED` elsewhere, `EPERM` surfaced as
  `JX_RT_ERR_PRIVILEGE`). `deploy_posix.c` (`jx_deploy_run`) is the on-target
  glue: it `dlopen()`s the cross-compiled controller `.so`, resolves the
  three-symbol controller ABI (`jx_controller_init` / `_step` / `_period_ns`),
  threads the bump arena into init, places the thread for RT, and drives the
  step from the cycle scheduler. Distinct negative `JX_DEPLOY_ERR_*` codes for
  each setup failure.

### Hardware-in-the-loop & benchmarking

- **HIL harness** (`jaxility.hil`). A deployed artifact runs on the target and
  emits a per-cycle JSON-Lines trace; the harness parses it and compares against
  the host reference under the documented `(target_family, dtype, quantity)`
  tolerances, so HIL parity routes through the same equivalence surface as every
  other check (invariant 6). `run_hil(...) -> HILReport`; `TargetRunner` with
  `LocalRunner` (subprocess) and `SshRunner` (runs a binary on a tethered
  target; `deploy_binary` ships a cross-built artifact over); `StateSchema` /
  `parse_trace` / `CARTPOLE_SCHEMA` are the self-describing trace contract,
  where every malformed / misaligned / miscounted trace is a loud `HILError`.
- **acados controller HIL binary** (`jaxility.hil.controller`). Generates and
  builds a closed-loop binary around the real acados controller (OCP control +
  an acados ERK *sim* over the same CasADi-from-JAX dynamics), emitting the
  trace each cycle. `CARTPOLE_LQR_SCHEMA` is the nx=4 trace schema. Measured
  host-vs-host divergence ~1e-14.
- **Runs on real Cortex-A76 silicon.** `build_controller_on_target` ships the
  host-generated acados C to a tethered Pi 5 and builds it natively;
  `test/hil/test_on_pi_controller.py` asserts step-locked parity against the
  host reference — measured **~1e-15 (ULP)**, cross-architecture (macOS-arm64
  host ↔ Pi-aarch64).
- **Benchmark harness** (`jaxility.bench`). `BenchRecord` / `SolveTiming`
  structured per-cycle stats (min/mean/p50/p99/σ, jitter, peak RSS,
  `meets_1khz`), tied to the source manifest hash. `generate_controller_bench_source`
  times the bare `acados_solve` with `CLOCK_MONOTONIC` after warm-up; CLI
  `jaxility bench cartpole --target {host,pi5}`. Measured on the Pi 5
  (performance governor, 2.4 GHz, n=5000): **mean 98 µs, p50 96 µs, p99 134 µs,
  max 265 µs, σ 7.4 µs, RSS 5.9 MiB — meets 1 kHz with ~3.8× worst-case
  margin.** Energy is not measured (no on-board sensor; out of scope).

### Cartpole end-to-end & robot-rooted dynamics

- **Cartpole closed-form dynamics fitted from the Robot.** `jaxility/zoo/cartpole`'s
  `_dynamics_factory` no longer hand-hardcodes `(g, mc, mp, L)` — it reads them
  from the upstream Jaxterity Robot via `jaxterity.zoo.cartpole.reduced_params(robot)`,
  so calibrating a mass or length propagates into the lowered binary: the
  attestation handle and the deployed closed-form dynamics move together
  ("one model, one truth", at the parameter level). ADR-016 stands — the
  deployed plant is the closed-form, not MJX. Byte-identical for the
  uncalibrated zoo robot.
- **`examples/cartpole_end_to_end.py`** — the integrated launch pipeline:
  calibrated Jaxterity Robot → export → acados LQR controller → host build →
  closed-loop HIL parity → solve-time bench → attestation manifest. The
  controller is built **from the robot**, so recalibrating the pole moves the
  compiled artifact (nominal vs 2× mass → the artifact hash and source handle
  both change). When `JAXILITY_HIL_SSH_HOST` names a tethered Pi 5, the script
  also ships the controller to the board, compiles it natively, and verifies
  **on-Pi HIL parity + a 1 kHz bench** (solve mean ~103 µs / max ~175 µs).

### Foundation — attestation, manifest, coverage

- **Equivalence-check protocol + tolerance policy.** `jaxility.testing.compare(...)`
  returns an `EquivalenceReport` with per-quantity divergence and a structured
  suggestion on failure. `TOLERANCE_TABLE` + `test/EQUIVALENCE.md` are the
  contract; tests cross-check the two stay in sync. A violation requires both
  the abs and rel bounds to be exceeded at the same step.
- **`Target` abstraction** (ADR-003). Frozen Pydantic v2 model (`name`,
  `family`, `schema_version`, `toolchain`, `vector_extensions`, `npu`, `memory`,
  `realtime`, `vendor_sdk_paths`, `quirks`); `.hash` is BLAKE3 over the
  canonical-JSON encoding. Ships `MOCK_CORTEX_A` / `MOCK_CORTEX_M` and bundles
  `jaxility.manifest.canonical_dumps`.
- **Manifest schema v0 + signing + verify** (ADR-005, ADR-008). `Manifest` with
  the eight chain fields; `HexBytes` annotated type for bytes-as-hex JSON
  round-trips; `Signer` Protocol with the OSS `HashChainSigner` (unsigned hash
  chain); `verify_manifest` returns a `ChainReport` with an optional
  out-of-band `expected_content_hash` for tamper detection. CLI `jaxility verify
  <manifest> [--expected-hash <hex>]`, exit codes 0 / 2 / 40. The build
  timestamp is factored *out* of the hashed payload, so identical builds at
  different times stay byte-identical (invariant 5).
- **Coverage declaration + `CoverageError`.** The `(op, dtype, target_family) →
  CoverageEntry` table the lowering gates every op against; `assert_supported`
  raises a structured `CoverageError` on unsupported combinations and unknown
  keys. CLI `jaxility coverage [--target <family>]` emits the matrix as
  Markdown. Seeds the `JaxilityError` hierarchy.
- **`Artifact` + write-once content-addressed cache.** Frozen Pydantic v2
  (`payload`, `content_hash` verified against `blake3(payload)` at construction,
  `source_manifest_hash`, `target_profile_hash`, `build_log`). `BuildLogEntry`
  uses a relative `offset_us`, not wall-clock (invariant 5). `ArtifactCache` is
  a write-once store at `~/.cache/jaxility/artifacts/<hash>/`;
  `JAXILITY_CACHE_ROOT` relocates the root.
- **End-to-end mock lowering pipeline.** `jaxility.testing.mock_lower(...)`
  composes the contracts into a Python-only build producing a `MockArtifactBundle`
  (`Artifact` + `Manifest` + `.simulate()`). `Source` Protocol + `MockSource`
  form the duck-typed interface. Confirms invariant 5 end-to-end: same source +
  target + dtype + n_steps → byte-identical artifact regardless of build
  timestamp.
- **Jaxterity attestation-chain integration** (invariant 2). `JaxteritySource`
  wraps a real `jaxterity.robot.Robot` and exposes the `Source` Protocol. The
  full chain (`Robot.attestation_handle` → manifest → `artifact.source_manifest_hash`
  → `manifest.content_hash`) verifies end-to-end; mutating any of the four
  canonical handle inputs changes the artifact hash; tampering with
  `source_attestation_handle` is caught by `verify_manifest(expected_content_hash=...)`.
  A `require_calibration_state` gate lets production callers refuse uncalibrated
  robots.
- **Reference robot zoo.** `jaxility.zoo` registry + four entries — `cartpole`
  (LQR, `mock-cortex-a`, real), `so100` (WBC, `mock-cortex-a`, real),
  `crazyflie` (TrackingMPC, `mock-cortex-m`, `float32`, **stub**), and
  `berkeley_humanoid_lite` (CentroidalMPC, `mock-cortex-a`, **stub**). Each
  mock-builds end-to-end with a verifying manifest and a distinct artifact hash;
  each README documents source, license, upstream status, and remaining work.

### Packaging, CLI, MCP & docs

- **Repository skeleton.** `pyproject.toml` with the core deps (`jaxterity`,
  `casadi>=3.7,<3.8`, `pydantic>=2.6`, `cryptography`, `blake3`) and the
  optional-extras surface (`[acados]`, `[litert]`, `[executorch]`, `[fvp]`,
  `[bench]`, `[targets-cortex-m]`, `[targets-cortex-a]`, `[targets-qualcomm]`,
  `[policy]`, `[mcp]`, `[dev]`, `[all]`). Subpackages under `jaxility/`; `test/`
  with a smoke suite; `runtime-c/` for the MISRA-aware on-target C runtime.
- **CLI dispatcher** (`jaxility/cli`) — `verify` / `coverage` / `build` /
  `bench` / `mcp serve`.
- **MCP scaffold** (`jaxility/mcp/server.py`) — FastMCP server with a `TOOLS`
  source-of-truth registry that wraps the CLI (ADR-013).
- **Claude Code skill** — `.claude/skills/jaxility/SKILL.md` with a root
  `SKILL.md` symlink keeping it discoverable.
- **Continuous integration.** GitHub Actions installs Jaxility with `--no-deps`
  plus an explicit dev-tools list (ADR-015), since the upstream packages are
  private; structural cross-compile, Tier-B Cortex-M, and Tier-B Pi 5 jobs run
  per PR, and a `cross-deps` job (`workflow_dispatch` + weekly) exercises the
  real aarch64 cross-build.
- **Contract docs.** README support table + lowering diagrams;
  [`CLAIMS.md`](CLAIMS.md) / [`KNOWN_GAPS.md`](KNOWN_GAPS.md) ledger kept
  symmetric with anti-drift tests; `AGENTS/` orientation set including the
  upstream license analysis (CasADi LGPL flagged) and the toolchain-pinning
  policy.

### Fixed

- **Arena absolute-address alignment.** `runtime-c/src/arena.c`
  (`jx_arena_alloc_aligned`) now aligns the *absolute* address `base + used`
  rather than the offset `used` alone, so an alignment request stronger than the
  base buffer's own alignment no longer returns an under-aligned pointer.
  Surfaced on the Pi 5, where the runtime's static buffer landed at
  `addr % 64 == 16`; the host had silently masked it. It matters because the
  arena backs acados working memory and `dcache_aarch64.c` operates at 64-byte
  cache-line granularity. A regression test constructs the exact
  `addr % 64 == 16` condition so it fails on every platform if the bug returns.
- **Loud-failure hardening across the contract surfaces.** The manifest,
  coverage, artifact-cache, and equivalence surfaces were hardened so no
  uncovered or mismatched path passes silently: the full `JaxilityError`
  hierarchy (`EquivalenceError`, `ManifestError`, `TargetError`, `SourceError`,
  `HILError`, `BenchmarkError`, `ToolchainError`, `CoverageError`,
  `ArtifactError`) is present and importable; the per-build coverage gate fires
  unconditionally on an empty `(family, dtype)`; `ArtifactCache.load` verifies
  the stored hash; `verify_manifest` catches an unknown signer and returns a
  structured report; hex parsing rejects embedded whitespace; hashed records
  carry `schema_version`; and the TOCTOU window in `ArtifactCache.store` is
  closed. This closed all findings (11 major + 5 minor) from the first skeptical
  review pass.
