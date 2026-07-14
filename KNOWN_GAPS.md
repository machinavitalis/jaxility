# Known gaps

> **Maintainer note (the doc-drift rule).** This file is half of a
> two-document symmetric contract with [`CLAIMS.md`](CLAIMS.md);
> the `README.md` quick-reference table is the third leg. **When a
> gap closes in code, the PR that closes it must delete (or rewrite)
> the entry below AND add the corresponding claim to `CLAIMS.md` AND
> bump the README row if the gap was reflected there.** Anti-drift
> tests in `test/unit/test_top_level_docs.py` enforce structural
> rules (target enumeration, cross-references, key phrases) but
> cannot catch a closed gap that nobody moves. See
> [`AGENTS/CONTEXT.md`](AGENTS/CONTEXT.md) §"Contract docs and the
> doc-drift rule" for the full governance.
>
> Phrases this file uses for not-yet-implemented work — "pending",
> "deferred", "still out of scope", "queued", "not yet" — are the
> grep targets future agents use to find candidate gaps when they
> ship a capability. If your PR resolves one of those phrases,
> delete it.

This document is the symmetric counterpart to [`CLAIMS.md`](CLAIMS.md):
the list of things Jaxility **explicitly does not do** at this
version. Every gap is documented with a pointer to where the gap is
codified — a coverage row, an ADR, a task ID — so a downstream
consumer never has to guess what's a bug vs. what's deliberate.

Stage in the roadmap: the Pi 5 runtime/HIL/bench lane
and the learned-policy lane have both landed.

The rule (PATTERNS §6, invariant 7): every gap below is announced
loudly at the boundary that would have used the missing capability.
There are no silent defaults. If you exercise a path that lands in a
gap, you get a structured error naming the documented workaround.

---

## JAX-side translation gaps

These are surfaces the coverage table (`jaxility/lowering/coverage.py`)
explicitly marks unsupported. Every row carries a `suggestion` field
naming the documented workaround.

### MJX-driven Robot dynamics

Calibrated robots from [Jaxterity](https://github.com/machinavitalis/jaxterity)
that expose their dynamics through MJX (MuJoCo JAX) **do not flow
through the JAX → CasADi translator**.

Why: tracing the cartpole MJX ode surfaces 34 distinct primitives
across 2,375 jaxpr equations, including an unconditional `while_loop`
(constraint solver) plus several primitives with no handler.
Smoothing was rejected for prohibitive unroll cost, accuracy
degradation during contact phases, and upstream coordination cost.

**Documented in:** `AGENTS/DECISIONS.md` ADR-016.
**Coverage row:** `lax.while_loop` (suggestion names MJX explicitly).
**Workaround:** per-robot **closed-form** dynamics function. The zoo
entries in `jaxility/zoo/` follow this contract: `cartpole` uses an
analytical four-state model, `crazyflie` (T-110) a 13-state Newton-Euler
floating-base model (both matching the MJX reference to ~ULP), and
`so100` (T-111) a Featherstone **Articulated Body Algorithm** — a
manipulator needs `M(q)⁻¹`, which the coverage table has no solve for,
and ABA computes `q̈` in O(n) using only spatial matmuls and scalar
reciprocals, so it lowers. All three lower to CasADi. The scalars/spatial
tree are sourced from the calibrated Robot (not hardcoded), so
calibration propagates to the deployed dynamics even though the
*structure* is a closed form, not MJX. Joint damping/friction remain
sim-only by design.

**so100 fidelity is manipulator-grade, not ULP** (`~1e-5` rel vs MJX,
tested at a `1e-4` bound). An *independent* ABA and MuJoCo's internal
`cinert` CRB diverge by ~1e-6 on this featherweight arm (distal inertias
~1e-5) — a deterministic representational floor, not a bug or round-off.
The flyers hit ULP because they are well-conditioned with explicit closed
forms; recursive manipulator dynamics inherently cannot.

Note the closed-form dynamics **lower**; the crazyflie (quaternion
tracking MPC) and so100 (WBC) *OCP templates* are follow-ons, so
`jaxility build {crazyflie,so100}` fails structurally until they land.

### `lax.while_loop` for any reason

Dynamic iteration counts are not consumed by acados' OCP formulation.

**Coverage row:** `lax.while_loop`.
**Workaround:** bound the loop statically (Python `range`) or fold
the loop body into the OCP horizon. For MJX-emitted while_loops
(constraint solver), supply a closed-form per-robot dynamics function
instead of going through MJX.

### `lax.cond` / `jnp.where` over **traced** predicates

Non-smooth branches do not survive the CasADi → acados pipeline.

**Coverage rows:** `lax.cond[traced]`, `jnp.where[traced]`.
**Workaround:** smoothing approximation (sigmoid/softmax with a named
sharpness factor; mirrors Jaxterity invariant 6) or restructure the
dynamics to avoid the conditional.

### `dynamic_slice` over **traced** start indices

Traced start indices require a CasADi conditional at every load; this
exits the smooth-op subset.

**Coverage row:** `dynamic_slice[traced]`.
**Workaround:** use `jax.lax.dynamic_slice` with literal start indices
(supported via the `dynamic_slice[static]` row, A1) or rewrite as a
plain `operand[a:b]` slice.

### Dynamic array shapes

Incompatible with embedded codegen.

**Coverage row:** `dynamic_shape`.
**Workaround:** fix the shape at trace time, or move the dynamic-shape
code outside the lowered region.

---

## Template gaps

### Hierarchical WBC

The **weighted** WBC template ships; tasks are expressed as
relative cost weights, not strict hierarchies. Hierarchical WBC
(QP-staged priorities) is planned.

**Workaround for now:** if you need a strict task priority, scale the
high-priority cost weight up by enough orders of magnitude that the
weighted solution converges to the hierarchical one. We do not promise
the convergence is exact.

### Multi-contact Centroidal MPC

**Single-contact** centroidal MPC ships. The contact-phase
schedule has length 1; the controller cannot plan a heel-strike or a
toe-off. Multi-contact lands later.

---

## Build / deployment gaps

### Real-hardware execution in *standing CI*

Real-hardware execution itself is **no longer a gap**: the acados Cartpole
controller runs on real Pi 5 / Cortex-A76 silicon — native and
pinned-cross-compiled — with step-locked HIL parity and a 1 kHz benchmark, and
the robot-rooted launch demo (Jaxterity T-102) runs there too. See
[`CLAIMS.md`](CLAIMS.md) §"Hardware-in-the-loop on real silicon".

**What remains:** the on-silicon HIL/bench tests are **opt-in** — they need a
tethered board (`JAXILITY_HIL_SSH_HOST`) and self-skip in the hardware-free
per-PR CI. There is no standing *automated* hardware gate; the silicon runs are
operator-driven.

### Tier B of T-031 — closed (both lanes live)

**Cortex-M lane.** `arm-none-eabi-gcc 15.2.1` runs locally on the dev
host (Homebrew `gcc-arm-embedded`) and on CI runners.
`verify_toolchain_installed(CORTEX_M4)` matches the pin; the end-to-end
real-compile test produces an ELF `.o` for Cortex-M4. The
`_FAMILY_CFLAGS` table carries rows for `cortex-m4`, `ethos-u55`, and
`ethos-u65` (the M55 host inside each Ethos pairing). M-profile family
flags include `-c` so the output is a relocatable object the
deployment runtime project links into its final ELF (linker scripts +
startup files are T-052's surface).

**Pi 5 lane.** Arm GNU `aarch64-none-linux-gnu-gcc 15.2.1` runs on
**CI Linux x86_64 runners** (installed via the
`arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz`
archive). Arm does **not** ship a darwin-arm64 host build of this
chain in 15.2.Rel1 — that's a real upstream constraint, not a
Jaxility decision — so on Apple Silicon dev hosts the Pi 5 Tier-B
tests skip locally and run only in CI. The CI step
`.github/workflows/ci.yml`/`Install Arm GNU aarch64-linux toolchain
(15.2.Rel1)` does the install; a cache key keyed on the release name
keeps it cheap. `test_real_aarch64_detect_matches_pin` and
`test_real_aarch64_compile_minimal_cortex_a76` exercise detection
and a real ELF shared-object compile, respectively.

**Linker gap (Pi 5 / aarch64 lane — closed in code; CI-verification
pending).** The cross-build orchestrator `jaxility.builder_deps`
(ADR-018, T-031 follow-up) cross-builds the acados / blasfeo / hpipm
static archives **from source** for an Arm target via a CMake toolchain
file (`cmake/toolchains/aarch64-none-linux-gnu.cmake`), and
`cross_build_for_target(..., deps=...)` links the controller against
them — producing a real ELF shared object instead of dying at link.
The Tier-B test `test_cross_build_for_target_links_with_deps` proves
this end-to-end when both the aarch64 toolchain **and** an acados
source tree are present. **Still pending:** the standing CI gate does
not provision an acados source checkout, so the per-PR run skips the
real cross-build; the `cross-deps` `workflow_dispatch` job exercises it
on demand. Promoting that to a blocking gate is queued behind T-034.

**Cortex-M (`arm-none-eabi`) lane — still open.** The bare-metal
archives are out of scope for ADR-018 (hosted-ABI aarch64 only); they
are T-052 and materially harder (no OS/malloc; newlib +
embedded blasfeo config).

**Archive byte-determinism — closed (T-113).** Upstream acados/blasfeo
invoke `ar` without the deterministic (`D`) flag, so their `.a` archives
embed build member timestamps/uid/gid. `execute_dep_build` now normalizes
each archive with `objcopy --enable-deterministic-archives` before hashing,
so a rebuild from identical inputs yields byte-identical archives — the
`dep-archive:<name>` manifest hashes are reproducible, not merely recorded.
Verified by the Tier-B `test_build_cross_deps_is_byte_deterministic`
(builds twice, asserts equal hashes). *Object-level* determinism of the
individual `.o` members (compiler-embedded) is not separately claimed; the
archive-metadata source of drift — the documented one — is closed.

**Local dev gap on Apple Silicon (Pi 5 lane).** No darwin-arm64 host
build of `aarch64-none-linux-gnu-gcc` is published by Arm, so the
aarch64 dep cross-build cannot run on Apple Silicon dev hosts.
Workarounds for local iteration: (a) cross-compile inside an aarch64
Linux Docker container, or (b) push to a feature branch and let CI
exercise the test. Option (b) is the recommended path; the Tier-B tests
light up automatically wherever the toolchain + acados source are
present.

**Tracked in:** `jaxility.builder_deps`; ADR-018; the
`extra_link_args` / `deps=` seam in `plan_cross_compile` /
`cross_build_for_target`; the CI `cross-deps` job; this section.

### Pi 5 runtime layer — partial

The first runtime ingredients landed in `runtime-c/`: the no-malloc
arena, the POSIX cycle scheduler with drift correction, and the
D-cache / I-cache coherence sequence (`DC CIVAC` + `IC IVAU` + `ISB`)
closing the PI5 quirk `d-cache-clean-required-for-codegen-buffers`.
The Python build orchestrator (`jaxility.runtime.c_runtime`) produces
`libjaxility_runtime_cortex-a76.a` on CI Linux runners (verified by
`test_build_real_pi5_runtime_archive`).

**PREEMPT_RT thread-affinity helpers — landed (T-032).** `rt_posix.c`
provides `jx_rt_pin_to_core` / `jx_rt_set_realtime_priority` /
`jx_rt_configure`, host-tested on CI Linux (real path) and macOS
(unsupported branch), and built into `libjaxility_runtime_cortex-a76.a`.
Calling them at the right point in the on-Pi deployment loop is part of
the deployment-glue work still pending below.

**Deployment launcher — landed (T-032).** `deploy_posix.c`
(`jx_deploy_run`, ABI in `jaxility_runtime/deploy.h`) dlopen()s the
controller `.so`, threads the arena into it, and drives it from the
cycle loop; `jaxility.runtime.deploy.plan_deploy_launcher` cross-links
the `jx_deploy_<family>` launcher binary against the runtime archive.
The host test exercises the dlopen / arena-threading / cycle wiring end
to end without hardware. **Still pending:** the controller side of the
ABI — the thin shim that implements `jx_controller_init/step/period_ns`
over acados `acados_create` / `acados_solve` / `acados_get` — is
generated alongside the controller C in T-034, not here; and the
launcher's real cross-link + on-Pi run is HIL-validated in T-033.

**Still out of scope (next runtime PRs):** GPIO bring-up for the
Cartpole hardware; GICv2 IRQ priority tuning (PI5 quirk
`gicv2-irq-priority-needs-tuning`); the final link that combines the
launcher + controller `.so` + runtime archive against
acados / blasfeo / hpipm (the cross-built static archives now exist via
`jaxility.builder_deps` / ADR-018 — see the "Linker gap" entry above;
what remains is threading the runtime `.a` into the same link).

**Tracked in:** T-032 follow-up PRs; PI5 quirks
`gicv2-irq-priority-needs-tuning` and `preempt-rt-soft-not-hard`;
the `extra_link_args` seam in `plan_cross_compile`.

### HIL parity

**Harness landed (T-033); generated-artifact parity still pending.** The
step-locked HIL harness now exists and runs on real Cortex-A76 silicon:
`jaxility.hil.run_hil` drives a deployed artifact through a
`TargetRunner` (`LocalRunner` subprocess, or `SshRunner` to a tethered
Pi 5), parses its per-cycle JSONL trace (`jaxility.hil.parse_trace`),
and compares it against the host reference under the documented
`cortex-a76` × `float32` tolerances (`test/EQUIVALENCE.md`). It is
validated end to end on the launch hardware against a deterministic
fixture (`test/hil/fixtures/cartpole_hil.c`).

**Generated controller HIL — on host (T-034 slice 1).** The real acados
Cartpole controller is now HIL-validated *on the host*:
`jaxility.hil.controller` generates + builds a closed-loop binary (acados
OCP control + acados sim plant) that passes step-locked parity against the
host reference (acados control + JAX ERK4 plant) under `cortex-a76` ×
`float64`, measured to ~1e-14 (ULP). This is the host stand-in for the
on-silicon gate.

**Generated controller HIL — on real silicon (T-034 / T-035).** The acados
Cartpole controller now runs **on the Pi 5**: `build_controller_on_target`
ships the generated C and builds it natively on the Pi, and
`test/hil/test_on_pi_controller.py` asserts step-locked parity vs the host
reference at `cortex-a76` × `float64` — measured ~1e-15 (ULP). The
benchmark (`jaxility bench cartpole --target pi5`) measures the solve at
mean 98 µs / max 265 µs, meeting 1 kHz.

**Attested artifact — DONE (T-037).** The controller is now also built with
the **pinned Arm GNU 15.2.Rel1** toolchain in a Docker container
(`docker/cross-aarch64.Dockerfile` + `scripts/cross_build_attested_pi5.sh`),
linking acados/blasfeo/hpipm cross-built as static archives. Its manifest
records `aarch64-none-linux-gnu-gcc 15.2.1` + the dep-archive BLAKE3 hashes
(invariant 3) and verifies. Deployed to the Pi, it passes the same HIL parity
(~1e-15) and benchmarks *faster + tighter* than the native build (mean 87 µs
/ max 121 µs vs 98 / 265 µs). The glibc concern proved a non-issue — the
artifact references only `GLIBC ≤ 2.34` (≤ the Pi's 2.36), so it loads
dynamically with no static-libc workaround needed.

### Extended target families in `_FAMILY_CFLAGS`

The whole **aarch64-linux lane now cross-compiles** (T-114): `_FAMILY_CFLAGS`
carries A-profile rows for `cortex-a76`, `cortex-a55`, `cortex-a78`,
`cortex-a710`, `neoverse-n1`, and `qualcomm-iq10` — they share one toolchain
and differ only by `-mcpu`, each with a real ELF `.so` Tier-B compile test.

**Still missing** `_FAMILY_CFLAGS` cross-compile rows:
- **Apple Silicon** (`apple-silicon`) — cross-Apple-Silicon dispatch needs the
  Universal2 / clang refinement noted under "Apple clang version regex".
- **Bare-metal Cortex-M beyond `cortex-m4`** and the **NPU codegen** families
  (Ethos-U, Hexagon) are separate lanes, not a missing cflags row — see the
  Cortex-M (T-052) and NPU-codegen gaps.

Calling `plan_cross_compile` for a family without a row still raises
`TargetError("no cross-compile cflags registered for target family ...")`
naming the table to extend (loud-fail keeps it from shipping by accident).

---

## Manifest / toolchain pin gaps

### Real `acados` ToolchainPin

**Closed (A6).** `Manifest.toolchain_versions` now records
`"acados-template"` and `"acados-library"` separately via the
detection in `jaxility.manifest.toolchain_detect`. Pinned upstreams
are importable constants
(`JAXILITY_ACADOS_TEMPLATE_PIN`, `JAXILITY_ACADOS_LIBRARY_PIN`).
The `acados_template` Python interface's silent `"unknown"` fallback
is gone; detection raises `ToolchainError` when the package is
missing, and the C-library detection returns a self-explaining
`library-unknown:<reason>` marker when the local source tree is
absent — never a silent placeholder.

### Arm GNU toolchain SHA-256 not yet pinned

PI5 and all extended Arm toolchain pins carry the `UNVERIFIED_SHA256`
sentinel (`expected_sha256="unverified"`) — no real archive/binary hash
is pinned yet, so toolchain-acquisition integrity is **not enforced**.

What T-112 *did* close: the cross-build no longer silently implies the
toolchain was checked. `cross_build_for_target` calls
`resolve_toolchain_integrity`, which records
`toolchain-integrity:<binary>` in the manifest as `"unverified"` (loud,
not silent) and — the moment a real SHA-256 is pinned — verifies the
installed binary and **hard-fails the build on a mismatch**. So the
verified path is wired and enforced; what remains is pinning the actual
hashes (host-specific per Arm release) for the shipped targets.

**Tracked in:** `resolve_toolchain_integrity` / `verify_toolchain_integrity`
(`jaxility.builder_cross`); PI5 + extended-targets pins.

### Apple clang version regex

`APPLE_SILICON.toolchain.version_regex` matches the Apple-clang
banner, not LLVM upstream. Cross-Apple-Silicon dispatch (Universal2
shells) will need refinement before macOS becomes a real deployment
target.

---

## Learned-policy lane

Later-roadmap work; in progress (T-040 landed).

- **JAX → ONNX export** (T-040) — **landed for the smooth-MLP subset.**
  `jaxility.policy.export_policy_to_onnx` handles dense layers
  (`dot_general` → `MatMul`), biases, and smooth activations (`tanh` /
  `sigmoid` / `relu`). **Still out of scope here:** basic CNN/RNN ops
  (conv, pooling, GRU/LSTM cells) — the next T-040 increment; and
  **transformers** (attention, layernorm), deferred to v0.2. Exports are
  single-observation (unbatched); a batched export path is not wired.
  Unsupported primitives raise `CoverageError` (no silent mis-export).
- **ONNX → LiteRT / ExecuTorch** (T-041) — **landed.** LiteRT
  (`jaxility.policy.litert`, `onnx2tf` → `.tflite`) is the priority path and is
  parity-verified (~7e-8 vs ONNX in the `[litert]` env); ExecuTorch
  (`jaxility.policy.executorch`) is the parallel path with the same interface.
  The converters are external toolchains gated behind the `[litert]` /
  `[executorch]` extras (the acados pattern). **Still out of scope here:**
  ExecuTorch parity is verified only where its (heavy) torch+executorch tooling
  is installed; a per-target op-coverage table beyond the smooth-MLP set lands
  with the CNN/RNN handlers (T-040).
- **Quantization recipes** with parity tests (T-042) — **landed.**
  `jaxility.policy.quantize` ships `float16` / `dynamic_int8` (verified ~3.4e-4
  / ~7.6e-3 vs float32) and `static_int8` (full-integer, representative-dataset
  calibration). Each has a documented degradation budget; parity checks against
  it. **Still out of scope here:** `static_int8` full-integer output is
  model-dependent (onnx2tf skips it for some small MLPs) — the recipe fails
  loudly rather than falling back; broad static-int8 coverage + on-target int8
  benchmarks are the follow-up.
- **Dual-path runtime composition** (acados MPC + learned policy,
  T-043) — **landed.** `jaxility.compose.CompositionPlan` + `arbitrate`
  declare the rates, arbitration mode, named `SafetyEnvelope`, and fallback;
  the arbiter always clamps into the envelope and falls back to the MPC on
  timeout / envelope breach (invariant 8). **Still out of scope here:** the
  on-target C arbiter (T-044 mirrors this reference in the runtime), and the
  ADR-017 L4CasADi embedded-learned-function sub-lane (Proposed; not adopted —
  the composition is tooling-agnostic).
- **Cartpole dual-path demo on Pi 5** (T-044) — **landed.**
  `jaxility.compose.codegen` emits a C binary (acados MPC + embedded MLP policy
  + T-043 arbiter); the **composition runs + HIL-matches on a real Pi 5** and
  the **fallback is exercised in HIL** (host). **Still out of scope here:** the
  on-Pi run embeds the MLP forward in C (not the LiteRT runtime — that path is
  T-041), and the policy is a small fixed MLP, not a trained one (training is
  upstream, out of Jaxility scope per CONTEXT).

The `NPUFamily.APPLE`, `NPUFamily.QUALCOMM`, `NPUFamily.ETHOS_U55`,
`NPUFamily.ETHOS_U65` *labels* exist on the extended target rows; the
**codegen** for any of them does not.

### Prior art: L4CasADi for embedded-learned-function MPC

For the distinct case of an MPC whose cost-to-go, terminal value, or
contact-dynamics block is itself a learned model evaluated *inside*
the OCP each SQP iteration, **[L4CasADi](https://github.com/Tim-Salzmann/l4casadi)**
is the closest prior art. It compiles PyTorch models into C++ shared
libraries that acados loads via the
`ocp.solver_options.model_external_shared_lib_dir` /
`...shared_lib_name` external-function hook, handles
Jacobian/Hessian generation, and has solved the CMake + scikit-build
packaging surface. MIT-licensed, v2 just shipped, 575★, actively
maintained.

**Status in Jaxility:** ADR-017 (Proposed) records the integration
shape — depend on or vendor L4CasADi rather than re-deriving its
FFI seam from scratch — and routes the JAX side through
`JAX → ONNX → torch (or direct torch alongside JAX) → L4CasADi → acados`.
Implementation gates on owner approval of ADR-017.

L4CasADi's source restriction is PyTorch-only. The standalone
learned-policy lane (T-040..T-042; policy runs *alongside* the
acados controller, not embedded in the OCP) is unaffected and
still goes JAX → ONNX → LiteRT/ExecuTorch.

---

## Open questions still pending

These are tracked in `AGENTS/DECISIONS.md` as `OQ-*` rows. They block
specific later work but do not affect the shipped structural
claims:

- **OQ-4** — Apple Neural Engine programming interface (non-CoreML
  codegen path). `APPLE_SILICON.npu.peak_tops` is conservatively 0.0
  until this resolves.

OQ-1..OQ-3 and OQ-5 are closed by their respective ADRs. OQ-6 was
closed by ADR-015 (CI install via `--no-deps`).

---

## What is *not* a gap

For clarity, these are **not** gaps — they are deliberate design
decisions:

- **Data-only Targets** (no per-Target subclasses) is ADR-003. Adding
  a new SoC is a `Target(...)` row, never a class.
- **CasADi as the JAX → acados bridge** is ADR-001. CasADi is the
  symbolic boundary, not a deferred decision.
- **BLAKE3 hash chain** for the manifest is ADR-005, not an
  artefact of `hashlib` being unavailable.
- **`extra="forbid"` + `frozen=True`** on every Pydantic model is the
  loud-fail invariant 7, not paranoia.
- **No silent fallback in the coverage table** is invariant 7.
  `assert_supported` raises `CoverageError`; it never returns a
  best-guess answer.

If something in this list reads like a bug to you, file an issue —
the doc is wrong, not the code.

---

For the **load-bearing list of things Jaxility does guarantee**, see
[`CLAIMS.md`](CLAIMS.md).
