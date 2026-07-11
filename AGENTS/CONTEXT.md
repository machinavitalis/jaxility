# Jaxility — Project Context

Start at AGENTS/README.md, then read this file, then PATTERNS.md and
DECISIONS.md before writing code, and RULES.md once for operating
principles and workflow.

If you have not yet read the Jaxonomy and Jaxterity `AGENTS/CONTEXT.md`
files, read them first. Jaxility sits at the bottom of the three-package
stack and consumes both. This document describes only what is specific
to Jaxility.

## What Jaxility is

Jaxility is a deployment artifact factory. It takes a `CalibratedRobot`
from Jaxterity, plus a target Arm-based SoC profile, and produces a
self-contained binary (or C project, or compiled module) that runs the
robot's controller on the target hardware at the required cycle time,
with a signed attestation manifest proving how it was built.

It is the *compiler* of the three-package stack. Jaxonomy and Jaxterity
are libraries; you import them and call functions. Jaxility is mostly
operated as a CLI and a build system. Its Python API exists, but the
center of gravity is `jaxility build <robot> --target <soc>`.

The target users are deployment engineers at humanoid, quadruped, drone,
and manipulator companies who have a calibrated simulation model and
need a runnable binary on their target chip. The secondary user is the
robotics OEM evaluating embedded silicon — Jaxility's benchmark output
is the canonical "this is how fast a JAX-trained policy runs on this
chip" reference.

Jaxility's core toolchain is distributed under the MIT license. Any
commercial or enterprise extras ship under a separate license in a
separate repository (`jaxility-enterprise`).

## Relationship to Jaxonomy, Jaxterity, and the broader ecosystem

Jaxility is the third and most opinionated package of the stack:

- **Jaxonomy** — the engine. Hybrid dynamical simulation in JAX.
  Jaxility imports it transitively through Jaxterity but rarely
  touches its API directly.
- **Jaxterity** — the robotics layer. Produces calibrated robots.
  Jaxility consumes `CalibratedRobot` objects and their attestation
  handles.
- **Jaxility** — this package. Consumes Jaxterity outputs, produces
  embedded artifacts.

The boundary rule:

- If a feature is useful for non-deployment workflows (sim, sysid,
  policy design, visualization), it belongs in Jaxonomy or Jaxterity.
- If a feature produces or operates on a binary, a compiled C project,
  a signed manifest, a hardware benchmark, or a cross-compilation
  toolchain invocation, it belongs in Jaxility.

Jaxility imports `jaxonomy`, `jaxterity`, `casadi`, `acados-template`,
`tflite-runtime` (or its 2026 equivalent), and a curated set of vendor
SDKs for the target SoCs. It does **not** vendor or re-implement any of
these. It is glue with strong type discipline.

Jaxility relies on external toolchains it cannot replace:

- **acados** for MPC/WBC code generation. Mature, well-validated.
- **CasADi** as the JAX-to-acados translation surface, used at exactly
  one boundary (the dynamics lowering). See DECISIONS.md ADR-001 for
  why this is a pipeline component, not a substrate.
- **Arm GCC / LLVM** for cross-compilation. Vendor-supplied.
- **LiteRT (formerly TensorFlow Lite) / ExecuTorch** for on-device
  learned-policy inference. Dual path: acados for the MPC/WBC layer,
  LiteRT/ExecuTorch for the learned-policy layer when present.
- **Vendor SDKs** for the specific SoCs (Qualcomm Sensors Execution
  Environment, Arm Compute Library, STM32Cube, etc.). Wrapped behind
  a unified `Target` abstraction.
- **Arm Fixed Virtual Platforms (FVP)** for CI testing without
  physical hardware.

## Design philosophy: correctness, attestation, portability

Jaxility is designed around three theses:

**1. Correctness is non-negotiable; performance is a benchmark.** A
generated binary that produces incorrect output is worse than no binary
at all. Every code generation path has a numerical equivalence test
between the generated artifact and the source Jaxterity simulation, run
on the target (or on FVP), with documented tolerances. If the
equivalence check fails, the build fails. Performance is then a
secondary axis the benchmark page measures and the user optimizes
against.

**2. Attestation is a first-class output, not an afterthought.** Every
artifact Jaxility produces carries a signed manifest binding the
fitted parameters, the source telemetry hash, the toolchain versions,
the target SoC profile, and the generated binary's content hash into a
single record. This matters for regulated domains (DO-178C for aviation,
ISO 26262 for automotive, IEC 62304 for medical, EN ISO 13849 for
industrial robotics), where a verifiable provenance trail from telemetry
to binary is increasingly expected.

**3. Portability across the Arm-based ecosystem, not just one SoC.**
Jaxility's initial benchmark surface spans a range of Arm-based SoCs
(Cortex-A55/A76/A78/A710/N1, Cortex-M4/M7, Ethos-U55/U65, Qualcomm
Dragonwing IQ10, Apple Silicon as it becomes accessible). The unifying
abstraction is the `Target` profile; specific SoCs are configurations
of it. The package is hardware-portable within the Arm-based ecosystem,
which is where the early work is focused.

When in doubt about a design choice, prefer the option that (a)
preserves the attestation chain, (b) keeps the cross-SoC abstraction
clean, or (c) makes the failure mode louder rather than quieter.

## What Jaxility is NOT

- **It is not a training framework.** Jaxility does not train policies.
  Policies arrive trained (from K-Sim, RSL-RL, LeRobot, or any other
  upstream stack), as either a JAX checkpoint or an ONNX/LiteRT
  artifact. Jaxility deploys; it does not train.
- **It is not a simulation tool.** Jaxonomy and Jaxterity handle that.
  Jaxility uses simulation only for build-time correctness checks
  (does the generated binary match the source simulation?) and for
  HIL co-simulation against the deployed artifact.
- **It is not a code-generator from scratch.** acados generates the C
  code for MPC/WBC; LiteRT and ExecuTorch generate the inference
  code; Jaxility orchestrates them. Where Jaxility does emit code
  directly (the I/O glue, the manifest binding, the safety
  envelopes), the emitted code is small, MISRA-aware, and reviewed
  by a human before each release.
- **It is not a fleet management tool.** OTA updates, telemetry
  ingestion, fleet-level diagnostics, A/B deployment — these are
  commercial concerns that live in the `jaxility-enterprise`
  repository, not in the OSS package.
- **It is not a certification body.** Jaxility produces *attestation
  artifacts* — evidence of how a binary was built — that a
  certifying authority can consume. It does not certify anything
  itself.
- **It does not target x86 or NVIDIA today.** The initial work is
  focused on Arm-based silicon, so the pipeline is built around it for
  now. Other architectures are a later phase.

## Scope — what Jaxility owns

Jaxility provides the deployment artifact pipeline for the stack.
Its scope includes:

- **JAX → acados lowering pipeline**
  - JAX-traced dynamics, costs, and constraints from a Jaxterity
    `CalibratedRobot` and a controller spec → CasADi `SX`/`MX` graph
    → acados problem definition.
  - Restricted to the JAX subset acados can consume reliably:
    smooth dynamics, smooth costs, smooth or smoothed constraints.
    Non-smooth operations (`jnp.where` over traced values, `lax.cond`)
    are either rejected with a structured error or routed through
    smoothing approximations that are named and parameterized
    (mirrors Jaxterity invariant 6).
  - The lowering is structured so that a future JAX → MLIR → embedded
    path can replace the CasADi step without touching the rest of the
    pipeline.
- **acados problem template library**
  - Pre-built problem templates for common robot classes: LQR
    stabilizer (Cartpole-class), trajectory-tracking MPC (Crazyflie-
    class), task-space WBC (manipulator-class), centroidal MPC
    (humanoid-class).
  - Templates parametrize over the calibrated robot, the horizon,
    the sample rate, and the cost weights. Users supply the
    `CalibratedRobot` and the template fills in the rest.
- **Learned-policy deployment path**
  - JAX checkpoint → ONNX → LiteRT or ExecuTorch, with a documented
    op-coverage table per target.
  - Quantization recipes (static, dynamic, per-channel) with parity
    tests against the original float32 policy.
  - Dual-path runtime composition: the acados MPC runs at high rate;
    the learned policy runs at a lower rate; the runtime glue
    arbitrates between them. The composition pattern is documented
    and tested.
- **Cross-compilation toolchain wrappers**
  - Unified `Target` abstraction. Each supported SoC has a `Target`
    profile declaring its toolchain, ABI, vector extensions, NPU
    presence, memory constraints, and known quirks.
  - Initial targets: Cortex-A55, A76, A78, A710, N1; Cortex-M4, M7;
    Ethos-U55, U65; Qualcomm Dragonwing IQ10. Raspberry Pi 5 (A76)
    is the launch target.
  - Toolchain invocations are subprocess-driven and fully reproducible
    (toolchain hashes pinned in the manifest).
- **Runtime support library**
  - Minimal MISRA-aware C runtime that wires generated artifacts to
    target I/O: motor controllers, IMU drivers, encoder readers.
  - Vendor-agnostic at the interface, vendor-specific at the
    implementation. Implementations for the initial target list ship
    with the package.
  - Real-time scheduling primitives (cyclic executive on Cortex-M,
    PREEMPT_RT-aware scheduling on Cortex-A).
- **Attestation manifest pipeline**
  - Manifest schema: source attestation handle (from Jaxterity),
    toolchain versions, target profile, generated artifact content
    hashes, build timestamp, signer identity (optional for OSS,
    required for enterprise).
  - Signing infrastructure: stub OSS implementation (unsigned hash
    chain); pluggable signer interface so `jaxility-enterprise` can
    swap in real signing.
  - Verification CLI: `jaxility verify <manifest>` checks the
    artifact against its claimed provenance.
- **Benchmark harness**
  - `jaxility bench <robot> --target <soc>` runs a standardized
    workload (cycle time, jitter, memory footprint, energy if
    measurable) on physical hardware or FVP.
  - Results are emitted as a structured JSON record committed to a
    benchmark database (initially a git repo,
    `jaxility-benchmarks`). The public benchmark page is generated
    from this database.
  - Reproducibility is the bar: every benchmark record includes the
    full attestation manifest of the binary under test plus the
    benchmark harness version.
- **HIL co-simulation**
  - Run the Jaxterity simulation in JAX on the host and the deployed
    binary on the target (or on FVP), step-locked, comparing every
    state at every cycle. Divergence reports point at the offending
    code generation step.
- **CLI and MCP server**
  - Top-level CLI: `jaxility build`, `jaxility verify`,
    `jaxility bench`, `jaxility hil`, `jaxility targets`,
    `jaxility skills`.
  - MCP server exposing the build, verify, bench, and HIL operations
    as agent-callable tools. Same `@<type>[<hash>]` stable-handle
    pattern as Jaxterity.

## Scope — what belongs elsewhere

- **Policy training** — upstream (K-Sim, RSL-RL, LeRobot, custom).
- **Calibration and sysid** — Jaxterity.
- **Simulation and physics** — Jaxonomy and Jaxterity.
- **Fleet management, OTA updates, hosted compilation** —
  `jaxility-enterprise`.
- **Certification** — the certifying authority. Jaxility produces
  evidence; it does not grant certificates.
- **x86 / Jetson / RISC-V targets** — a later phase.
- **GUI tools** — not in scope.

## Architectural heritage

Jaxility draws specific patterns from a small set of upstream projects:

- **acados** — the MPC codegen target. The CasADi → acados pipeline is
  acados' established workflow; Jaxility uses it as designed and does
  not innovate at this layer.
- **CasADi** — the symbolic intermediate representation. Used at one
  integration boundary; not a substrate. See ADR-001.
- **TensorFlow Lite Micro / LiteRT / ExecuTorch** — the on-device
  inference path. Jaxility follows whichever has the cleanest
  embedded story per target at build time.
- **Arm CMSIS-DSP / CMSIS-NN** — vendor-supplied kernels for
  Cortex-M targets. Used where they outperform compiler-generated
  code.
- **Bazel** for the cross-compilation orchestration, where appropriate.
  Falls back to Make-based builds for older toolchains.
- **in-toto / SLSA** — supply-chain attestation patterns. The
  manifest schema is loosely SLSA-aligned without claiming SLSA
  compliance (that's an enterprise concern).
- **Drake** — reference for trajectory optimization problem
  formulations; not a runtime dependency.

## Architecture overview

Top-level package layout (under `jaxility/`):

- `lowering/` — JAX → CasADi → acados pipeline. The single biggest
  technical complexity in Jaxility lives here. `lowering/jax_to_casadi.py`
  is the translator; `lowering/casadi_to_acados.py` is the problem
  builder; `lowering/coverage.py` declares the supported JAX op set.
- `templates/` — acados problem templates (LQR, MPC,
  trajectory-tracking, WBC, centroidal MPC).
- `policy/` — Learned-policy deployment. `policy/onnx_export.py`,
  `policy/litert.py`, `policy/executorch.py`, `policy/quantize.py`.
- `targets/` — `Target` abstraction and per-SoC profiles. Each
  supported SoC is a subpackage with its toolchain spec, its quirks,
  and its runtime support.
- `runtime/` — The MISRA-aware C runtime that wires generated
  artifacts to target I/O. Per-vendor implementations under
  `runtime/<vendor>/`.
- `compose/` — Dual-path composition (acados MPC + learned policy
  arbitration).
- `manifest/` — Attestation manifest schema, signing, verification.
- `bench/` — Benchmark harness and result schema.
- `hil/` — Hardware-in-the-loop co-simulation.
- `cli/` — Command-line entry points.
- `mcp/` — FastMCP server.
- `testing/` — Test utilities, FVP integration, golden artifact
  comparison helpers.

## Key abstractions and invariants

These extend, not replace, the Jaxonomy and Jaxterity invariants.

- **Target** — A `Target` is a complete profile of a deployment SoC:
  toolchain (compiler, linker, ABI), vector extensions (NEON, SVE,
  Helium), NPU presence and capability, memory constraints, real-time
  guarantees, vendor SDK paths, known quirks. `Target` is the
  pluggable surface; adding a new SoC is adding a `Target` profile.
- **Artifact** — The output of a build. Carries: the binary or C
  project bytes, the source attestation handle, the build manifest,
  the toolchain version pins, the target profile hash. Artifacts are
  content-addressed.
- **Manifest** — The signed (or hash-chained, in OSS) record binding
  source robot, toolchain, target, and artifact. The contract with
  any downstream attestation consumer.
- **Coverage** — A declarative description of which JAX operations the
  lowering supports for which targets. A build fails clearly if the
  source uses an unsupported op for the chosen target.
- **CompositionPlan** — A declarative description of how an MPC layer
  and a learned-policy layer share execution time on the target.
  Rate, priority, fallback behavior on policy timeout, safety
  envelope.
- **GoldenArtifact** — A reference binary committed to the benchmark
  database with its manifest. Used to detect regressions in the
  generated code across Jaxility releases.

### Invariants that must hold

1. **Source-to-binary numerical equivalence.** Every generated artifact
   passes a numerical equivalence check against the source Jaxterity
   simulation before the build is considered successful. Tolerances
   are documented per-(target, dtype, quantity) in
   `test/EQUIVALENCE.md`. A build that does not pass equivalence is
   not shippable, regardless of pressure.
2. **Attestation chain unbroken.** Every artifact's manifest can be
   traced back through Jaxterity's attestation handle to the original
   telemetry hash. If the chain breaks at any step, the build fails
   loudly. Silent regeneration of intermediate hashes is forbidden.
3. **No untracked toolchain versions.** Every external tool (compiler,
   linker, acados, CasADi, LiteRT) used in a build has its version
   recorded in the manifest. Builds that depend on
   "whatever is on PATH" are not reproducible and not acceptable.
4. **Target portability of the API.** A user's build script is
   portable across targets by changing the `--target` flag.
   Target-specific code in user-facing APIs is forbidden; it lives in
   the `Target` profile and the runtime layer.
5. **Deterministic builds.** Same source robot, same toolchain
   versions, same target → byte-identical artifact (modulo
   timestamps, which are factored out of the content hash).
6. **HIL parity is the release gate.** A target is "supported" only
   when its HIL parity tests pass against the reference robot zoo.
   Adding a target without HIL is forbidden.
7. **No code generation without coverage.** Every JAX operation
   emitted into generated code is declared in
   `lowering/coverage.py`. Adding new op support is a documented PR
   touching the coverage file; the lowering implementation cannot
   silently support a new op.
8. **Safety envelopes are explicit.** Any deployment with a learned
   policy in the loop carries a documented safety envelope (joint
   limits, velocity limits, contact constraints) enforced by the
   acados layer underneath. The envelope is named, parameterized,
   and verifiable; it is not buried in template code.

## Supported versions and dependencies

- Python: 3.10+
- Jaxonomy: pinned through Jaxterity
- Jaxterity: pinned to current minor (`>=0.2,<0.3` at v0.1 of Jaxility)
- CasADi: pinned tight (v3.7.x at v0.1)
- acados: pinned tight (latest stable at v0.1)
- JAX: inherited from Jaxonomy
- LiteRT: pinned per-release; ExecuTorch parallel-supported behind
  an extra
- ONNX: pinned for policy export
- Optional extras: `[litert]`, `[executorch]`, `[fvp]` (Arm FVP
  integration), `[bench]` (benchmark harness extras),
  `[targets-cortex-m]`, `[targets-cortex-a]`, `[targets-qualcomm]`,
  `[all]`

`[all]` is large. It pulls vendor SDKs that have non-trivial system
requirements (Arm GCC, vendor-specific compilers, FVP binaries). The
default `pip install jaxility` installs only the lowering pipeline and
the manifest tooling; cross-compilation requires target-specific
extras.

## Repository layout at root

- `jaxility/` — the package itself
- `test/` — pytest test suite
- `test/equivalence/` — numerical equivalence tests (source vs
  generated)
- `test/hil/` — hardware-in-the-loop tests (requires hardware or FVP)
- `test/golden/` — golden artifact regression tests
- `docs/` — MkDocs source
- `examples/` — end-to-end examples, one per zoo robot
- `targets/` — `Target` profiles (also referenced in package as
  `jaxility.targets`)
- `runtime-c/` — the MISRA-aware C runtime source, with per-vendor
  subdirectories
- `AGENTS/` — this directory; agent-oriented documentation
- `.claude/` — Claude Code project configuration and skills
- `SKILL.md` — root Claude Code Skill
- `pyproject.toml`, `LICENSE.md`, `README.md` — standard
- `pypi-placeholder/` — name-reservation placeholder package
  (version `0.0.0`, dependency-free) that ships to PyPI to claim
  the `jaxility` name. The real package replaces it at the v0.0.1
  release. Mirrors the jaxonomy / jaxterity placeholder shape. See
  `pypi-placeholder/UPLOAD.md` for the one-shot upload procedure.

## Current state (as of project bootstrap)

This is a new project. The foundation phase establishes the lowering
pipeline contract, the `Target` abstraction, the manifest schema, and
the equivalence-check harness before any actual code generation is
attempted. See [`../CHANGELOG.md`](../CHANGELOG.md) for what shipped.

The first major milestone is the Cartpole nanojaxility demo: a
calibrated Cartpole `Robot` from Jaxterity → an LQR stabilizer
generated via acados → cross-compiled to a Raspberry Pi 5 → runs at
1kHz with a signed manifest. This is the launch artifact.

## Positioning summary

Jaxility's one-line description: *the open-source compiler from
JAX-trained robotics to Arm silicon, with signed attestation
manifests.*

What makes it unique relative to existing tools:

- **acados** generates excellent C code but does not consume JAX; the
  user writes the dynamics in CasADi by hand. Jaxility automates that
  step from a calibrated Jaxterity `Robot`.
- **JAX → MLIR → embedded** paths exist in principle but are not
  mature for robotics workloads as of 2026. Jaxility is the pragmatic
  bridge until that path matures, structured so the substrate can
  swap.
- **NVIDIA Isaac and Jetson** are GPU-only and lock the user into a
  single vendor. Jaxility is portable across the Arm-based ecosystem.
- **Vendor-supplied toolchains** (Qualcomm, Apple, etc.) do not
  produce attested artifacts. Jaxility does.

The value is the integration plus the attestation, not any single
component: the JAX → CasADi → acados translator, the dual-path runtime,
the cross-SoC benchmark, and the attestation infrastructure working
together.

## When modifying Jaxility

- Check DECISIONS.md before making architectural choices. The
  CasADi-as-component-not-substrate decision (ADR-001) and the
  manifest-schema decisions (ADR-005, ADR-006) are particularly
  load-bearing.
- Follow PATTERNS.md for coding conventions. Jaxility inherits
  Jaxonomy's and Jaxterity's conventions and adds compiler-specific
  ones (target dispatch, MISRA discipline in emitted C, manifest
  serialization).
- Respect the invariants above. Source-to-binary numerical
  equivalence (invariant 1) and the unbroken attestation chain
  (invariant 2) are the load-bearing properties; everything else
  derives from them.
- When in doubt about scope ("does this belong in Jaxterity or
  Jaxility?"), ask: "does this produce, transform, or verify a
  binary or signed manifest?" If yes, it's Jaxility. If no, it's
  probably Jaxterity.
- Every new target is a `Target` profile + a runtime implementation
  + a HIL test suite. Adding a target without all three is forbidden
  by invariant 6.
- Every change touching the lowering pipeline or the manifest schema
  is reviewed for attestation-chain integrity. The handle algorithm
  is versioned (consistent with Jaxterity ADR-008); changing it is
  a major version bump.
- If a change touches generated code, golden artifact regression
  tests must be updated in the same PR, with a justification.

## Contract docs and the doc-drift rule

The three repo-root files are the *load-bearing contract* with
downstream users and must stay in lockstep with the code:

- [`README.md`](../README.md) — the human-readable quick-reference
  table.
- [`CLAIMS.md`](../CLAIMS.md) — the exhaustive list of what Jaxility
  guarantees, with code citations.
- [`KNOWN_GAPS.md`](../KNOWN_GAPS.md) — the symmetric list of what
  Jaxility explicitly does NOT do, with workarounds and ADR pointers.

**The rule.** Any PR that:

1. **Closes a gap in code** must move the corresponding text from
   `KNOWN_GAPS.md` into `CLAIMS.md` and (if the gap was reflected
   in the README quick-reference table) bump the README row in the
   same PR.
2. **Opens a gap in code** (deliberately ships a missing piece or
   removes a capability) must add a row to `KNOWN_GAPS.md` naming
   the workaround and the tracking task.
3. **Changes a pin, target profile, family flag, or coverage row**
   must update the corresponding CLAIMS / KNOWN_GAPS / README rows
   in the same PR.

**Mechanical safety net.** `test/unit/test_top_level_docs.py` runs on
every CI build and asserts:

- Both contract docs cross-reference each other and the README
  cross-references them.
- The CLAIMS targets table lists every `Target` symbol that exists
  in code AND its current pinned `toolchain.version`.
- The README lists the same set of `Target` symbols with the same
  pinned version for PI5.
- The README and CLAIMS name the four acados templates.
- KNOWN_GAPS names MJX and ADR-016 (the single largest documented
  rejection).
- KNOWN_GAPS names the Tier-B cross-compile gap or "Cortex-M lane".
- The README uses the canonical `smooth-op` phrasing so search
  terms cross-link.

The tests catch *specific* drift (versions, symbols, key phrases).
They do NOT catch:

- A gap closing in code without text being moved from KNOWN_GAPS
  to CLAIMS (an agent has to do this deliberately).
- A new capability landing without being added to CLAIMS.
- A new limitation appearing without being added to KNOWN_GAPS.
- Downstream-package (Jaxonomy / Jaxterity) changes that affect
  what Jaxility can consume.

When in doubt: run `python -m pytest test/unit/test_top_level_docs.py
-v` after any change to the lowering pipeline, the target profiles,
or the manifest schema. If your code change resolves a documented
"still out of scope," "pending," or "deferred" phrase, the PR should
also delete that phrase.
