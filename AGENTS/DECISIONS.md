# Jaxility — Architectural Decisions

This file records architectural decisions that have already been made for
Jaxility. Same ADR format as Jaxterity DECISIONS.md.

ADRs follow the `ADR-NNN` numbering. Jaxility ADRs start at `ADR-001`
within this package's namespace.

---

## ADR-001 — CasADi is a pipeline component, not a substrate

**Status:** Accepted (per Jaxonomy-vs-PathSim-vs-Archimedes analysis)

### Context

Jaxility needs to translate JAX-traced dynamics into acados-consumable
problems. CasADi is acados' established input format and has a mature
codegen story. The question is whether CasADi becomes a substrate
(internal representation throughout the pipeline) or a component (used
at exactly one boundary).

### Decision

CasADi is used at exactly one boundary: the JAX → acados translation
step. Jaxility's intermediate representation, target dispatch, manifest
schema, and runtime layer do not depend on CasADi.

The lowering pipeline is structured as: `JAX dynamics → CasADi graph →
acados OCP → C code`. Each arrow is a distinct module that can be
swapped. A future JAX → MLIR → acados (or MLIR → IREE) path would
replace only the first arrow.

### Alternatives considered

1. **CasADi as substrate.** Tempting because acados speaks CasADi
   natively. Rejected: it would couple every Jaxility module to a
   single symbolic backend, blocking any future migration to MLIR-based
   paths.
2. **Direct JAX → C codegen, bypass CasADi.** Considered. Rejected for
   v1: JAX's MLIR pipeline can in principle target embedded backends,
   but the toolchain (IREE for robotics) is not mature in 2026. CasADi
   is the safer v1 bet.
3. **PyTorch → ONNX → embedded.** Different stack. Rejected: incompatible
   with the JAX-throughout-the-pipeline architectural commitment.

### Consequences

- The lowering module structure is `jax_to_casadi` and
  `casadi_to_acados` as separate passes, not a fused translator.
- CasADi appears only in the lowering subpackage. Targets, runtime,
  manifest, benchmark — none import CasADi.
- A future JAX → MLIR replacement is a swap of `jax_to_casadi`, not a
  rewrite of the package.
- The version pin on CasADi can be tight (one minor) without
  destabilizing the rest of Jaxility.

---

## ADR-002 — Dual deployment path: acados for control, LiteRT/ExecuTorch for learning

**Status:** Accepted

### Context

Modern robotics deployments mix model-based control (MPC, WBC) and
learned policies (PPO-trained motor policies, VLA action heads, learned
state estimators). Each has different runtime requirements: MPC needs
hard real-time guarantees and ~1kHz cycles; learned policies need
optimized inference kernels and tolerate slower cycle rates. A single
runtime can't serve both well.

### Decision

Two deployment paths, composed at runtime:

- **acados path** for model-based control. Generates MPC/WBC C code,
  runs at the high rate (1kHz typical), provides hard real-time
  guarantees and a safety envelope.
- **LiteRT (priority) / ExecuTorch (parallel)** for learned policies.
  Inference runtime running at a lower rate (10–100Hz typical),
  consuming the safety envelope as constraints.

The two paths share a `CompositionPlan` data structure declaring how
they interact: rate, priority, fallback on learned-policy timeout,
safety-envelope enforcement.

### Alternatives considered

1. **acados only.** Rejected: forces learned policies to be
   approximated as MPC cost terms, which loses the value of learning.
2. **LiteRT/ExecuTorch only.** Rejected: gives up the hard
   real-time guarantee and the safety envelope that the
   regulatory-credibility tier needs.
3. **A unified custom runtime.** Tempting (one runtime, less to
   maintain) but rejected because building a competitor to LiteRT or
   acados is out of scope. Use what exists; compose it.
4. **TensorFlow Lite Micro as the only learning path.** Outdated by
   2026; LiteRT is the supported successor.

### Consequences

- Two extras: `[litert]` and `[executorch]`. Users install whichever
  they need; both can coexist.
- The `CompositionPlan` becomes load-bearing for safety arguments —
  the safety envelope is declarative in `CompositionPlan`, not buried
  in code.
- Adding a third inference runtime (some 2027 successor) is adding a
  parallel path, not displacing one.

---

## ADR-003 — Targets are Pydantic data, not subclasses

**Status:** Accepted

### Context

A new SoC could be added by subclassing a `Target` base class (with
virtual methods for `toolchain()`, `vector_ops()`, etc.), or by
filling out a Pydantic data model with declarative fields. The
subclass path is more flexible; the data path is more constrained.

### Decision

`Target` is a Pydantic data model. Adding a target is filling out
the model. Behavior that depends on target capability is registered
as a strategy keyed off capability flags, not virtual methods on the
target.

### Alternatives considered

1. **`Target` as ABC with virtual methods.** Rejected: targets
   become Python code that has to be imported and instantiated;
   serialization for the manifest gets weird; capability detection
   gets imperative.
2. **Targets as YAML files.** Considered. Rejected because Pydantic
   gives us static type-checking and validation that YAML doesn't.
3. **Targets as a TOML configuration.** Rejected for the same
   reasons as YAML.

### Consequences

- Adding a target is a documented data-entry task, accessible to
  contributors who do not know how to subclass.
- Target metadata is serializable; the target hash is well-defined
  and goes into the manifest.
- Target-dependent behavior lives in strategy registrations, which
  are centralized and inspectable.
- A `jaxility targets` CLI can list and detail every target without
  side effects.

---

## ADR-004 — Target rollout: start with Raspberry Pi 5, then expand across Arm-based targets

**Status:** Accepted

### Context

Bringing up a broad range of Arm-based targets at once is infeasible
within the initial launch scope. The choice is where to start and how
to sequence the rollout.

### Decision

We start with the Raspberry Pi 5 (Cortex-A76) as the demonstrated
hardware target for v0.1 — used for the launch demos and for HIL
validation. From there, support expands to other Arm-based targets
next, with non-Arm architectures a possible later phase.

The `Target` profile abstraction is designed to generalize across
targets, but at v0.1 only the Pi 5 has a passing HIL test suite and
runtime support code.

### Alternatives considered

1. **Pi 5 only, with no rollout beyond it.** Rejected: the value here
   depends on portability across Arm-based targets. Committing to a
   single target makes us look like a Pi project.
2. **Several hardware targets at launch.** Rejected: bandwidth. Each
   hardware bring-up takes weeks; starting from one demonstrated target
   is the honest path.
3. **Scaffolding every target at varying depth.** Rejected: targets
   without HIL validation multiply confusion; better to be clear about
   what's demonstrated and what isn't.

### Consequences

- CLAIMS.md stays precise about what's demonstrated: the Pi 5 with HIL
  at v0.1, with other targets following.
- The benchmark page at launch starts with the Pi 5, which is honest.
  It scales out one target at a time.

---

## ADR-005 — Manifest schema v0: hash chain over canonical JSON

**Status:** Accepted

### Context

The attestation manifest binds source to artifact. The schema must be
stable, byte-canonical for hashing, signable, and forward-compatible.
At OSS-minimum, signing infrastructure is too heavy; a hash chain is
sufficient.

### Decision

Manifest schema v0 fields:

```
schema_version: int
source_attestation_handle: bytes        # from Jaxterity
toolchain_versions: dict[str, str]      # tool → semver
target_profile_hash: bytes
artifact_content_hash: bytes
build_timestamp_utc: int                # microseconds; not hashed
signer_identity: Optional[str]
signature: Optional[bytes]              # None at OSS level
```

Serialized via the canonical JSON serializer (PATTERNS.md §3.1).
Hashed with BLAKE3 (matches Jaxterity ADR-008).

The hash chain: each manifest's hash includes the source's hash. A
chain of `(source telemetry hash) → (Jaxterity attestation handle) →
(Jaxility manifest hash) → (deployed binary hash)` is verifiable
end-to-end via `jaxility verify`.

### Alternatives considered

1. **Sigstore / SLSA from day one.** Rejected: too heavy for v0.1.
   Schema v1 will move toward SLSA alignment.
2. **GPG-signed manifests.** Rejected: key management is a separate
   problem; defer to enterprise tier.
3. **JWT-based attestation.** Rejected: doesn't fit the
   content-addressed model.
4. **CycloneDX SBOM as the manifest format.** Considered. Rejected
   because CycloneDX optimizes for software composition; the
   robotics-specific fields (source attestation handle, target
   profile) don't fit cleanly.

### Consequences

- OSS users get reproducibility and hash-chain verifiability without
  signing infrastructure.
- The `Signer` interface is pluggable; enterprise plugs in real
  signing.
- Schema v1 (when it lands) is a coordinated migration; v0 manifests
  stay readable forever (the verify tool dispatches on version).

---

## ADR-006 — Determinism over performance, where they conflict

**Status:** Accepted

### Context

In a few places — choosing between threading models, choosing between
acados solver settings, choosing between aggressive and conservative
compiler optimization — Jaxility faces a determinism-versus-performance
tradeoff. Cross-build reproducibility (invariant 5) requires
determinism; benchmark scores favor performance.

### Decision

Default to determinism. The artifact content hash must be stable
across builds; if a compiler flag introduces nondeterminism, it is
not used in default builds. Performance-tuned builds are opt-in
(`--release --aggressive`) and produce a non-content-addressed
artifact (the manifest records the flags).

### Alternatives considered

1. **Default to performance.** Rejected: silently nondeterministic
   builds break the attestation chain. Users who do not understand
   the tradeoff would be on the wrong side of it.
2. **Document but do not enforce.** Rejected: invariant 5 is
   load-bearing for the regulatory tier; "we document it" is too
   weak.

### Consequences

- Default builds are slightly slower than the maximum achievable.
  Benchmark records publish both default and aggressive results.
- `--aggressive` is explicit and recorded in the manifest. An
  auditor can see whether nondeterministic optimizations were used.
- Reproducibility tests in CI use default builds only.

---

## ADR-007 — Equivalence checks on the host first, on target second

**Status:** Accepted

### Context

Numerical equivalence between the source JAX simulation and the
generated artifact is invariant 1. The question is where the check
runs: on the host before deployment, on the target after deployment,
or both.

### Decision

Both. Every artifact is host-verified at build time (the generated C
is run on the host, in single-precision matching the target where
possible, and its output is compared bit-exact-modulo-ULP against
the source JAX). Targets with HIL support add a target-side
equivalence check at deploy time.

Host equivalence is a build gate; target equivalence is a deploy
gate. Both block release.

### Alternatives considered

1. **Host-only equivalence.** Insufficient: host floating-point
   behavior can differ from target behavior (denormals, rounding,
   fused-multiply-add). Catches most bugs but not all.
2. **Target-only equivalence.** Costs HIL time for every build, even
   pre-merge. Slow. Rejected.

### Consequences

- The host check runs on every PR and is fast.
- The target check runs nightly or on protected-branch merges.
- A regression that passes host equivalence but fails target
  equivalence is a real bug in the codegen and gets a high-severity
  label.

---

## ADR-008 — Enterprise extras live in a separate repository

**Status:** Accepted (mirrors Jaxterity ADR-013)

### Context

`jaxility` is MIT-licensed. Any commercial or enterprise extras must not
live in this repository; the question is where such code lives.

### Decision

`jaxility` is fully MIT. Commercial extras live in
`jaxility-enterprise` under a separate private repository and a
separate license. Extension points in `jaxility` (the `Signer`
protocol, the `BenchmarkUploader` protocol, the
`TelemetryStorageBackend` protocol) are public interfaces; enterprise
plugs in real implementations.

### Alternatives considered

Same as Jaxterity ADR-013. Conclusions transfer.

### Consequences

- The OSS user gets a working stub (`HashChainSigner`,
  `LocalBenchmarkUploader`, `LocalTelemetryStorage`); enterprise
  swaps in audited implementations.
- Adding a new extension point is an ADR.

---

## ADR-009 — Benchmark database is a git repo, not a hosted service

**Status:** Accepted

### Context

Benchmark records need to be persistent, verifiable, and shareable.
A hosted database (Postgres, a SaaS) is heavier than the project can
maintain. A git repo with structured JSON files is lighter.

### Decision

The benchmark database is `jaxility-benchmarks`, a separate public
git repo containing one JSON file per measurement, organized by
`(target, robot, version)`. Records are append-only; the public
benchmark page is generated by static-site tooling from the repo.

Contributing a benchmark is a PR. The PR's CI verifies the record's
manifest chain before accepting.

### Alternatives considered

1. **Self-hosted database.** Rejected: maintenance surface.
2. **A SaaS benchmark service.** Rejected for the same reasons.
3. **Inline benchmarks in `jaxility` repo.** Rejected because the
   database grows without bound; the main repo stays small.

### Consequences

- Benchmark records are fork-and-PR contributions; community can
  upload.
- The page is reproducible from the repo state at any commit.
- Migration to a hosted service later, if needed, is a tooling
  change, not a schema change.

---

## ADR-010 — Cortex-A and Cortex-M share the `Target` abstraction; they do not share runtime code

**Status:** Accepted

### Context

Cortex-A and Cortex-M targets have radically different runtime
characteristics: Cortex-A has an MMU, runs Linux (often
PREEMPT_RT), has gigabytes of memory, runs in user space.
Cortex-M is bare-metal, has kilobytes of memory, runs ISR-driven.
The `Target` abstraction could cover both, or split.

### Decision

`Target` covers both. They share the abstraction (declarative spec
of toolchain, ABI, capabilities) but not the runtime code. The
`runtime-c/` directory is partitioned: `runtime-c/cortex-a/`,
`runtime-c/cortex-m/`, with shared headers under
`runtime-c/common/` for things that genuinely cross (e.g., the
manifest header, the fixed-point math primitives).

### Alternatives considered

1. **Split `TargetA` and `TargetM` abstractions.** Rejected: doubles
   the dispatch logic and makes target-portable code harder to
   write.
2. **A single shared runtime.** Rejected: cortex-A runtime assumes
   things cortex-M can't provide (Linux syscalls, dynamic
   allocation).

### Consequences

- The runtime split surfaces in directory layout but not in the
  Python API.
- A user's Python build script is identical across Cortex-A and
  Cortex-M targets (modulo the `--target` flag).
- Adding RISC-V (future) is adding `runtime-c/riscv/` and the
  corresponding target profiles; the abstraction holds.

---

## ADR-011 — HIL tests gate target supportedness

**Status:** Accepted

### Context

A target could be claimed "supported" because its `Target` profile is
filled out, because host equivalence passes, or because HIL parity
passes. Each is a different bar.

### Decision

A target is "supported" only when its HIL parity tests pass against
the reference robot zoo. Profile-only and host-equivalence-only
targets are "experimental"; the CLAIMS.md page makes the distinction
explicit.

This is invariant 6 from CONTEXT.md, locked here.

### Alternatives considered

1. **Looser support definition.** Rejected: would let claims drift.
2. **Stricter (signed manifest required).** Rejected: signing is
   enterprise; OSS should ship something useful.

### Consequences

- The benchmark page distinguishes supported from experimental
  targets visually.
- A new target's PR series is: profile → host equivalence →
  runtime → HIL → benchmark. Only the last unlocks "supported".

---

## ADR-012 — Ship the Claude Code skill on day one (mirrors Jaxterity ADR-012)

**Status:** Accepted

### Context, Decision, Alternatives, Consequences

Same as Jaxterity ADR-012. The Jaxility skill is for an agent
helping a deployment engineer; the workflow tree is build → verify
→ bench → HIL.

---

## ADR-013 — `jaxility build` is the canonical operation; the Python API is secondary

**Status:** Accepted

### Context

Library users expect a Python API. Compiler users expect a CLI.
Jaxility is a compiler that ships from a Python ecosystem. Where
does the primary surface live?

### Decision

The CLI is the canonical surface. `jaxility build`, `jaxility verify`,
`jaxility bench`, `jaxility hil`. Python API exists and is documented,
but the docs lead with the CLI. The MCP server wraps the CLI, not the
Python API.

### Alternatives considered

1. **Python API first.** Rejected: a compiler that requires a Python
   shell to invoke is awkward in CI, in build systems, in shell
   pipelines.
2. **CLI only, no Python API.** Rejected: testing requires a Python
   API; eat your own dog food.

### Consequences

- The CLI gets first-class attention: structured JSON output, exit
  codes, comprehensive `--help`.
- The Python API is documented as "what the CLI is built on" rather
  than "the main interface."
- Agentic workflows (Claude Code, Cursor) reach for the MCP tools,
  which wrap the CLI.

---

## ADR-014 — Vendor SDK dependencies are optional extras

**Status:** Accepted

### Context

Some targets require vendor SDKs that are large, sometimes
license-restricted, sometimes require manual install steps. Forcing
every user to install them poisons the funnel.

### Decision

Each vendor SDK is behind a target-specific extra. `pip install
jaxility[targets-qualcomm]` pulls (or scripts the install of) the
Qualcomm SDK. `pip install jaxility[targets-cortex-m]` pulls Arm GCC
for bare-metal. Default `pip install jaxility` is small and lets the
user inspect targets, coverage, and manifests without any
cross-compilation infrastructure.

### Alternatives considered

1. **One mega-extra `[all-targets]`.** Considered for convenience.
   Provided but documented as huge. Most users want one extra.
2. **No vendor SDK installation; user installs separately.**
   Rejected: friction at adoption. We script the install when the
   license permits.

### Consequences

- Per-target install docs live with each target's documentation.
- CI matrix runs targets only when their extras are installed; tests
  skip gracefully (and the skip is visible) when extras are absent.
- A future commercial-license SDK (per vendor) is handled by an
  override env var pointing at the user's licensed install.

---

## ADR-015 — CI installs Jaxility with ``--no-deps`` while upstreams stay private

**Status:** Accepted (resolves OQ-6)

### Context

Jaxility imports Jaxterity, which imports Jaxonomy. At v0.0.1 both
upstream packages live in private GitHub repositories
(``machinavitalis/jaxterity``, ``machinavitalis/jaxonomy``) and are
not on PyPI. A faithful ``pip install -e .[dev]`` on a clean GitHub
Actions runner fails at dependency resolution because the standard
package indexes do not see them. This blocked T-002 (CI workflow)
across the early milestones — surfacing repeatedly as OQ-6.

### Decision

CI installs Jaxility via a **two-step recipe**:

1. ``pip install -e . --no-deps`` registers the editable install
   without trying to resolve upstreams from any index.
2. An explicit list of PyPI-resolvable dependencies pulls everything
   the test suite needs to import: ``ruff``, ``mypy``, ``pytest``,
   ``hypothesis``, ``numpy``, ``jax[cpu]``, ``casadi``, ``pydantic``,
   ``blake3``, ``cryptography``, ``fastmcp``.

Tests that need ``jaxterity`` (the chain-integration tier in
``test_jaxterity_chain.py``, parts of ``test_zoo.py``) gate on
``pytest.importorskip("jaxterity")`` and skip cleanly. Tests that
need acados or ``t_renderer`` skip via fixture-level
``skipif``-on-env-var-presence. The CI job exercises the
pure-Python contracts (~250+ tests) on every push / PR.

### Alternatives considered

1. **PAT secret with cross-repo pip-from-git install.** Most faithful
   to local development; requires a GitHub Actions secret with
   ``repo`` scope on the two upstreams. Rejected for v0.1 because it
   couples the OSS project to a credential that has to be rotated
   and audited.
2. **Flip Jaxterity + Jaxonomy public.** Lowest CI friction but a
   strategic decision about repo visibility that should not be
   forced by a CI choice. Re-evaluated when the upstream packages
   are ready to publish to PyPI.
3. **Vendor minimal stubs of Jaxterity in Jaxility's test tree.**
   Would let CI run the full suite. Rejected because the stubs
   drift; the initial review explicitly mocked Jaxterity surfaces in
   only one place (``MockSource``) and tracking a second stub of the
   same upstream is anti-pattern.
4. **Stay deferred.** Rejected once the real-target work began — the
   discipline collapses without CI past a certain code volume, and
   the host build path had already produced ~300 tests.

### Consequences

- The PR-time CI gate is a *subset* of the local preflight: tests
  needing private upstreams or proprietary build environments skip.
  Surface explicitly in PR bodies so reviewers see what's covered.
- The two-step install is documented in
  ``.github/workflows/ci.yml`` and in ``AGENTS/TOOLCHAINS.md`` for
  contributors who reproduce CI locally.
- The PAT-secret + public-upstream alternatives stay viable as a
  follow-up; flipping the install path is a one-block change.
- The ``ToolchainPin`` work (T-030) and the
  ``Robot.to_diagram`` → JAX-dynamics adapter (next pickup item)
  both add tests that ride this CI path.

---

## ADR-016 — MJX-driven dynamics do not flow through the JAX → CasADi translator; closed-form per-robot dynamics is the contract

**Status:** Accepted (closes the "real Jaxterity Robot dynamics through the lowering pipeline" question)

### Context

Jaxility's lowering pipeline (ADR-001) translates JAX dynamics into
CasADi → acados → embedded C. The host-build CLI work
(``jaxility build cartpole --target host``) raised the obvious
question: can the Jaxility translator consume the dynamics function
that ``jaxterity.robot.Robot.build_system(...).ode`` exposes? That
function is the natural surface — every Jaxterity Robot has it, and
plumbing it through Jaxility would let any calibrated Robot land on
host (and eventually on Pi 5) without a hand-written analytical
fallback.

The Robot's ``ode`` is implemented via MJX (the JAX port of MuJoCo).
A trace of the cartpole ``ode`` jaxpr emits 34 distinct primitives
across 2,375 top-level equations, including:

- A single top-level ``while_loop`` — MJX's iterative constraint
  solver. The loop runs unconditionally on every forward step, even
  for an unconstrained model with no contacts and no active joint
  limits.
- 30 ``scatter`` / ``scatter_add`` operations for body-state updates.
- 15 ``gather`` operations.
- Standard arithmetic and shape ops.

### Decision

**The JAX → CasADi translator does not, and will not in Jaxility's
v0.x line, accept MJX-emitted dynamics directly.** The
``while_loop`` is the load-bearing reason: acados' OCP formulation
is a fixed-size SQP graph, and ``while_loop`` over a traced
predicate cannot be unrolled at OCP construction time. Smoothing
wrappers (sigmoid-approximated branches) work for ``jnp.where`` over
traced predicates, but there is no analogous smoothing for an
iterative-solver fixpoint.

**Closed-form per-robot dynamics functions are the contract.** Each
zoo entry that wants to ride the lowering pipeline supplies a JAX
function ``f(state, control) -> dx`` written in the smooth-op subset
the translator covers. Cartpole ships this today; SO-100 needs the
6-DOF inertia-matrix form or it stays CLI-blocked at host build.

### Alternatives considered

1. **Smoothing the constraint-solver ``while_loop`` to a fixed-iteration
   unrolled equivalent.** Investigated: MJX's solver runs to fixpoint
   in 5–20 iterations for typical motion, so a 20-iteration unrolled
   approximation could in principle replace the loop. Rejected for
   v0.x because (a) the unrolled cost on every OCP stage is
   prohibitive, (b) numerical accuracy across the smoothing window
   degrades silently for active-contact phases, (c) upstream MJX
   would have to expose the unrolled path explicitly and the
   coordination cost is high.
2. **Upstream Jaxterity to expose a symbolic-dynamics path alongside
   MJX.** Considered. Rejected for v0.x because it doubles the
   Jaxterity API surface and asks upstream to invest in a
   non-MJX backend for our use case. Revisit when Jaxterity gains
   symbolic / Pinocchio-style dynamics natively.
3. **JAX → MLIR → embedded path that bypasses CasADi (and therefore
   the while-loop issue).** ADR-001's swap-point. Documented as the
   later replacement once IREE / robotics-MLIR matures; not v0.x.
4. **Continue hitting ``CoverageError`` silently in the CLI.** What
   the gap looked like before this ADR. Rejected — the CLI error
   has to name *why* and point at the closed-form fallback.

### Consequences

- Cartpole zoo entry ships an analytical ``cartpole(state, control)``
  in ``jaxility/zoo/cartpole/__init__.py`` (the same dynamics the
  T-027 host-equivalence tests use). The Jaxterity Robot still
  supplies the attestation handle via the source factory, so the
  manifest chain remains anchored upstream — the lowering pipeline
  consumes a *different* JAX function than the simulation does.
  - **T-101 refinement:** the closed-form is still a
    different (and reduced) function than the MJX sim, but its scalars
    ``(g, mc, mp, L)`` are no longer hand-hardcoded — they are sourced
    from the upstream Robot's fitted leaves via
    ``jaxterity.zoo.cartpole.reduced_params``. So calibration propagates
    into the lowered binary and the handle and deployed dynamics move
    together (Jaxterity Invariant 1 at the parameter level). This does
    **not** reopen the decision: the deployed plant is still the
    frictionless closed-form, not MJX. Joint damping/friction remain in
    the sim model only.
- SO-100 zoo entry has no analytical fallback bundled, so the CLI
  surfaces a structured "no jax_dynamics_factory" error.
  Promoting SO-100 to a buildable target requires either a
  hand-written 6-DOF closed-form or an upstream Jaxterity symbolic
  dynamics path.
- ``lax.while_loop``'s ``CoverageError`` suggestion explicitly names
  MJX so the user reading the error sees the architectural
  consequence, not just the generic "loop not allowed" hint.
- ``dynamic_slice[static]`` is added as a separate supported row to
  cover the legitimate user-written JAX pattern
  (``jax.lax.dynamic_slice(x, (literal,), (n,))``); the
  ``dynamic_slice[traced]`` row stays unsupported with a documented
  fallback to plain slicing.
- The "real Jaxterity Robot dynamics through Jaxility's lowering
  pipeline" question is closed at the architectural level. Future
  proposals to reopen it have to start by addressing the
  while_loop / fixpoint-solver problem head-on.

---

## ADR-017 — Learned-policy lane reuses L4CasADi's acados FFI seam (proposed)

**Status:** Proposed (pending owner approval; do not start T-040..T-045 implementation against this ADR until accepted)

### Context

The learned-policy lane (T-040..T-045) is the dual-path runtime:
acados handles the MPC / WBC layer (its shipped contract) while a learned
policy runs alongside, with the safety envelope enforced by acados.
The current plan is JAX policy → ONNX → LiteRT/ExecuTorch (separate
runtime).

A distinct, narrower case is **MPC with an embedded learned
function**: the controller's cost-to-go, terminal value, or contact
dynamics is itself a neural net evaluated *inside* the OCP each SQP
iteration. acados already exposes an external-function hook
(`ocp.solver_options.model_external_shared_lib_dir` /
`...shared_lib_name`) for this — the OCP calls into a foreign C++
function for the value + Jacobian + Hessian of the embedded model.

**The reusable prior art is [L4CasADi](https://github.com/Tim-Salzmann/l4casadi).**
L4CasADi compiles a PyTorch model into a C++ shared library that
implements the external-function contract acados expects. Their
v2 has solved:

1. The shared-library packaging (CMake + scikit-build + Ninja).
2. The Jacobian (and optional Hessian) generation that acados'
   SQP demands.
3. The exact `model_external_shared_lib_dir` / `shared_lib_name`
   wiring on the acados side.
4. A `NaiveL4CasADiModule` path that recreates small models in pure
   CasADi MX — equivalent to what Jaxility's translator does for the
   smooth-op subset (every supported primitive has a CasADi-pure
   handler). Architecture aligns.

L4CasADi's source restriction is **PyTorch only.** Jaxility's source
is JAX. The reusable layer is everything *below* the framework
loader: the shared-library packaging, the acados FFI seam, the
Jacobian generation, and the small-model symbolic-recreation pattern.
The framework-loading layer is replaceable.

### Decision (proposed)

For the *MPC-with-embedded-learned-function* lane (T-043 in
particular), **Jaxility depends on or vendors L4CasADi rather than
re-deriving the acados FFI seam from scratch.** The integration
shape:

1. JAX policy → ONNX (via `jax2tf` + `tf2onnx`, or a direct JAX-to-
   ONNX exporter; T-040 territory).
2. ONNX → L4CasADi-compatible PyTorch model, OR direct PyTorch
   export from an equivalent torch model trained alongside the JAX
   one (the simpler short-term path).
3. L4CasADi compiles the model into a shared library.
4. The acados OCP's `model_external_shared_lib_*` fields point at
   the L4CasADi-produced library.

The **standalone learned-policy lane** (T-040..T-042; JAX policy
running alongside, not embedded in the OCP, talking to acados
through the safety envelope at runtime) is unaffected by this
ADR. That lane still goes JAX → ONNX → LiteRT/ExecuTorch and runs
as a separate process / runtime as planned.

### Alternatives considered

1. **Reimplement L4CasADi's FFI seam in Jaxility.** Rejected because
   the FFI surface (Jacobian/Hessian generation, the acados external-
   function ABI, the CMake + scikit-build packaging) is exactly the
   bit that's hardest to get right and most fragile across acados
   versions. L4CasADi has 575 stars, v2 just shipped, and tracks
   acados upstream. Re-deriving it is ~3-4 weeks of work to land at
   the same place; ADR-014 (vendor SDK dependencies are optional
   extras) gives us precedent for opt-in dependence.
2. **Direct JAX → CasADi MX recreation for learned policies.** Works
   only for the small-model case (the smooth-op subset Jaxility
   already supports). Falls apart for any policy with ReLU
   activations, MaxPool, or non-smooth ops — those need the FFI
   shared-lib path because CasADi MX can't represent them. L4CasADi
   covers both cases; reimplementing the small-model path in our
   translator and outsourcing the FFI path to L4CasADi is the same
   architectural split L4CasADi already uses.
3. **Skip the embedded-learned-function case entirely; let users
   write closed-form value functions.** Loses the v0.x
   differentiator Jaxility's strategy doc names. Rejected.
4. **Wait for L4CasADi to support JAX directly.** Their issue tracker
   doesn't show this on the roadmap. Open question for them; not a
   Jaxility blocker.

### Consequences

If accepted:

- A new optional extra `jaxility[learned-policy-mpc]` pulls in
  L4CasADi alongside the existing PyTorch dep. ADR-014 governs the
  extras schema.
- The learned-policy PRs cite this ADR when they wire the FFI seam; the dual-
  path runtime composition test (T-043) becomes "round-trip a small
  PyTorch model through L4CasADi into acados, solve, compare with
  pure-JAX reference" rather than "implement the FFI seam from
  scratch."
- KNOWN_GAPS.md "Learned-policy lane" section already documents
  L4CasADi as a prior-art comparator (this PR landed that row); the
  ADR is what governs the actual integration.
- `CLAIMS.md` does not change at v0.x because the learned-policy lane has not
  started. Once T-043 lands, the dual-path runtime entry moves from
  KNOWN_GAPS to CLAIMS with the L4CasADi citation.
- A new ADR-grade question opens: which version of L4CasADi do we
  pin against? Their v2 just shipped with breaking changes. Pin
  v2.0.x as the floor and version-gate in CI per the standard
  toolchain-pin pattern (`ToolchainPin`-style, not via the Python
  package's `__version__` which is unreliable per A6's lesson with
  acados_template).

### What still needs to be decided to accept this ADR

1. Owner approval (standing rule for new ADRs).
2. Confirm the licensing path is clean — L4CasADi is MIT, Jaxility
   is MIT, no friction. Verified.
3. Decide whether to **depend on** L4CasADi (lightweight) or
   **vendor a subset** of its codegen (heavier but version-locked).
   Recommendation: depend, version-pin v2.0.x via `extras`.
4. Decide whether the small-model symbolic-recreation path
   (L4CasADi's `NaiveL4CasADiModule`) should be replicated in
   `jaxility.policy` for JAX-source learned policies, or whether
   *every* embedded-learned-function case routes through the FFI
   shared-lib path. Recommendation: route everything through FFI to
   keep the integration surface narrow; the smooth-op subset is
   already what Jaxility's translator handles for dynamics, and
   conflating policies with dynamics there muddies the contract.

---

## ADR-018 — Cross-target acados/blasfeo/hpipm archives are built from source per target, never vendored as binary blobs

**Status:** Accepted

### Context

A cross-compiled controller artifact (`jaxility.builder_cross`) emits
objects that reference acados / hpipm / blasfeo symbols. On the build
host those symbols live in the shared libraries acados builds by
default; for an Arm cross target they do not exist until the three
libraries are cross-built **as static archives** and fed into the link
step through the `extra_link_args` / `extra_include_dirs` seam in
`plan_cross_compile`. Until this ADR that seam was empty and
`cross_build_for_target` died at link with a controlled `ToolchainError`
(KNOWN_GAPS.md "Linker gap"; the T-031 follow-up).

The TODO framed the choice as "cross-build **and vendor (or pin
upstream)**". This is the cross-target instance of OQ-1 (vendor a
minimal acados vs pin upstream), deferred to T-021 for the *host* path
and resolved there by pinning an upstream acados checkout
(`JAXILITY_ACADOS_LIBRARY_PIN`, AGENTS/TOOLCHAINS.md). The cross target
needs the same answer made explicit.

### Decision

The acados / blasfeo / hpipm static archives a deployment artifact
links against are **built from source for the target**, reusing the
same pinned upstream acados checkout the host path already depends on,
routed through the target's Arm GNU toolchain via a CMake toolchain
file (`cmake/toolchains/<target>.cmake`). Prebuilt binary archives are
**never** committed to the repo or shipped as blobs.

`jaxility.builder_deps` implements this with the standard plan/execute
split: `plan_dep_build` composes a pure-data `DepBuildPlan` (the cmake
configure/build argv, install prefix, expected archives, include dirs,
link args); `execute_dep_build` runs it; `build_cross_deps` is the
one-call convenience. The configure differs from the host build in
exactly three knobs: `-DBUILD_SHARED_LIBS=OFF` (emit `.a` not `.so`),
`-DBLASFEO_TARGET=<per family>` (pin blasfeo's micro-arch kernels —
`ARMV8A_ARM_CORTEX_A76` for the Pi 5), and
`-DCMAKE_TOOLCHAIN_FILE=<target>` (route every compile through the
cross toolchain). The resulting `CrossBuiltDeps` plugs into
`cross_build_for_target(..., deps=...)`, which merges its include dirs +
link args into the controller cross-compile and records each archive's
BLAKE3 hash in the manifest under `dep-archive:<name>`.

### Alternatives considered

1. **Vendor prebuilt `.a` archives in the repo (or via a release /
   LFS).** Rejected. Binary blobs in git carry a license/provenance
   burden, cannot be audited by reading the tree, and cut against the
   pin-everything reproducibility model (ADR-005/006, invariants 3/5).
   The point of `ToolchainPin` is that every binary touching a build is
   reconstructible from a pinned source + pinned toolchain; a vendored
   blob is a binary nobody can re-derive.
2. **`find_package(acados)` against a system install on the build
   host.** Rejected — system install is host-state the manifest can't
   pin, and there is no aarch64-target acados system package.
3. **acados' own auto-download of prebuilt blasfeo.** Not available for
   cross targets, and would reintroduce the unpinned-blob problem (cf.
   the from-source `t_renderer` decision in TOOLCHAINS.md).

### Consequences

- The Pi 5 link gap closes: with `deps=` supplied,
  `cross_build_for_target` produces a real ELF shared object. The
  Tier-B test
  `test/unit/test_builder_deps.py::test_cross_build_for_target_links_with_deps`
  exercises this end-to-end when both the aarch64 toolchain and an
  acados source tree are present.
- Adding a new Arm family requires two explicit rows (no silent
  default, invariant 7): a blasfeo `TARGET` in
  `_BLASFEO_TARGET_FOR_FAMILY` and a CMake toolchain file registered in
  `_TOOLCHAIN_FILE_FOR_BINARY`.
- The bare-metal Cortex-M (`arm-none-eabi`) archives are **out of scope
  here** — they belong to T-052 and are materially harder
  (no OS/malloc; newlib + embedded blasfeo config). This ADR covers the
  hosted-ABI aarch64-linux lane only.
- **Reproducibility caveat:** upstream acados/blasfeo do not invoke
  `ar` in deterministic (`D`) mode, so the produced archives are not
  guaranteed byte-identical across builds. Jaxility records their
  content hashes for *provenance* (the manifest names exactly which
  archives an artifact linked) but does not yet claim bit-reproducible
  third-party archives. Tracked in KNOWN_GAPS.md.
- Resolves the cross-target half of OQ-1.

---

## Open questions

These are explicitly tracked as undecided. Do not default them.

### OQ-1 — Whether to vendor a minimal acados or pin upstream

acados v0.5+ has gotten more flexible about its build but still
requires building from source for embedded targets. The choice is to
vendor a known-good acados tarball into Jaxility's build process or
to pin a tagged upstream version and depend on the user having it.
Decision deferred to T-021.

### OQ-2 — FreeRTOS vs. bare-metal on Cortex-M

Both are viable for the Cortex-M targets. FreeRTOS gives portability
and scheduling primitives; bare-metal gives the most direct control
and lowest overhead. Decision deferred to T-052 with a benchmark
on Crazyflie.

### OQ-3 — Which JAX subset to support in the lowering

The full JAX op set is too large to translate. The smooth-op subset
that acados can consume is well-defined. The question is whether to
also support a "smoothed" wrapper for `jnp.where` over traced
predicates (replacing it with a sigmoid approximation parametrized
by a sharpness factor), or to reject those programs entirely.
Decision deferred to T-020.

### OQ-4 — How to handle the Apple Silicon target

Apple Silicon is in the strategic SoC list, but Apple's toolchain
and SDK story is more closed than the Arm reference. The build host
likely has to be macOS. Whether to support Apple Silicon at v0.x is
deferred; it is a later-roadmap task at the earliest.

### OQ-5 — Coordination with Jaxterity on the `Robot` serialization

Jaxterity's ADR open question OQ-3 asks the same thing from the
other side. The decision needs to be coordinated. Default assumption
in Jaxility: import Jaxterity directly and pass `CalibratedRobot` in
memory; the serialization-based handoff is a v0.2 consideration.
