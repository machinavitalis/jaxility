# Jaxility Skill

You are using Jaxility, the open-source compiler from JAX-trained
robotics to Arm silicon. This file is your operating manual. Read it
before suggesting any Jaxility code.

## What Jaxility does

Jaxility takes a calibrated robot from Jaxterity (plus optionally a
trained policy) and produces a binary that runs on a target Arm-based SoC at
the required cycle time, with a signed attestation manifest binding the
build to its source.

It is the deployment layer at the base of a larger JAX-native stack — for where
it sits and what feeds it, see `AGENTS/CONTEXT.md`.

## When to use Jaxility

Use it when the user is:

- Deploying a calibrated Jaxterity `Robot` to embedded hardware.
- Cross-compiling an MPC, LQR, WBC, or centroidal controller.
- Deploying a JAX-trained policy to a Cortex-A or Cortex-M chip.
- Composing a model-based controller with a learned policy at the
  runtime layer.
- Verifying that a deployed binary's provenance matches its claimed
  source.
- Benchmarking a controller-policy combination across Arm-based SoCs.

## When NOT to use Jaxility

- **Training policies.** Use K-Sim, RSL-RL, LeRobot, or the user's
  training stack. Jaxility deploys; it does not train.
- **Calibrating a robot.** Use Jaxterity's `sysid` recipes. Jaxility
  consumes the calibrated robot.
- **Simulating a robot.** Use Jaxonomy or Jaxterity directly.
- **Deploying to x86, NVIDIA Jetson, or RISC-V.** A later phase; not
  supported yet.
- **Generic C/C++ codegen from JAX.** Jaxility is robotics-shaped. For
  generic JAX → embedded code, point the user at JAX's MLIR
  pipelines or at IREE.
- **Fleet management, OTA updates, hosted compilation.** That's the
  `jaxility-enterprise` package. Out of scope for the OSS library.

## Decision tree for the typical workflow

When the user asks for help with Jaxility, route through this tree:

1. **Does the user have a `CalibratedRobot` from Jaxterity?**
   - Yes → continue.
   - No → first calibrate the robot with Jaxterity. Jaxility cannot
     deploy an uncalibrated robot (the attestation chain breaks).

2. **What target SoC?**
   - Raspberry Pi 5 (Cortex-A76) → fully supported at v0.1; launch target.
   - Other Arm-based SoCs → check `jaxility targets`; most are
     "experimental" until HIL parity passes.
   - Non-Arm (x86, Jetson, RISC-V) → a later phase; explain and stop.

3. **What controller?**
   - LQR (Cartpole-class) → `jaxility.templates.LQR`.
   - Trajectory-tracking MPC (Crazyflie-class) →
     `jaxility.templates.TrackingMPC`.
   - Task-space WBC (manipulator-class) →
     `jaxility.templates.WBC`.
   - Centroidal MPC (humanoid-class) →
     `jaxility.templates.CentroidalMPC`.
   - Custom dynamics → write a template; this is more advanced.

4. **Is there a learned policy on top?**
   - No → straight acados-only build.
   - Yes → use the dual-path runtime; declare a `CompositionPlan`
     specifying rate, safety envelope, and fallback behavior.

5. **What's the user actually trying to do right now?**
   - "Build a binary" → `jaxility build`.
   - "Check that a binary matches its source" → `jaxility verify`.
   - "Measure performance" → `jaxility bench`.
   - "Run hardware-in-the-loop" → `jaxility hil`.
   - "See what's supported" → `jaxility targets`,
     `jaxility coverage`.

## Core surface, in order of how often agents will use it

### Building

The canonical operation is the CLI:

```
jaxility build <robot> --target <soc> --template <controller>
```

The Python API mirror:

```python
from jaxility import build
artifact = build(
    robot=calibrated_robot,         # from Jaxterity
    target="pi5",                   # Cortex-A76 profile
    template="LQR",
    template_options={"Q": Q_matrix, "R": R_matrix},
)
```

`artifact` carries the binary bytes (or generated C project), the full
manifest, and the build log. The artifact's manifest references the
robot's attestation handle, so the chain is intact.

### Verification

```
jaxility verify <manifest_path>
```

Walks the hash chain from the deployed binary up through the manifest
to the source robot's attestation handle. Returns 0 on a valid chain,
40 on a manifest error.

### Benchmarking

```
jaxility bench cartpole --target pi5
```

Runs the standardized workload, captures cycle time, jitter, memory,
energy (if measurable). Emits a JSON record committed to the benchmark
database. Reproducibility is the bar.

### Hardware-in-the-loop

```
jaxility hil cartpole --target pi5
```

Runs the source simulation on the host and the deployed binary on the
target (or FVP), step-locked, comparing every state at every cycle.
Divergence points at the offending codegen step.

### MCP tools (when running under MCP)

The `jaxility.mcp` server exposes:

- `build(@robot, target, template, options) -> @artifact[<hash>]`
- `verify(@artifact) -> ChainReport`
- `bench(@artifact) -> BenchmarkRecord`
- `hil(@artifact, scenario) -> HILReport`
- `list_targets() -> [Target]`
- `coverage(target) -> CoverageReport`

Stable handles follow the Jaxterity pattern. The source robot's
`attestation_handle` is the same hash that becomes
`@robot[<hash>]` in the MCP layer.

## Three-package chain

The chain Jaxility participates in:

```
telemetry → CalibratedRobot.attestation_handle → Manifest.source_handle
         (Jaxterity)                          (Jaxility)
                  → Artifact.manifest_hash → Binary.embedded_hash
                                          (deployed)
```

When the user asks "is this binary actually from this telemetry?",
`jaxility verify` walks the chain and answers yes or no. Do not
short-circuit any step of this chain. Do not silently regenerate any
hash.

## What you must NOT do

- **Modify the manifest after a build.** Manifests are immutable;
  changes require rebuilds.
- **Run a build with a missing toolchain.** The target loader
  validates toolchain presence at startup; if it fails, do not work
  around it. Point the user at the install docs.
- **Suggest a target without HIL coverage for production deployment.**
  Profile-only and host-only targets are "experimental"; flag this
  clearly.
- **Bypass the equivalence check.** Every build runs it; suggesting
  the user disable it is wrong. Failures mean the codegen broke; fix
  the build, not the check.
- **Re-implement parts of acados, CasADi, LiteRT, or vendor SDKs.**
  Use them as designed. If they fall short, upstream the fix.
- **Use floats for timestamps or for hashed payloads.** Integer
  microseconds-since-epoch; canonical JSON.
- **Mix in target-specific Python imports in user-facing code.** Use
  the `--target` flag; let Jaxility dispatch.

## Coverage and limits

The lowering pipeline supports a subset of JAX:

**Supported:** `+`, `-`, `*`, `/`, `**`, `jnp.sin`, `jnp.cos`,
`jnp.tan`, `jnp.exp`, `jnp.log`, `jnp.sqrt`, `jnp.where` (static
predicates only), matrix multiplication, static-index slicing.

**Not supported (yet):** `lax.cond` over traced predicates,
`lax.while_loop`, dynamic shapes, dynamic indexing, custom
`jax.custom_vjp` not in the equivalence-validated set.

`jaxility coverage --target <soc>` prints the current matrix.
If the user hits a coverage error, the error message names the op and
suggests either (a) a smoothing approximation, (b) a different
template that doesn't need that op, or (c) waiting for a future
release.

## Supported targets at v0.1

| Target          | SoC           | Status         | Notes                          |
|-----------------|---------------|----------------|--------------------------------|
| `pi5`           | Cortex-A76    | Supported      | Full HIL; launch target        |
| Other Arm-based | Various       | Expanding      | Check `jaxility targets`       |

Use `jaxility targets --detail` to see the full matrix at any moment.

## Three-package safety chain

Jaxility's safety story is collective, not local:

- **Jaxonomy** ensures the simulation is mathematically correct.
- **Jaxterity** ensures the robot model is calibrated against real
  telemetry.
- **Jaxility** ensures the deployed binary matches the simulated
  model and carries proof of how it was built.

A user deploying without all three (e.g., simulating in MuJoCo,
hand-translating to acados, deploying without Jaxility) loses the
chain. When advising users, the chain is the value proposition:
piecewise replacement gives up the property.

## Pitfalls & idioms (from consumer sessions)

- **The OSS signer has no signature — that's healthy.** `HashChainSigner.sign`
  returns `None`; integrity comes from the BLAKE hash chain, so
  `verify_manifest(...).signature_status == "absent"` is the *expected* state
  (cryptographic signing is an enterprise add-on). Don't assert `"verified"`.
  `verify_manifest` compares the *recomputed chain hash*
  (`report.recomputed_content_hash_hex`), not the artifact payload hash.
- **A fixed feedback law deploys without acados.** acados is only needed for the
  OCP templates (`lqr`, `tracking_mpc`, …). A smooth fixed law (e.g. an FOC PI +
  decoupling loop) lowers straight through CasADi: `translate(jax_fn,
  in_shapes=..., target_family="cortex-m4")` returns a `CasadiFunction` whose
  `.fn.generate("x.c")` emits C to cross-compile with the pinned
  `arm-none-eabi-gcc`. `cf.primitives_used` is the manifest audit trail.
- **Reductions don't lower — unroll them.** `translate` rejects `reduce_*`
  primitives, so `jnp.sum(forces, axis=0)` (and `mean`/`prod`/`max`) fails.
  Write a fixed-size unrolled version (`f0 + f1`, static slices `u[0:3]` — not
  `reshape`) for the deploy path; keep the `jnp.sum` form for
  planning/simulation.
- **Don't gate acados on `import acados_template`.** The Python package can
  import while the compiled library is unconfigured (`ACADOS_SOURCE_DIR` unset,
  no `t_renderer`). Probe the real prerequisites before claiming an OCP build
  will run.
- **Deploy a *calibrated* control law so recalibration moves the artifact.**
  `Robot.attestation_handle` is a property (BLAKE3 over model + params); build
  the manifest with `source_attestation_handle=bytes.fromhex(handle)` so
  recalibration changes both the handle and the artifact hash, and
  `verify_manifest` chains them (no stale-binary pairing).

## When to escalate to the human

- **Coverage error on an op the user thinks should work.** Could be a
  bug in the coverage table, could be a real limit. Escalate.
- **Equivalence check fails on a build that "should" work.** A
  generated binary diverges from the source simulation. This is a
  serious bug; never advise the user to suppress the check.
- **HIL divergence in a production-like scenario.** Stop and escalate.
- **Manifest verification fails on an artifact the user trusts.**
  Either the chain is broken (real bug) or the user's trust is
  misplaced (their problem). Either way, human decision.
- **Any deployment in safety-critical contexts** (robotic surgery,
  automotive, aerospace). Jaxility's manifests are evidence, not
  certification. Always make this clear.
- **Adding a new target.** This is a later-roadmap task with a defined
  series (profile → host equivalence → runtime → HIL → benchmark).
  Don't add a target as part of a build session.

## Where to find more

- `AGENTS/CONTEXT.md` — the contributor-facing architectural
  orientation.
- `AGENTS/PATTERNS.md` — coding conventions.
- `AGENTS/DECISIONS.md` — ADRs.
- `AGENTS/TOOLCHAINS.md` — toolchain version pinning policy.
- `docs/CLAIMS.md` — what Jaxility actually does, measured.
- `docs/KNOWN_GAPS.md` — what it does not do.
- `examples/` — end-to-end examples, one per zoo robot.
- `docs/targets/<target>/` — per-target details, quirks, performance.
