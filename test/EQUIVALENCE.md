# Jaxility — Numerical equivalence contract

This document is the human-readable tolerance contract for the
numerical equivalence check (invariant 1, `AGENTS/CONTEXT.md`). The
machine-readable source of truth is
[`jaxility/testing/tolerances.py`](../jaxility/testing/tolerances.py)
(`TOLERANCE_TABLE`); the tests in
[`test/unit/test_equivalence.py`](unit/test_equivalence.py) assert the
two stay in sync.

If you change a tolerance — loosening or tightening — the change
travels through PR review with a justification in the `rationale`
field of the `Tolerance` row *and* a corresponding line in this file.
There are no magic numbers (`PATTERNS §7.4`).

## Lookup key

The tolerance lookup is `(target_family, dtype, quantity)`:

- **`target_family`** — the SoC family identifier
  (`mock-cortex-a`, `mock-cortex-m`, `cortex-a76`, `cortex-m7`, ...).
  Lookup is exact. Every Cortex-A76 deployment shares the
  `cortex-a76` family bounds.
- **`dtype`** — `"float32"` or `"float64"`. `float32` is the
  embedded-default precision and gets correspondingly looser bounds
  than `float64`.
- **`quantity`** — the trajectory key
  (`joint_position`, `joint_velocity`, `actuator_torque`, ...).

A miss raises `KeyError`. Adding a quantity is a deliberate
table entry, not a silent default.

## Grades

The `grade` field on a `Tolerance` documents the *intent* of the bound:

| Grade           | Meaning                                                                                                            |
|-----------------|--------------------------------------------------------------------------------------------------------------------|
| `bit-exact`     | Bit-identical floats. Used only for content hashes and integer-coded quantities; not for floating-point quantities. |
| `ulp-bounded`   | Within a small ULP count of the source. Embedded-default for properly-implemented codegen on a single precision.    |
| `approximate`   | Engineering-tolerance level. Used for cross-precision comparisons (source `float64` vs. candidate `float32`).       |

## Mock-target entries (mock targets only)

The mock-target rows cover only the mock targets. Real-target entries
land with the corresponding target profile.

### `mock-cortex-a` × `float64`

The mock pipeline (T-015) runs at `float64` by default; bounds
are tight because the candidate just wraps the source JAX in a
function-call interface.

| Quantity            | abs_tol   | rel_tol   | Grade         | Rationale                                                                                |
|---------------------|-----------|-----------|---------------|------------------------------------------------------------------------------------------|
| `joint_position`    | `1e-12`   | `1e-10`   | `ulp-bounded` | Mock pipeline wraps source JAX in float64; mirrors JAX's own jit ULP envelope.           |
| `joint_velocity`    | `1e-10`   | `1e-9`    | `ulp-bounded` | Velocities accumulate one integration step of position error; one OOM looser than position. |
| `actuator_torque`   | `1e-10`   | `1e-9`    | `ulp-bounded` | Actuator outputs are linear in state at the mock layer; same envelope as joint_velocity. |

### `mock-cortex-a` × `float32`

Cross-precision (source at `float64`, candidate at `float32`); looser
to reflect single-precision rounding accumulation.

| Quantity            | abs_tol  | rel_tol  | Grade         | Rationale                                                                                 |
|---------------------|----------|----------|---------------|-------------------------------------------------------------------------------------------|
| `joint_position`    | `1e-5`   | `1e-4`   | `approximate` | float32 candidate vs. float64 source; reflects single-precision rounding accumulation.    |
| `joint_velocity`    | `1e-4`   | `1e-3`   | `approximate` | float32 velocity inherits one extra integration step of float32 noise.                    |
| `actuator_torque`   | `1e-4`   | `1e-3`   | `approximate` | float32 actuator outputs; matches velocity bound at mock scale.                           |

### `mock-cortex-m` × `float32`

Embedded-default precision. Bounds are identical to `mock-cortex-a` ×
`float32` because the *mock* layer does not introduce target-specific
drift; the divergence between Cortex-A and Cortex-M shows up at the
real-codegen layer.

| Quantity            | abs_tol  | rel_tol  | Grade         | Rationale                                                                       |
|---------------------|----------|----------|---------------|---------------------------------------------------------------------------------|
| `joint_position`    | `1e-5`   | `1e-4`   | `approximate` | Identical to mock-cortex-a float32; mock layer does not differentiate yet.      |
| `joint_velocity`    | `1e-4`   | `1e-3`   | `approximate` | Matches mock-cortex-a float32 joint_velocity; mock target does not differentiate. |
| `actuator_torque`   | `1e-4`   | `1e-3`   | `approximate` | Matches mock-cortex-a float32 actuator_torque; mock target does not differentiate. |

## Host build path entries

The host build path (T-026) runs the full JAX → CasADi → acados pipeline
and produces a real shared library. Equivalence is checked against a
JAX-side ERK4 forward simulation. Both backends compute at `float64`,
but they do not agree to ULP because acados defaults to multiple ERK
sub-steps per OCP stage while the JAX reference does a single ERK4
step. The bound is therefore **engineering-tight**, not ULP-tight, on
the host build path.

Tightening to ULP is a **T-027 follow-up**: configure the JAX
integrator to mirror acados' sub-step count and the bound drops to the
single-step round-off envelope. The HIL gate (T-033) replaces
the JAX reference with a sim-locked target run, at which point the
relevant comparison is "did codegen drift between host and target",
not "did the JAX integrator match the host integrator", and the ULP
envelope returns.

### `host-darwin` × `float64` and `host-linux` × `float64`

Identical rows for the two host families because the host build does
not distinguish darwin from linux at the dynamics layer.

| Quantity            | abs_tol   | rel_tol  | Grade         | Rationale                                                                                              |
|---------------------|-----------|----------|---------------|--------------------------------------------------------------------------------------------------------|
| `joint_position`    | `1e-2`    | `3e-1`   | `approximate` | Host build at float64; bound reflects integrator-step mismatch between acados sub-stepping and JAX RK4. |
| `joint_velocity`    | `5e-2`    | `5e-1`   | `approximate` | Velocity inherits one extra integration step of integrator-mismatch error; ~5× looser than position.     |
| `actuator_torque`   | `1e-10`   | `1e-9`   | `ulp-bounded` | Controls round-trip through LINEAR_LS cost without passing through the integrator; ULP envelope holds.   |

## HIL entries (real silicon)

The HIL gate (T-033) runs the deployed artifact on the real target —
the Raspberry Pi 5 / Cortex-A76 launch hardware — and compares it,
step-locked, against the host reference simulation. The candidate
executes at the embedded-default `float32` precision; the host
reference is `float64`. The bound is **engineering-tight**, dominated
by single-precision rounding accumulated over the control horizon, not
by integrator mismatch (the T-033 fixture and its host reference share
the same explicit-Euler recurrence exactly).

These rows are measured against the deterministic HIL fixture
(`test/hil/fixtures/cartpole_hil.c`), which stands in for the generated
acados controller until T-034. When the real shim lands it emits the
same trace contract and these bounds tighten toward the ULP envelope as
the integrators are matched.

### `cortex-a76` × `float32`

| Quantity            | abs_tol  | rel_tol  | Grade         | Rationale                                                                                          |
|---------------------|----------|----------|---------------|----------------------------------------------------------------------------------------------------|
| `joint_position`    | `5e-4`   | `2e-2`   | `approximate` | float32 artifact on Cortex-A76 vs float64 host reference; single-precision accumulation over a few hundred 1 kHz Euler steps on a decaying plant. |
| `joint_velocity`    | `1e-3`   | `2e-2`   | `approximate` | Angular rate carries one extra integration step of float32 noise; ~2× looser than joint_position.   |
| `actuator_torque`   | `5e-3`   | `2e-2`   | `approximate` | Actuator command is a fixed linear gain on the state, so it amplifies the state's float32 error by the gain magnitude. |

### `cortex-a76` × `float64`

The real **acados controller** HIL (T-034). The candidate is the generated
closed-loop binary — acados OCP control + acados sim (ERK) plant, at acados'
native `float64`. The source is the host closed loop: the same acados control
with a JAX ERK4 plant. **Measured host-vs-host divergence is ~1e-14 (ULP)** —
acados' ERK sim integrator matches the JAX ERK4 reference and the Python and C
codegen agree bit-for-bit — so the bound is **ULP-bounded, not engineering
tolerance**. The `1e-9` / `1e-7` envelope sits ~5 orders above the measured
host floor to carry cross-architecture libm (`sin`/`cos`) and FMA-contraction
differences on real Cortex-A76, to be re-confirmed when the on-Pi gate lands
(T-036). A looser bound would hide real codegen divergence. Quantities follow
the nx=4 Cartpole layout (`joint_position`/`joint_velocity` are 2-vectors).

| Quantity            | abs_tol  | rel_tol  | Grade         | Rationale                                                                                          |
|---------------------|----------|----------|---------------|----------------------------------------------------------------------------------------------------|
| `joint_position`    | `1e-9`   | `1e-7`   | `ulp-bounded` | Closed-loop acados controller vs host acados control + JAX ERK4 plant; measured ~1e-14 host-vs-host, bound carries cross-arch libm/FMA margin for Cortex-A76. |
| `joint_velocity`    | `1e-9`   | `1e-7`   | `ulp-bounded` | Same closed-loop ULP regime as joint_position.   |
| `actuator_torque`   | `1e-9`   | `1e-7`   | `ulp-bounded` | Control round-trips through the acados solve without an extra integration step; same ULP regime. |

## Violation rule

A quantity fails the equivalence check when **both** the per-step
absolute and per-step relative errors exceed their tolerances at the
same step:

```
per_step_abs_error > tolerance.abs_tol AND per_step_rel_error > tolerance.rel_tol
```

This avoids two pathologies:

- Pure relative-error overshoot at very small absolute magnitudes
  (e.g., a quantity near zero) — that's numerical noise, not a real
  divergence.
- Pure absolute-error overshoot at very large absolute magnitudes
  (e.g., a quantity in the millions of units) — that's a real
  divergence we *do* want to catch; the relative-error gate is
  permissive at large scale.

The first time step where the conjunction holds is the
`first_violation_step` on the report; the `_suggest` heuristic uses
that step plus the quantity name to pick a structured hint.

## How to add a quantity

1. Decide the `(target_family, dtype, quantity)` it applies to.
2. Add a row to `TOLERANCE_TABLE` in
   [`jaxility/testing/tolerances.py`](../jaxility/testing/tolerances.py)
   with a one-line `rationale`.
3. Add the corresponding row in this document under the right
   subsection (creating the subsection if needed).
4. Run the test suite — the cross-check test enforces parity between
   the table and this document.
5. PR title `T-NNN: extend equivalence contract for <quantity>` so
   review history records the change.

## How to add a target

A new target family in `TOLERANCE_TABLE` is a later-roadmap task. The
review checklist:

- Real bound numbers come from running the lowering pipeline on the
  reference robot zoo and observing the actual drift, not from
  guessing.
- A new family always lands with bounds that are *no looser* than the
  embedded-default `cortex-a76`-equivalent without a documented
  hardware reason in `rationale`.
- The `grade` reflects what the codegen produces, not what we wish it
  produced. Wishful grading is a CI failure.

## Cross-reference

- `AGENTS/CONTEXT.md` invariant 1 — equivalence is a build gate.
- `AGENTS/DECISIONS.md` ADR-007 — equivalence runs on host first,
  target second.
- `AGENTS/PATTERNS.md` §7 — test patterns;
  §7.4 — tolerances from this table.
