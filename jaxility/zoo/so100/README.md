# SO-100 — Jaxility zoo entry

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Name             | `so100`                                        |
| Source           | `jaxterity.zoo.load("so100")` (real robot)     |
| Target           | `mock-cortex-a`                                |
| Controller       | WBC (OCP template: T-024 follow-on)            |
| Real target lane | Post-launch (T-070)                            |
| Status           | real robot; closed-form ABA dynamics, OCP pending |
| License          | MIT                                            |

## Dynamics (T-111)

SO-100 / SO-101 is a serial **6-DoF revolute manipulator** (fixed base;
`nq = nv = 6`). Like the flyers, the deployed plant is a closed form, not MJX —
but a manipulator's forward dynamics need `M(q)⁻¹`, and the lowering coverage
table has no linear solve. The lowerable route is **Featherstone's Articulated
Body Algorithm** (`_dynamics.py`): O(n) forward dynamics using only spatial
(6-vector) matmuls and scalar reciprocals `1/D_i`, all inside the smooth-op
subset, so it translates to CasADi.

`_dynamics_factory` returns `(f, (12,), (6,))` with `state = [q(6), q̇(6)]` and
`control = τ(6)`. The spatial tree — parent→child transforms, joint axes, and
per-link mass/com/inertia — is read from the calibrated Robot, so a
recalibration propagates into the lowered binary.

**Fidelity is manipulator-grade, not ULP.** The ABA matches the robot's MJX
`functional_dynamics` to `~1e-5` rel in a contact-free, non-singular regime
(tested at a `1e-4` bound). An *independent* recursive algorithm and MuJoCo's
internal `cinert` CRB diverge by ~1e-6 on this featherweight arm (distal
inertias ~1e-5) — a representational floor, not a bug (see `KNOWN_GAPS.md`).
Verified in `test/unit/test_jaxterity_chain.py`:

- `test_so100_closed_form_matches_mjx_reference` — ABA vs MJX within the bound.
- `test_so100_calibration_propagates_into_deployed_dynamics` — mass change moves
  both the attestation handle and the deployed dynamics.
- `test_so100_closed_form_lowers_to_casadi` — no `CoverageError`.

## Remaining work

1. Wire the **WBC OCP template** for the 12-state manipulator (T-024). Until
   then `jaxility build so100` fails *structurally* with a clear "template not
   wired" reason — the dynamics translate, but the controller does not build yet.
2. Land the Jaxterity `Task` DSL consumer (T-024).
3. Bring up a real-target deployment lane post-launch (T-070).
