# Crazyflie — Jaxility zoo entry

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Name             | `crazyflie`                                    |
| Source           | Real Jaxterity robot (`JaxteritySource.from_robot`) |
| Target           | `mock-cortex-m`                                |
| Controller       | TrackingMPC (OCP template: T-110 follow-on)    |
| Real target lane | STM32H7 / Cortex-M7 (T-050..T-053)             |
| Status           | real robot; closed-form dynamics, OCP pending  |
| License          | MIT (Jaxterity zoo + Jaxility zoo entry)       |

## Dynamics (T-110)

Crazyflie is a real Jaxterity zoo robot: a single free rigid body on a
`FLOATING` base (`nq == 7`: position + unit quaternion; `nv == 6`). Like
Cartpole, the **deployed plant is a closed form**, not MJX — MJX exposes the
dynamics through a constraint-solver `while_loop` that cannot be lowered to
acados' fixed-size SQP graph (ADR-016). A quadrotor is a single free body, so
its closed form is the Newton-Euler equations of a floating base:

```
state   = [ pos(3), quat_wxyz(4), v_world(3), omega_body(3) ]   # 13
control = [ thrust, Mx, My, Mz ]                                # 4  (body frame)
```

`_dynamics_factory` returns `(f, (13,), (4,))`. The scalars `(m, I, g)` are read
from the calibrated Robot (`_reduced_params`), so a mass/inertia recalibration
propagates into the lowered binary — the attestation handle and the deployed
dynamics move together (one model, one truth). The closed form assumes the COM
sits at the body-frame origin (true for the vendored `cf2.xml`).

It stays inside the smooth-op subset, so it **lowers to CasADi** — verified in
`test/unit/test_jaxterity_chain.py`:

- `test_crazyflie_closed_form_matches_mjx_reference` — the closed form matches
  `jaxterity.zoo.crazyflie.thrust_dynamics()` (the MJX reference) to ~ULP.
- `test_crazyflie_calibration_propagates_into_deployed_dynamics` — doubling the
  mass moves both the attestation handle and the deployed dynamics.
- `test_crazyflie_closed_form_lowers_to_casadi` — no `CoverageError`.

## Remaining work

1. Wire the **quaternion-aware tracking-MPC OCP template** (follow-on to T-110;
   attitude in acados needs a unit-norm handling choice). Until then
   `jaxility build crazyflie` fails *structurally* with a clear "template not
   wired" reason — the dynamics translate, but the controller does not build yet.
2. Cortex-M7 cross-compilation lane + linker scripts (T-051/T-052).
3. FVP-driven HIL parity (T-053).
