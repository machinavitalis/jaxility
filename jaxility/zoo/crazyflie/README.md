# Crazyflie — Jaxility zoo entry

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Name             | `crazyflie`                                    |
| Source           | Real Jaxterity robot (`JaxteritySource.from_robot`) |
| Target           | `mock-cortex-m`                                |
| Controller       | TrackingMPC (hover / small-attitude regulation) |
| Real target lane | STM32H7 / Cortex-M7 (T-050..T-053)             |
| Status           | real robot; **builds end-to-end** (Newton-Euler + tracking MPC) |
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

## Controller (T-110b)

`jaxility build crazyflie --target host` now produces a real artifact: a
tracking MPC that regulates hover at the origin with identity attitude. The
13-state cost penalizes quaternion error in the Euclidean/tangent sense, which
is correct **near identity (hover / small attitude)**; the reference is the
weight-cancelling hover thrust, with thrust/moment bounds. Tested end-to-end in
`test/unit/test_cli_zoo_build.py`.

## Remaining work

1. **Large-attitude quaternion MPC** (T-110b follow-on): a geodesic quaternion
   cost + unit-norm handling to generalize beyond hover to aggressive maneuvers.
2. Cortex-M7 cross-compilation lane + linker scripts (T-051/T-052).
3. FVP-driven HIL parity (T-053).
