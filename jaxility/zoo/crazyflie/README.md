# Crazyflie — Jaxility zoo entry

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Name             | `crazyflie`                                    |
| Source           | **Stub** (`MockSource`); see "Upstream gap"    |
| Target           | `mock-cortex-m`                                |
| Controller       | TrackingMPC (T-023)                            |
| Real target lane | STM32H7 / Cortex-M7 (T-050..T-053)             |
| Status           | mock pipeline; **stub source**                 |
| License          | MIT                                            |

## Upstream gap (the load-bearing note for review)

Crazyflie ships in the Jaxility documentation set (CONTEXT.md, SKILL.md,
the v0.1 launch plan) but **is not yet in the Jaxterity zoo**.
Jaxterity will need to land:

- A Crazyflie URDF (or MJCF) with the floating base.
- Rotor modelling on each motor.
- Contact modelling for ground interaction.

Once those land in Jaxterity's zoo, this entry's `source_factory` swaps
from `MockSource` to `JaxteritySource.from_robot(load("crazyflie"))`
and the `_CRAZYFLIE_HANDLE_SALT` constant is removed. The Jaxility-side
contract (target, template, dtype, n_steps, license, remaining_work)
does not change.

## Mock contract today

The stub `MockSource` uses a deterministic handle derived from
`_CRAZYFLIE_HANDLE_SALT` so that mock-pipeline tests can assert "Crazyflie
artifact hash is distinct from Cartpole / SO-100 / Berkeley humanoid"
without depending on a real upstream model. The trajectory is the same
synthetic the other zoo entries use — it satisfies the equivalence
check trivially.

## Remaining work

1. Promote to real-robot when Jaxterity ships a Crazyflie zoo entry
   (rotor + contact modelling required).
2. Land the trajectory-tracking MPC template (T-023).
3. Cortex-M7 cross-compilation lane + linker scripts (T-051).
4. FVP-driven HIL parity (T-053).
