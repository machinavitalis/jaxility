# Berkeley Humanoid Lite — Jaxility zoo entry

| Field            | Value                                            |
|------------------|--------------------------------------------------|
| Name             | `berkeley_humanoid_lite`                         |
| Source           | **Stub** (`MockSource`); see "Upstream gap"      |
| Target           | `mock-cortex-a`                                  |
| Controller       | CentroidalMPC (T-025)                            |
| Real target lane | Post-launch (T-080)                              |
| Status           | mock pipeline; **stub source**                   |
| License          | MIT                                              |

## Upstream gap

Berkeley Humanoid Lite is in the Jaxility documentation but **not yet
in the Jaxterity zoo**. Jaxterity has floating-base
support (its ADR-018) but needs:

- A humanoid URDF / MJCF.
- Per-joint actuator modelling at humanoid scale.
- Contact modelling for foot strikes.

Once those land, swap `MockSource` for `JaxteritySource.from_robot` and
drop `_BHL_HANDLE_SALT`. The Jaxility-side contract is stable.

## Mock contract today

The stub uses a 25-DoF synthetic with a unique handle salt. The mock-pipeline
tests assert "humanoid artifact hash is distinct from the other zoo
entries" without depending on a real humanoid URDF.

## Remaining work

1. Promote to real-robot when Jaxterity ships a humanoid zoo entry.
2. Land the centroidal MPC template (T-025).
3. Real-target deployment lane post-launch (T-080).
