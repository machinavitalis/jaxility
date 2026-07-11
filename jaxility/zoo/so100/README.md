# SO-100 — Jaxility zoo entry

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Name             | `so100`                                        |
| Source           | `jaxterity.zoo.load("so100")` (real robot)     |
| Target           | `mock-cortex-a`                                |
| Controller       | WBC (T-024)                                    |
| Real target lane | Post-launch (T-070)                            |
| Status           | mock pipeline                                  |
| License          | MIT                                            |

## What this entry exercises today

The mock pipeline runs end-to-end on Jaxterity's real SO-101
URDF (called `so100` in the Jaxterity registry): the attestation
chain is intact, the manifest verifies, and the equivalence check
passes against the synthetic source trajectory.

## Remaining work

1. Replace the synthetic `simulate` with MJX-driven trajectory
   via `robot.to_diagram()` (T-026).
2. Land the WBC template + Jaxterity `Task` DSL consumer (T-024).
3. Bring up a real-target deployment lane post-launch (T-070).
