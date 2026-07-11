# Cartpole — Jaxility zoo entry

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Name             | `cartpole`                                     |
| Source           | `jaxterity.zoo.load("cartpole")` (real robot)  |
| Target           | `mock-cortex-a`                                |
| Controller       | LQR (T-022)                                    |
| Real target lane | Pi 5 / Cortex-A76 (T-030..T-034)               |
| Status           | mock pipeline                                  |
| License          | MIT                                            |

## What this entry exercises today

The mock pipeline runs end-to-end on the real Jaxterity
`Robot`: the attestation chain is intact, the manifest verifies under
`HashChainSigner`, and the equivalence check passes against the
(synthetic) source trajectory.

## Remaining work to land the real deployment

1. Replace the synthetic `simulate` with MJX-driven trajectory
   via `robot.to_diagram()` (T-026).
2. Land the real LQR template (T-022).
3. Wire to Pi 5 / Cortex-A76 toolchain (T-031).
4. HIL parity tests against the deployed binary (T-033).
5. Benchmark on Pi 5 (T-035).

## Why Cartpole leads the zoo

It is the simplest controller class (linear, single-mode), making it
the fastest path to first end-to-end working Cortex-A artifact at
launch. The launch demo target is *Cartpole on Pi 5 at 1 kHz with a
signed manifest*.
