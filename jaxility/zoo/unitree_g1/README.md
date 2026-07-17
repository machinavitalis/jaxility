# Unitree G1 — Jaxility zoo entry

| Field            | Value                                              |
|------------------|----------------------------------------------------|
| Name             | `unitree_g1`                                       |
| Source           | `jaxterity.zoo.load("unitree_g1")` (real robot)    |
| Target           | `mock-cortex-a`                                     |
| Controller       | WBC (single joint-space regulation task)           |
| Real target lane | Post-launch (T-070)                                |
| Status           | real robot; **first branched entry**, generator-sourced |
| License          | MIT                                                |

## Dynamics (T-126)

Unitree G1 is a **29-DoF humanoid** presented fixed-base (welded pelvis;
`nq = nv = 29`), all-revolute, and **branched** — the pelvis carries three
children (both legs + torso), and the torso branches again to the two arms and
head. The hand-written serial-chain ABA (`zoo/so100/_dynamics.py`) structurally
cannot express a branch, so G1's deployable dynamics are **generated** from the
robot's MJCF:

```python
generate_dynamics(robot.to_mjcf(), source_format="mjcf", tree_source="mujoco")
```

`tree_source="mujoco"` builds the rigid-body tree from the *compiled* MuJoCo
model — the authoritative source for an MJX-native robot — folding fixed
appendage bodies into their moving ancestor and reading `dof_armature` directly.
Armature (~0.004–0.025 here) dominates the effective inertia of the light distal
joints and is what Pinocchio's MJCF parser silently zeroes, so reading it from
MuJoCo is what makes the deployed dynamics track MJX.

**Fidelity.** The generated dynamics matches the *unconstrained* rigid-body
dynamics (Pinocchio ABA with the model's armature) to `~1.7e-7`. The reference is
unconstrained on purpose: MJX forward dynamics for a 29-DoF humanoid has
joint-limit constraint forces active at essentially every pose (0 of 200 random
poses were constraint-free), and the deployed model excludes those by design —
limits live in the OCP, not the dynamics (see `KNOWN_GAPS.md`, T-124/T-122).
Verified in `test/unit/test_zoo_unitree_g1.py` and
`test/unit/test_pinocchio_rbd.py::test_mujoco_tree_source_branched_humanoid`.

## Controller (T-024)

`jaxility build unitree_g1 --target host` produces a real artifact: the WBC
template builds a 58-state / 29-input OCP from a single joint-space regulation
task (hold the neutral pose), with the generated dynamics. The acados codegen +
compile of a 58-state model is slow (minutes), so the end-to-end build test is
marked `slow`.

## Remaining work

1. Richer, multi-task whole-body control via the Jaxterity `Task` DSL (T-024);
   the current entry uses one joint-space regulation task.
2. Floating base + contact for locomotion — parked (T-122); a fixed-base
   humanoid is contact-free by construction.
3. Bring up a real-target deployment lane post-launch (T-070).
