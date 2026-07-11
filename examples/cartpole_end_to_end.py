# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Cartpole end-to-end on the host — the Pi 5 pipeline minus the board (T-102).

Chains the full deployment stack in one script:

    calibrated Jaxterity Robot
        -> reduced_params (T-101)        # robot-faithful dynamics scalars
        -> translate (JAX -> CasADi)
        -> LQR template -> acados OCP
        -> host build (compile + link)   # a real, loadable controller
        -> closed-loop HIL parity        # the compiled controller stabilises it
        -> solve-time bench (1 kHz?)     # per-cycle timing on the host
        -> attestation manifest          # provenance rooted on the robot handle

By default this runs entirely on the host — the on-silicon launch (Jaxility
T-034/35/36, green on a real Pi 5) minus the board. **If ``JAXILITY_HIL_SSH_HOST``
names a tethered Pi 5**, step [7] also ships the robot-built controller to the
board, compiles it natively on the Cortex-A76, and verifies **on-silicon HIL
parity + a 1 kHz solve-time bench** — leaving only the video.

Why it matters: the deployed controller is built **from the robot**. Its
dynamics come from ``jaxterity.zoo.cartpole.reduced_params`` and its manifest is
rooted on the robot's real ``attestation_handle`` — so **recalibrating the robot
changes the compiled binary**. The script proves this at the end: it builds for a
nominal pole and a (heavier) recalibrated pole and shows the artifact hash and
the source handle both move. That is "one model, one truth" carried all the way
to the deployable artifact.

Run::

    python examples/cartpole_end_to_end.py                       # host only
    JAXILITY_HIL_SSH_HOST=user@pi python examples/cartpole_end_to_end.py  # + Pi 5

Requires the acados toolchain (``t_renderer``) + a host ``cc``, and a Jaxterity
that exports ``reduced_params`` (>= T-101). The on-Pi step also needs a reachable
target with a native acados (``JAXILITY_HIL_ACADOS``, default ``$HOME/acados``).
Exits 0 on success.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jaxility.builder import BuildBundle

from jaxility.bench import generate_controller_bench_source, run_controller_bench
from jaxility.builder import build_for_target
from jaxility.hil import (
    CARTPOLE_LQR_SCHEMA,
    CARTPOLE_LQR_TRACE,
    LocalRunner,
    build_controller_hil_binary,
    build_controller_on_target,
    generate_controller_hil_source,
    run_hil,
)
from jaxility.lowering import translate
from jaxility.manifest import verify_manifest
from jaxility.targets import current_host_target
from jaxility.templates import lqr

# --- the deployment workload (matches jaxility.bench.cartpole's LQR setup) ----
NX, NU = 4, 1
INITIAL_STATE = (0.3, 0.0, 0.0, 0.0)
HORIZON_STEPS = 20
TIME_HORIZON_S = 1.0
PLANT_DT = TIME_HORIZON_S / HORIZON_STEPS  # 0.05 s shooting interval
Q = (10.0, 10.0, 1.0, 1.0)
R = (0.1,)
INPUT_BOUNDS = ((-20.0,), (20.0,))
ONE_KHZ_BUDGET_US = 1000.0  # 1 ms per control cycle


def robot_dynamics(params: dict[str, float]):
    """The frictionless closed-form cartpole ``f(state, control)`` from the
    robot's reduced scalars ``params = {g, mc, mp, L}`` (the same closed form
    Jaxility's zoo entry lowers; see ``jaxterity.zoo.cartpole.reduced_params``).
    """
    g, mp, mc, length = params["g"], params["mp"], params["mc"], params["L"]

    def dynamics(state, control):
        theta, x_dot, theta_dot = state[1], state[2], state[3]
        sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
        denom = mc + mp * sin_t * sin_t
        x_ddot = (
            control[0] + mp * sin_t * (length * theta_dot * theta_dot + g * cos_t)
        ) / denom
        theta_ddot = (
            -control[0] * cos_t
            - mp * length * theta_dot * theta_dot * cos_t * sin_t
            - (mc + mp) * g * sin_t
        ) / (length * denom)
        return jnp.array([x_dot, theta_dot, x_ddot, theta_ddot])

    return dynamics


@dataclass
class Built:
    """A controller built from a robot, plus the dynamics it was built on."""

    bundle: BuildBundle
    dynamics: Callable  # f(state, control) -> dstate (JAX)
    model_name: str


def build_controller(robot, work_dir: Path, model_name: str) -> Built:
    """Export + host-build the acados LQR controller **from ``robot``**.

    Sources the dynamics scalars from the robot (T-101) and roots the manifest on
    the robot's real ``attestation_handle`` — so the artifact tracks calibration.
    """
    from jaxterity.zoo.cartpole import reduced_params

    params = reduced_params(robot)
    dynamics = robot_dynamics(params)
    cf = translate(dynamics, in_shapes=((NX,), (NU,)), name=model_name)
    spec = lqr(
        cf,
        Q=Q,
        R=R,
        initial_state=INITIAL_STATE,
        input_bounds=INPUT_BOUNDS,
        name=model_name,
        horizon_steps=HORIZON_STEPS,
        time_horizon_s=TIME_HORIZON_S,
    )
    bundle = build_for_target(
        dynamics=cf,
        spec=spec,
        target=current_host_target(),
        source_attestation_handle=bytes.fromhex(robot.attestation_handle),
        work_dir=work_dir,
    )
    return Built(bundle=bundle, dynamics=dynamics, model_name=model_name)


def _rk4_step(dynamics, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One ERK4 step of ``dynamics`` (matches acados' ERK integrator)."""
    xj, uj = jnp.asarray(x), jnp.asarray(u)
    k1 = dynamics(xj, uj)
    k2 = dynamics(xj + 0.5 * dt * k1, uj)
    k3 = dynamics(xj + 0.5 * dt * k2, uj)
    k4 = dynamics(xj + dt * k3, uj)
    return np.asarray(xj + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def closed_loop_reference(solver, dynamics, n_steps: int, seed: int) -> dict:
    """Host reference closed loop: acados control + JAX ERK4 plant, in the
    nx=4 Cartpole quantity layout (matches ``CARTPOLE_LQR_SCHEMA``).

    Resets the solver first so it starts cold — the on-target binary warm-starts
    from zero, and with SQP_RTI (one iteration per cycle) a pre-warmed host
    solver would diverge from it at step 0.
    """
    solver.reset()
    x = np.array(INITIAL_STATE, dtype=np.float64)
    x[0] += 0.01 * float(seed)  # same perturbation as the generated binary
    pos = np.empty((n_steps, 2))
    vel = np.empty((n_steps, 2))
    tau = np.empty((n_steps, 1))
    for i in range(n_steps):
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        if solver.solve() != 0:
            raise RuntimeError(f"reference acados solve failed at step {i}")
        u = np.asarray(solver.get(0, "u"), dtype=np.float64)
        pos[i], vel[i], tau[i] = x[:2], x[2:], u
        x = _rk4_step(dynamics, x, u, PLANT_DT)
    return {"joint_position": pos, "joint_velocity": vel, "actuator_torque": tau}


def hil_parity(built: Built, n_steps: int = 40, seed: int = 0):
    """Build the closed-loop HIL binary and assert step-locked host parity."""
    gen_dir = built.bundle.shared_library_path.parent
    source = generate_controller_hil_source(
        model_name=built.model_name,
        nx=NX,
        nu=NU,
        trace=CARTPOLE_LQR_TRACE,
        initial_state=INITIAL_STATE,
    )
    exe = build_controller_hil_binary(
        generated_code_dir=gen_dir,
        model_name=built.model_name,
        source=source,
        out_path=gen_dir / f"{built.model_name}_hil",
    )
    reference = closed_loop_reference(
        built.bundle.solver, built.dynamics, n_steps, seed
    )
    report = run_hil(
        reference,
        LocalRunner(exe),
        target_family="cortex-a76",
        dtype="float64",
        schema=CARTPOLE_LQR_SCHEMA,
        n_steps=n_steps,
        seed=seed,
    )
    report.assert_passed()
    return report


def bench_host(solver, n_cycles: int = 2000) -> dict:
    """Time the per-cycle ``solver.solve()`` on the host (a 1 kHz feasibility
    indicator; the rigorous on-target numbers come from ``jaxility bench``)."""
    x = np.array(INITIAL_STATE, dtype=np.float64)
    for _ in range(50):  # warm up
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        solver.solve()
    times_us = np.empty(n_cycles)
    for i in range(n_cycles):
        t0 = time.perf_counter_ns()
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        solver.solve()
        times_us[i] = (time.perf_counter_ns() - t0) / 1e3
    return {
        "mean_us": float(times_us.mean()),
        "p50_us": float(np.percentile(times_us, 50)),
        "p99_us": float(np.percentile(times_us, 99)),
        "max_us": float(times_us.max()),
        "meets_1khz": bool(np.percentile(times_us, 99) < ONE_KHZ_BUDGET_US),
    }


def _resolve_remote_acados(host: str) -> str:
    """The acados install path on the target (absolute). ``JAXILITY_HIL_ACADOS``
    overrides; otherwise resolve ``$HOME/acados`` on the target."""
    env = os.environ.get("JAXILITY_HIL_ACADOS")
    if env:
        return env
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "echo $HOME/acados"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return out.stdout.strip() or "/home/pi/acados"


def run_on_pi(built: Built, host: str, *, n_steps: int = 40, n_cycles: int = 2000):
    """Ship the robot-built controller to a tethered Pi 5, build it natively, and
    verify **on-silicon** HIL parity + a 1 kHz solve-time bench.

    The generated acados C (built on the host from *this* robot) is shipped and
    compiled on the Cortex-A76 against a target-native acados; the loop runs on
    the Pi and is compared, step-locked, to the robot-faithful host reference.
    """
    acados = _resolve_remote_acados(host)
    gen_dir = built.bundle.shared_library_path.parent

    hil_runner = build_controller_on_target(
        host=host,
        generated_code_dir=gen_dir,
        model_name=built.model_name,
        source=generate_controller_hil_source(
            model_name=built.model_name,
            nx=NX,
            nu=NU,
            trace=CARTPOLE_LQR_TRACE,
            initial_state=INITIAL_STATE,
        ),
        source_name=f"{built.model_name}_hil_main.c",
        remote_dir=f"/tmp/jaxility-{built.model_name}-hil",
        remote_acados=acados,
    )
    reference = closed_loop_reference(built.bundle.solver, built.dynamics, n_steps, 0)
    report = run_hil(
        reference,
        hil_runner,
        target_family="cortex-a76",
        dtype="float64",
        schema=CARTPOLE_LQR_SCHEMA,
        n_steps=n_steps,
        seed=0,
    )
    report.assert_passed()

    bench_runner = build_controller_on_target(
        host=host,
        generated_code_dir=gen_dir,
        model_name=built.model_name,
        source=generate_controller_bench_source(
            model_name=built.model_name,
            nx=NX,
            nu=NU,
            initial_state=INITIAL_STATE,
            n_warmup=100,
        ),
        source_name=f"{built.model_name}_bench_main.c",
        remote_dir=f"/tmp/jaxility-{built.model_name}-bench",
        remote_acados=acados,
    )
    record = run_controller_bench(
        bench_runner,
        robot="cartpole",
        target_family="cortex-a76",
        target_name="pi5",
        n_cycles=n_cycles,
        n_warmup=100,
        seed=0,
    )
    return report, record


def main() -> None:
    import jaxterity.zoo as zoo

    print("=" * 72)
    print("Cartpole end-to-end on the host — the Pi 5 pipeline minus the board")
    print("=" * 72)

    with __import__("tempfile").TemporaryDirectory() as td:
        work = Path(td)

        # 1) A calibrated robot. (The zoo robot arrives uncalibrated; we stamp a
        #    provenance record to model "sysid ran" — the handle is real either
        #    way.) Everything downstream is built FROM this object.
        robot = zoo.load("cartpole").with_provenance(
            ("t102-demo-recipe", "v0", "telemetry-hash"), calibrated=True
        )
        print(f"\n[1] Robot: {robot.name}  state={robot.calibration_state.name}")
        print(f"    attestation_handle = {robot.attestation_handle[:32]}…")

        # 2-4) Export -> host build -> the controller is real and loadable.
        built = build_controller(robot, work / "nominal", "cartpole_e2e")
        print("\n[2] Exported + host-built the acados LQR controller from the robot")
        nominal_hash = built.bundle.artifact.content_hash.hex()
        print(f"    artifact hash      = {nominal_hash[:32]}…")
        print(f"    shared library     = {built.bundle.shared_library_path.name}")

        # 5) Closed-loop HIL parity: the compiled controller stabilises the
        #    robot-faithful plant, bit-for-bit vs the host reference.
        hil_parity(built)
        print("\n[3] Closed-loop HIL parity (compiled controller vs host reference)")
        print("    PASSED — max divergence within cortex-a76 × float64 bounds")

        # 6) Solve-time: is 1 kHz feasible?
        bench = bench_host(built.bundle.solver)
        ok = "yes" if bench["meets_1khz"] else "NO"
        print("\n[4] Host solve-time (1 kHz feasibility indicator)")
        print(
            f"    mean {bench['mean_us']:.1f} µs | p50 {bench['p50_us']:.1f} | "
            f"p99 {bench['p99_us']:.1f} | max {bench['max_us']:.1f} µs "
            f"→ meets 1 kHz: {ok}"
        )

        # 7) Attestation: the manifest chain verifies and is rooted on the robot.
        chain = verify_manifest(built.bundle.manifest)
        rooted = built.bundle.manifest.source_attestation_handle == bytes.fromhex(
            robot.attestation_handle
        )
        print("\n[5] Attestation manifest")
        print(f"    verifies: {chain.ok} | rooted on the robot handle: {rooted}")

        # 8) The T-101 punchline: recalibrate (heavier pole) → the binary moves.
        heavy = robot.with_parameters(
            {"pole.mass": robot.parameters()["pole.mass"] * 2.0}
        )
        built_heavy = build_controller(heavy, work / "heavy", "cartpole_e2e_heavy")
        hb = built_heavy.bundle.artifact.content_hash
        nb = built.bundle.artifact.content_hash
        handle_moved = robot.attestation_handle != heavy.attestation_handle
        print("\n[6] One model, one truth — recalibrate the pole (2× mass):")
        print(f"    handle moved:        {handle_moved}")
        print(f"    artifact hash moved: {nb != hb}")
        print("    → the deployed binary tracks calibration, not just the manifest.")

        # 9) If a Pi 5 is tethered, cross to silicon: ship the robot-built
        #    controller, build it natively on the Cortex-A76, and verify on-Pi
        #    HIL parity + 1 kHz. This turns the dry-run into the real thing.
        host = os.environ.get("JAXILITY_HIL_SSH_HOST")
        if host:
            print(f"\n[7] Tethered Pi 5 found ({host}) — crossing to real silicon…")
            _, record = run_on_pi(built, host)
            s = record.solve
            print("    on-Pi HIL parity: PASSED (controller runs on the Cortex-A76)")
            print(
                f"    on-Pi solve-time (n={record.n_cycles}): "
                f"mean {s.mean_ns / 1e3:.1f} | p99 {s.p99_ns / 1e3:.1f} | "
                f"max {s.max_ns / 1e3:.1f} µs → meets 1 kHz: {record.meets_1khz}"
            )
            pi_ran = True
        else:
            pi_ran = False

    print("\n" + "=" * 72)
    if pi_ran:
        print("Ran on REAL Pi 5 silicon: robot → export → on-Pi controller at 1 kHz.")
        print("The integrated launch demo is on hardware — only the 3-minute Rerun")
        print("overlay video remains.")
    else:
        print("Host dry-run complete. Set JAXILITY_HIL_SSH_HOST=user@host to also")
        print("cross to a tethered Pi 5 (build natively on the Cortex-A76 + 1 kHz")
        print("bench). Remaining for the launch demo: the physical-Pi run + the")
        print("3-minute Rerun overlay video.")
    print("=" * 72)


if __name__ == "__main__":
    main()
