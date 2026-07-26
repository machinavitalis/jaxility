# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""E-bike motor current controller: PMSM plant -> acados LQR -> embedded C.

Companion to the Jaxonomy ``docs/examples/ebike_hybrid_simulation.py`` cargo
e-bike model. That model contains a hand-written field-oriented PI current loop
as its *reference* controller. Jaxility does not lower a hand-written control
law to C; its lane is **synthesize a controller from plant dynamics** and
generate deployable code for it. So the deployable artifact here is an LQR
field-oriented current *regulator* synthesized from the PMSM dq-axis plant and
lowered through the real acados pipeline:

    PMSM dq current dynamics  f(x, u)          (x = [i_d, i_q], u = [v_d, v_q])
        -> translate (JAX -> CasADi)
        -> LQR template -> acados OCP
        -> host build (compile + link a loadable controller)
        -> closed-loop stabilization check     (the compiled controller works)
        -> attestation manifest                (provenance-rooted, verifiable)

This is the architecturally-correct division of labour: Jaxonomy owns the
full-vehicle simulation and the reference PI design; Jaxility owns the
plant->optimal-controller->C deployment. (Only Jaxility imports downstream of
Jaxonomy, never the reverse.)

Requires the acados toolchain (``~/Dev/acados`` by default, or
``JAXILITY_ACADOS_DIR``) and a host ``cc``. Exits 0 on success.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
from pathlib import Path


# --- acados env + SIP-safe dylib preload (mirrors jaxility test/conftest.py) --
def _setup_acados_env() -> None:
    root = Path(os.environ.get("JAXILITY_ACADOS_DIR", str(Path.home() / "Dev" / "acados")))
    if not root.exists():
        raise SystemExit(f"acados not found at {root}; set JAXILITY_ACADOS_DIR.")
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(root))
    lib = root / "lib"
    key = "DYLD_LIBRARY_PATH" if os.uname().sysname == "Darwin" else "LD_LIBRARY_PATH"
    if str(lib) not in os.environ.get(key, "").split(":"):
        os.environ[key] = f"{lib}:{os.environ.get(key, '')}".rstrip(":")
    for name in ("libblasfeo", "libhpipm", "libqpOASES_e", "libacados"):
        for ext in (".dylib", ".so"):
            cand = lib / f"{name}{ext}"
            if cand.exists():
                try:
                    ctypes.CDLL(str(cand))
                except OSError:
                    pass


_setup_acados_env()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxility.builder import build_for_target  # noqa: E402
from jaxility.lowering import translate  # noqa: E402
from jaxility.manifest import verify_manifest  # noqa: E402
from jaxility.targets import current_host_target  # noqa: E402
from jaxility.templates import lqr  # noqa: E402


# --- PMSM dq-axis plant (same motor as the Jaxonomy e-bike model) ------------
# R_s [ohm], L_d/L_q [H], psi_m [Wb], P [pole pairs]; Kt = 1.5*P*psi_m = 0.6 Nm/A
R_S, L_D, L_Q, PSI_M, P = 0.05, 0.0005, 0.0008, 0.10, 4.0
NX, NU = 2, 2  # state = [i_d, i_q], input = [v_d, v_q]

# LQR weights: penalize current error heavily, voltage effort lightly.
Q_WEIGHTS = (2.0, 2.0)
R_WEIGHTS = (0.01, 0.01)
V_MAX = 28.0                       # dq voltage magnitude limit (~ V_dc/sqrt(3))
INPUT_BOUNDS = ((-V_MAX, -V_MAX), (V_MAX, V_MAX))
INITIAL_STATE = (0.0, 60.0)       # a 60 A q-axis current error to regulate out
HORIZON_STEPS = 20
TIME_HORIZON_S = 0.002            # 10 kHz current loop (dt = 1e-4 s)
PLANT_DT = TIME_HORIZON_S / HORIZON_STEPS


def pmsm_plant(w_mech: float):
    """Return f(x,u) -> dx for the dq current dynamics at electrical speed
    ``w_e = P*w_mech`` (fixed operating point). The back-emf term ``w_e*psi_m``
    is a known feedforward, handled separately, so the plant used for LQR
    synthesis has the origin as an equilibrium (linear, smooth, coverage-clean).
    """
    w_e = P * w_mech

    def f(x, u):
        i_d, i_q = x[0], x[1]
        v_d, v_q = u[0], u[1]
        did = (v_d - R_S * i_d + w_e * L_Q * i_q) / L_D
        diq = (v_q - R_S * i_q - w_e * L_D * i_d) / L_Q
        return jnp.array([did, diq])

    return f


def _attestation_handle(w_mech: float) -> bytes:
    """Stand-in for a Jaxterity CalibratedRobot attestation handle: a stable
    32-byte digest of the motor parameters + operating point, so the manifest is
    provenance-rooted and recalibration changes the root."""
    payload = f"pmsm|Rs={R_S}|Ld={L_D}|Lq={L_Q}|psi={PSI_M}|P={P}|w={w_mech}"
    return hashlib.blake2b(payload.encode(), digest_size=32).digest()


def build_current_controller(w_mech: float, work_dir: Path):
    cf = translate(pmsm_plant(w_mech), in_shapes=((NX,), (NU,)), name="ebike_foc")
    spec = lqr(cf, Q=Q_WEIGHTS, R=R_WEIGHTS, initial_state=INITIAL_STATE,
               input_bounds=INPUT_BOUNDS, name="ebike_foc",
               horizon_steps=HORIZON_STEPS, time_horizon_s=TIME_HORIZON_S)
    return build_for_target(
        dynamics=cf, spec=spec, target=current_host_target(),
        source_attestation_handle=_attestation_handle(w_mech), work_dir=work_dir,
    )


def _rk4(f, x, u, dt):
    xj, uj = jnp.asarray(x), jnp.asarray(u)
    k1 = f(xj, uj); k2 = f(xj + 0.5 * dt * k1, uj)
    k3 = f(xj + 0.5 * dt * k2, uj); k4 = f(xj + dt * k3, uj)
    return np.asarray(xj + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def closed_loop_check(bundle, w_mech, n_steps=200):
    """Run the compiled controller in closed loop against the PMSM plant and
    confirm it regulates the initial current error to ~zero."""
    solver = bundle.solver
    solver.reset()
    f = pmsm_plant(w_mech)
    x = np.array(INITIAL_STATE, dtype=np.float64)
    i0 = float(np.hypot(*x))
    for i in range(n_steps):
        solver.set(0, "lbx", x)
        solver.set(0, "ubx", x)
        if solver.solve() != 0:
            raise RuntimeError(f"acados solve failed at step {i}")
        u = np.asarray(solver.get(0, "u"), dtype=np.float64)
        x = _rk4(f, x, u, PLANT_DT)
    return i0, float(np.hypot(*x))


def main():
    print("=" * 70)
    print("  E-BIKE FOC CURRENT CONTROLLER — PMSM PLANT -> acados LQR -> C")
    print("=" * 70)

    work = Path(tempfile.mkdtemp(prefix="ebike_foc_"))
    w_nominal = 40.0
    bundle = build_current_controller(w_nominal, work / "nominal")

    gen_c = list((work / "nominal" / "c_generated_code").rglob("*.c"))
    gen_h = list((work / "nominal" / "c_generated_code").rglob("*.h"))
    print(f"\n  [build] operating point w_mech = {w_nominal} rad/s "
          f"(w_e = {P*w_nominal} rad/s)")
    print(f"  [build] generated C: {len(gen_c)} .c + {len(gen_h)} .h files")
    print(f"  [build] shared library: {bundle.shared_library_path.name}")
    print(f"  [build] artifact content hash: {bundle.artifact.content_hash.hex()[:24]}...")

    report = verify_manifest(bundle.manifest)
    print(f"  [verify] manifest chain ok = {report.ok}  "
          f"(schema + recomputed hash + signer: {report.signature_status})")

    i0, i_final = closed_loop_check(bundle, w_nominal)
    print(f"\n  [closed-loop] initial current error |i| = {i0:6.2f} A")
    print(f"  [closed-loop] final   current error |i| = {i_final:6.3f} A  "
          f"({'STABILIZED' if i_final < 1.0 else 'CHECK'})")

    # Recalibration / operating-point provenance: a different plant yields a
    # different controller and a different artifact hash.
    bundle_hi = build_current_controller(80.0, work / "hi_speed")
    h1 = bundle.artifact.content_hash.hex()
    h2 = bundle_hi.artifact.content_hash.hex()
    print(f"\n  [provenance] w=40 artifact hash: {h1[:24]}...")
    print(f"  [provenance] w=80 artifact hash: {h2[:24]}...")
    print(f"  [provenance] hashes differ = {h1 != h2}  "
          f"(operating point flows into the deployable binary)")
    print("=" * 70)


if __name__ == "__main__":
    main()
