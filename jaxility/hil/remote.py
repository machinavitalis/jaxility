# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Build a generated controller binary natively on a tethered target (T-034/T-035).

This is the **build-on-target proof path** chosen for getting
the real acados controller onto the Pi 5 quickly: ship the host-generated acados
C to the target, compile it there with the target's native toolchain against a
target-native acados install, and hand back an :class:`jaxility.hil.SshRunner`
pointed at the result.

It is explicitly *not* the attested artifact path. The signed launch artifact
(T-036) is cross-compiled with the pinned Arm GNU toolchain (Docker / Linux) so
the manifest records a reproducible toolchain — the native build records the
target's own `gcc`, which is fine for a HIL/benchmark proof but not for
attestation. Use this to *measure* and *validate* on silicon; use the cross
build to *ship*.

The host-compiled CasADi model objects (`*.o`) are the host's object format
(Mach-O on a macOS dev box), so they cannot link on the target — this compiles
the model `*.c` sources on the target instead.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from ..errors import HILError, ToolchainError
from .runner import SshRunner

_DEFAULT_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8")


def _run(argv: list[str], *, what: str, timeout_s: float) -> str:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        raise HILError(f"{what} timed out after {timeout_s}s") from exc
    if completed.returncode != 0:
        raise HILError(
            f"{what} exited {completed.returncode}. stderr:\n"
            f"{completed.stderr.strip() or '(empty)'}"
        )
    return completed.stdout


def build_controller_on_target(
    *,
    host: str,
    generated_code_dir: Path,
    model_name: str,
    source: str,
    source_name: str,
    remote_dir: str,
    remote_acados: str,
    cc: str = "gcc",
    ssh_opts: tuple[str, ...] = _DEFAULT_SSH_OPTS,
    timeout_s: float = 300.0,
) -> SshRunner:
    """Ship the generated acados C to ``host``, compile ``source`` there.

    ``generated_code_dir`` is the host ``c_generated_code`` directory from
    ``build_for_target``; ``source`` is the C `main` (HIL or benchmark) to build
    against it, written as ``source_name``. ``remote_acados`` is the absolute path
    to the target's acados install (its ``lib`` must be on the target's loader
    path — e.g. registered via ``ldconfig``). Returns an :class:`SshRunner` for
    the compiled executable.
    """
    gen = Path(generated_code_dir)
    if not gen.is_dir():
        raise HILError(f"generated_code_dir not found: {gen}")
    if not source_name.endswith(".c"):
        raise ToolchainError(f"source_name must end with .c: {source_name!r}")

    # Stage the source into the local gen dir so a single recursive copy ships
    # everything the target compile needs.
    (gen / source_name).write_text(source)

    remote_dir = remote_dir.rstrip("/")
    aca = remote_acados.rstrip("/")
    remote_exe = f"{remote_dir}/{source_name[:-2]}"

    _run(
        [
            "ssh",
            *ssh_opts,
            host,
            f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}",
        ],
        what=f"prepare {remote_dir} on {host}",
        timeout_s=timeout_s,
    )
    _run(
        ["scp", "-rq", *ssh_opts, f"{gen}/.", f"{host}:{remote_dir}/"],
        what=f"ship generated code to {host}",
        timeout_s=timeout_s,
    )

    includes = " ".join(
        f"-I{p}"
        for p in (
            remote_dir,
            f"{aca}/include",
            f"{aca}/include/acados",
            f"{aca}/include/blasfeo/include",
            f"{aca}/include/hpipm/include",
        )
    )
    sources = " ".join(
        [
            f"{remote_dir}/{source_name}",
            f"{remote_dir}/acados_solver_{model_name}.c",
            f"{remote_dir}/acados_sim_solver_{model_name}.c",
            f"{remote_dir}/{model_name}_model/*.c",
        ]
    )
    link = f"-L{aca}/lib -lacados -lhpipm -lblasfeo -lm -Wl,-rpath,{aca}/lib"
    compile_cmd = (
        f"{shlex.quote(cc)} -std=c99 -O2 {includes} {sources} {link} "
        f"-o {shlex.quote(remote_exe)}"
    )
    _run(
        ["ssh", *ssh_opts, host, compile_cmd],
        what=f"compile {source_name} on {host}",
        timeout_s=timeout_s,
    )
    return SshRunner(host=host, remote_executable=remote_exe, ssh_opts=ssh_opts)
