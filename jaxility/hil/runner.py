# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Target runners — execute a HIL-instrumented artifact, return its trace.

A :class:`TargetRunner` is the transport seam between the host harness
and wherever the deployed artifact actually runs. The harness does not
care *how* the artifact executes — only that, given a cycle count and a
seed, it returns the artifact's raw stdout trace (parsed downstream by
:mod:`jaxility.hil.trace`). Two runners ship today:

* :class:`LocalRunner` — runs a host-native binary as a subprocess. The
  everywhere-available path (CI, dev box) for validating the harness and
  the codegen logic without hardware.
* :class:`SshRunner` — runs a binary already present on a remote target
  over SSH. The real-silicon path: the Raspberry Pi 5 launch target, or
  any reachable Cortex-A box. :meth:`SshRunner.deploy_binary` ships a
  locally-built (cross-compiled, T-034) artifact to the target first.

Both honour the loud-failure rule (invariant 7): a non-zero exit, a
timeout, or a transport error raises :class:`~jaxility.errors.HILError`
rather than returning a partial trace. Subprocess calls go through
``subprocess.run`` directly, matching the existing toolchain wrappers
(``jaxility.builder_cross``); the planned central ``subprocess_runner``
(PATTERNS §2.1) is not wired yet.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..errors import HILError

DEFAULT_TIMEOUT_S = 30.0
"""Default wall-clock budget for a single target run."""


@runtime_checkable
class TargetRunner(Protocol):
    """Runs a HIL artifact for ``n_steps`` cycles and returns raw stdout."""

    @property
    def label(self) -> str:
        """Human-readable identity of the transport (for the HIL report)."""

    def run(self, *, n_steps: int, seed: int) -> str:
        """Execute the artifact and return its captured stdout trace."""


def _check(completed: subprocess.CompletedProcess[str], *, what: str) -> str:
    """Return stdout, or raise HILError with stderr context on failure."""
    if completed.returncode != 0:
        raise HILError(
            f"{what} exited {completed.returncode}. stderr:\n"
            f"{completed.stderr.strip() or '(empty)'}"
        )
    return completed.stdout


@dataclass(frozen=True)
class LocalRunner:
    """Run a host-native HIL binary as a local subprocess."""

    executable: Path
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def label(self) -> str:
        return f"local:{self.executable}"

    def run(self, *, n_steps: int, seed: int) -> str:
        exe = Path(self.executable)
        if not exe.exists():
            raise HILError(f"HIL executable not found: {exe}")
        argv = [str(exe), str(n_steps), str(seed)]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise HILError(
                f"local HIL run timed out after {self.timeout_s}s: {shlex.join(argv)}"
            ) from exc
        return _check(completed, what=f"local HIL run {shlex.join(argv)}")


@dataclass(frozen=True)
class SshRunner:
    """Run a HIL binary already present on a remote target over SSH.

    ``host`` is any destination ``ssh`` understands (``user@host``, or a
    ``~/.ssh/config`` alias). ``remote_executable`` is an absolute path
    on the target. ``ssh_opts`` are extra ``ssh`` flags (e.g. a
    ``BatchMode=yes`` / ``ConnectTimeout`` pair); they default to a
    non-interactive, fail-fast set so a missing key never hangs the run.
    """

    host: str
    remote_executable: str
    ssh_opts: tuple[str, ...] = (
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
    )
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def label(self) -> str:
        return f"ssh:{self.host}:{self.remote_executable}"

    def run(self, *, n_steps: int, seed: int) -> str:
        remote_cmd = f"{shlex.quote(self.remote_executable)} {int(n_steps)} {int(seed)}"
        argv = ["ssh", *self.ssh_opts, self.host, remote_cmd]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise HILError(
                f"ssh HIL run timed out after {self.timeout_s}s on {self.host}"
            ) from exc
        return _check(completed, what=f"ssh HIL run on {self.host}")

    @classmethod
    def deploy_binary(
        cls,
        host: str,
        local_executable: Path,
        *,
        remote_dir: str,
        ssh_opts: tuple[str, ...] = (
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
        ),
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> SshRunner:
        """Ship a locally-built target binary to ``host`` and return a runner.

        Used by the cross-compiled deployment path (T-034): the artifact
        is built on the host for the target ABI, copied over, made
        executable, and then run via :meth:`run`. ``scp`` and the
        ``chmod`` go through the same loud-failure discipline as the run.
        """
        local = Path(local_executable)
        if not local.exists():
            raise HILError(f"local HIL binary to deploy not found: {local}")
        remote_path = f"{remote_dir.rstrip('/')}/{local.name}"
        # scp uses -O (legacy protocol) implicitly on modern clients; the
        # default sftp path is fine here. Keep the same non-interactive opts.
        scp_argv = ["scp", *ssh_opts, str(local), f"{host}:{remote_path}"]
        try:
            scp = subprocess.run(
                scp_argv, capture_output=True, text=True, timeout=timeout_s
            )
            _check(scp, what=f"scp HIL binary to {host}")
            chmod_argv = [
                "ssh",
                *ssh_opts,
                host,
                f"chmod +x {shlex.quote(remote_path)}",
            ]
            chmod = subprocess.run(
                chmod_argv, capture_output=True, text=True, timeout=timeout_s
            )
            _check(chmod, what=f"chmod HIL binary on {host}")
        except subprocess.TimeoutExpired as exc:
            raise HILError(
                f"deploying HIL binary to {host} timed out after {timeout_s}s"
            ) from exc
        return cls(
            host=host,
            remote_executable=remote_path,
            ssh_opts=ssh_opts,
            timeout_s=timeout_s,
        )
