# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Structural cross-compile wrapper (T-031).

The host build path (:mod:`jaxility.builder`) lets ``acados`` generate
*and* compile C against the host's compiler. Cross-compilation reuses
the same code-generation stage but routes the resulting C tree
through a target-specific toolchain (Arm GNU ``aarch64-none-linux-gnu-gcc``
for Pi 5; Arm ``arm-none-eabi-gcc`` for Cortex-M; etc.).

This module is the *structural* half of T-031 — the parts that work
without the cross-compiler installed:

* :class:`CrossCompilePlan` — a frozen record of the toolchain to
  invoke, the C sources to compile, the flags to pass, and the
  output path. Pure data; ``__init__`` does no I/O.
* :func:`plan_cross_compile` — assembles a :class:`CrossCompilePlan`
  from a :class:`~jaxility.targets.Target`, the acados-generated C
  source directory, and an output path. Tested in CI without the
  toolchain present.
* :func:`cflags_for_family` — the per-family ``-mcpu / -march / ...``
  flag composition. New target profiles plug in new family rows
  without touching the executor.
* :func:`verify_toolchain_installed` — runs ``target.toolchain.detect_command``
  and matches the captured stdout against ``target.toolchain.version_regex``.
  Raises :class:`~jaxility.errors.ToolchainError` cleanly when the
  binary is missing or its version is wrong.
* :func:`execute_cross_compile` — runs the plan's ``compiler_argv``
  via :mod:`subprocess`. Subprocess-bound; the Tier-A test suite
  exercises it with a Python-fake toolchain stub so we do not have
  to install Arm GCC on CI runners.

The ``acados``-side code generation is reused as-is: the cross path
calls :func:`build_ocp` and constructs an :class:`AcadosOcpSolver`
in *generate-only* mode (``build=False`` on the modern interface),
captures the resulting C tree under ``work_dir/c_generated_code/``,
then drives the cross-compiler against it.

The artifact manifest records the cross-compiled toolchain version,
not the host compiler — this is the manifest's load-bearing reason
to exist (invariant 5: byte-identical inputs → byte-identical
artifact + manifest; cross-target deployments must be distinguishable
from host builds at the manifest layer).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import blake3

from .builder import BuildBundle
from .builder_deps import CrossBuiltDeps
from .errors import TargetError, ToolchainError
from .lowering import CasadiFunction, OcpTemplateSpec, build_ocp
from .manifest import (
    SCHEMA_VERSION_V0,
    Artifact,
    BuildLogEntry,
    Manifest,
)
from .manifest.toolchain_detect import detect_toolchain_versions
from .targets import Target

# ---------------------------------------------------------------------------
# Per-family compile flags
# ---------------------------------------------------------------------------

# Keyed by ``Target.family``. The flag set is intentionally narrow —
# everything that affects ABI / ISA selection plus -O3 / -fPIC for a
# shared library. Per-deployment overrides (LTO, stack-protector, etc.)
# layer on top via ``extra_cflags`` so this table stays close to a
# single source of truth for "what does this family even mean."
_FAMILY_CFLAGS: dict[str, tuple[str, ...]] = {
    # ------------------------------------------------------------------
    # A-profile (linux ABI; -fPIC -shared → .so)
    # ------------------------------------------------------------------
    # Pi 5. `-mcpu=cortex-a76` already pins the right `-march`
    # (armv8.2-a) and ISA feature set; specifying `-march` explicitly
    # alongside `-mcpu` triggers a "switch ... conflicts with ..."
    # error in Arm GNU 15.2.Rel1 (cc1 enforces the constraint that
    # newer releases tightened).
    "cortex-a76": (
        "-mcpu=cortex-a76",
        "-O3",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",  # acados-generated C trips this prolifically
    ),
    # ------------------------------------------------------------------
    # M-profile (bare-metal; -c → relocatable .o; no -shared / -fPIC).
    # Linker scripts, startup files, and reset vectors are the runtime's
    # job (T-032 / T-052 territory). The cross-compile here produces an
    # *object* file that the runtime project links into its final ELF.
    # ------------------------------------------------------------------
    "cortex-m4": (
        "-mcpu=cortex-m4",
        "-mthumb",
        "-mfpu=fpv4-sp-d16",
        "-mfloat-abi=hard",
        "-O3",
        "-c",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
    ),
    # Ethos-U55 / U65 NPU pairings ship a Cortex-M55 host; the M55
    # host's Helium (MVE) extension is what -march captures here. The
    # Vela compiler handles the NPU side separately (see Ethos quirks).
    "ethos-u55": (
        "-mcpu=cortex-m55",
        "-mthumb",
        "-mfpu=auto",
        "-mfloat-abi=hard",
        "-O3",
        "-c",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
    ),
    "ethos-u65": (
        "-mcpu=cortex-m55",
        "-mthumb",
        "-mfpu=auto",
        "-mfloat-abi=hard",
        "-O3",
        "-c",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
    ),
}


def cflags_for_family(family: str) -> tuple[str, ...]:
    """Return the ABI / ISA / optimisation flags for ``family``.

    Raises
    ------
    TargetError
        If the family has no entry yet. Adding a new target family is
        an explicit PR that touches this table — there is no silent
        default (invariant 7).
    """
    try:
        return _FAMILY_CFLAGS[family]
    except KeyError:
        known = ", ".join(sorted(_FAMILY_CFLAGS)) or "(none)"
        raise TargetError(
            f"no cross-compile cflags registered for target family "
            f"{family!r}; add a row to jaxility.builder_cross._FAMILY_CFLAGS. "
            f"Known families: {known}."
        ) from None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossCompilePlan:
    """A materialised cross-compile command + metadata.

    The plan is *pure data*: constructing it does no I/O, runs no
    subprocess, and does not require the toolchain to be installed.
    :func:`execute_cross_compile` is what actually invokes the compiler.

    The separation lets the structural surface (composition, naming,
    flag selection, sources list) be tested in CI on hosts without the
    cross-toolchain present (the typical case today).
    """

    target: Target
    """The deployment target the plan was built for."""

    c_source_dir: Path
    """Root of the acados-generated C tree (under ``work_dir``)."""

    sources: tuple[Path, ...]
    """``.c`` files to compile, in stable order. Includes acados-
    generated solver / cost / dynamics / constraints sources for the
    model. Discovered by :func:`plan_cross_compile`."""

    include_dirs: tuple[Path, ...]
    """Header search roots: the generated source dir + any
    user-supplied extras (acados / blasfeo / hpipm cross-built headers
    will plug in here in T-032)."""

    output_path: Path
    """Where :func:`execute_cross_compile` writes the resulting .so."""

    compiler_argv: tuple[str, ...]
    """Fully composed ``[toolchain, *flags, -o, output, *includes, *sources]``
    argv. Stable across runs for a given (target, c_source_dir, output_path)
    triple — that stability is what makes the manifest content hash
    deterministic across hosts (invariant 5)."""

    extra_link_args: tuple[str, ...]
    """Caller-supplied flags appended after the sources list (typically
    static-archive paths for the cross-built blasfeo / hpipm; deferred
    to T-032 — empty in Tier A)."""


# ---------------------------------------------------------------------------
# Source-tree discovery
# ---------------------------------------------------------------------------


def _discover_c_sources(c_source_dir: Path, model_name: str) -> tuple[Path, ...]:
    """Find the model-specific ``.c`` sources under the acados tree.

    The acados code-generation step writes a model-keyed subdirectory
    under ``c_source_dir`` (e.g. ``c_generated_code/lqr_model/``). The
    cross-compile pulls every ``.c`` file inside that subdirectory plus
    the top-level ``acados_solver_<model>.c`` glue. The result is sorted
    for byte-stable plan composition across hosts.
    """
    if not c_source_dir.exists():
        raise TargetError(
            f"acados code-gen directory does not exist: {c_source_dir}. "
            "Run the code-generation step before planning the cross-compile."
        )
    sources = sorted(c_source_dir.rglob("*.c"))
    # Filter to the model-related ones — acados drops some host-only
    # helpers (.c files for the Python ctypes shim) that the cross
    # target should not link.
    model_token = model_name.lower()
    model_sources = tuple(
        s for s in sources if model_token in s.name.lower() or s.parent != c_source_dir
    )
    if not model_sources:
        raise TargetError(
            f"no model-related .c sources under {c_source_dir} for model "
            f"{model_name!r}. The acados code generator produced an empty "
            "tree — this is almost always an upstream acados version mismatch."
        )
    return model_sources


# ---------------------------------------------------------------------------
# Plan composer
# ---------------------------------------------------------------------------


def plan_cross_compile(
    *,
    target: Target,
    c_source_dir: Path,
    output_path: Path,
    model_name: str,
    extra_include_dirs: tuple[Path, ...] = (),
    extra_link_args: tuple[str, ...] = (),
) -> CrossCompilePlan:
    """Compose a :class:`CrossCompilePlan` for ``target``.

    The composition is deterministic — same inputs → byte-identical
    ``compiler_argv``. That property is load-bearing because the
    manifest's artifact-content hash depends on the compiled output;
    if the argv drifts across hosts the cross-compiled artifact won't
    reproduce.

    Args
    ----
    target : Target
        The deployment target. Its ``toolchain.name`` is the compiler
        binary; its ``family`` keys the flag table.
    c_source_dir : Path
        The acados-generated C source root (``work_dir/c_generated_code``).
    output_path : Path
        Where the resulting shared library lands. Parent directory must
        exist (the function does not create it — that's a deployer
        decision).
    model_name : str
        The acados model name (used to discover the model-specific
        ``.c`` sources).
    extra_include_dirs : tuple[Path, ...]
        Additional header search roots — typically the cross-built
        acados / blasfeo / hpipm install prefixes. Tier A
        leaves this empty; T-032 wires the runtime install path.
    extra_link_args : tuple[str, ...]
        Additional flags appended after the sources list (cross-built
        static-archive paths in T-032).

    Returns
    -------
    CrossCompilePlan
        Pure-data plan. Pass to :func:`execute_cross_compile` to run.

    Raises
    ------
    TargetError
        Unknown target family, missing C source dir, or no model-
        related sources under ``c_source_dir``.
    """
    flags = cflags_for_family(target.family)
    sources = _discover_c_sources(c_source_dir, model_name)
    include_dirs = (c_source_dir, *extra_include_dirs)

    argv: list[str] = [target.toolchain.name]
    argv.extend(flags)
    argv.extend(f"-I{p}" for p in include_dirs)
    argv.extend(("-o", str(output_path)))
    argv.extend(str(s) for s in sources)
    argv.extend(extra_link_args)

    return CrossCompilePlan(
        target=target,
        c_source_dir=c_source_dir,
        sources=sources,
        include_dirs=include_dirs,
        output_path=output_path,
        compiler_argv=tuple(argv),
        extra_link_args=extra_link_args,
    )


# ---------------------------------------------------------------------------
# Toolchain detection
# ---------------------------------------------------------------------------


def verify_toolchain_installed(target: Target) -> str:
    """Verify the cross-toolchain is on PATH and matches the pinned version.

    Runs ``target.toolchain.detect_command`` and matches the combined
    stdout/stderr against ``target.toolchain.version_regex``. Returns
    the captured version string on success.

    Raises
    ------
    ToolchainError
        Binary not on PATH; binary ran but did not match the pin's
        version regex; binary ran but reported a version different
        from the pin's ``version`` field.
    """
    binary = target.toolchain.detect_command[0]
    if shutil.which(binary) is None:
        raise ToolchainError(
            f"cross-toolchain {binary!r} not found on PATH. "
            f"Install Arm GNU Toolchain {target.toolchain.version} from "
            f"{target.toolchain.download_url} and re-run."
        )

    try:
        completed = subprocess.run(
            list(target.toolchain.detect_command),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError as exc:
        raise ToolchainError(f"failed to invoke {binary!r}: {exc}") from exc

    output = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(target.toolchain.version_regex, output)
    if match is None:
        raise ToolchainError(
            f"{binary!r} ran but its output did not match the pin's "
            f"version regex {target.toolchain.version_regex!r}. "
            f"Captured first 200 chars: {output[:200]!r}."
        )

    captured = match.group(1) if match.groups() else match.group(0)
    if captured != target.toolchain.version:
        raise ToolchainError(
            f"{binary!r} reports version {captured!r} but the Target "
            f"pin requires {target.toolchain.version!r}. Install the "
            "exact pinned version (PATTERNS §2.2 — toolchain pins are "
            "exact, not lower-bounds)."
        )
    return captured


def verify_toolchain_integrity(target: Target) -> str:
    """Verify the installed toolchain binary's SHA-256 matches the pin (M-7).

    Closes audit finding M-7: the previous design recorded
    ``expected_sha256="0" * 64`` placeholders that *looked* like real
    SHA digests but were unverified fiction. The new contract:

    * Pins with :attr:`~ToolchainPin.expected_sha256` ==
      :data:`~jaxility.targets.UNVERIFIED_SHA256` declare integrity
      is not yet pinned. This function raises ``ToolchainError`` with
      a clear message rather than silently passing.
    * Pins with a hex SHA-256 are checked against the installed binary
      via :class:`hashlib.sha256`.

    Returns the captured SHA-256 hex string on success.

    Raises
    ------
    ToolchainError
        Binary not on PATH; pin is unverified; binary's SHA does not
        match the pin.
    """
    import hashlib

    binary = target.toolchain.detect_command[0]
    binary_path = shutil.which(binary)
    if binary_path is None:
        raise ToolchainError(
            f"cross-toolchain {binary!r} not found on PATH; cannot verify integrity."
        )
    if not target.toolchain.has_pinned_integrity():
        raise ToolchainError(
            f"toolchain pin for target {target.name!r} carries "
            f"expected_sha256={target.toolchain.expected_sha256!r} "
            "(the 'unverified' sentinel). Integrity is not yet pinned; "
            "compute the SHA-256 of the upstream archive at "
            f"{target.toolchain.download_url!r} and bump the pin "
            "before relying on integrity verification."
        )
    digest = hashlib.sha256()
    with open(binary_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    captured = digest.hexdigest()
    if captured != target.toolchain.expected_sha256.lower():
        raise ToolchainError(
            f"binary at {binary_path!r} hashes to {captured!r} but the "
            f"pin requires {target.toolchain.expected_sha256!r}. Either "
            "re-install the pinned binary or bump the pin (PATTERNS §2.2)."
        )
    return captured


def resolve_toolchain_integrity(target: Target) -> tuple[str, str | None]:
    """Resolve the cross-toolchain integrity status for the manifest (T-112).

    Wraps :func:`verify_toolchain_integrity` with the build-path policy:

    * **Pinned integrity** (a real SHA-256): verify the installed binary. A
      mismatch — or a missing binary — raises ``ToolchainError`` and aborts the
      build. We never ship an artifact whose toolchain integrity we set out to
      check and could not confirm.
    * **Unverified sentinel** (:data:`~jaxility.targets.UNVERIFIED_SHA256`): the
      integrity hash is not pinned yet. Rather than silently implying the
      toolchain was verified, the build records ``"unverified"`` in the manifest
      and continues — loud, not silent (invariant 7). Pin a real SHA-256
      (PATTERNS §2.2) to promote this to enforcement.

    Returns ``(manifest_status, warning)``: ``manifest_status`` is stored in the
    manifest under ``toolchain-integrity:<binary>``; ``warning`` is a build-log
    message, ``None`` when integrity was verified.
    """
    if target.toolchain.has_pinned_integrity():
        sha = verify_toolchain_integrity(target)
        return f"sha256:{sha}", None
    return (
        target.toolchain.expected_sha256,  # the "unverified" sentinel
        (
            f"toolchain integrity NOT verified for {target.toolchain.name!r}: the "
            "pin carries the unverified sentinel. The manifest records this under "
            "'toolchain-integrity'; pin a real SHA-256 (PATTERNS §2.2) to enforce it."
        ),
    )


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def execute_cross_compile(plan: CrossCompilePlan, *, timeout_s: float = 120.0) -> Path:
    """Run the cross-compile and return the path to the produced .so.

    Args
    ----
    plan : CrossCompilePlan
        Composed by :func:`plan_cross_compile`.
    timeout_s : float
        Subprocess timeout. Defaults to 120 s; bump for very large
        models or under heavy host load.

    Returns
    -------
    Path
        ``plan.output_path``, after the compile succeeded.

    Raises
    ------
    ToolchainError
        Compiler exit code ≠ 0, or compiler invocation timed out, or
        the compiler ran successfully but did not write ``output_path``.
    """
    try:
        completed = subprocess.run(
            list(plan.compiler_argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise ToolchainError(
            f"failed to launch cross-compiler {plan.compiler_argv[0]!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolchainError(
            f"cross-compile timed out after {timeout_s}s (target={plan.target.name!r})"
        ) from exc

    if completed.returncode != 0:
        raise ToolchainError(
            f"cross-compile failed for target {plan.target.name!r} "
            f"with exit code {completed.returncode}.\n"
            f"argv: {plan.compiler_argv!r}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not plan.output_path.exists():
        raise ToolchainError(
            f"cross-compile reported success but no file at "
            f"{plan.output_path}. Compiler stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return plan.output_path


# ---------------------------------------------------------------------------
# End-to-end glue
# ---------------------------------------------------------------------------


def cross_build_for_target(
    *,
    dynamics: CasadiFunction,
    spec: OcpTemplateSpec,
    target: Target,
    source_attestation_handle: bytes,
    work_dir: Path,
    output_path: Path | None = None,
    extra_include_dirs: tuple[Path, ...] = (),
    extra_link_args: tuple[str, ...] = (),
    deps: CrossBuiltDeps | None = None,
    build_timestamp_utc: int | None = None,
) -> BuildBundle:
    """Cross-compile a controller for ``target`` and package the artifact.

    Generates acados C the same way the host path does, then drives
    the cross-compiler through :func:`execute_cross_compile`. The
    returned :class:`BuildBundle.solver` is the *host-side* solver
    instance used to drive code generation — it cannot execute the
    cross-compiled artifact (different ABI); the HIL test harness
    (T-033) loads ``shared_library_path`` on the actual deployment
    target.

    Args
    ----
    dynamics : CasadiFunction
        From :func:`jaxility.lowering.translate`.
    spec : OcpTemplateSpec
        From a template factory or hand-rolled.
    target : Target
        Cross-compilation deployment target (e.g. ``PI5``).
    source_attestation_handle : bytes
        BLAKE3 digest from the upstream Source.
    work_dir : Path
        Directory acados writes its generated code into.
    output_path : Path | None
        Override for the cross-compiled artifact location. Defaults to
        ``work_dir / f"lib{target.name}_{model.name}.so"``.
    extra_include_dirs : tuple[Path, ...]
        Additional header roots (cross-built acados / blasfeo / hpipm
        prefixes — T-032 territory; empty in Tier A).
    extra_link_args : tuple[str, ...]
        Additional link arguments.
    deps : CrossBuiltDeps | None
        Cross-built acados / blasfeo / hpipm archives (from
        :func:`jaxility.builder_deps.build_cross_deps`). When provided,
        the dep include dirs + link args are merged into the
        cross-compile (after any caller-supplied ``extra_*``), and each
        archive's BLAKE3 hash is recorded in the manifest's
        ``toolchain_versions`` under a ``dep-archive:<name>`` key for
        provenance. Without it the link step fails (no acados symbols) —
        that is the documented Tier-A path.
    build_timestamp_utc : int | None
        Override for the manifest timestamp.

    Returns
    -------
    BuildBundle
        Cross-compiled artifact + manifest + host solver + library path.

    Raises
    ------
    ToolchainError
        Cross-toolchain missing, version mismatch, or compile failed.
    TargetError
        Unknown family, acados code-gen empty, or shared library
        not at expected location.
    """
    from acados_template import AcadosOcpSolver  # noqa: PLC0415

    if deps is not None:
        extra_include_dirs = (*extra_include_dirs, *deps.include_dirs)
        extra_link_args = (*extra_link_args, *deps.link_args)

    work_dir.mkdir(parents=True, exist_ok=True)

    log: list[BuildLogEntry] = []
    log.append(
        BuildLogEntry(
            offset_us=0,
            stage="plan",
            level="info",
            message=(
                f"cross_build_for_target target={target.name!r} "
                f"spec.name={spec.name!r} horizon={spec.horizon_steps} "
                f"nx={dynamics.input_shapes[0][0]}"
            ),
            detail={
                "target_profile_hash_hex": target.hash.hex(),
                "source_attestation_handle_hex": source_attestation_handle.hex(),
                "target_family": target.family,
            },
        )
    )

    detected_version = verify_toolchain_installed(target)
    log.append(
        BuildLogEntry(
            offset_us=1,
            stage="plan",
            level="info",
            message=(
                f"cross-toolchain {target.toolchain.name!r} detected at "
                f"version {detected_version!r} (matches pin)"
            ),
            detail={
                "toolchain_binary": target.toolchain.name,
                "toolchain_version": detected_version,
            },
        )
    )

    # T-112: verify (or honestly record non-verification of) the cross-toolchain
    # binary integrity, and carry the result into the manifest so a verifier can
    # tell whether provenance covers toolchain integrity. A real-pin mismatch
    # raises inside ``resolve_toolchain_integrity`` and aborts the build here.
    integrity_status, integrity_warning = resolve_toolchain_integrity(target)
    log.append(
        BuildLogEntry(
            offset_us=1,
            stage="verify",
            level="warn" if integrity_warning else "info",
            message=(
                integrity_warning
                or f"cross-toolchain {target.toolchain.name!r} integrity verified"
            ),
            detail={
                "toolchain_binary": target.toolchain.name,
                "toolchain_integrity": integrity_status,
            },
        )
    )

    ocp = build_ocp(dynamics, spec)
    log.append(
        BuildLogEntry(
            offset_us=2,
            stage="lower",
            level="info",
            message="acados OCP constructed",
            detail={"model_name": ocp.model.name},
        )
    )

    # acados writes generated C under cwd; force it into ``work_dir``
    # and run code generation only (no host link step).
    json_filename = f"{ocp.model.name}.json"
    old_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        # ``build=False`` skips the host-side ``make`` invocation; the
        # cross step replaces it.
        try:
            solver = AcadosOcpSolver(
                ocp, json_file=json_filename, verbose=False, build=False, generate=True
            )
        except TypeError:
            # Older acados versions do not expose ``build`` /
            # ``generate`` — fall back to the full host build and ignore
            # the resulting host .so.
            solver = AcadosOcpSolver(ocp, json_file=json_filename, verbose=False)
    finally:
        os.chdir(old_cwd)

    c_source_dir = work_dir / "c_generated_code"
    model_name = ocp.model.name

    if output_path is None:
        output_path = work_dir / f"lib{target.name}_{model_name}.so"

    plan = plan_cross_compile(
        target=target,
        c_source_dir=c_source_dir,
        output_path=output_path,
        model_name=model_name,
        extra_include_dirs=extra_include_dirs,
        extra_link_args=extra_link_args,
    )
    log.append(
        BuildLogEntry(
            offset_us=3,
            stage="compile",
            level="info",
            message="cross-compile plan composed",
            detail={
                "argv_head": " ".join(plan.compiler_argv[:8]),
                "n_sources": str(len(plan.sources)),
            },
        )
    )

    library_path = execute_cross_compile(plan)
    payload = library_path.read_bytes()
    log.append(
        BuildLogEntry(
            offset_us=4,
            stage="package",
            level="info",
            message="cross-compiled shared library packaged into artifact",
            detail={
                "library_path": str(library_path),
                "payload_bytes": str(len(payload)),
                "host_python": platform.python_version(),
                "host_machine": platform.machine(),
            },
        )
    )

    if build_timestamp_utc is None:
        build_timestamp_utc = int(time.time() * 1_000_000)

    toolchain_versions = detect_toolchain_versions(target)
    # The cross path overwrites the deployment-target toolchain key with
    # the *detected* version captured at verify_toolchain_installed
    # time, so the manifest carries the actual binary's report rather
    # than only the pinned value.
    toolchain_versions[target.toolchain.name] = detected_version
    # T-112: record whether the toolchain binary's integrity was verified
    # ("sha256:<hex>") or not ("unverified"), so attestation never silently
    # implies a check that did not happen.
    toolchain_versions[f"toolchain-integrity:{target.toolchain.name}"] = (
        integrity_status
    )
    if deps is not None:
        # Record dependency-binary provenance: each cross-built archive's
        # BLAKE3 hash travels in the manifest so a verifier knows exactly
        # which acados / blasfeo / hpipm binaries the artifact linked.
        for archive_name, archive_hash in deps.archive_hashes:
            toolchain_versions[f"dep-archive:{archive_name}"] = archive_hash.hex()
    manifest = Manifest(
        schema_version=SCHEMA_VERSION_V0,
        source_attestation_handle=source_attestation_handle,
        toolchain_versions=toolchain_versions,
        target_profile_hash=target.hash,
        artifact_content_hash=blake3.blake3(payload).digest(),
        build_timestamp_utc=build_timestamp_utc,
    )

    artifact = Artifact.build(
        payload=payload,
        source_manifest_hash=manifest.content_hash(),
        target_profile_hash=target.hash,
        build_log=tuple(log),
    )

    return BuildBundle(
        artifact=artifact,
        manifest=manifest,
        target=target,
        solver=solver,
        shared_library_path=library_path,
    )


# Suppress unused-import warning for the type-only ``Any`` re-export.
_ = Any
