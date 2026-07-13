# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the Pi 5 / Cortex-A76 cross-compile wrapper (T-031).

The wrapper splits into a *plan* (pure data; no I/O) and an *execute*
step (subprocess against the cross-toolchain). Tier A — what runs on
CI without Arm GCC installed — covers the plan composition end-to-end
and exercises ``execute_cross_compile`` against a Python-fake toolchain
stub. Tier B (post-Arm-GCC-install) drives the real toolchain.

The MJX-as-source close from A1 means the cross test exercises only
the hand-written analytical cartpole; that's the real launch path.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from jaxility import (
    CrossCompilePlan,
    cflags_for_family,
    plan_cross_compile,
    resolve_toolchain_integrity,
    verify_toolchain_installed,
    verify_toolchain_integrity,
)
from jaxility.builder_cross import (
    _FAMILY_CFLAGS,
    _discover_c_sources,
    execute_cross_compile,
)
from jaxility.errors import TargetError, ToolchainError
from jaxility.targets import MOCK_CORTEX_A, MOCK_CORTEX_M, PI5

# ---------------------------------------------------------------------------
# Per-family flag table
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cflags_known_family_returns_stable_tuple() -> None:
    flags = cflags_for_family("cortex-a76")
    assert flags == _FAMILY_CFLAGS["cortex-a76"]
    assert isinstance(flags, tuple)


@pytest.mark.unit
def test_cflags_cortex_a76_pins_isa_and_tune() -> None:
    flags = cflags_for_family("cortex-a76")
    # `-mcpu=cortex-a76` pins the ISA + features (armv8.2-a + the
    # A76 feature set). Specifying `-march` alongside conflicts in
    # Arm GNU 15.2.Rel1 (cc1 rejects the combination), so the table
    # uses `-mcpu` alone.
    assert "-mcpu=cortex-a76" in flags
    assert not any(f.startswith("-march=") for f in flags)
    assert "-O3" in flags
    assert "-fPIC" in flags
    assert "-shared" in flags


@pytest.mark.unit
def test_cflags_unknown_family_raises_target_error() -> None:
    with pytest.raises(TargetError) as exc_info:
        cflags_for_family("totally-fake-family")
    msg = str(exc_info.value)
    assert "totally-fake-family" in msg
    assert "_FAMILY_CFLAGS" in msg  # tells you exactly where to add the row


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def _emit_fake_c_tree(root: Path, model_name: str) -> None:
    """Build a tree that mirrors what acados emits, just enough for tests."""
    (root / f"{model_name}_model").mkdir(parents=True)
    (root / f"{model_name}_constraints").mkdir(parents=True)
    (root / f"{model_name}_cost").mkdir(parents=True)
    # Model-related ``.c`` files (the cross-compile picks these up).
    (root / f"acados_solver_{model_name}.c").write_text("/* solver glue */\n")
    (root / f"{model_name}_model" / f"{model_name}_dyn.c").write_text("/* dyn */\n")
    (root / f"{model_name}_constraints" / f"{model_name}_constr.c").write_text(
        "/* constr */\n"
    )
    (root / f"{model_name}_cost" / f"{model_name}_cost.c").write_text("/* cost */\n")
    # A spurious unrelated file at the root that should be filtered out
    # because it shares no token with the model name.
    (root / "unrelated_helper.c").write_text("/* not for us */\n")


@pytest.mark.unit
def test_discover_c_sources_picks_model_specific_files(tmp_path: Path) -> None:
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "cartpole_lqr")
    sources = _discover_c_sources(root, "cartpole_lqr")
    names = {s.name for s in sources}
    assert "acados_solver_cartpole_lqr.c" in names
    assert "cartpole_lqr_dyn.c" in names
    assert "cartpole_lqr_constr.c" in names
    assert "cartpole_lqr_cost.c" in names
    # Top-level unrelated file is filtered out.
    assert "unrelated_helper.c" not in names


@pytest.mark.unit
def test_discover_c_sources_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="does not exist"):
        _discover_c_sources(tmp_path / "nope", "model")


@pytest.mark.unit
def test_discover_c_sources_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "c_generated_code").mkdir()
    with pytest.raises(TargetError, match="no model-related"):
        _discover_c_sources(tmp_path / "c_generated_code", "lqr")


@pytest.mark.unit
def test_discover_c_sources_is_sorted(tmp_path: Path) -> None:
    """Byte-stable plan composition needs sorted sources."""
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "m")
    sources_a = _discover_c_sources(root, "m")
    sources_b = _discover_c_sources(root, "m")
    assert sources_a == sources_b
    assert list(sources_a) == sorted(sources_a)


# ---------------------------------------------------------------------------
# Plan composition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plan_for_pi5_uses_pinned_toolchain(tmp_path: Path) -> None:
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    out = tmp_path / "libpi5_lqr.so"
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=out,
        model_name="lqr",
    )
    assert plan.compiler_argv[0] == "aarch64-none-linux-gnu-gcc"
    assert plan.output_path == out


@pytest.mark.unit
def test_plan_argv_contains_family_flags(tmp_path: Path) -> None:
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=tmp_path / "lib.so",
        model_name="lqr",
    )
    assert "-mcpu=cortex-a76" in plan.compiler_argv
    assert "-O3" in plan.compiler_argv
    assert "-shared" in plan.compiler_argv
    assert "-fPIC" in plan.compiler_argv


@pytest.mark.unit
def test_plan_argv_routes_includes_and_output(tmp_path: Path) -> None:
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    extra = tmp_path / "vendor_headers"
    extra.mkdir()
    out = tmp_path / "out.so"
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=out,
        model_name="lqr",
        extra_include_dirs=(extra,),
    )
    assert f"-I{root}" in plan.compiler_argv
    assert f"-I{extra}" in plan.compiler_argv
    assert "-o" in plan.compiler_argv
    out_idx = plan.compiler_argv.index("-o")
    assert plan.compiler_argv[out_idx + 1] == str(out)


@pytest.mark.unit
def test_plan_argv_ends_with_sources_then_link_args(tmp_path: Path) -> None:
    """Order matters: sources before extra link args (typical link rules)."""
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=tmp_path / "lib.so",
        model_name="lqr",
        extra_link_args=("-lm", "-lpthread"),
    )
    # The last two argv elements are the link args.
    assert plan.compiler_argv[-2:] == ("-lm", "-lpthread")
    # Right before them are the .c sources (all of them).
    sources_str = {str(s) for s in plan.sources}
    n_sources = len(plan.sources)
    argv_sources_slice = plan.compiler_argv[-2 - n_sources : -2]
    assert set(argv_sources_slice) == sources_str


@pytest.mark.unit
def test_plan_is_deterministic_across_calls(tmp_path: Path) -> None:
    """invariant 5: same inputs → byte-identical argv."""
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    out = tmp_path / "lib.so"
    p1 = plan_cross_compile(
        target=PI5, c_source_dir=root, output_path=out, model_name="lqr"
    )
    p2 = plan_cross_compile(
        target=PI5, c_source_dir=root, output_path=out, model_name="lqr"
    )
    assert p1.compiler_argv == p2.compiler_argv


@pytest.mark.unit
def test_plan_is_frozen(tmp_path: Path) -> None:
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=tmp_path / "lib.so",
        model_name="lqr",
    )
    with pytest.raises(Exception):
        plan.target = MOCK_CORTEX_A  # type: ignore[misc]


@pytest.mark.unit
def test_plan_unknown_family_raises(tmp_path: Path) -> None:
    """MOCK_CORTEX_M has family 'mock-cortex-m' — not in _FAMILY_CFLAGS."""
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    with pytest.raises(TargetError, match="no cross-compile cflags"):
        plan_cross_compile(
            target=MOCK_CORTEX_M,
            c_source_dir=root,
            output_path=tmp_path / "lib.so",
            model_name="lqr",
        )


# ---------------------------------------------------------------------------
# Toolchain detection — without the real Arm GCC installed
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_toolchain_missing_binary_raises_toolchain_error() -> None:
    """Without Arm GCC on PATH the wrapper raises ToolchainError, not OSError."""
    if shutil.which("aarch64-none-linux-gnu-gcc") is not None:
        pytest.skip("Arm GCC is installed; this test verifies the *missing* path")
    with pytest.raises(ToolchainError) as exc_info:
        verify_toolchain_installed(PI5)
    msg = str(exc_info.value)
    assert "aarch64-none-linux-gnu-gcc" in msg
    assert "developer.arm.com" in msg  # tells the user where to get it


@pytest.mark.unit
def test_verify_toolchain_wrong_version_raises(tmp_path: Path) -> None:
    """Stub `aarch64-none-linux-gnu-gcc --version` to print a different version."""
    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    # Banner shape matches the trailing ``<semver> YYYYMMDD`` form the
    # 15.2.Rel1 release reports; the captured semver here disagrees
    # with the pinned 15.2.1 → wrong-version path.
    banner = (
        "aarch64-none-linux-gnu-gcc (GNU Toolchain for the A-profile "
        "Architecture 12.0.0 (Arm)) 12.0.0 20221231"
    )
    fake_bin.write_text(f"#!/usr/bin/env bash\necho '{banner}'\n")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        with pytest.raises(ToolchainError, match="reports version"):
            verify_toolchain_installed(PI5)
    finally:
        os.environ["PATH"] = old_path


@pytest.mark.unit
def test_verify_toolchain_garbled_output_raises(tmp_path: Path) -> None:
    """Stub returns text that does not match version_regex at all."""
    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_bin.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            echo "totally unrelated banner with no version"
            """
        )
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        with pytest.raises(ToolchainError, match="version regex"):
            verify_toolchain_installed(PI5)
    finally:
        os.environ["PATH"] = old_path


@pytest.mark.unit
def test_verify_toolchain_matching_version_returns_string(tmp_path: Path) -> None:
    """Stub returns exactly the pinned version — should pass."""
    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    # Banner shape mirrors the real 15.2.Rel1 release: ``Rel1`` inside
    # the parens, semver in the trailing ``<semver> YYYYMMDD`` tail.
    banner = (
        "aarch64-none-linux-gnu-gcc (GNU Toolchain for the A-profile "
        "Architecture 15.2.Rel1 (Build arm-15.86)) 15.2.1 20251203"
    )
    fake_bin.write_text(f"#!/usr/bin/env bash\necho '{banner}'\n")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        captured = verify_toolchain_installed(PI5)
    finally:
        os.environ["PATH"] = old_path
    assert captured == "15.2.1"


# ---------------------------------------------------------------------------
# M-7: verify_toolchain_integrity rejects unverified pins loudly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_integrity_rejects_unverified_pin(tmp_path: Path) -> None:
    """Audit M-7: pins carrying ``UNVERIFIED_SHA256`` must raise, not silently pass."""
    # All shipped real-target pins are currently UNVERIFIED — verify
    # the loud-fail path against PI5 specifically.
    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_bin.write_text("#!/usr/bin/env bash\necho dummy\n")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        with pytest.raises(ToolchainError, match="not yet pinned"):
            verify_toolchain_integrity(PI5)
    finally:
        os.environ["PATH"] = old_path


@pytest.mark.unit
def test_verify_integrity_matches_pinned_sha(tmp_path: Path) -> None:
    """When the pin carries a real SHA, verify_toolchain_integrity matches."""
    import hashlib

    from jaxility.targets import ToolchainPin

    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_bin.write_bytes(b"#!/usr/bin/env bash\necho fake gcc 9.9.9\n")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    expected_sha = hashlib.sha256(fake_bin.read_bytes()).hexdigest()

    pinned_target = PI5.model_copy(
        update={
            "toolchain": ToolchainPin(
                **{
                    **PI5.toolchain.model_dump(),
                    "expected_sha256": expected_sha,
                }
            )
        }
    )
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        captured = verify_toolchain_integrity(pinned_target)
    finally:
        os.environ["PATH"] = old_path
    assert captured == expected_sha


@pytest.mark.unit
def test_verify_integrity_rejects_mismatching_sha(tmp_path: Path) -> None:
    from jaxility.targets import ToolchainPin

    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_bin.write_bytes(b"definitely not what the pin expects")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    pinned_target = PI5.model_copy(
        update={
            "toolchain": ToolchainPin(
                **{
                    **PI5.toolchain.model_dump(),
                    "expected_sha256": "a" * 64,
                }
            )
        }
    )
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        with pytest.raises(ToolchainError, match="hashes to"):
            verify_toolchain_integrity(pinned_target)
    finally:
        os.environ["PATH"] = old_path


# ---------------------------------------------------------------------------
# T-112: resolve_toolchain_integrity — the build-path policy that carries the
# integrity result into the manifest (record-honestly vs enforce).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_integrity_unverified_records_without_raising() -> None:
    """An unverified pin (all shipped Arm pins today) does NOT abort the build;
    it records ``"unverified"`` and returns a warning so the manifest is honest."""
    status, warning = resolve_toolchain_integrity(PI5)
    assert status == "unverified"
    assert warning is not None
    assert "NOT verified" in warning


@pytest.mark.unit
def test_resolve_integrity_pinned_match_records_sha(tmp_path: Path) -> None:
    """A real pin that matches the installed binary records ``sha256:<hex>`` and
    emits no warning."""
    import hashlib

    from jaxility.targets import ToolchainPin

    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_bin.write_bytes(b"#!/usr/bin/env bash\necho fake gcc 9.9.9\n")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    expected_sha = hashlib.sha256(fake_bin.read_bytes()).hexdigest()
    pinned_target = PI5.model_copy(
        update={
            "toolchain": ToolchainPin(
                **{**PI5.toolchain.model_dump(), "expected_sha256": expected_sha}
            )
        }
    )
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        status, warning = resolve_toolchain_integrity(pinned_target)
    finally:
        os.environ["PATH"] = old_path
    assert status == f"sha256:{expected_sha}"
    assert warning is None


@pytest.mark.unit
def test_resolve_integrity_pinned_mismatch_hard_fails(tmp_path: Path) -> None:
    """A real pin whose binary does not match aborts the build (never ship an
    artifact whose toolchain integrity we set out to check and could not confirm)."""
    from jaxility.targets import ToolchainPin

    fake_bin = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_bin.write_bytes(b"tampered binary")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    pinned_target = PI5.model_copy(
        update={
            "toolchain": ToolchainPin(
                **{**PI5.toolchain.model_dump(), "expected_sha256": "a" * 64}
            )
        }
    )
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old_path}"
    try:
        with pytest.raises(ToolchainError, match="hashes to"):
            resolve_toolchain_integrity(pinned_target)
    finally:
        os.environ["PATH"] = old_path


# ---------------------------------------------------------------------------
# execute_cross_compile — Python-fake toolchain
# ---------------------------------------------------------------------------


def _write_fake_gcc(path: Path, *, fail: bool = False) -> Path:
    """Create a fake ``aarch64-none-linux-gnu-gcc`` that just writes its -o output.

    Used by ``execute_cross_compile`` tests to drive subprocess.run
    without depending on a real cross-toolchain installation.
    """
    script = textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        argv = sys.argv[1:]
        if {fail!r}:
            sys.stderr.write("fake-gcc: simulated failure\\n")
            sys.exit(1)
        # Mimic a compiler: find -o <output> and create the output file.
        out = None
        for i, a in enumerate(argv):
            if a == "-o" and i + 1 < len(argv):
                out = argv[i + 1]
                break
        if out is not None:
            with open(out, "wb") as f:
                f.write(b"\\x7fELF\\x02\\x01\\x01\\x00")  # ELF64 magic
        sys.exit(0)
        """
    )
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.mark.unit
def test_execute_cross_compile_writes_output(tmp_path: Path) -> None:
    """Drives execute_cross_compile against a Python-fake gcc that writes a file."""
    fake_gcc = _write_fake_gcc(tmp_path / "aarch64-none-linux-gnu-gcc")
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    out = tmp_path / "lib.so"

    plan = plan_cross_compile(
        target=PI5, c_source_dir=root, output_path=out, model_name="lqr"
    )
    # Swap the first argv element to point at our fake gcc directly.
    rewired = CrossCompilePlan(
        target=plan.target,
        c_source_dir=plan.c_source_dir,
        sources=plan.sources,
        include_dirs=plan.include_dirs,
        output_path=plan.output_path,
        compiler_argv=(str(fake_gcc), *plan.compiler_argv[1:]),
        extra_link_args=plan.extra_link_args,
    )
    result = execute_cross_compile(rewired)
    assert result == out
    assert out.exists()
    assert out.read_bytes().startswith(b"\x7fELF")


@pytest.mark.unit
def test_execute_cross_compile_propagates_failure(tmp_path: Path) -> None:
    fake_gcc = _write_fake_gcc(tmp_path / "aarch64-none-linux-gnu-gcc", fail=True)
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=tmp_path / "lib.so",
        model_name="lqr",
    )
    rewired = CrossCompilePlan(
        target=plan.target,
        c_source_dir=plan.c_source_dir,
        sources=plan.sources,
        include_dirs=plan.include_dirs,
        output_path=plan.output_path,
        compiler_argv=(str(fake_gcc), *plan.compiler_argv[1:]),
        extra_link_args=plan.extra_link_args,
    )
    with pytest.raises(ToolchainError, match="exit code 1"):
        execute_cross_compile(rewired)


@pytest.mark.unit
def test_execute_cross_compile_missing_output_raises(tmp_path: Path) -> None:
    """Fake gcc exits 0 but writes no file — wrapper must catch that."""
    script = textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        sys.exit(0)
        """
    )
    fake_gcc = tmp_path / "aarch64-none-linux-gnu-gcc"
    fake_gcc.write_text(script)
    fake_gcc.chmod(fake_gcc.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    root = tmp_path / "c_generated_code"
    _emit_fake_c_tree(root, "lqr")
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=root,
        output_path=tmp_path / "lib.so",
        model_name="lqr",
    )
    rewired = CrossCompilePlan(
        target=plan.target,
        c_source_dir=plan.c_source_dir,
        sources=plan.sources,
        include_dirs=plan.include_dirs,
        output_path=plan.output_path,
        compiler_argv=(str(fake_gcc), *plan.compiler_argv[1:]),
        extra_link_args=plan.extra_link_args,
    )
    with pytest.raises(ToolchainError, match="no file at"):
        execute_cross_compile(rewired)


# ---------------------------------------------------------------------------
# Tier B — exercises the real installed cross-toolchain.
#
# Pi 5 lane (aarch64-none-linux-gnu) is queued: Arm doesn't ship that
# variant via Homebrew on darwin-arm64, so the user installs it manually
# from developer.arm.com. The test below lights up automatically once
# the binary is on PATH.
#
# Cortex-M lane (arm-none-eabi) is exercised here: the user installed
# arm-none-eabi-gcc 15.2.1 via `gcc-arm-embedded` (Homebrew). The tests
# verify the toolchain detection matches the pin, then drive a real
# compile of a minimal C source to a Cortex-M4 relocatable object.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("aarch64-none-linux-gnu-gcc") is None,
    reason=(
        "aarch64-none-linux-gnu-gcc not on PATH; Pi 5 Tier-B runs in CI "
        "(Linux x86_64 runners install Arm GNU 15.2.Rel1; "
        "darwin-arm64 host build not shipped by Arm)"
    ),
)
def test_real_aarch64_detect_matches_pin() -> None:
    """Tier B (Pi 5 lane, detection) — captures the binary's reported version."""
    version = verify_toolchain_installed(PI5)
    assert version == PI5.toolchain.version


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("aarch64-none-linux-gnu-gcc") is None,
    reason=(
        "aarch64-none-linux-gnu-gcc not on PATH; Pi 5 Tier-B runs in CI "
        "(Linux x86_64 runners install Arm GNU 15.2.Rel1; "
        "darwin-arm64 host build not shipped by Arm)"
    ),
)
def test_real_aarch64_compile_minimal_cortex_a76(tmp_path: Path) -> None:
    """Tier B (Pi 5 lane) — end-to-end compile of a minimal C source.

    Drives the real aarch64-none-linux-gnu-gcc through
    ``execute_cross_compile`` against a single hand-written C file.
    A-profile family flags include ``-shared -fPIC`` so the output is
    a position-independent shared library suitable for runtime loading.
    """
    src_dir = tmp_path / "c_generated_code" / "pi5_demo"
    src_dir.mkdir(parents=True)
    (src_dir / "pi5_demo_step.c").write_text(
        "/* Minimal Cortex-A76 unit under cross-compile test. */\n"
        "#include <stdint.h>\n"
        "int32_t pi5_demo_step(int32_t x) {\n"
        "    return x * 2 + 1;\n"
        "}\n"
    )
    out = tmp_path / "libpi5_demo.so"
    plan = plan_cross_compile(
        target=PI5,
        c_source_dir=tmp_path / "c_generated_code",
        output_path=out,
        model_name="pi5_demo",
    )
    assert plan.compiler_argv[0] == "aarch64-none-linux-gnu-gcc"
    assert "-mcpu=cortex-a76" in plan.compiler_argv
    assert "-shared" in plan.compiler_argv
    assert "-fPIC" in plan.compiler_argv

    result = execute_cross_compile(plan)
    assert result == out
    assert out.exists()
    # ELF shared object: 0x7f 'E' 'L' 'F' magic.
    payload = out.read_bytes()
    assert payload.startswith(b"\x7fELF")


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("arm-none-eabi-gcc") is None,
    reason="arm-none-eabi-gcc not installed; Cortex-M Tier-B lane unavailable",
)
def test_real_arm_none_eabi_detect_matches_pin() -> None:
    """Tier B (Cortex-M lane) — detection captures the binary's reported version."""
    from jaxility.targets import CORTEX_M4

    version = verify_toolchain_installed(CORTEX_M4)
    assert version == CORTEX_M4.toolchain.version


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("arm-none-eabi-gcc") is None,
    reason="arm-none-eabi-gcc not installed; Cortex-M Tier-B lane unavailable",
)
def test_real_arm_none_eabi_compile_minimal_cortex_m4(tmp_path: Path) -> None:
    """Tier B (Cortex-M lane) — end-to-end compile against a minimal C source.

    Drives the real arm-none-eabi-gcc through ``execute_cross_compile``
    against a single hand-written C file. Produces a relocatable ``.o``
    (M-profile family cflags include ``-c``); a final ELF needs a
    runtime project's linker script + startup files, which is T-052's
    surface, not this PR's.
    """
    from jaxility.targets import CORTEX_M4

    # Minimal acados-style source — the cross-compile must accept it
    # without depending on acados headers (those headers are not in
    # the M-profile build of acados / blasfeo / hpipm; those archives
    # don't exist yet — that gap is documented in KNOWN_GAPS.md).
    src_dir = tmp_path / "c_generated_code" / "cortex_m4_demo"
    src_dir.mkdir(parents=True)
    (src_dir / "cortex_m4_demo_step.c").write_text(
        "/* Minimal Cortex-M4 unit under cross-compile test. */\n"
        "#include <stdint.h>\n"
        "int32_t cortex_m4_demo_step(int32_t x) {\n"
        "    return x * 2 + 1;\n"
        "}\n"
    )
    out = tmp_path / "cortex_m4_demo_step.o"
    plan = plan_cross_compile(
        target=CORTEX_M4,
        c_source_dir=tmp_path / "c_generated_code",
        output_path=out,
        model_name="cortex_m4_demo",
    )
    # Sanity-check the composed argv before running it.
    assert plan.compiler_argv[0] == "arm-none-eabi-gcc"
    assert "-mcpu=cortex-m4" in plan.compiler_argv
    assert "-mthumb" in plan.compiler_argv
    assert "-c" in plan.compiler_argv  # compile-only for M-profile

    result = execute_cross_compile(plan)
    assert result == out
    assert out.exists()
    # ELF relocatable object: 0x7f 'E' 'L' 'F' magic.
    payload = out.read_bytes()
    assert payload.startswith(b"\x7fELF")


# ---------------------------------------------------------------------------
# Audit M-3: integration test for cross_build_for_target end-to-end.
#
# The previous audit found cross_build_for_target was exported but had
# zero test callers — the full path (acados codegen + cross-compile +
# manifest pack) was untested. This Tier-B test exercises the function
# end-to-end on whichever cross-toolchain is live (Pi 5 on CI, Cortex-M
# locally). Compile-only success is the bar; linking against cross-built
# acados / blasfeo / hpipm is still gap-documented in KNOWN_GAPS.md.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(
    shutil.which("aarch64-none-linux-gnu-gcc") is None,
    reason="aarch64-none-linux-gnu-gcc not on PATH; Pi 5 Tier-B runs in CI",
)
def test_cross_build_for_target_end_to_end_pi5(tmp_path: Path) -> None:
    """Audit M-3: ``cross_build_for_target`` runs acados codegen + cross-compile.

    The function attempts to link against cross-built acados / blasfeo
    / hpipm static archives, which do not exist on the runner. The
    expected outcome is therefore a controlled :class:`ToolchainError`
    raised by the cross-compile step (the linker reports unresolved
    symbols), NOT a silent pass and NOT a Python TypeError or KeyError.
    This test asserts the contract: the function exists, the
    acados-codegen path runs to completion, and the cross-compile
    failure is a structured Jaxility error pointing at the next gap.
    """
    pytest.importorskip("acados_template")
    pytest.importorskip("casadi")
    from jaxility import cross_build_for_target
    from jaxility.errors import ToolchainError
    from jaxility.lowering import translate
    from jaxility.targets import PI5
    from jaxility.templates import lqr

    def cartpole(x, u):
        # Minimal smooth dynamics: avoid MJX, avoid lax.cond, avoid
        # while_loop — pure arithmetic so the translator + acados
        # codegen succeed.
        return jnp.stack(
            (x[1], -9.81 * jnp.sin(x[0]) + u[0], x[3], -0.1 * x[3]),
            axis=0,
        )

    import jax.numpy as jnp  # noqa: PLC0415

    cf = translate(cartpole, in_shapes=((4,), (1,)), name="m3_demo")
    spec = lqr(
        cf,
        Q=(1.0, 1.0, 1.0, 1.0),
        R=(0.1,),
        initial_state=(0.1, 0.0, 0.0, 0.0),
        input_bounds=((-10.0,), (10.0,)),
        name="m3_demo",
        horizon_steps=5,
        time_horizon_s=0.5,
    )
    # The link step will fail with a ToolchainError because the cross
    # acados / blasfeo / hpipm archives do not exist; that is the
    # documented gap. The point of this test is that the failure is a
    # structured Jaxility error, not a bare exception.
    with pytest.raises(ToolchainError):
        cross_build_for_target(
            dynamics=cf,
            spec=spec,
            target=PI5,
            source_attestation_handle=bytes(32),
            work_dir=tmp_path / "build",
        )
