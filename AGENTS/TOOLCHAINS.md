# Jaxility — Toolchain Pinning Policy

This file documents how Jaxility pins external toolchains (compilers,
linkers, code generators, vendor SDKs, FVPs) and how to update those pins.
A toolchain version is a **first-class manifest field** (invariant 3): a
build that does not record the exact versions of every tool it shelled
out to is not reproducible and not acceptable. Cross-reference:
PATTERNS §2, DECISIONS.md ADR-006 (determinism > performance),
LIBRARIES.md (license analysis of each toolchain).

The Python-side dependencies (CasADi, Pydantic, acados-template, …) are
pinned in `pyproject.toml` and follow standard Python packaging
conventions; this file is about the *external binaries* Jaxility
subprocess-invokes.

## Principles

1. **Every binary that touches a build is pinned to an exact version.**
   "Whatever is on PATH" is not a build; it is a build-by-accident.
2. **Pins live in the `Target` profile, not in user code.** Adding a
   target is filling out the profile, including its toolchain pin
   (ADR-003).
3. **Toolchains are detected at startup, not on demand** (PATTERNS §2.2).
   A `Target` load fails immediately if its toolchain is absent or at the
   wrong version, with a user-actionable install hint.
4. **The pinned version travels in the manifest** (invariant 3,
   ADR-005). Every artifact carries `toolchain_versions: dict[str, str]`;
   `jaxility verify` rejects manifests with version drift if the verifier
   demands strict mode.
5. **Determinism over performance.** A toolchain flag that introduces
   nondeterministic output (e.g. `-flto` in some GCC versions, parallel
   linking in some lld releases) is opt-in (`--release --aggressive`) and
   the manifest records it (ADR-006).
6. **Each pin has a documented upgrade path.** Upgrading is a documented
   PR that updates the pin, regenerates the golden artifacts (PATTERNS
   §7.3), and runs the HIL parity tier for every target affected.

## How a pin is expressed

A toolchain pin in a `Target` profile looks like:

```python
toolchain = ToolchainPin(
    name="aarch64-none-linux-gnu-gcc",
    version="15.2.1",
    distribution="arm-gnu-toolchain-15.2.rel1",
    download_url="https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/...",
    expected_sha256="<hash>",  # or "unverified" when not yet pinned (M-7)
    detect_command=["aarch64-none-linux-gnu-gcc", "--version"],
    version_regex=r"(\d+\.\d+\.\d+)\s+\d{8}",
)
```

Each pin records: the canonical binary name, the exact upstream release,
the official download URL, the binary's expected SHA-256 (so the user can
verify what they installed), the detect command (used at `Target` load),
and the regex that extracts the version from the detect command's
stdout.

Pins are **data**, not code. They serialise into the manifest.

## Current pins

| Target                                 | Toolchain                                 | Status       |
|----------------------------------------|-------------------------------------------|--------------|
| `mock-cortex-a`, `mock-cortex-m`       | n/a — Python stub                       | shipped T-011 |
| **acados runtime**                     | acados (source build) + tera_renderer    | **shipped T-021** — see notes below |
| `pi5` (Cortex-A76)                     | `aarch64-none-linux-gnu-gcc` 15.2.1       | **live in CI** — Linux x86_64 runners install Arm GNU 15.2.Rel1; darwin-arm64 host build not shipped by Arm |
| `cortex-m4` / `ethos-u55` / `ethos-u65` | `arm-none-eabi-gcc` 15.2.1                | **live locally + CI** — Tier B real-compile test exercises this |
| `stm32h7` (Cortex-M7)                  | `arm-none-eabi-gcc` 15.2.1                | pending T-050 / T-051 |
| Arm FVP                                | Arm Fast Models FVP                       | pending T-053 |

For the not-yet-shipped rows the exact version strings are placeholders; the
*load-bearing* commitment is that whatever lands in those PRs is
pinned, hashed, and detectable. The acados row is live and documented
in detail below.

### acados (T-021+)

acados is the MPC code generator at the second of the three lowering
stages (ADR-001). The lowering pipeline consumes it via the Python package
`acados_template`, which wraps the C library `libacados`.

| Component        | Provenance                            | Acceptance                                                   |
|------------------|----------------------------------------|--------------------------------------------------------------|
| `libacados`      | source build at `~/Dev/acados` | `libacados.dylib` produced; loadable via `ctypes.CDLL`       |
| `acados_template` | `pip install -e $ACADOS/interfaces/acados_template` | importable; `AcadosOcp` / `AcadosOcpSolver` construct        |
| `t_renderer`     | source build at `$ACADOS/interfaces/acados_template/tera_renderer` via `cargo build --release` | binary at `$ACADOS/bin/t_renderer` (`AcadosOcpSolver.generate()` renders) |
| `libqpOASES_e`   | built as part of `libacados`           | sibling dylib at `$ACADOS/lib/`                              |
| `libhpipm`       | built as part of `libacados`           | sibling dylib at `$ACADOS/lib/`                              |
| `libblasfeo`     | built as part of `libacados`           | sibling dylib at `$ACADOS/lib/`                              |

**Environment expectations:**

- `ACADOS_SOURCE_DIR` points at the acados source tree
  (`~/Dev/acados` by default).
- `DYLD_LIBRARY_PATH` (macOS) / `LD_LIBRARY_PATH` (Linux) includes
  `$ACADOS_SOURCE_DIR/lib`.
- `$ACADOS_SOURCE_DIR/bin/t_renderer` is executable.

`test/conftest.py` populates the env vars automatically when
`~/Dev/acados` exists, or whatever path
`$JAXILITY_ACADOS_DIR` points at. The conftest also performs a
**ctypes preload** of `libblasfeo` / `libhpipm` / `libqpOASES_e` /
`libacados` because macOS SIP strips `DYLD_LIBRARY_PATH` from inherited
env when launching SIP-protected subprocesses, and acados spawns one
to compile its generated C code. Without the preload, the dyld linker
fails to find sibling libraries even though the parent process sees
them. On Linux this layer collapses to a normal `LD_LIBRARY_PATH`
lookup.

**To install acados on a fresh host (macOS):**

```sh
git clone --recursive https://github.com/acados/acados.git ~/Dev/acados
cd ~/Dev/acados && mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make -j install
cd ../interfaces/acados_template/tera_renderer
cargo build --release
mkdir -p ../../../bin && cp target/release/t_renderer ../../../bin/
pip install -e ../  # interfaces/acados_template
```

Then point Jaxility at it via `JAXILITY_ACADOS_DIR=~/Dev/acados` or
let the canonical path be picked up automatically.

**Pinned upstreams (A6):**

The exact upstream versions Jaxility's test suite currently runs
against are pinned in `jaxility/manifest/toolchain_detect.py` as
importable constants:

| Constant                          | Value                | What it pins                 |
|-----------------------------------|----------------------|------------------------------|
| `JAXILITY_ACADOS_TEMPLATE_PIN`    | `0.5.1`              | acados Python interface      |
| `JAXILITY_ACADOS_LIBRARY_PIN`     | `v0.5.4-7-gdc6668f85`| acados C library (`git describe`) |

The detection functions in the same module — `detect_acados_template_version`,
`detect_acados_library_version`, `detect_casadi_version`,
`detect_toolchain_versions` — return the *currently installed*
versions and never the placeholder `"unknown"` (invariant 7 — loud
fallback). When the local acados source tree is missing or not a git
checkout, `detect_acados_library_version` returns a self-explaining
`library-unknown:<reason>` marker that names exactly why detection
failed; the caller can react to that without papering over it.

The `Manifest.toolchain_versions` dict carries four keys when the
build path runs:
`{target.toolchain.name, "acados-template", "acados-library", "casadi"}`.
Upgrading any pin is a documented PR that bumps both the constant and
the recorded value in this table.

**acados caveats:**
- `t_renderer` is built from the Rust source in
  `interfaces/acados_template/tera_renderer/`. The acados Python
  package can also auto-download a prebuilt binary from
  `github.com/acados/tera_renderer/releases`; we build from source
  to keep the toolchain provenance auditable.
- macOS SIP is the load-bearing reason for the ctypes preload. If a
  future Linux host runs into the same kind of issue, the fallback
  is `install_name_tool -change` on `libacados.dylib` to embed the
  absolute path to its sibling libs.
- Deprecation noise: acados 0.5.4 deprecates `ocp.dims.N`,
  `ocp.json_file`, `ocp.acados_include_path` in favour of
  `ocp.solver_options.N`, `ocp.code_gen_opts.*`. Jaxility code uses
  the new APIs; the deprecation warnings in test output come from
  acados' internal compatibility shims, not from this codebase.

## Update protocol

Upgrading a toolchain pin is a PR with the following shape:

1. Update the `Target` profile's `toolchain.version` and
   `expected_sha256`.
2. Re-run the equivalence suite (PATTERNS §7) for that target. Document
   any tolerance drift in `test/EQUIVALENCE.md`.
3. Regenerate the golden artifact for every (template, target) pair the
   toolchain affects, in the same PR (PATTERNS §7.3).
4. Re-run the HIL parity tier for every (robot, target) pair the
   toolchain affects.
5. Update LIBRARIES.md if the toolchain's license string changed.
6. Tag the PR with `toolchain-upgrade` so downstream consumers can grep
   for them.

The PR is reviewed under the same "attestation-chain integrity" lens as
schema changes (CONTEXT.md "When modifying Jaxility").

## CI alignment

The CI workflow (T-002) will:

- Detect-and-pin the toolchain at the start of every job; fail the job
  immediately if the detected version drifts from the expected pin.
- Cache the toolchain binary across runs by its SHA-256 (faster than
  re-downloading on every PR).
- Run the equivalence suite against the pinned toolchain only; jobs that
  detect a pin mismatch surface a structured `ToolchainError`, not an
  obscure pytest failure.

## Vendor SDKs

Vendor SDKs (Qualcomm, STMicro, Apple) follow the same pinning policy
but with two additional constraints:

- The download URL points at the vendor's official site, not a mirror.
  License terms require the user to accept the vendor's EULA before
  installing.
- Some vendor SDKs are click-through and not script-installable. In
  those cases the `Target` profile carries an
  `installation_instructions_url` and the toolchain detect fails with a
  message pointing the user at the install docs.

See LIBRARIES.md "Vendor SDKs" for the per-vendor license picture.

## Anti-patterns

- **Floating version strings** (`>=13.2`, `~13.2.0`) in a `Target`
  profile. Pins are exact. Ranges are for `pyproject.toml`, not for
  toolchains.
- **Using `which gcc` to find the compiler.** The detect command runs
  the canonical binary name (`aarch64-none-linux-gnu-gcc`), not
  whatever shadows it on PATH.
- **Falling back to the system toolchain if the pinned one is missing.**
  Loud failure; no silent fallback. The user installs the pinned tool.
- **Untracked flag changes.** A change to a default compile flag is a
  schema-equivalent change: it must be in the PR title and the manifest
  field that records the flag set.
- **Skipping the regenerate-golden step on a toolchain upgrade.** The
  golden artifact is the regression net for codegen changes; skipping
  the regen means the next contributor inherits a broken net.

## Cross-reference

- License compatibility per toolchain: `LIBRARIES.md`.
- Why pins are part of the manifest: invariant 3 in `CONTEXT.md` and
  ADR-005 in `DECISIONS.md`.
- Why determinism beats performance: ADR-006.
- Where Cortex-A and Cortex-M runtimes diverge: ADR-010.
