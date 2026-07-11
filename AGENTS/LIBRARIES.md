# Jaxility — Upstream Libraries, Attribution, and License Analysis

Jaxility composes existing compiler, code-generation, and embedded-runtime
toolchains rather than re-implementing them (ADR-001, ADR-002). This file
records, for every upstream project Jaxility depends on, borrows from, or
shells out to: its role, how Jaxility uses it (runtime dependency, vendored
code, subprocess invocation, or reference only), its reported license, and
the license-compatibility note for Jaxility's MIT distribution.

## Methodology and caveats

- Licenses below were checked against each project's public repository on
  the date noted and are recorded as **reported**. Open-source and vendor
  licenses can change between releases. **Before any upstream code is
  vendored or copied into Jaxility (as opposed to merely imported or
  invoked via subprocess), re-confirm the license against the pinned
  version's `LICENSE` / `NOTICE` files** and record the exact commit / tag.
- "Incorporation mode" is the load-bearing column:
  - **dependency** — imported at runtime; we ship no upstream code. License
    compatibility is trivial (permissive licenses impose no obligation on a
    mere importer).
  - **subprocess** — invoked as an external binary via
    `jaxility.runtime.subprocess_runner` (PATTERNS §2.1). We ship no
    upstream code; the *user* must have the binary installed. License
    obligations apply to the user's binary, not to Jaxility's distribution.
  - **vendored** — upstream source is copied into Jaxility (e.g. small
    headers, build templates). The upstream license and notices travel
    with the copied files; this is where compatibility actually matters.
  - **reference** — used only as an algorithm / spec reference for clean-room
    implementation. No code is taken; no license obligation, but
    attribution is courteous and recorded here.
- "Vendor SDK" entries reflect proprietary or click-through licenses that
  are **not redistributable**. Jaxility never ships these; it documents
  them and instructs the user to install them locally.
- Last verified at initial bootstrap (most entries marked
  *reported — confirm at integration*; tighten as each integration lands).

## Permissive-license compatibility primer

MIT, BSD-2-Clause, and BSD-3-Clause are permissive and combine freely with
Jaxility's MIT license; vendoring requires retaining the upstream copyright
and license text in the copied files. Apache-2.0 is also compatible to
combine, but vendoring Apache-2.0 code additionally requires preserving
`NOTICE` content and is one-directional (Apache-2.0 files keep their
license; they are not relicensed to MIT). LGPL is *compatible to link
against* dynamically with documented obligations; static-link or copy is
GPL-equivalent and is a red flag (escalate via ADR before adoption). Full
GPL / AGPL is incompatible with Jaxility's MIT distribution and must not
be vendored.

Vendor SDKs (Qualcomm, STMicro, Apple) generally ship under proprietary
click-through licenses that allow personal / commercial use of the binaries
but forbid redistribution. Jaxility's MIT distribution does not include
them; the user installs them per the vendor's terms. See "Vendor SDKs"
below.

## Runtime dependencies (Python)

| Project       | Role in Jaxility                                                    | Incorporation         | Upstream                              | License *(reported)*       | Note                                                                                                                                                                              |
|---------------|---------------------------------------------------------------------|-----------------------|---------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Jaxterity** | Source of `CalibratedRobot` and `attestation_handle` (core dep)  | core dependency       | machinavitalis/jaxterity (private)    | MIT                        | Imported only; attestation chain consumer.                                                                                                                                        |
| **Jaxonomy**  | Simulation engine (transitive via Jaxterity)                        | transitive dependency | machinavitalis/jaxonomy (private)     | MIT                        | Not imported directly from Jaxility code (CONTEXT.md).                                                                                                                            |
| **CasADi**    | JAX → acados symbolic boundary (ADR-001)                            | core dependency       | casadi/casadi                         | LGPL-3.0 *(reported)*      | **Action item** — LGPL needs link-mode review at packaging time. Used at one boundary; not vendored.                                                                              |
| **Pydantic**  | Schema for `Target`, `Manifest`, `Coverage`, `Artifact` (PATTERNS §3.3) | core dependency       | pydantic/pydantic                     | MIT *(reported)*           | Importer only.                                                                                                                                                                    |
| **blake3**    | Canonical hash for manifest chain (ADR-005)                         | core dependency       | oconnor663/blake3-py                  | Apache-2.0 OR MIT *(reported)* | Importer only; algorithm consistent with Jaxterity ADR-008.                                                                                                                       |
| **cryptography** | Signing primitives behind the `Signer` protocol (ADR-005)         | core dependency       | pyca/cryptography                     | Apache-2.0 OR BSD-3 *(reported)* | OSS ships `HashChainSigner`; enterprise plugs in real signing.                                                                                                                    |
| **FastMCP**   | MCP server framework (ADR-012)                                      | dependency (`[mcp]`)  | jlowin/fastmcp                        | Apache-2.0 *(reported — confirm)* | Importer only; T-004.                                                                                                                                                             |
| **acados-template** | acados problem builder (Python side)                          | dependency (`[acados]`) — *not consistently on PyPI* | acados/acados                  | BSD-2-Clause *(reported)*  | acados itself is BSD-2; the Python template package is in the same tree. T-021; see OQ-1 (vendor vs. pin upstream).                                                     |
| **LiteRT runtime** | On-device inference for learned policies (ADR-002)             | dependency (`[litert]`) | google-ai-edge/LiteRT                 | Apache-2.0 *(reported — confirm)* | LiteRT is the TensorFlow Lite successor; expect the canonical Python wheel to land before T-041.                                                                       |
| **ExecuTorch runtime** | Parallel inference path (ADR-002)                           | dependency (`[executorch]`) | pytorch/executorch                  | BSD-3-Clause *(reported — confirm)* | PyTorch's embedded path; supported in parallel with LiteRT.                                                                                                                       |
| **ONNX**      | Policy export intermediate (T-040)                                  | dependency (`[litert]`/`[executorch]`) | onnx/onnx                           | Apache-2.0 *(reported)*    | Importer only.                                                                                                                                                                    |

## External toolchains (subprocess)

These are invoked via `jaxility.runtime.subprocess_runner`. Jaxility does
not ship them; the user installs them per the toolchain's terms.

| Toolchain           | Role in Jaxility                                | Incorporation | Upstream / vendor                   | License *(reported)*       | Note                                                                                                                                                                                                                       |
|---------------------|-------------------------------------------------|---------------|-------------------------------------|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **acados**          | MPC / WBC C-code generator (ADR-001)            | subprocess + dependency | acados/acados                       | BSD-2-Clause *(reported)*  | The C library is BSD-2; Python template package travels in the same tree. T-021. See OQ-1.                                                                                                                       |
| **Arm GCC (aarch64)** | Cross-compiler for Cortex-A targets           | subprocess    | Arm Ltd. (binary releases)          | GPL-3.0 *(toolchain itself)* | GPL applies to the toolchain binaries; *output* is not GPL-encumbered (GCC runtime-library exception). Documented in TOOLCHAINS.md.                                                                                        |
| **Arm GCC (arm-none-eabi)** | Bare-metal cross-compiler for Cortex-M  | subprocess    | Arm Ltd.                            | GPL-3.0                    | Same exception applies.                                                                                                                                                                                                    |
| **LLVM / Clang**    | Alternative cross-compiler (post-v0.1)          | subprocess    | llvm/llvm-project                   | Apache-2.0 WITH LLVM-exception | Permissive; preferred for new toolchains where available.                                                                                                                                                                  |
| **Arm FVP**         | Fixed Virtual Platforms (FVP) for HIL without hardware | subprocess  | Arm Ltd.                            | proprietary EULA           | Click-through; redistribution forbidden. User installs per Arm's terms. T-053.                                                                                                                                   |
| **Bazel**           | Optional cross-compile orchestration            | subprocess    | bazelbuild/bazel                    | Apache-2.0 *(reported)*    | Optional; falls back to Make. Documented in TOOLCHAINS.md.                                                                                                                                                                 |

## Embedded runtime libraries (linked into the artifact)

These libraries' code is **linked into the deployed binary** when their
target is selected. License compatibility actually matters here.

| Project           | Role                                                            | Incorporation                | Upstream / vendor      | License *(reported)*    | Note                                                                                                                                                                                                                        |
|-------------------|-----------------------------------------------------------------|------------------------------|------------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CMSIS-DSP**     | Fixed- and floating-point DSP kernels for Cortex-M (PATTERNS §4) | linked, **vendored at build** | ARM-software/CMSIS-DSP | Apache-2.0 *(reported)* | Apache headers travel with copied files; preserve `NOTICE`.                                                                                                                                                                 |
| **CMSIS-NN**      | NPU / SIMD-aware NN kernels for Cortex-M / Ethos                | linked, **vendored at build** | ARM-software/CMSIS-NN  | Apache-2.0 *(reported)* | Same as CMSIS-DSP.                                                                                                                                                                                                          |
| **Arm Compute Library** | NEON / SVE / SVE2 SIMD kernels for Cortex-A               | linked, **vendored at build** | ARM-software/ComputeLibrary | MIT *(reported)*        | MIT; combines freely.                                                                                                                                                                                                       |
| **LiteRT C runtime** | Embedded inference runtime for learned policies              | linked, **vendored at build** | google-ai-edge/LiteRT  | Apache-2.0 *(reported — confirm)* | Apache headers + `NOTICE`.                                                                                                                                                                                                   |
| **ExecuTorch C runtime** | Parallel inference path                                  | linked, **vendored at build** | pytorch/executorch     | BSD-3-Clause *(reported — confirm)* | BSD-3 header.                                                                                                                                                                                                                |
| **acados runtime (C)** | OCP solver runtime linked into the artifact                 | linked, **vendored at build** | acados/acados          | BSD-2-Clause *(reported)* | BSD-2 header.                                                                                                                                                                                                                |
| **FreeRTOS** *(if used; OQ-2)* | RTOS for Cortex-M (alternative to bare-metal)         | linked, **vendored at build** | FreeRTOS-Kernel        | MIT *(reported)*        | OQ-2 decides whether to use FreeRTOS at all on Cortex-M (T-052).                                                                                                                                                   |
| **PREEMPT_RT Linux** *(target-side, Cortex-A)* | Real-time Linux kernel patches           | host environment (not vendored) | kernel.org             | GPL-2.0                 | Runs on the *user's* device. Jaxility's userspace binary links against glibc only; GPL kernel patches do not encumber the artifact.                                                                                          |

## Vendor SDKs (per-target extras)

Vendor SDKs ship under proprietary click-through licenses that **are not
redistributable**. Jaxility's MIT distribution does not contain any vendor
SDK code. The user installs the SDK per the vendor's terms; Jaxility's
build process invokes it via subprocess.

| Vendor SDK                            | Role                                          | Extras                  | License *(reported)*               | Note                                                                                                                                                                                                                     |
|---------------------------------------|-----------------------------------------------|-------------------------|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Qualcomm Sensors Execution Environment** *(Dragonwing IQ10)* | NPU compiler / runtime for Qualcomm targets | `[targets-qualcomm]`    | proprietary, click-through         | T-063.                                                                                                                                                                                                          |
| **STM32Cube**                         | Drivers / linker scripts / startup for STM32 series | `[targets-cortex-m]`    | mixed proprietary + permissive components | Most ST drivers ship under BSD-3 or Apache-2.0; HAL components occasionally proprietary. **Re-verify per file before vendoring.** T-051.                                                                          |
| **Apple Silicon SDK**                 | Apple-side toolchain (Xcode CLT) for Apple-Silicon targets | `[targets-apple]` *(deferred)* | proprietary EULA                   | OQ-4 (post-v0.1).                                                                                                                                                                                                         |
| **Raspberry Pi userland**             | Pi 5 userspace headers                        | bundled in Raspberry Pi OS | mixed BSD-3 / proprietary firmware | Userspace is BSD-3; firmware blobs proprietary. We link userspace only. T-030.                                                                                                                                  |

## Reference-only influences (no code incorporated)

These shape Jaxility's design (per CONTEXT.md "Architectural heritage") but
no source is taken; clean-room implementation.

| Project          | What we take (concept)                                                          | Upstream                    | License *(reported)*          |
|------------------|---------------------------------------------------------------------------------|-----------------------------|-------------------------------|
| **acados**       | CasADi-to-OCP-to-C codegen workflow as designed                                  | acados/acados               | BSD-2-Clause                  |
| **TensorFlow Lite Micro** | On-device inference patterns; deprecated in favour of LiteRT                | tensorflow/tflite-micro     | Apache-2.0                    |
| **in-toto / SLSA** | Supply-chain attestation patterns; manifest schema is loosely SLSA-aligned     | in-toto/in-toto, slsa.dev   | Apache-2.0                    |
| **Sigstore**     | Signature primitives reference (enterprise tier, not v0.1)                       | sigstore/sigstore           | Apache-2.0                    |
| **CycloneDX**    | SBOM schema reference (considered for manifest, rejected — ADR-005)              | CycloneDX/specification     | Apache-2.0                    |
| **Drake**        | Trajectory-optimization problem-formulation reference                            | RobotLocomotion/drake       | BSD-3-Clause                  |
| **JAX → MLIR (IREE)** | Future substrate path the lowering is structured around (ADR-001)            | iree-org/iree               | Apache-2.0 WITH LLVM-exception |

## Action items before each integration

- **CasADi (LGPL)** — confirm link-mode (importer only, no copying or
  derivative-work compilation that would trip LGPL section 4). The CasADi
  Python wheel is the standard distribution mode; importing it does not
  create a derivative work of CasADi. Reconfirm before publishing the
  Jaxility wheel.
- **acados** — OQ-1 (vendor a known-good tarball vs. pin upstream) is open
  and resolved at T-021. Either way, BSD-2 lets us redistribute if we
  preserve the header.
- **CMSIS-DSP / CMSIS-NN / Arm Compute Library** — Apache-2.0 `NOTICE`
  handling: when copying source into a generated artifact tree, copy the
  upstream `NOTICE` to the artifact's root and preserve per-file headers.
- **Vendor SDKs** — never copied into the Jaxility distribution. Per-target
  install docs link to the vendor's official download.
- **LiteRT / ExecuTorch** — confirm the canonical wheel name and license
  string at integration time (T-041).
- **FreeRTOS** — OQ-2 (FreeRTOS vs. bare-metal on Cortex-M) decides whether
  this entry stays or moves to "Reference-only".

## Cross-reference

For toolchain *version* policy (how we pin and update), see
`TOOLCHAINS.md`. For architectural reasoning on why CasADi is a component
not a substrate, see DECISIONS.md ADR-001.
