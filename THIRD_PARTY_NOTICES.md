# Third-Party Notices

Jaxility is licensed under the [MIT License](LICENSE.md). It **depends on** the
packages below (declared in `pyproject.toml`) but does not bundle or redistribute
their source — each remains under its own license.

> License identifiers reflect each project's stated license to the best of our
> knowledge; the authoritative license is the upstream project's. Verify there
> if it matters for your use.

## Runtime (pip) dependencies

| Package | License (typical SPDX) | Note |
|---|---|---|
| jaxterity | MIT | sibling project (pulls jaxonomy, MIT, transitively) |
| pydantic | MIT | |
| blake3 (python binding) | Apache-2.0 / MIT | |
| cryptography | Apache-2.0 OR BSD-3-Clause | |
| **casadi** | **LGPL-3.0-or-later** | ⚠️ weak copyleft — see below |

### Note on CasADi (LGPL-3.0)

CasADi is the only non-permissive dependency. It is used as a **separately
installed library** (a normal pip dependency that Jaxility imports — not vendored,
not statically linked into Jaxility's own distribution). The LGPL permits use by
differently-licensed software under these conditions, so **Jaxility itself remains
MIT-licensed**. If you redistribute a bundle that *statically links or vendors*
CasADi, review the LGPL's relinking/source-availability obligations for that
bundle. Using Jaxility as a normal Python package does not trigger them.

## Build-time / toolchain components (cross-compiled from source, not pip deps)

The attested deployment lane builds these from source inside a pinned toolchain
container; their archive hashes are recorded in the manifest:

| Component | License |
|---|---|
| acados | BSD-2-Clause |
| BLASFEO | BSD-2-Clause (recent releases) |
| HPIPM | BSD-2-Clause (recent releases) |
| Arm GNU Toolchain | GPL-3.0 (the *compiler*; this does not affect the license of code it compiles) |

## Provenance

Jaxility is original work that sits at the bottom of the stack, consuming
Jaxterity and Jaxonomy (both MIT). It is not derived from any other codebase.
