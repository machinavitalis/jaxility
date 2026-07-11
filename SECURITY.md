# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately via GitHub's **Report a vulnerability** flow:
**Security → Advisories → Report a vulnerability**
(`https://github.com/machinavitalis/jaxility/security/advisories/new`).

Include the affected version/commit, a minimal reproduction, and the impact you
observed. We aim to acknowledge within a few business days.

## Supported versions

Security fixes target the latest `main` and the most recent tagged release.

## Scope notes for users

Jaxility is a **compiler and deployment tool** — it ingests models and emits C
projects, then can cross-compile and run them. Some behaviors are powerful by
design; know what you're running:

- **Untrusted models / inputs.** Lowering ingests a model and generates code
  from it. Only compile models you trust — treat a model file like source code.
- **Generated artifacts execute.** The emitted C, the cross-compiled binaries,
  and the on-target runtime run real code on your machine/board. Review and run
  generated artifacts only from inputs you trust.
- **Toolchain integrity.** Cross-compilation pulls a pinned Arm toolchain and
  builds third-party deps (acados/blasfeo/hpipm) from source; their hashes are
  recorded in the manifest. Verify the manifest chain (`jaxility verify`) before
  trusting an artifact's provenance.

If you can escalate beyond these documented behaviors (e.g. code execution from
a path documented as safe, or forging a manifest that `verify` accepts), that
*is* a vulnerability — please report it.
