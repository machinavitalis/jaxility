# Contributing to Jaxility

Thanks for your interest in contributing. Jaxility is the deployment compiler for
the stack — it takes a calibrated robot (from [Jaxterity](https://github.com/machinavitalis/jaxterity),
on [Jaxonomy](https://github.com/machinavitalis/jaxonomy)) to an Arm-SoC C
project with a signed attestation manifest (lane: JAX → CasADi → acados →
embedded C). Released under the [MIT License](LICENSE.md); by contributing you
agree your contributions are licensed under the same terms.

## Development setup

Jaxility requires **Python 3.10+**. It sits at the bottom of the stack and is
installed against sibling checkouts of jaxterity/jaxonomy:

```bash
git clone https://github.com/machinavitalis/jaxility.git
cd jaxility
python -m pip install --upgrade pip
# install the upstream stack editable first, then jaxility with --no-deps
pip install -e ../jaxonomy --no-deps
pip install -e ../jaxterity --no-deps
pip install -e . --no-deps
pip install "pytest>=8.0"   # plus other test deps as needed
```

The `--no-deps` pattern is intentional: it keeps the editable upstream checkouts
authoritative instead of pulling published wheels over them.

## Running the tests

```bash
pytest test/ -q
```

Tests that need the upstream layer gate on `pytest.importorskip("jaxterity")`,
so a partial install degrades gracefully rather than erroring at collection.
The cross-compile / on-silicon (HIL) paths are operator-driven and not part of
the default per-PR gate.

## What we value in a change

- **Lowering coverage is explicit.** Unsupported ops raise a structured error
  with a documented workaround — never add a silent fallback.
- **Reproducibility.** Manifests are canonical-JSON, BLAKE3 hash-chained, and
  byte-identical for identical inputs; don't introduce nondeterminism into the
  artifact or manifest path.
- **Claims stay scoped.** On-silicon results are tied to the specific target
  that ran them; keep `CLAIMS.md` / `KNOWN_GAPS.md` honest about what has and
  hasn't touched hardware.
- **Tests + docstrings** ship with the change; user-visible changes get a
  `CHANGELOG` entry.

The `AGENTS/` directory holds the architecture, decision records, and toolchain
notes — skim the relevant ones before a non-trivial change.

## Pull request process

1. Branch from `main`; keep the change focused.
2. Ensure `pytest test/ -q` passes and CI is green.
3. Describe the change and its motivation; link any related issue.

## Reporting bugs & security issues

Open a GitHub issue for bugs. For anything with security impact, **do not open a
public issue** — follow [SECURITY.md](SECURITY.md).
