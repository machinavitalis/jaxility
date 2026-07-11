# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Top-level CLI.

The CLI is the canonical surface (ADR-013). The Python API exists and is
documented but the docs lead with the CLI. The ``jaxility`` console entry
point is wired in pyproject.toml ``[project.scripts]`` and dispatches
through :func:`main` below.

Live subcommands:

* ``jaxility mcp serve`` — launch the FastMCP server (T-004).
* ``jaxility verify <manifest>`` — verify an attestation manifest (T-012).
* ``jaxility coverage [--target <family>]`` — print the coverage matrix
  as Markdown (T-013).

The substantive build / bench / hil / targets subcommands listed in
``AGENTS/CONTEXT.md`` arrive alongside the capabilities they expose.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the ``jaxility`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="jaxility",
        description=(
            "Jaxility — open-source compiler from JAX-trained robotics to Arm "
            "silicon, with signed attestation manifests."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mcp = sub.add_parser("mcp", help="Model Context Protocol server commands.")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_sub.add_parser("serve", help="Run the Jaxility MCP server on stdio.")

    verify = sub.add_parser(
        "verify",
        help=(
            "Verify a Jaxility attestation manifest against schema v0. "
            "Emits a structured ChainReport on stdout (PATTERNS §8.2)."
        ),
    )
    verify.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest JSON file to verify.",
    )
    verify.add_argument(
        "--expected-hash",
        dest="expected_hash",
        default=None,
        help=(
            "Hex-encoded BLAKE3 digest that an out-of-band source claims "
            "this manifest produces (e.g. a benchmark-registry entry). "
            "When supplied, verification fails on mismatch — which is how "
            "tampering is detected at the OSS hash-chain level."
        ),
    )

    coverage = sub.add_parser(
        "coverage",
        help=(
            "Print the JAX-op coverage matrix as Markdown (invariant 7). "
            "Optionally filtered to a single target family."
        ),
    )
    coverage.add_argument(
        "--target",
        dest="target_family",
        default=None,
        help=(
            "Restrict the table to this target family "
            "(e.g. ``mock-cortex-a``, ``cortex-a76``). "
            "Default emits every family."
        ),
    )

    build = sub.add_parser(
        "build",
        help=(
            "Build a deployable artifact for a zoo entry on the chosen "
            "target. Currently ships the ``host`` target only; cross-compile "
            "targets land later."
        ),
    )
    build.add_argument(
        "zoo_name",
        help="Zoo entry name (see ``jaxility.zoo.available()``).",
    )
    build.add_argument(
        "--target",
        dest="target_name",
        default="host",
        help=(
            "Target identifier. Accepts ``host`` (auto-detects "
            "darwin/linux), ``host-darwin``, or ``host-linux``. "
            "Default ``host``."
        ),
    )
    build.add_argument(
        "--work-dir",
        dest="work_dir",
        default=None,
        help=(
            "Directory acados writes generated C and compiled binaries "
            "into. Defaults to ``~/.cache/jaxility/builds/<artifact-hash>/``."
        ),
    )

    bench = sub.add_parser(
        "bench",
        help=(
            "Benchmark a controller's per-cycle solve time, jitter, and "
            "memory on a target (T-035). Emits a BenchRecord as JSON."
        ),
    )
    bench.add_argument("robot", help="Robot to benchmark (currently ``cartpole``).")
    bench.add_argument(
        "--target",
        dest="target_name",
        default="host",
        help=(
            "``host`` (local subprocess; a smoke check) or ``pi5`` (build + "
            "run on a tethered Pi 5 via ``JAXILITY_HIL_SSH_HOST``). "
            "Default ``host``."
        ),
    )
    bench.add_argument("--cycles", dest="n_cycles", type=int, default=1000)
    bench.add_argument("--warmup", dest="n_warmup", type=int, default=100)
    bench.add_argument("--seed", dest="seed", type=int, default=0)
    bench.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Write the BenchRecord JSON here (default stdout).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``jaxility`` console script.

    Args
    ----
    argv : list[str] | None
        Argument vector excluding the program name. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code (PATTERNS §8.3): ``0`` success, ``2`` invalid
        arguments, ``40`` manifest error.
    """
    args = build_parser().parse_args(argv)

    if args.command == "mcp" and args.mcp_command == "serve":
        from ..mcp.server import serve

        serve()
        return 0

    if args.command == "verify":
        from ..manifest.verify import verify_cli

        return verify_cli(args.manifest, args.expected_hash)

    if args.command == "coverage":
        from ..lowering.coverage import coverage_markdown

        print(coverage_markdown(args.target_family), end="")
        return 0

    if args.command == "build":
        from .build_cmd import run_build

        return run_build(
            zoo_name=args.zoo_name,
            target_name=args.target_name,
            work_dir=args.work_dir,
        )

    if args.command == "bench":
        from .bench_cmd import run_bench

        return run_bench(
            robot=args.robot,
            target_name=args.target_name,
            n_cycles=args.n_cycles,
            n_warmup=args.n_warmup,
            seed=args.seed,
            out=args.out,
        )

    # argparse's required subparsers make this unreachable today, but a
    # defined return keeps the type-checker happy and survives future
    # commands.
    return 2
