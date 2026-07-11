# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""Tests for the MCP server scaffold (T-004).

The server requires the optional ``[mcp]`` extra (``fastmcp``); when it is
not installed these tests skip. The ``[dev]`` extra pulls ``fastmcp`` so the
local preflight and (future) CI matrix exercise them.
"""

import asyncio

import pytest

pytest.importorskip("fastmcp")

from jaxility.cli import build_parser  # noqa: E402
from jaxility.mcp import server  # noqa: E402


@pytest.mark.unit
def test_ping_returns_pong() -> None:
    """The ``ping`` health-check tool returns ``"pong"``."""
    assert server.ping() == "pong"


@pytest.mark.unit
def test_server_builds_and_enumerates_tools() -> None:
    """The server constructs and exposes a non-empty tool list including ping."""
    mcp = server.build_server()
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) >= 1
    assert "ping" in {t.name for t in tools}


@pytest.mark.unit
def test_tools_registry_non_empty() -> None:
    """The source-of-truth ``TOOLS`` registry is non-empty (regression guard)."""
    assert len(server.TOOLS) >= 1


@pytest.mark.unit
def test_cli_exposes_mcp_serve() -> None:
    """The ``jaxility mcp serve`` command parses (the CLI wiring exists)."""
    args = build_parser().parse_args(["mcp", "serve"])
    assert args.command == "mcp"
    assert args.mcp_command == "serve"
