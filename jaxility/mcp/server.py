# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors

"""FastMCP server exposing Jaxility as agent-callable tools.

Initial scaffold: a single ``ping`` health-check tool, enough to prove an
MCP-aware client (Claude Code, Cursor) can launch the server and enumerate a
tool. The build / verify / bench / hil tools advertised in
``.claude/skills/jaxility/SKILL.md`` are registered here later
(T-015 onward — once their CLIs exist; the MCP server wraps the CLI per
ADR-013). Launch with ``jaxility mcp serve`` (stdio transport).

``fastmcp`` is an optional dependency (the ``[mcp]`` extra); importing this
module requires it. The ``jaxility.mcp`` package ``__init__`` does not import
this module, so the rest of the package imports without ``fastmcp`` installed.
"""

from __future__ import annotations

from fastmcp import FastMCP

SERVER_NAME = "jaxility"


def ping() -> str:
    """Health check: confirm the Jaxility MCP server is reachable.

    Returns
    -------
    str
        Always ``"pong"``.

    Examples
    --------
    >>> ping()
    'pong'
    """
    return "pong"


# Source-of-truth registry of the tools exposed on the server. Later phases
# append build / verify / bench / hil / list_targets / coverage here as the
# corresponding CLI subcommands land (ADR-013).
TOOLS = (ping,)


def build_server(name: str = SERVER_NAME) -> FastMCP:
    """Construct the Jaxility FastMCP server with every tool in ``TOOLS``.

    Args
    ----
    name : str
        The server name advertised to MCP clients. Defaults to ``"jaxility"``.

    Returns
    -------
    FastMCP
        A configured server, ready to ``.run()`` or to introspect.
    """
    mcp = FastMCP(name)
    for fn in TOOLS:
        mcp.tool(fn)
    return mcp


def serve() -> None:
    """Run the Jaxility MCP server on stdio (blocking).

    stdio is FastMCP's default transport — the one MCP clients spawn as a
    subprocess. Invoked by ``jaxility mcp serve``.
    """
    build_server().run()
