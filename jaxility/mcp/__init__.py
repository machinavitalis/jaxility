# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxility Authors
"""FastMCP server exposing build / verify / bench / hil as agent tools.

The MCP server wraps the CLI, not the Python API (ADR-013). Stable
``@<type>[<hash>]`` handles follow the Jaxterity pattern; the source
robot's ``attestation_handle`` becomes ``@robot[<hash>]`` at the MCP
boundary. Server scaffold + a trivial ``ping`` tool land in T-004.
"""
