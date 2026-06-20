"""Expose the PTP MCP tools to the Claude (Anthropic) agent SDK (A4).

The Phase 0/1 prototype already drives LangChain self-selection from the
``PTP_TOOLS`` descriptors (`mcp/langchain_tools.py`). This adapter does the same
for the Anthropic tool-use API, so the **identical** self-selection-optimized
descriptions drive a second framework — closing the Phase 1 DoD requirement that
the MCP wrapper be tested against two agent frameworks.

The Anthropic tool schema differs from MCP only in the key name: tools are
``{"name", "description", "input_schema"}`` (snake case) rather than
``inputSchema``. Building the tool list is pure dict-shaping, so this module
imports without the ``anthropic`` SDK installed; only an optional live test
needs the SDK + an API key.
"""

from __future__ import annotations

from typing import Any

from .server import PTP_TOOLS, PTPToolHandler


def build_anthropic_tools() -> list[dict[str, Any]]:
    """Return ``PTP_TOOLS`` in the Anthropic tool-use schema.

    The ``name`` and ``description`` are carried over verbatim (they drive tool
    self-selection); ``inputSchema`` is renamed to ``input_schema``.
    """
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["inputSchema"],
        }
        for t in PTP_TOOLS
    ]


def dispatch_tool_use(handler: PTPToolHandler, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Execute a Claude ``tool_use`` block against the PTP store.

    Returns the handler's result dict; a caller wraps it in a ``tool_result``
    content block to feed back to the model. Auth failures surface as the
    handler's ``{"status": 403, ...}`` payload (it maps ``AuthError`` itself).
    """
    return handler.call(name, tool_input)
