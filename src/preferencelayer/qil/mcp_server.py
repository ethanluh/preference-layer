"""QIL MCP server: exposes quality intelligence as agent tools.

Mirrors the PTP MCP server (``mcp/server.py``). An agent calls ``get_quality``
to retrieve use-profile-conditioned quality posteriors before recommending a
product, and ``compare_quality`` to choose between two candidates. The tool
*logic* lives in :class:`QILToolHandler`, testable without the MCP SDK;
:func:`build_server` wires it into a live ``mcp.server.Server`` when ``mcp`` is
installed.
"""

from __future__ import annotations

from typing import Any

from .query import QualityService

QIL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_quality",
        "description": (
            "Retrieve use-profile-conditioned quality intelligence for a product: "
            "per-dimension posterior quality (mean + 90% credible interval), an "
            "estimated failure rate, and how much evidence backs it. Call this when "
            "evaluating whether a specific product is good FOR THIS USER'S KIND OF "
            "USE — population star ratings do not answer that. Provide the user's "
            "use_profile (e.g. 'gaming', 'travel', 'professional')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "use_profile": {"type": "string", "description": "How the user will use it, e.g. 'heavy_use'."},
                "dimensions": {"type": "array", "items": {"type": "string"},
                               "description": "Optional subset of quality dimensions."},
            },
            "required": ["product_id", "use_profile"],
        },
    },
    {
        "name": "compare_quality",
        "description": (
            "Compare two products on use-profile-conditioned quality. Returns, per "
            "dimension, the posterior quality difference and the probability that A "
            "is better than B for the given use_profile. Call this to choose between "
            "two shortlisted products rather than relying on aggregate ratings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_id_a": {"type": "string"},
                "product_id_b": {"type": "string"},
                "use_profile": {"type": "string"},
            },
            "required": ["product_id_a", "product_id_b", "use_profile"],
        },
    },
]


class QILToolHandler:
    """Dispatches QIL MCP tool calls to a :class:`QualityService`."""

    def __init__(self, service: QualityService):
        self.service = service

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_quality":
            return self.service.quality(
                product_id=arguments["product_id"],
                use_profile=arguments["use_profile"],
                dimensions=arguments.get("dimensions"),
            )
        if name == "compare_quality":
            return self.service.compare(
                product_id_a=arguments["product_id_a"],
                product_id_b=arguments["product_id_b"],
                use_profile=arguments["use_profile"],
            )
        return {"status": 400, "detail": f"unknown tool '{name}'"}


def build_server(service: QualityService):  # pragma: no cover - requires mcp SDK
    """Construct a live MCP ``Server`` exposing the QIL tools (needs ``mcp`` extra)."""
    import json

    from mcp.server import Server
    from mcp.types import TextContent, Tool

    handler = QILToolHandler(service)
    server = Server("preferencelayer-qil")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in QIL_TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = handler.call(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server
