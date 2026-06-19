"""PTP MCP server (PTP spec §6).

Exposes the three PTP operations as MCP tools whose descriptions are written for
agent self-selection — an agent should reach for ``get_preference`` before
ranking products, ``submit_outcome`` after a transaction, and
``request_elicitation`` when confidence is low.

The tool *logic* lives in :class:`PTPToolHandler`, which wraps a
:class:`~preferencelayer.ptp.store.CredentialStore` and is fully testable without
the MCP SDK installed. :func:`build_server` wires the handler into a live
``mcp.server.Server`` when the optional ``mcp`` dependency is present.
"""

from __future__ import annotations

from typing import Any

from ..ptp.store import AuthError, CredentialStore

# Tool descriptors (also serialized into the MCP server's tool list). Descriptions
# are deliberately phrased to guide correct, unprompted tool selection by agents.
PTP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_preference",
        "description": (
            "Retrieve the user's portable preference credential for a product "
            "category. Call this BEFORE ranking, filtering, or recommending "
            "products so results reflect the user's known preferences across "
            "platforms. Returns a signed, query-scoped preference graph plus a "
            "confidence score; if confidence is low, follow up with "
            "request_elicitation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Product category, e.g. 'laptops'."},
                "query_context": {"type": "string", "description": "Free-text description of the current query/use case."},
                "disclosure_scope": {"type": "array", "items": {"type": "string"}, "description": "Optional attribute ids to limit disclosure."},
                "min_confidence": {"type": "number", "description": "Optional: only return nodes at/above this confidence."},
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_outcome",
        "description": (
            "Submit a purchase, return, dwell, or rating signal to update the "
            "user's preference model. Call this AFTER a transaction or a "
            "significant interaction so the credential improves over time. The "
            "update is differentially private and computed on-device."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "product_id": {"type": "string"},
                "outcome_type": {"type": "string", "enum": ["purchase", "return", "dwell", "rating", "elicitation"]},
                "use_context": {"type": "string"},
                "rating": {"type": "number", "description": "Only for outcome_type=rating, in [0,1]."},
            },
            "required": ["category", "product_id", "outcome_type"],
        },
    },
    {
        "name": "request_elicitation",
        "description": (
            "Request a short sequence of high-information-gain questions to raise "
            "preference confidence for weak attributes. Use when get_preference "
            "reports low confidence and the user is available for a brief "
            "interaction. Submit answers back via submit_outcome with "
            "outcome_type='elicitation'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "attribute_focus": {"type": "array", "items": {"type": "string"}},
                "max_questions": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["category"],
        },
    },
]


class PTPToolHandler:
    """Dispatches MCP tool calls to a credential store under one agent token."""

    def __init__(self, store: CredentialStore, agent_token: str):
        self.store = store
        self.token = agent_token

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "get_preference":
                return self.store.get_preference(
                    self.token,
                    category=arguments["category"],
                    query_context=arguments.get("query_context", ""),
                    disclosure_scope=arguments.get("disclosure_scope"),
                    min_confidence=arguments.get("min_confidence", 0.0),
                )
            if name == "submit_outcome":
                return self.store.submit_outcome(
                    self.token,
                    category=arguments["category"],
                    product_id=arguments["product_id"],
                    outcome_type=arguments["outcome_type"],
                    use_context=arguments.get("use_context", ""),
                    rating=arguments.get("rating"),
                    elicitation_weights=arguments.get("elicitation_weights"),
                )
            if name == "request_elicitation":
                return self.store.elicit(
                    self.token,
                    category=arguments["category"],
                    attribute_focus=arguments.get("attribute_focus"),
                    max_questions=arguments.get("max_questions", 3),
                )
            return {"status": 400, "detail": f"unknown tool '{name}'"}
        except AuthError as e:
            return {"status": 403, "detail": str(e)}


def build_server(store: CredentialStore, agent_token: str):  # pragma: no cover - requires mcp SDK
    """Construct a live MCP ``Server`` exposing the PTP tools.

    Requires the optional ``mcp`` dependency (``pip install preferencelayer[mcp]``).
    """
    import json

    from mcp.server import Server
    from mcp.types import TextContent, Tool

    handler = PTPToolHandler(store, agent_token)
    server = Server("preferencelayer-ptp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in PTP_TOOLS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = handler.call(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server
