"""Expose the PTP MCP tools as LangChain tools.

A real LangChain agent does not know about :class:`PTPToolHandler`; it is handed a
list of tools and selects among them from their names/descriptions. This adapter
turns ``PTP_TOOLS`` + a :class:`PTPToolHandler` into LangChain ``StructuredTool``
objects whose ``name``/``description`` are exactly the MCP descriptors, so the
same self-selection-optimized descriptions drive both frameworks.

Requires the optional ``langchain-core`` dependency; importing this module
without it raises a clear error (the rest of the package is unaffected).
"""

from __future__ import annotations

import json
from typing import Any

from .server import PTP_TOOLS, PTPToolHandler

try:
    from langchain_core.tools import StructuredTool
    from pydantic import create_model
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "LangChain integration requires 'langchain-core': pip install 'preferencelayer[langchain]'"
    ) from exc

_JSON_TYPE_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model(tool_name: str, input_schema: dict):
    """Build a pydantic args model from an MCP JSON inputSchema.

    Required properties are required fields; everything else is optional. This is
    what lets LangChain pass ``category=...`` etc. through to the handler rather
    than inferring an empty signature.
    """
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop, spec in props.items():
        py = _JSON_TYPE_TO_PY.get(spec.get("type", "string"), str)
        if prop in required:
            fields[prop] = (py, ...)
        else:
            fields[prop] = (py | None, None)
    return create_model(f"{tool_name}_Args", **fields)


def build_langchain_tools(handler: PTPToolHandler) -> list["StructuredTool"]:
    """Build LangChain ``StructuredTool``s bound to a PTP tool handler.

    Each tool carries the MCP descriptor's name, description, and a pydantic args
    schema derived from its ``inputSchema``; the ``func`` dispatches through
    ``handler.call`` and returns the result dict serialized to JSON (LangChain
    tools return strings).
    """
    tools: list[StructuredTool] = []
    for descriptor in PTP_TOOLS:
        name = descriptor["name"]

        def _make(tool_name: str):
            def _call(**kwargs: Any) -> str:
                # Drop unset optional args so the handler sees a clean dict.
                args = {k: v for k, v in kwargs.items() if v is not None}
                return json.dumps(handler.call(tool_name, args))
            return _call

        tools.append(
            StructuredTool.from_function(
                func=_make(name),
                name=name,
                description=descriptor["description"],
                args_schema=_args_model(name, descriptor["inputSchema"]),
            )
        )
    return tools
