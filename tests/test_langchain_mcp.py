"""LangChain integration + live MCP smoke tests for the PTP tools (A4).

Two things are proven here:

1. **Self-selection in LangChain.** The PTP tools are exposed as real LangChain
   ``StructuredTool`` objects (same descriptions as the MCP descriptors). An
   *unprompted* agent — modeled by a deterministic description-driven selector,
   so CI needs no LLM/network — picks the right tool for the rank /
   post-purchase / low-confidence situations, then the selected tool executes
   end-to-end through the real ``PTPToolHandler`` (auth + selective disclosure +
   signing). The selector reads ONLY the tool descriptions, so a correct pick is
   evidence the descriptions are discriminative.

2. **Live MCP server.** When the optional ``mcp`` SDK is installed, the real
   ``build_server`` is exercised: its ``list_tools`` advertises the three PTP
   tools and ``call_tool`` round-trips a ``get_preference`` call returning a
   signed credential.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from preferencelayer.mcp.server import PTP_TOOLS, PTPToolHandler
from preferencelayer.ptp.credential import (
    AttributeNode,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.store import CredentialStore


def _store_and_token(confidence: float = 0.8):
    sk, did = new_user_keypair(seed=b"3" * 32)
    store = CredentialStore(sk, did)
    store.put_credential(PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", 0.8, confidence),
            AttributeNode("portability", 0.6, confidence),
        ],
    )))
    token = store.authorize_agent("agent.shop", scope=["laptops"])
    return sk, store, token


# --------------------------------------------------------------------------- #
# Deterministic, offline model of an unprompted agent's tool self-selection.    #
# Scores each tool by keyword overlap between the situation and the tool's       #
# DESCRIPTION ONLY (never its name), mirroring how an LLM routes on descriptions.#
# --------------------------------------------------------------------------- #
def _select_tool(situation: str, tools: list) -> str:
    words = {w.strip(".,;:'\"()").lower() for w in situation.split()}
    best, best_score = None, -1
    for tool in tools:
        desc = tool.description.lower()
        score = sum(1 for w in words if len(w) > 3 and w in desc)
        if score > best_score:
            best, best_score = tool.name, score
    return best


@pytest.fixture
def langchain_tools():
    pytest.importorskip("langchain_core")
    from preferencelayer.mcp.langchain_tools import build_langchain_tools

    _, store, token = _store_and_token()
    return build_langchain_tools(PTPToolHandler(store, token))


def test_langchain_tools_built_from_descriptors(langchain_tools):
    names = {t.name for t in langchain_tools}
    assert names == {t["name"] for t in PTP_TOOLS}
    # Descriptions carried over verbatim (they drive self-selection).
    for t in langchain_tools:
        descriptor = next(d for d in PTP_TOOLS if d["name"] == t.name)
        assert t.description == descriptor["description"]


@pytest.mark.parametrize("situation,expected", [
    ("Rank these laptops and recommend the best one for the user before showing results",
     "get_preference"),
    ("The user just completed a purchase; record this transaction to update their model",
     "submit_outcome"),
    ("Preference confidence is low; ask the user a few clarifying questions to improve it",
     "request_elicitation"),
])
def test_unprompted_agent_selects_right_tool(langchain_tools, situation, expected):
    assert _select_tool(situation, langchain_tools) == expected


def test_selected_tool_executes_end_to_end(langchain_tools):
    """The 'rank' situation -> get_preference -> a real signed credential comes back."""
    name = _select_tool("rank and recommend products for the user", langchain_tools)
    tool = next(t for t in langchain_tools if t.name == name)
    out = json.loads(tool.invoke({"category": "laptops"}))
    assert out["status"] == 200
    assert set(out["coverage"]) == {"performance", "portability"}


def test_langchain_tool_respects_auth_boundary():
    pytest.importorskip("langchain_core")
    from preferencelayer.mcp.langchain_tools import build_langchain_tools

    _, store, token = _store_and_token()
    store.revoke_agent("agent.shop")  # token no longer valid
    tools = build_langchain_tools(PTPToolHandler(store, token))
    tool = next(t for t in tools if t.name == "get_preference")
    out = json.loads(tool.invoke({"category": "laptops"}))
    assert out["status"] == 403  # PTPToolHandler maps AuthError -> 403


# --------------------------------------------------------------------------- #
# Live MCP server smoke test (requires the optional `mcp` SDK).                  #
# --------------------------------------------------------------------------- #
def test_live_mcp_server_lists_and_calls_tools():
    pytest.importorskip("mcp")
    from preferencelayer.mcp.server import build_server

    sk, store, token = _store_and_token()
    server = build_server(store, token)

    async def _run():
        # The decorators register handlers on the Server; resolve and invoke them
        # the way the MCP runtime would.
        tools = await _invoke_list_tools(server)
        names = {t.name for t in tools}
        assert names == {"get_preference", "submit_outcome", "request_elicitation"}
        # call_tool -> get_preference
        contents = await _invoke_call_tool(server, "get_preference", {"category": "laptops"})
        payload = json.loads(contents[0].text)
        assert payload["status"] == 200
        cred = PreferenceCredential.from_dict(payload["credential"])
        assert cred.verify(sk.verify_key)

    asyncio.run(_run())


async def _invoke_list_tools(server):
    """Call the registered list_tools handler across MCP SDK versions."""
    from mcp import types

    handler = server.request_handlers.get(types.ListToolsRequest)
    assert handler is not None, "server did not register a list_tools handler"
    result = await handler(types.ListToolsRequest(method="tools/list"))
    return result.root.tools if hasattr(result, "root") else result.tools


async def _invoke_call_tool(server, name, arguments):
    from mcp import types

    handler = server.request_handlers.get(types.CallToolRequest)
    assert handler is not None, "server did not register a call_tool handler"
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await handler(req)
    return result.root.content if hasattr(result, "root") else result.content
