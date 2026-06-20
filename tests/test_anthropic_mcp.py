"""Claude (Anthropic) agent SDK integration for the PTP tools (A4).

Mirrors ``test_langchain_mcp.py`` for the second framework required by the
Phase 1 DoD. Proves:

1. **Schema shape.** ``PTP_TOOLS`` maps cleanly to the Anthropic tool-use schema
   (``input_schema``, not ``inputSchema``) with names/descriptions carried over
   verbatim.
2. **Self-selection.** An *unprompted* agent — modeled by the same deterministic,
   description-only selector used for LangChain, so CI needs no LLM/network —
   picks the right tool for rank / post-purchase / low-confidence situations.
3. **End-to-end dispatch.** The selected tool round-trips through the real
   ``PTPToolHandler`` (auth + selective disclosure + signing).
4. **Auth boundary.** A revoked token surfaces as a 403 result payload.
5. **(Optional) Live model.** When ``ANTHROPIC_API_KEY`` is set and the SDK is
   installed, the real Claude API is handed the tool list and must select
   ``get_preference`` for a ranking prompt.
"""

from __future__ import annotations

import os

import pytest

from preferencelayer.mcp.anthropic_tools import build_anthropic_tools, dispatch_tool_use
from preferencelayer.mcp.server import PTP_TOOLS, PTPToolHandler
from preferencelayer.ptp.credential import (
    AttributeNode,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.store import CredentialStore


def _store_and_token(confidence: float = 0.8):
    sk, did = new_user_keypair(seed=b"8" * 32)
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


# Same description-only selector as the LangChain test: scores each tool by
# keyword overlap with the situation, reading the DESCRIPTION only (never names).
def _select_tool(situation: str, tools: list[dict]) -> str:
    words = {w.strip(".,;:'\"()").lower() for w in situation.split()}
    best, best_score = None, -1
    for tool in tools:
        desc = tool["description"].lower()
        score = sum(1 for w in words if len(w) > 3 and w in desc)
        if score > best_score:
            best, best_score = tool["name"], score
    return best


def test_anthropic_tools_built_from_descriptors():
    tools = build_anthropic_tools()
    assert {t["name"] for t in tools} == {t["name"] for t in PTP_TOOLS}
    for t in tools:
        descriptor = next(d for d in PTP_TOOLS if d["name"] == t["name"])
        # Anthropic uses input_schema (snake case); description carried verbatim.
        assert set(t) == {"name", "description", "input_schema"}
        assert t["description"] == descriptor["description"]
        assert t["input_schema"] == descriptor["inputSchema"]


@pytest.mark.parametrize("situation,expected", [
    ("Rank these laptops and recommend the best one for the user before showing results",
     "get_preference"),
    ("The user just completed a purchase; record this transaction to update their model",
     "submit_outcome"),
    ("Preference confidence is low; ask the user a few clarifying questions to improve it",
     "request_elicitation"),
])
def test_unprompted_agent_selects_right_tool(situation, expected):
    assert _select_tool(situation, build_anthropic_tools()) == expected


def test_selected_tool_dispatches_end_to_end():
    sk, store, token = _store_and_token()
    handler = PTPToolHandler(store, token)
    tools = build_anthropic_tools()
    name = _select_tool("rank and recommend products for the user", tools)
    out = dispatch_tool_use(handler, name, {"category": "laptops"})
    assert out["status"] == 200
    assert set(out["coverage"]) == {"performance", "portability"}
    assert PreferenceCredential.from_dict(out["credential"]).verify(sk.verify_key)


def test_dispatch_respects_auth_boundary():
    _, store, token = _store_and_token()
    handler = PTPToolHandler(store, token)
    store.revoke_agent("agent.shop")  # token no longer valid
    out = dispatch_tool_use(handler, "get_preference", {"category": "laptops"})
    assert out["status"] == 403


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY; live test skipped")
def test_live_claude_selects_get_preference():  # pragma: no cover - network/LLM
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        tools=build_anthropic_tools(),
        messages=[{
            "role": "user",
            "content": "Rank these laptops for me and recommend the best one before showing results.",
        }],
    )
    used = [b.name for b in resp.content if getattr(b, "type", None) == "tool_use"]
    assert "get_preference" in used
