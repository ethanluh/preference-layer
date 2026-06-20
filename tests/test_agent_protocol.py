"""Tests for the protocol-level agent (ranks via the PTP + QIL MCP tool handlers)."""

import numpy as np

from preferencelayer.agent import combine
from preferencelayer.agent.protocol import (
    ProtocolAgent,
    credential_from_arrays,
    quality_from_response,
    score_from_credential,
)
from preferencelayer.attributes import AttributeSchema
from preferencelayer.data import integrated
from preferencelayer.eval import metrics
from preferencelayer.mcp.server import PTPToolHandler
from preferencelayer.ptp.credential import (
    AttributeNode,
    Edge,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.store import CredentialStore
from preferencelayer.qil.aggregate import QualityAggregator
from preferencelayer.qil.extract import ExtractedSignal
from preferencelayer.qil.mcp_server import QILToolHandler
from preferencelayer.qil.query import QualityService

SCHEMA = AttributeSchema.for_category("laptops")


def _col(name):
    return SCHEMA.index(name)


def _attrs(**vals):
    """Build a candidate attribute row with named shared attributes set."""
    x = np.zeros(SCHEMA.dim)
    for name, v in vals.items():
        x[_col(name)] = v
    return x


# --------------------------------------------------------- score_from_credential
def test_score_uses_node_weights():
    cred = PreferenceCredential("did:key:zTest", PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", weight=1.0, confidence=0.8)],
    ))
    attrs = np.stack([_attrs(performance=0.9), _attrs(performance=0.1)])
    scores = score_from_credential(cred, attrs, SCHEMA)
    assert scores[0] > scores[1]


def test_score_skips_unknown_attribute_ids():
    cred = PreferenceCredential("did:key:zTest", PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", 1.0, 0.8), AttributeNode("not_a_real_attr", 5.0, 0.8)],
    ))
    attrs = np.stack([_attrs(performance=1.0), _attrs(performance=0.0)])
    scores = score_from_credential(cred, attrs, SCHEMA)
    # The bogus node is ignored; ranking is driven purely by 'performance'.
    assert scores[0] > scores[1]


def test_score_includes_edge_interaction():
    cred = PreferenceCredential("did:key:zTest", PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", 0.0, 0.8), AttributeNode("portability", 0.0, 0.8)],
        edges=[Edge("performance", "portability", weight=1.0)],
    ))
    attrs = np.stack([_attrs(performance=1.0, portability=1.0), _attrs(performance=1.0, portability=0.0)])
    scores = score_from_credential(cred, attrs, SCHEMA)
    assert scores[0] > scores[1]   # only the first has the interaction active


# ------------------------------------------------------------ quality_from_response
def test_quality_from_response_paths():
    ok = {"status": 200, "dimensions": {"thermal": {"posterior_mean": 0.8}, "display": {"posterior_mean": 0.6}}}
    assert quality_from_response(ok) == 0.7
    assert quality_from_response({"status": 404}) == 0.5          # neutral fallback
    with_fail = {"status": 200, "dimensions": {"thermal": {"posterior_mean": 0.8}}, "failure_rate": 0.4}
    assert quality_from_response(with_fail, failure_penalty=1.0) < 0.8


# ---------------------------------------------------------------- end-to-end agent
def _qil_handler():
    sigs = []
    for pid, mean in (("good", 0.85), ("bad", 0.25)):
        for _ in range(12):
            for dim in ("thermal", "build_quality"):
                sigs.append(ExtractedSignal(pid, "laptops", "gaming", "performance", None, dim, mean, 0.9))
    return QILToolHandler(QualityService(QualityAggregator().fit(sigs)))


def _store_with_credential(confidence=0.8):
    sk, did = new_user_keypair()
    store = CredentialStore(sk, did)
    cred = PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", 0.9, confidence), AttributeNode("portability", 0.3, confidence)],
    ))
    store.put_credential(cred)
    return store, did


def test_protocol_agent_end_to_end():
    store, _ = _store_with_credential(confidence=0.8)
    token = store.authorize_agent("agent.shop", scope=["laptops"])
    agent = ProtocolAgent(PTPToolHandler(store, token), _qil_handler(), SCHEMA)
    cand_ids = ["good", "bad"]
    attrs = np.stack([_attrs(performance=0.8), _attrs(performance=0.8)])  # equal preference
    rec = agent.recommend("laptops", "gaming", cand_ids, attrs)
    assert rec.status == 200
    assert rec.pref.shape == (2,) and rec.blended.shape == (2,)
    # Preference is tied; quality breaks it, so 'good' ranks first.
    assert rec.order[0] == 0
    # Alpha is the documented function of the credential's disclosed confidence.
    assert np.isclose(rec.alpha, combine.alpha_from_confidence(rec.confidence))
    assert set(rec.coverage) == {"performance", "portability"}


def test_protocol_agent_denied_after_revocation():
    store, _ = _store_with_credential()
    token = store.authorize_agent("agent.shop", scope=["laptops"])
    store.revoke_agent("agent.shop")
    agent = ProtocolAgent(PTPToolHandler(store, token), _qil_handler(), SCHEMA)
    rec = agent.recommend("laptops", "gaming", ["good", "bad"],
                          np.stack([_attrs(performance=0.5), _attrs(performance=0.5)]))
    assert rec.status == 403
    assert rec.order == []          # no ranking without an authorized preference


def test_credential_roundtrip_ranks_relevant_set():
    """A credential built from a user's preference, served over PTP + QIL, ranks well."""
    s = integrated.generate(n_users=48, seed=23)
    idx = s.product_index()
    qil = QILToolHandler(QualityService(QualityAggregator().fit(s.signals)))
    blend, pref_only = [], []
    for ui, u in enumerate(s.users):
        sk, did = new_user_keypair()
        store = CredentialStore(sk, did)
        store.put_credential(credential_from_arrays(
            s.schema, s.theta[ui], s.phi_pairs, s.phi[ui],
            category="laptops", issuer_did=did, node_confidence=u.mean_confidence))
        token = store.authorize_agent("a", scope=["laptops"])
        agent = ProtocolAgent(PTPToolHandler(store, token), qil, s.schema)
        cand_attrs = np.stack([idx[c].attributes for c in u.candidates])
        rec = agent.recommend("laptops", u.use_profile, u.candidates, cand_attrs)
        rel = set(u.relevant)
        blend.append(metrics.ndcg_at_k([u.candidates[i] for i in rec.order], rel, 10))
        pref_only.append(metrics.ndcg_at_k([u.candidates[i] for i in np.argsort(-rec.pref)], rel, 10))
    # Well above the ~0.18 random floor, and quality adds to preference alone.
    assert np.mean(blend) > 0.5
    assert np.mean(blend) >= np.mean(pref_only)


# ------------------------------------------------ ProtocolAgent over LangChain tools
import json  # noqa: E402

import pytest  # noqa: E402


class _LangChainPTPShim:
    """A ``.call``-compatible adapter that routes through LangChain PTP tools.

    Lets the existing :class:`ProtocolAgent` (which expects ``call(name, args) ->
    dict``) drive the credential store *through* the LangChain ``StructuredTool``
    layer, so the same agent path is exercised across frameworks. The tools return
    JSON strings (LangChain contract); this shim invokes the matching tool and
    parses the result back to a dict.
    """

    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def call(self, name: str, arguments: dict) -> dict:
        args = {k: v for k, v in (arguments or {}).items() if v is not None}
        return json.loads(self._tools[name].invoke(args))


def test_protocol_agent_over_langchain_tools_round_trip():
    pytest.importorskip("langchain_core")
    from preferencelayer.mcp.langchain_tools import build_langchain_tools

    store, _ = _store_with_credential(confidence=0.8)
    token = store.authorize_agent("agent.shop", scope=["laptops"])
    ptp_shim = _LangChainPTPShim(build_langchain_tools(PTPToolHandler(store, token)))

    agent = ProtocolAgent(ptp_shim, _qil_handler(), SCHEMA)
    attrs = np.stack([_attrs(performance=0.8), _attrs(performance=0.8)])
    rec = agent.recommend("laptops", "gaming", ["good", "bad"], attrs)
    assert rec.status == 200
    assert rec.order[0] == 0  # quality breaks the preference tie, same as the direct path
    assert set(rec.coverage) == {"performance", "portability"}


def test_protocol_agent_over_langchain_tools_denied_after_revocation():
    pytest.importorskip("langchain_core")
    from preferencelayer.mcp.langchain_tools import build_langchain_tools

    store, _ = _store_with_credential()
    token = store.authorize_agent("agent.shop", scope=["laptops"])
    store.revoke_agent("agent.shop")
    ptp_shim = _LangChainPTPShim(build_langchain_tools(PTPToolHandler(store, token)))
    agent = ProtocolAgent(ptp_shim, _qil_handler(), SCHEMA)
    rec = agent.recommend("laptops", "gaming", ["good", "bad"],
                          np.stack([_attrs(performance=0.5), _attrs(performance=0.5)]))
    assert rec.status == 403
    assert rec.order == []
