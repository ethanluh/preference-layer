import copy

from preferencelayer.ptp import (
    AttributeNode,
    Edge,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)


def _make_cred():
    sk, did = new_user_keypair(seed=b"0" * 32)
    graph = PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", 0.8, 0.7),
            AttributeNode("portability", 0.6, 0.2),
            AttributeNode("price_sensitivity", -0.3, 0.6),
        ],
        edges=[Edge("performance", "portability", -0.4)],
    )
    return sk, did, PreferenceCredential(did, graph)


def test_sign_and_verify_roundtrip():
    sk, did, cred = _make_cred()
    cred.sign(sk)
    assert cred.verify(sk.verify_key)
    assert did.startswith("did:key:z")


def test_tamper_detected():
    sk, _, cred = _make_cred()
    cred.sign(sk)
    # Mutate a weight after signing -> signature must fail.
    cred.graph.attributeNodes[0].weight = 0.1
    assert not cred.verify(sk.verify_key)


def test_serialization_roundtrip_preserves_validity():
    sk, _, cred = _make_cred()
    cred.sign(sk)
    doc = cred.to_dict()
    reloaded = PreferenceCredential.from_dict(copy.deepcopy(doc))
    assert reloaded.verify(sk.verify_key)
    assert reloaded.graph.category == "laptops"
    assert len(reloaded.graph.attributeNodes) == 3


def test_selective_disclosure_redacts_nodes_and_edges():
    sk, _, cred = _make_cred()
    cred.sign(sk)
    scoped = cred.scoped(disclosure_scope=["performance"], min_confidence=0.0)
    ids = [n.id for n in scoped.graph.attributeNodes]
    assert ids == ["performance"]
    # The edge referenced portability, which is now redacted.
    assert scoped.graph.edges == []


def test_min_confidence_filter():
    sk, _, cred = _make_cred()
    cred.sign(sk)
    scoped = cred.scoped(min_confidence=0.5)
    ids = sorted(n.id for n in scoped.graph.attributeNodes)
    assert ids == ["performance", "price_sensitivity"]  # portability (0.2) dropped


def test_context_includes_required_uris():
    sk, _, cred = _make_cred()
    doc = cred.sign(sk).to_dict()
    assert "https://www.w3.org/ns/credentials/v2" in doc["@context"]
    assert doc["type"] == ["VerifiableCredential", "PreferenceCredential"]
