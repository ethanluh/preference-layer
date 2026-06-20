import tempfile
from pathlib import Path

import pytest

from preferencelayer.ptp import (
    AttributeNode,
    AuthError,
    CredentialStore,
    DPConfig,
    PreferenceCredential,
    PreferenceGraph,
    context_to_nodes,
    new_user_keypair,
)


def _store(dp: DPConfig | None = None):
    sk, did = new_user_keypair(seed=b"2" * 32)
    g = PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", 0.5, 0.6),
            AttributeNode("portability", 0.4, 0.2),
            AttributeNode("price_sensitivity", -0.2, 0.3),
        ],
    )
    store = CredentialStore(sk, did, dp=dp)
    store.put_credential(PreferenceCredential(did, g))
    return sk, did, store


def test_authorize_and_scope_enforced():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    assert store.get_preference(token, "laptops")["status"] == 200
    with pytest.raises(AuthError):
        store.get_preference(token, "headphones")


def test_revocation_invalidates_token():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    assert store.revoke_agent("agent.a") == 1
    with pytest.raises(AuthError):
        store.get_preference(token, "laptops")


def test_get_preference_scoped_and_signed():
    sk, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    res = store.get_preference(token, "laptops", disclosure_scope=["performance"], min_confidence=0.0)
    assert res["coverage"] == ["performance"]
    cred = PreferenceCredential.from_dict(res["credential"])
    assert cred.verify(sk.verify_key)  # returned credential is validly re-signed


def test_submit_outcome_updates_and_resigns():
    sk, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    res = store.submit_outcome(token, "laptops", "thinkpad", "purchase",
                               use_context="sustained compute, gaming")
    assert res["status"] == 202
    assert "performance" in res["affected_nodes"]
    # Credential remains valid after the in-place update + re-sign.
    after = store.get_preference(token, "laptops")
    assert PreferenceCredential.from_dict(after["credential"]).verify(sk.verify_key)


def test_submit_outcome_budget_exhaustion_returns_429():
    sk, _, store = _store(dp=DPConfig(epsilon=2.0, budget_max=3.0))
    token = store.authorize_agent("agent.a", scope=["laptops"])
    first = store.submit_outcome(token, "laptops", "thinkpad", "purchase",
                                 use_context="sustained compute")
    assert first["status"] == 202
    # Second update would exceed budget_max (2 + 2 > 3): refused, not a 500.
    second = store.submit_outcome(token, "laptops", "thinkpad", "purchase",
                                  use_context="sustained compute")
    assert second["status"] == 429
    assert second["consent_required"] is True
    # Refusal left the credential valid and unchanged (still signed).
    assert PreferenceCredential.from_dict(
        store.get_preference(token, "laptops")["credential"]).verify(sk.verify_key)


def test_reset_budget_requires_consent():
    sk, _, store = _store(dp=DPConfig(epsilon=2.0, budget_max=3.0))
    token = store.authorize_agent("agent.a", scope=["laptops"])
    store.submit_outcome(token, "laptops", "thinkpad", "purchase", use_context="sustained compute")

    # Without consent: refused with 403, budget untouched.
    refused = store.reset_budget(token, "laptops")
    assert refused["status"] == 403 and refused["consent_required"] is True

    # With explicit consent: budget reset, credential re-signed, updates flow again.
    ok = store.reset_budget(token, "laptops", consent=True)
    assert ok["status"] == 200 and ok["privacy_budget_consumed"] == 0.0
    assert PreferenceCredential.from_dict(
        store.get_preference(token, "laptops")["credential"]).verify(sk.verify_key)
    assert store.submit_outcome(token, "laptops", "thinkpad", "purchase",
                                use_context="sustained compute")["status"] == 202


def test_elicit_orders_by_information_gain():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    res = store.elicit(token, "laptops", max_questions=3)
    igs = [q["information_gain"] for q in res["questions"]]
    assert igs == sorted(igs, reverse=True)
    # Lowest-confidence node (portability, conf 0.2 -> IG 0.8) should come first.
    assert res["questions"][0]["target_attribute"] == "portability"


def test_context_to_nodes_keyword_routing():
    nodes = context_to_nodes("frequent travel and commute", ["performance", "portability"])
    assert "portability" in nodes


def test_encrypted_persistence_roundtrip():
    sk, did, store = _store()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "store.cred"
        store.save(path)
        # Ciphertext must not contain plaintext category names.
        assert b"laptops" not in path.read_bytes()
        store2 = CredentialStore(sk, did)
        store2.load(path)
        assert store2.categories() == ["laptops"]
        assert PreferenceCredential.from_dict(
            store2.get_preference(store2.authorize_agent("x", ["laptops"]), "laptops")["credential"]
        ).verify(sk.verify_key)
