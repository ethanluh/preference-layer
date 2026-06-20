"""Tests for the PTP HTTP transport (spec §4).

Covers the three endpoints, the auth boundary (401 missing/expired, 403 scope,
404 absent credential), credential re-signing across HTTP, and a p95 latency
check on /preference (< 100 ms target, kickoff A2). Skipped cleanly if the
optional 'http'/'dev' extra (FastAPI + httpx) is not installed.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from preferencelayer.http import build_app  # noqa: E402
from preferencelayer.ptp.credential import (  # noqa: E402
    AttributeNode,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.store import CredentialStore  # noqa: E402


def _store():
    sk, did = new_user_keypair(seed=b"4" * 32)
    store = CredentialStore(sk, did)
    store.put_credential(PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", 0.8, 0.7),
            AttributeNode("portability", 0.6, 0.5),
            AttributeNode("price_sensitivity", -0.3, 0.6),
        ],
    )))
    return sk, did, store


def _client(store):
    return TestClient(build_app(store))


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------ happy paths
def test_get_preference_ok_and_resigned():
    sk, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"}, headers=_auth(token))
    assert r.status_code == 200
    body = r.json()
    # The internal store "status" key must not leak into the response body (§4.2).
    assert "status" not in body
    assert set(body["coverage"]) == {"performance", "portability", "price_sensitivity"}
    # The returned credential is freshly re-signed and verifies under the user key.
    cred = PreferenceCredential.from_dict(body["credential"])
    assert cred.verify(sk.verify_key)


def test_get_preference_disclosure_scope():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    client = _client(store)
    r = client.request("GET", "/preference",
                       json={"category": "laptops", "disclosure_scope": ["performance"]},
                       headers=_auth(token))
    assert r.status_code == 200
    assert r.json()["coverage"] == ["performance"]


def test_post_outcome_updates_and_resigns_over_http():
    sk, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    client = _client(store)
    before = store._creds["laptops"].graph.updateCount
    r = client.post("/outcome", headers=_auth(token), json={
        "category": "laptops", "product_id": "thinkpad", "outcome_type": "purchase",
        "use_context": "sustained compute, gaming", "timestamp": "2026-06-20T00:00:00Z",
    })
    assert r.status_code == 202
    body = r.json()
    # The internal store "status" key must not leak into the response body (§4.3).
    assert "status" not in body
    assert body["update_queued"] is True
    assert "performance" in body["affected_nodes"]
    # Re-signed credential survives a subsequent GET.
    after = store._creds["laptops"].graph.updateCount
    assert after == before + 1
    g = client.request("GET", "/preference", json={"category": "laptops"}, headers=_auth(token))
    assert PreferenceCredential.from_dict(g.json()["credential"]).verify(sk.verify_key)


def test_post_elicit_orders_by_information_gain():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    client = _client(store)
    r = client.post("/elicit", headers=_auth(token), json={"category": "laptops", "max_questions": 3})
    assert r.status_code == 200
    igs = [q["information_gain"] for q in r.json()["questions"]]
    assert igs == sorted(igs, reverse=True)


# ------------------------------------------------------------------ auth boundary
def test_missing_auth_header_401():
    _, _, store = _store()
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"})
    assert r.status_code == 401


def test_bad_scheme_401():
    _, _, store = _store()
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"},
                       headers={"Authorization": "Token abc"})
    assert r.status_code == 401


def test_invalid_token_401():
    _, _, store = _store()
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"}, headers=_auth("agt_nope"))
    assert r.status_code == 401


def test_out_of_scope_403():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["headphones"])
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"}, headers=_auth(token))
    assert r.status_code == 403


def test_revoked_token_rejected():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    store.revoke_agent("agent.a")
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"}, headers=_auth(token))
    assert r.status_code == 401


def test_absent_credential_404():
    sk, did = new_user_keypair(seed=b"5" * 32)
    store = CredentialStore(sk, did)
    token = store.authorize_agent("agent.a", scope=["laptops"])
    client = _client(store)
    r = client.request("GET", "/preference", json={"category": "laptops"}, headers=_auth(token))
    assert r.status_code == 404


# ------------------------------------------------------------------ latency target
def test_preference_p95_latency_under_100ms():
    _, _, store = _store()
    token = store.authorize_agent("agent.a", scope=["laptops"])
    client = _client(store)
    headers = _auth(token)
    # Warm up (import + first-request overhead is not representative of steady state).
    for _ in range(5):
        client.request("GET", "/preference", json={"category": "laptops"}, headers=headers)
    samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        r = client.request("GET", "/preference", json={"category": "laptops"}, headers=headers)
        samples.append((time.perf_counter() - t0) * 1000.0)
        assert r.status_code == 200
    samples.sort()
    p95 = samples[int(0.95 * len(samples)) - 1]
    assert p95 < 100.0, f"p95={p95:.2f}ms exceeds 100ms target"
