"""Tests for the RFC 8628 device flow exposed over HTTP (spec §4.1, kickoff A2).

Covers the full handshake over the wire (request code -> poll pending -> owner
approves -> poll yields a working token), the owner inspection/decision routes,
and the RFC 8628 error mappings (missing scope, bad grant_type, denied, expired).
Skipped cleanly if the optional 'http'/'dev' extra (FastAPI + httpx) is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from preferencelayer.http import build_app  # noqa: E402
from preferencelayer.http.app import DEVICE_CODE_GRANT  # noqa: E402
from preferencelayer.ptp.credential import (  # noqa: E402
    AttributeNode,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.device_flow import DeviceFlowAuthority  # noqa: E402
from preferencelayer.ptp.store import CredentialStore  # noqa: E402


def _store():
    sk, did = new_user_keypair(seed=b"7" * 32)
    store = CredentialStore(sk, did)
    for category in ("laptops", "headphones"):
        store.put_credential(PreferenceCredential(did, PreferenceGraph(
            category=category,
            attributeNodes=[AttributeNode("performance", 0.8, 0.7)],
        )))
    return store


def _client(store, *, interval: int = 0, code_ttl: int = 600):
    # interval=0 so the token endpoint can be polled back-to-back in tests.
    authority = DeviceFlowAuthority(store, interval=interval, code_ttl=code_ttl)
    return TestClient(build_app(store, device_authority=authority))


def _request_code(client, scope=("laptops",)):
    r = client.post("/device/code", json={"client_id": "agent.shop", "scope": list(scope)})
    assert r.status_code == 200
    return r.json()


def _poll(client, device_code):
    return client.post("/token", json={"grant_type": DEVICE_CODE_GRANT, "device_code": device_code})


# ----------------------------------------------------------------- happy path
def test_full_device_flow_yields_working_token():
    store = _store()
    client = _client(store)
    code = _request_code(client)
    assert code["user_code"] in code["verification_uri_complete"]

    # Before approval, the token endpoint reports authorization_pending (HTTP 400).
    pending = _poll(client, code["device_code"])
    assert pending.status_code == 400
    assert pending.json()["detail"]["error"] == "authorization_pending"

    # Owner inspects the pending request and sees exactly the requested scope.
    seen = client.get("/device", params={"user_code": code["user_code"]})
    assert seen.status_code == 200
    assert seen.json()["scope"] == ["laptops"]

    # Owner approves; the next poll returns a Bearer token.
    decision = client.post("/device/decision", json={"user_code": code["user_code"], "decision": "approve"})
    assert decision.status_code == 200
    granted = _poll(client, code["device_code"])
    assert granted.status_code == 200
    body = granted.json()
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] > 0
    token = body["access_token"]

    # The minted token authorizes the requested category over HTTP...
    ok = client.request("GET", "/preference", json={"category": "laptops"},
                        headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    # ...but is scoped: another category is forbidden.
    forbidden = client.request("GET", "/preference", json={"category": "headphones"},
                               headers={"Authorization": f"Bearer {token}"})
    assert forbidden.status_code == 403


def test_revoked_after_grant_rejected():
    store = _store()
    client = _client(store)
    code = _request_code(client)
    client.post("/device/decision", json={"user_code": code["user_code"], "decision": "approve"})
    token = _poll(client, code["device_code"]).json()["access_token"]
    assert store.revoke_agent("agent.shop") == 1
    r = client.request("GET", "/preference", json={"category": "laptops"},
                       headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


# ------------------------------------------------------------- error mappings
def test_missing_scope_is_400():
    client = _client(_store())
    r = client.post("/device/code", json={"client_id": "agent.shop"})
    assert r.status_code == 400
    # No implicit all-category wildcard (device_flow rejects empty scope).
    assert "error" in r.json()["detail"]


def test_unsupported_grant_type_is_400():
    store = _store()
    client = _client(store)
    code = _request_code(client)
    r = client.post("/token", json={"grant_type": "authorization_code", "device_code": code["device_code"]})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unsupported_grant_type"


def test_denied_request_maps_to_access_denied():
    store = _store()
    client = _client(store)
    code = _request_code(client)
    client.post("/device/decision", json={"user_code": code["user_code"], "decision": "deny"})
    r = _poll(client, code["device_code"])
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "access_denied"


def test_expired_device_code_maps_to_expired_token():
    store = _store()
    client = _client(store, code_ttl=0)  # codes expire immediately
    code = _request_code(client)
    r = _poll(client, code["device_code"])
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "expired_token"


def test_unknown_device_code_is_invalid_grant():
    client = _client(_store())
    r = _poll(client, "dev_does_not_exist")
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_grant"


def test_invalid_decision_value_is_400():
    store = _store()
    client = _client(store)
    code = _request_code(client)
    r = client.post("/device/decision", json={"user_code": code["user_code"], "decision": "maybe"})
    assert r.status_code == 400


def test_slow_down_when_polling_too_fast():
    store = _store()
    client = _client(store, interval=5)  # enforce the polling interval
    code = _request_code(client)
    first = _poll(client, code["device_code"])
    assert first.json()["detail"]["error"] == "authorization_pending"
    second = _poll(client, code["device_code"])  # immediate re-poll, within interval
    assert second.status_code == 400
    assert second.json()["detail"]["error"] == "slow_down"
