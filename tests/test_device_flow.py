"""Tests for the OAuth 2.0 device flow (RFC 8628) PTP agent auth."""

from __future__ import annotations

import time

import pytest

from preferencelayer.ptp import (
    AccessDenied,
    AttributeNode,
    AuthorizationPending,
    CredentialStore,
    DeviceFlowAuthority,
    ExpiredToken,
    InvalidDeviceCode,
    PreferenceCredential,
    PreferenceGraph,
    SlowDown,
    new_user_keypair,
)


def _store():
    sk, did = new_user_keypair(seed=b"6" * 32)
    store = CredentialStore(sk, did)
    store.put_credential(PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", 0.8, 0.7)],
    )))
    return store


def test_request_device_code_shape():
    auth = DeviceFlowAuthority(_store())
    resp = auth.request_device_code("agent.shop", scope=["laptops"])
    assert resp.device_code and resp.user_code and resp.verification_uri
    assert resp.user_code in resp.verification_uri_complete
    assert resp.expires_in > 0 and resp.interval > 0


def test_pending_then_approved_yields_working_token():
    store = _store()
    auth = DeviceFlowAuthority(store, interval=0)  # interval 0 so polling is unthrottled in tests
    resp = auth.request_device_code("agent.shop", scope=["laptops"])

    # Before approval, polling reports authorization_pending.
    with pytest.raises(AuthorizationPending):
        auth.poll_token(resp.device_code)

    auth.approve(resp.user_code)
    token = auth.poll_token(resp.device_code)

    # The minted token is a normal scoped store token: it works and is scoped.
    assert store.get_preference(token, "laptops")["status"] == 200


def test_device_code_is_single_use():
    store = _store()
    auth = DeviceFlowAuthority(store, interval=0)
    resp = auth.request_device_code("agent.shop", scope=["laptops"])
    auth.approve(resp.user_code)
    auth.poll_token(resp.device_code)
    # Second poll on a consumed device_code is rejected.
    with pytest.raises(InvalidDeviceCode):
        auth.poll_token(resp.device_code)


def test_denied_request():
    auth = DeviceFlowAuthority(_store(), interval=0)
    resp = auth.request_device_code("agent.shop", scope=["laptops"])
    auth.deny(resp.user_code)
    with pytest.raises(AccessDenied):
        auth.poll_token(resp.device_code)


def test_slow_down_when_polling_too_fast():
    auth = DeviceFlowAuthority(_store(), interval=5)
    resp = auth.request_device_code("agent.shop", scope=["laptops"])
    # First poll: pending. Immediate second poll: slow_down (within interval).
    with pytest.raises(AuthorizationPending):
        auth.poll_token(resp.device_code)
    with pytest.raises(SlowDown):
        auth.poll_token(resp.device_code)


def test_expired_device_code():
    auth = DeviceFlowAuthority(_store(), code_ttl=0, interval=0)
    resp = auth.request_device_code("agent.shop", scope=["laptops"])
    time.sleep(0.01)
    with pytest.raises(ExpiredToken):
        auth.poll_token(resp.device_code)


def test_approve_unknown_user_code():
    auth = DeviceFlowAuthority(_store())
    with pytest.raises(InvalidDeviceCode):
        auth.approve("ZZZZ-ZZZZ")


def test_revoke_after_device_flow_grant():
    store = _store()
    auth = DeviceFlowAuthority(store, interval=0)
    resp = auth.request_device_code("agent.shop", scope=["laptops"])
    auth.approve(resp.user_code)
    token = auth.poll_token(resp.device_code)
    assert store.revoke_agent("agent.shop") == 1
    # Token no longer authorizes after revocation (reuses the store's revoke path).
    from preferencelayer.ptp import AuthError
    with pytest.raises(AuthError):
        store.get_preference(token, "laptops")
