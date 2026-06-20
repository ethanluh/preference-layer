"""Tests for ciphertext-only cloud sync.

The load-bearing invariant: the sync server stores only ciphertext; no plaintext
credential content (category names, attribute ids, the DID) is recoverable from
what the server holds, and only the on-device key can decrypt.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from preferencelayer.ptp import (
    AttributeNode,
    CloudSyncClient,
    CredentialStore,
    PreferenceCredential,
    PreferenceGraph,
    SyncServerStub,
    new_user_keypair,
    open_payload,
    seal_payload,
    store_id_for,
)


def _keypair():
    return new_user_keypair(seed=b"8" * 32)


def test_seal_open_roundtrip():
    sk, _ = _keypair()
    blob = b'{"category":"laptops","secret":"performance"}'
    ct = seal_payload(sk, blob)
    assert ct != blob
    assert open_payload(sk, ct) == blob


def test_push_pull_via_client():
    sk, _ = _keypair()
    server = SyncServerStub()
    client = CloudSyncClient(sk, server)
    payload = json.dumps({"category": "laptops"}).encode()
    v1 = client.push_payload(payload)
    assert v1 == 1
    assert client.pull_payload() == payload
    # A second push bumps the version.
    assert client.push_payload(payload + b"x") == 2


def test_server_holds_no_plaintext():
    sk, did = _keypair()
    server = SyncServerStub()
    client = CloudSyncClient(sk, server)
    # A realistic credential payload with sensitive tokens in plaintext form.
    cred = PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("price_sensitivity", -0.3, 0.6)],
    ))
    client.push_payload(json.dumps(cred.to_dict()).encode())

    raw = server.raw_bytes(client.store_id)
    assert raw  # something was stored
    # None of the sensitive plaintext tokens appear in what the server can see.
    for needle in (b"laptops", b"price_sensitivity", did.encode(), b"preferenceGraph"):
        assert needle not in raw


def test_store_id_is_not_the_did_or_key():
    sk, did = _keypair()
    sid = store_id_for(sk)
    assert sid != did
    assert bytes(sk.verify_key).hex() not in sid
    # Stable across calls.
    assert store_id_for(sk) == sid


def test_wrong_key_cannot_decrypt():
    sk_a, _ = _keypair()
    sk_b, _ = new_user_keypair(seed=b"9" * 32)
    ct = seal_payload(sk_a, b"top secret")
    with pytest.raises(Exception):
        open_payload(sk_b, ct)


def test_sync_encrypted_store_db_file_is_ciphertext_only():
    """End-to-end: the persistent store's on-disk file syncs as ciphertext only."""
    sk, did = _keypair()
    store = CredentialStore(sk, did)
    store.put_credential(PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", 0.8, 0.7)],
    )))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "store.cred"
        store.save(path)  # already-encrypted blob on disk
        server = SyncServerStub()
        client = CloudSyncClient(sk, server)
        client.push_file(path)
        raw = server.raw_bytes(client.store_id)
        assert b"laptops" not in raw and b"performance" not in raw
        # Round-trips back to the exact on-disk bytes via the on-device key.
        assert client.pull_payload() == path.read_bytes()


def test_server_has_no_plaintext_accessor():
    """Defensive: the server type exposes no method to read plaintext."""
    server = SyncServerStub()
    assert not hasattr(server, "decrypt")
    assert not hasattr(server, "plaintext")
