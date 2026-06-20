"""Tests for the Phase 1 persistent, user-controlled credential store."""

import json
from pathlib import Path

import pytest

from preferencelayer.ptp import (
    AttributeNode,
    IdentityLocked,
    PersistentCredentialStore,
    PreferenceCredential,
    PreferenceGraph,
    StoreNotFound,
)


def _cred(did: str, category: str = "laptops") -> PreferenceCredential:
    g = PreferenceGraph(
        category=category,
        attributeNodes=[
            AttributeNode("performance", 0.5, 0.6),
            AttributeNode("portability", 0.4, 0.2),
            AttributeNode("price_sensitivity", -0.2, 0.3),
        ],
    )
    return PreferenceCredential(did, g)


def test_open_requires_init(tmp_path):
    with pytest.raises(StoreNotFound):
        PersistentCredentialStore.open(tmp_path / "missing", create=False)


def test_identity_and_credentials_persist_across_reopen(tmp_path):
    home = tmp_path / "store"
    s = PersistentCredentialStore.open(home, create=True, seed=b"7" * 32)
    did = s.issuer_did
    s.put_credential(_cred(did))
    s.close()

    s2 = PersistentCredentialStore.open(home)
    assert s2.issuer_did == did  # same persisted identity
    assert s2.categories() == ["laptops"]
    # The reloaded credential still verifies under the reloaded key.
    res = s2.get_preference(s2.authorize_agent("a", ["laptops"]), "laptops")
    assert PreferenceCredential.from_dict(res["credential"]).verify(s2.signing_key.verify_key)
    s2.close()


def test_tokens_persist_and_revoke_survives_reopen(tmp_path):
    home = tmp_path / "store"
    s = PersistentCredentialStore.open(home, create=True)
    s.put_credential(_cred(s.issuer_did))
    s.authorize_agent("agent.a", ["laptops"])
    s.close()

    s2 = PersistentCredentialStore.open(home)
    assert len(s2.agent_tokens()) == 1
    assert s2.revoke_agent("agent.a") == 1
    s2.close()

    s3 = PersistentCredentialStore.open(home)
    assert s3.agent_tokens() == []
    s3.close()


def test_submit_outcome_persists_updated_credential(tmp_path):
    home = tmp_path / "store"
    s = PersistentCredentialStore.open(home, create=True)
    s.put_credential(_cred(s.issuer_did))
    token = s.authorize_agent("a", ["laptops"])
    s.submit_outcome(token, "laptops", "thinkpad", "purchase", use_context="sustained compute")
    count_after = s._creds["laptops"].graph.updateCount
    assert count_after == 1
    s.close()

    s2 = PersistentCredentialStore.open(home)
    assert s2._creds["laptops"].graph.updateCount == count_after  # update was durable
    s2.close()


def test_encrypted_at_rest(tmp_path):
    home = tmp_path / "store"
    s = PersistentCredentialStore.open(home, create=True)
    s.put_credential(_cred(s.issuer_did))
    s.close()
    raw = (home / "store.db").read_bytes()
    # Neither the category nor an attribute id should appear in plaintext.
    assert b"laptops" not in raw
    assert b"performance" not in raw


def test_passphrase_locks_identity(tmp_path):
    home = tmp_path / "store"
    s = PersistentCredentialStore.open(home, create=True, passphrase="correct horse")
    s.close()
    # Identity file must not contain the raw seed key material.
    doc = json.loads((home / "identity.key").read_text())
    assert doc["kdf"] == "argon2id"
    with pytest.raises(IdentityLocked):
        PersistentCredentialStore.open(home)  # no passphrase
    with pytest.raises(IdentityLocked):
        PersistentCredentialStore.open(home, passphrase="wrong")
    s2 = PersistentCredentialStore.open(home, passphrase="correct horse")
    assert s2.issuer_did == s.issuer_did
    s2.close()


def test_export_bundle_is_signed(tmp_path):
    s = PersistentCredentialStore.open(tmp_path / "store", create=True)
    s.put_credential(_cred(s.issuer_did))
    bundle = s.export_bundle()
    assert bundle["issuer"] == s.issuer_did
    assert len(bundle["credentials"]) == 1
    assert PreferenceCredential.from_dict(bundle["credentials"][0]).verify(s.signing_key.verify_key)
    s.close()


def test_prune_expired_tokens(tmp_path):
    s = PersistentCredentialStore.open(tmp_path / "store", create=True)
    s.put_credential(_cred(s.issuer_did))
    s.authorize_agent("short", ["laptops"], ttl_seconds=-1)  # already expired
    s.authorize_agent("live", ["laptops"], ttl_seconds=3600)
    assert s.prune_expired() == 1
    assert [a.agent_id for a in s.agent_tokens()] == ["live"]
    s.close()


def test_delete_all_removes_files(tmp_path):
    home = tmp_path / "store"
    s = PersistentCredentialStore.open(home, create=True)
    s.put_credential(_cred(s.issuer_did))
    s.delete_all()
    assert not (home / "identity.key").exists()
    assert not (home / "store.db").exists()
