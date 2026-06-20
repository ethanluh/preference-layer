"""Optional cloud sync of the credential store — ciphertext only.

The architecture's invariant: *cloud sync stores client-side-encrypted ciphertext
only; the server can never read the plaintext credential*. This module makes that
concrete without weakening it.

What is uploaded is an **opaque blob** produced on-device:

* For the persistent store, the blob is the already-encrypted SQLite payload
  (per-row NaCl secret-box ciphertext) — or, more conservatively, a fresh
  secret-box sealing of an export bundle under a key derived from the identity
  key. Either way the sync server sees only ciphertext + a public ``store_id``.
* The identity key never leaves the device, so the server cannot derive the
  decryption key.

A :class:`SyncServerStub` models the server side for tests: it accepts and returns
blobs keyed by ``store_id`` and has **no** method to read plaintext. The included
test asserts no plaintext token (category, attribute id, DID) is recoverable from
what the server holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import nacl.encoding
import nacl.secret
import nacl.utils
from nacl import signing
from nacl.hash import blake2b


def _sync_box(signing_key: signing.SigningKey) -> nacl.secret.SecretBox:
    """Secret box keyed off the identity key (same scheme as the store at rest)."""
    key = blake2b(
        b"ptp-cloud-sync\x00" + bytes(signing_key),
        digest_size=nacl.secret.SecretBox.KEY_SIZE,
        encoder=nacl.encoding.RawEncoder,
    )
    return nacl.secret.SecretBox(key)


def store_id_for(signing_key: signing.SigningKey) -> str:
    """A stable, non-reversible sync identifier (keyed hash of the public key).

    Lets the server address a user's blob without learning the DID or any key
    material. Not the DID, not the public key — a one-way tag.
    """
    return blake2b(
        bytes(signing_key.verify_key),
        key=b"ptp-sync-id-v1\x00\x00",
        digest_size=16,
        encoder=nacl.encoding.HexEncoder,
    ).decode()


@dataclass
class SyncEnvelope:
    """What crosses the wire: a store id, ciphertext, and a monotone version."""

    store_id: str
    ciphertext: bytes
    version: int


def seal_payload(signing_key: signing.SigningKey, plaintext: bytes) -> bytes:
    """Encrypt a plaintext payload on-device. Returns ciphertext only."""
    return _sync_box(signing_key).encrypt(plaintext)


def open_payload(signing_key: signing.SigningKey, ciphertext: bytes) -> bytes:
    """Decrypt a previously-sealed payload on-device (server can never do this)."""
    return _sync_box(signing_key).decrypt(ciphertext)


class SyncServerStub:
    """In-memory model of the cloud sync server: holds ciphertext, nothing else.

    Deliberately exposes no plaintext accessor. The only data it retains is the
    opaque ``store_id``, the ciphertext blob, and a version counter.
    """

    def __init__(self) -> None:
        self._blobs: dict[str, SyncEnvelope] = {}

    def push(self, store_id: str, ciphertext: bytes) -> int:
        """Store/replace the blob for ``store_id``; returns the new version."""
        if not isinstance(ciphertext, (bytes, bytearray)):
            raise TypeError("sync server accepts ciphertext bytes only")
        prev = self._blobs.get(store_id)
        version = (prev.version + 1) if prev else 1
        self._blobs[store_id] = SyncEnvelope(store_id, bytes(ciphertext), version)
        return version

    def pull(self, store_id: str) -> SyncEnvelope | None:
        return self._blobs.get(store_id)

    def raw_bytes(self, store_id: str) -> bytes:
        """Everything the server can possibly see for a store (ciphertext only)."""
        env = self._blobs.get(store_id)
        return env.ciphertext if env else b""


class CloudSyncClient:
    """Client-side sync driver. All encryption happens here, on-device.

    Bind it to the user's signing key and a server (the stub, or any object with
    ``push(store_id, ciphertext)`` / ``pull(store_id)``). ``push_payload`` seals a
    plaintext blob and uploads only the ciphertext; ``pull_payload`` downloads and
    decrypts locally.
    """

    def __init__(self, signing_key: signing.SigningKey, server: SyncServerStub):
        self.signing_key = signing_key
        self.server = server
        self.store_id = store_id_for(signing_key)

    def push_payload(self, plaintext: bytes) -> int:
        ciphertext = seal_payload(self.signing_key, plaintext)
        return self.server.push(self.store_id, ciphertext)

    def pull_payload(self) -> bytes | None:
        env = self.server.pull(self.store_id)
        if env is None:
            return None
        return open_payload(self.signing_key, env.ciphertext)

    def push_file(self, path) -> int:
        """Sync an on-disk encrypted store file (e.g. the SQLite DB) as-is.

        The persistent store DB is already pure ciphertext; we re-seal it so the
        sync key is independent of the at-rest key, but either way only ciphertext
        leaves the device.
        """
        from pathlib import Path

        return self.push_payload(Path(path).read_bytes())
