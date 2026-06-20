"""Persistent, user-controlled credential store (PTP spec §4, Phase 1).

The Phase 0 :class:`~preferencelayer.ptp.store.CredentialStore` is in-memory with
an optional encrypted-blob ``save``/``load``. Phase 1's "Credential store"
deliverable (see ``docs/implementation-plan.md`` Work Stream A, Months 5–6) calls
for a local, user-owned store that *survives across processes* so the CLI
(``preflayer view`` / ``revoke`` / ``export`` / ``delete``) operates on real,
durable state.

:class:`PersistentCredentialStore` adds exactly that on top of the in-memory store:

* **Persistent identity.** The user's Ed25519 signing key lives in
  ``identity.key`` (0600). If ``PREFLAYER_PASSPHRASE`` is set, the seed is itself
  encrypted with an Argon2id-derived key; otherwise it is stored raw under
  owner-only permissions (the threat model is documented in ``view``/the kickoff
  doc — encryption-at-rest of the DB protects the *syncable* ciphertext, the key
  stays on device).
* **SQLite-backed credentials and agent tokens.** Credentials are stored as
  per-row ciphertext (NaCl secret box keyed off the identity key), matching the
  architecture's "cloud sync stores client-side-encrypted ciphertext only"
  invariant. Agent tokens persist so ``revoke <agent_id>`` works across CLI runs.

Everything is loaded into the in-memory dicts on open and written through on
mutation, so the inherited PTP operations (``get_preference`` / ``submit_outcome``
/ ``elicit``) run unchanged.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
from pathlib import Path

import nacl.encoding
import nacl.pwhash
import nacl.secret
import nacl.utils
from nacl import signing

from .credential import PreferenceCredential, did_key_from_public
from .store import AgentToken, CredentialStore
from .update import DPConfig

SCHEMA_VERSION = "1"
_DEFAULT_HOME = "~/.preflayer"


def default_home() -> Path:
    """Resolve the store directory: ``$PREFLAYER_HOME`` or ``~/.preflayer``."""
    return Path(os.environ.get("PREFLAYER_HOME", _DEFAULT_HOME)).expanduser()


# --------------------------------------------------------------------- identity


def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _b64d(text: str) -> bytes:
    return base64.b64decode(text)


def _write_identity(path: Path, seed: bytes, passphrase: str | None) -> None:
    if passphrase:
        salt = nacl.utils.random(nacl.pwhash.argon2id.SALTBYTES)
        key = nacl.pwhash.argon2id.kdf(
            nacl.secret.SecretBox.KEY_SIZE,
            passphrase.encode(),
            salt,
            opslimit=nacl.pwhash.argon2id.OPSLIMIT_MODERATE,
            memlimit=nacl.pwhash.argon2id.MEMLIMIT_MODERATE,
        )
        ct = nacl.secret.SecretBox(key).encrypt(seed)
        doc = {
            "kdf": "argon2id",
            "salt": _b64e(salt),
            "opslimit": nacl.pwhash.argon2id.OPSLIMIT_MODERATE,
            "memlimit": nacl.pwhash.argon2id.MEMLIMIT_MODERATE,
            "ciphertext": _b64e(ct),
        }
    else:
        doc = {"kdf": "none", "seed": _b64e(seed)}
    path.write_text(json.dumps(doc))
    path.chmod(0o600)


def _read_identity(path: Path, passphrase: str | None) -> bytes:
    doc = json.loads(path.read_text())
    if doc["kdf"] == "none":
        return _b64d(doc["seed"])
    if doc["kdf"] == "argon2id":
        if not passphrase:
            raise IdentityLocked(
                "identity is passphrase-encrypted; set PREFLAYER_PASSPHRASE to unlock"
            )
        key = nacl.pwhash.argon2id.kdf(
            nacl.secret.SecretBox.KEY_SIZE,
            passphrase.encode(),
            _b64d(doc["salt"]),
            opslimit=doc["opslimit"],
            memlimit=doc["memlimit"],
        )
        try:
            return nacl.secret.SecretBox(key).decrypt(_b64d(doc["ciphertext"]))
        except Exception as exc:  # noqa: BLE001 - surface a clean error
            raise IdentityLocked("wrong passphrase for identity") from exc
    raise ValueError(f"unknown identity kdf '{doc['kdf']}'")


class IdentityLocked(RuntimeError):
    """Raised when the on-disk identity cannot be unlocked (missing/bad passphrase)."""


class StoreNotFound(RuntimeError):
    """Raised when opening a store that has not been initialized."""


# --------------------------------------------------------------------- the store


class PersistentCredentialStore(CredentialStore):
    """A :class:`CredentialStore` whose credentials, tokens and identity persist.

    Open or create one with :meth:`open`. Mutations are written through to SQLite
    immediately, so a fresh process sees the same state.
    """

    def __init__(self, home: Path, signing_key: signing.SigningKey, issuer_did: str,
                 conn: sqlite3.Connection, dp: DPConfig | None = None):
        super().__init__(signing_key, issuer_did, dp=dp)
        self.home = home
        self._conn = conn
        self._load_all()

    # ----------------------------------------------------------------- lifecycle
    @classmethod
    def open(
        cls,
        home: str | Path | None = None,
        *,
        create: bool = False,
        passphrase: str | None = None,
        dp: DPConfig | None = None,
        seed: bytes | None = None,
    ) -> "PersistentCredentialStore":
        """Open an existing store, or create one when ``create=True``.

        ``passphrase`` defaults to ``$PREFLAYER_PASSPHRASE``. ``seed`` lets tests
        pin a deterministic identity on creation.
        """
        home = Path(home).expanduser() if home else default_home()
        passphrase = passphrase if passphrase is not None else os.environ.get("PREFLAYER_PASSPHRASE")
        key_path = home / "identity.key"
        db_path = home / "store.db"

        if key_path.exists():
            seed_bytes = _read_identity(key_path, passphrase)
            sk = signing.SigningKey(seed_bytes)
        elif create:
            home.mkdir(parents=True, exist_ok=True)
            seed_bytes = seed if seed is not None else nacl.utils.random(32)
            sk = signing.SigningKey(seed_bytes)
            _write_identity(key_path, seed_bytes, passphrase)
        else:
            raise StoreNotFound(f"no PreferenceLayer store at {home} (run `preflayer init`)")

        did = did_key_from_public(sk.verify_key)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        _init_schema(conn, did)
        return cls(home, sk, did, conn, dp=dp)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PersistentCredentialStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- load/persist
    def _cat_key(self, category: str) -> str:
        """Opaque, deterministic row key for a category (keyed hash, not reversible).

        Keeps the plaintext category out of the database so the DB is pure
        ciphertext (the real category lives inside the encrypted payload).
        """
        from nacl.hash import blake2b

        return self._keyed_hash(category)

    def _agent_key(self, agent_id: str) -> str:
        """Opaque, deterministic row key for an agent id (for revoke lookups)."""
        return self._keyed_hash(agent_id)

    def _keyed_hash(self, value: str) -> str:
        from nacl.hash import blake2b

        key = bytes(self.signing_key)[: nacl.secret.SecretBox.KEY_SIZE]
        return blake2b(value.encode(), key=key, encoder=nacl.encoding.HexEncoder).decode()

    def _load_all(self) -> None:
        box = self._secret_box()
        for row in self._conn.execute("SELECT ciphertext FROM credentials"):
            payload = box.decrypt(row["ciphertext"])
            cred = PreferenceCredential.from_dict(json.loads(payload.decode()))
            self._creds[cred.graph.category] = cred
        now = time.time()
        for row in self._conn.execute("SELECT ciphertext FROM tokens WHERE expires_at > ?", (now,)):
            t = json.loads(box.decrypt(row["ciphertext"]).decode())
            self._tokens[t["token"]] = AgentToken(t["token"], t["agent_id"], t["scope"], t["expires_at"])

    def _persist_credential(self, cred: PreferenceCredential) -> None:
        box = self._secret_box()
        ct = box.encrypt(json.dumps(cred.to_dict()).encode())
        self._conn.execute(
            "INSERT INTO credentials(cat_key, ciphertext, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(cat_key) DO UPDATE SET ciphertext=excluded.ciphertext, "
            "updated_at=excluded.updated_at",
            (self._cat_key(cred.graph.category), ct, _now_iso()),
        )
        self._conn.commit()

    # ----------------------------------------------------- write-through overrides
    def put_credential(self, cred: PreferenceCredential) -> None:
        super().put_credential(cred)  # signs + stores in memory
        self._persist_credential(self._creds[cred.graph.category])

    def authorize_agent(self, agent_id: str, scope: list[str], ttl_seconds: int = 86_400) -> str:
        token = super().authorize_agent(agent_id, scope, ttl_seconds)
        at = self._tokens[token]
        ct = self._secret_box().encrypt(json.dumps(
            {"token": at.token, "agent_id": at.agent_id, "scope": at.scope, "expires_at": at.expires_at}
        ).encode())
        self._conn.execute(
            "INSERT INTO tokens(agent_key, expires_at, ciphertext, created_at) VALUES (?,?,?,?)",
            (self._agent_key(agent_id), at.expires_at, ct, _now_iso()),
        )
        self._conn.commit()
        return token

    def revoke_agent(self, agent_id: str) -> int:
        n = super().revoke_agent(agent_id)
        self._conn.execute("DELETE FROM tokens WHERE agent_key = ?", (self._agent_key(agent_id),))
        self._conn.commit()
        return n

    def submit_outcome(self, token: str, category: str, *args, **kwargs) -> dict:
        result = super().submit_outcome(token, category, *args, **kwargs)
        if result.get("status") == 202 and category in self._creds:
            self._persist_credential(self._creds[category])  # update mutated + re-signed it
        return result

    # ------------------------------------------------------------------ queries
    def agent_tokens(self) -> list[AgentToken]:
        """Active (non-expired) tokens, most-recently-expiring first."""
        self.prune_expired()
        return sorted(self._tokens.values(), key=lambda a: -a.expires_at)

    def prune_expired(self) -> int:
        """Drop expired tokens from memory and disk; return how many were dropped."""
        now = time.time()
        dead = [t for t, a in self._tokens.items() if a.expires_at <= now]
        for t in dead:
            del self._tokens[t]
        if dead:
            self._conn.execute("DELETE FROM tokens WHERE expires_at <= ?", (now,))
            self._conn.commit()
        return len(dead)

    # ------------------------------------------------------------------- export
    def export_bundle(self) -> dict:
        """A portable, signed export of every credential (the VCs themselves)."""
        return {
            "issuer": self.issuer_did,
            "exportedAt": _now_iso(),
            "schemaVersion": SCHEMA_VERSION,
            "credentials": [self._creds[c].to_dict() for c in sorted(self._creds)],
        }

    # ------------------------------------------------------------------- delete
    def delete_all(self) -> None:
        """Irreversibly wipe the store: identity key and SQLite database."""
        self.close()
        for name in ("identity.key", "store.db"):
            p = self.home / name
            if p.exists():
                p.unlink()
        self._creds.clear()
        self._tokens.clear()


def _init_schema(conn: sqlite3.Connection, issuer_did: str) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS credentials(
            cat_key TEXT PRIMARY KEY,
            ciphertext BLOB NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tokens(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_key TEXT NOT NULL,
            expires_at REAL NOT NULL,
            ciphertext BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tokens_agent ON tokens(agent_key);
        """
    )
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO NOTHING",
        (SCHEMA_VERSION,),
    )
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('issuer_did', ?) ON CONFLICT(key) DO NOTHING",
        (issuer_did,),
    )
    conn.commit()


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
