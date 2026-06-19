"""PTP preference credential: build, sign, verify, scope.

A reference implementation of the credential described in ``docs/protocol-spec.md``
and ``docs/architecture.md``. The credential is a W3C-VC-shaped JSON document whose
``credentialSubject`` carries a sparse preference graph. It is signed with the
user's Ed25519 key; the issuer DID is a ``did:key`` derived from the public key.

The signature covers a canonical JSON serialization of the document with the
``proof.proofValue`` removed, so verification is order-independent and tamper
evident — any change to the graph invalidates the proof.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from nacl import signing

PTP_CONTEXT = "https://preferencelayer.io/context/v1"
W3C_VC_CONTEXT = "https://www.w3.org/ns/credentials/v2"
SCHEMA_VERSION = "0.1"

# Multicodec prefix for an Ed25519 public key (0xed01), used by did:key.
_ED25519_MULTICODEC = b"\xed\x01"
_BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = bytearray()
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_BASE58_ALPHABET[rem])
    for b in data:  # preserve leading zero bytes
        if b == 0:
            out.append(_BASE58_ALPHABET[0])
        else:
            break
    return out[::-1].decode()


def did_key_from_public(verify_key: signing.VerifyKey) -> str:
    """Derive a ``did:key`` identifier from an Ed25519 public key."""
    payload = _ED25519_MULTICODEC + bytes(verify_key)
    return "did:key:z" + _b58encode(payload)


@dataclass
class AttributeNode:
    id: str
    weight: float        # preference weight in [-1, 1]
    confidence: float    # confidence in [0, 1]
    embedding: list[float] | None = None


@dataclass
class Edge:
    source: str
    target: str
    weight: float        # interaction / tradeoff weight in [-1, 1]
    contextKey: str | None = None


@dataclass
class ContextConditioner:
    contextKey: str
    activeNodes: list[str]
    suppressedNodes: list[str] = field(default_factory=list)


@dataclass
class PreferenceGraph:
    category: str
    attributeNodes: list[AttributeNode]
    edges: list[Edge] = field(default_factory=list)
    contextConditioners: list[ContextConditioner] = field(default_factory=list)
    version: str = SCHEMA_VERSION
    updateCount: int = 0
    privacyBudgetConsumed: float = 0.0
    lastUpdated: str = ""
    coldStartPrior: str | None = None


def _canonical(obj: dict) -> bytes:
    """Deterministic JSON bytes for signing (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class PreferenceCredential:
    """A signable, verifiable preference credential."""

    def __init__(self, issuer_did: str, graph: PreferenceGraph, issuance_date: str | None = None):
        self.issuer = issuer_did
        self.graph = graph
        self.issuance_date = issuance_date or datetime.now(timezone.utc).isoformat()
        self.proof: dict | None = None

    # ------------------------------------------------------------- serialization
    def _unsigned_document(self) -> dict:
        g = asdict(self.graph)
        # Drop None / empty optional fields to keep the document tidy.
        g = {k: v for k, v in g.items() if v not in (None, [], "")}
        for node in g.get("attributeNodes", []):
            if node.get("embedding") is None:
                node.pop("embedding", None)
        return {
            "@context": [W3C_VC_CONTEXT, PTP_CONTEXT],
            "type": ["VerifiableCredential", "PreferenceCredential"],
            "issuer": self.issuer,
            "issuanceDate": self.issuance_date,
            "credentialSubject": {"id": self.issuer, "preferenceGraph": g},
        }

    def to_dict(self) -> dict:
        doc = self._unsigned_document()
        if self.proof is not None:
            doc["proof"] = self.proof
        return doc

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ------------------------------------------------------------------ signing
    def sign(self, signing_key: signing.SigningKey) -> "PreferenceCredential":
        message = _canonical(self._unsigned_document())
        sig = signing_key.sign(message).signature
        self.proof = {
            "type": "Ed25519Signature2020",
            "created": datetime.now(timezone.utc).isoformat(),
            "verificationMethod": f"{self.issuer}#key-1",
            "proofPurpose": "assertionMethod",
            "proofValue": "z" + base64.urlsafe_b64encode(sig).decode().rstrip("="),
        }
        return self

    def verify(self, verify_key: signing.VerifyKey) -> bool:
        if not self.proof or "proofValue" not in self.proof:
            return False
        raw = self.proof["proofValue"]
        if raw.startswith("z"):
            raw = raw[1:]
        sig = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        message = _canonical(self._unsigned_document())
        try:
            verify_key.verify(message, sig)
            return True
        except Exception:
            return False

    # ------------------------------------------------------ selective disclosure
    def scoped(
        self,
        disclosure_scope: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> "PreferenceCredential":
        """Return an unsigned copy exposing only the requested / confident nodes.

        Implements PTP selective disclosure: nodes outside ``disclosure_scope`` or
        below ``min_confidence`` are redacted, along with any edges that reference
        a redacted node. The result must be re-signed before transport.
        """
        keep = {
            n.id
            for n in self.graph.attributeNodes
            if n.confidence >= min_confidence and (disclosure_scope is None or n.id in disclosure_scope)
        }
        nodes = [n for n in self.graph.attributeNodes if n.id in keep]
        edges = [e for e in self.graph.edges if e.source in keep and e.target in keep]
        scoped_graph = PreferenceGraph(
            category=self.graph.category,
            attributeNodes=nodes,
            edges=edges,
            contextConditioners=self.graph.contextConditioners,
            version=self.graph.version,
            updateCount=self.graph.updateCount,
            privacyBudgetConsumed=self.graph.privacyBudgetConsumed,
            lastUpdated=self.graph.lastUpdated,
            coldStartPrior=self.graph.coldStartPrior,
        )
        return PreferenceCredential(self.issuer, scoped_graph, self.issuance_date)

    # ----------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, doc: dict) -> "PreferenceCredential":
        g = doc["credentialSubject"]["preferenceGraph"]
        graph = PreferenceGraph(
            category=g["category"],
            attributeNodes=[AttributeNode(**n) for n in g.get("attributeNodes", [])],
            edges=[Edge(**e) for e in g.get("edges", [])],
            contextConditioners=[ContextConditioner(**c) for c in g.get("contextConditioners", [])],
            version=g.get("version", SCHEMA_VERSION),
            updateCount=g.get("updateCount", 0),
            privacyBudgetConsumed=g.get("privacyBudgetConsumed", 0.0),
            lastUpdated=g.get("lastUpdated", ""),
            coldStartPrior=g.get("coldStartPrior"),
        )
        cred = cls(doc["issuer"], graph, doc.get("issuanceDate"))
        cred.proof = doc.get("proof")
        return cred


def new_user_keypair(seed: bytes | None = None) -> tuple[signing.SigningKey, str]:
    """Create an Ed25519 signing key and its did:key identifier."""
    sk = signing.SigningKey(seed) if seed else signing.SigningKey.generate()
    return sk, did_key_from_public(sk.verify_key)
