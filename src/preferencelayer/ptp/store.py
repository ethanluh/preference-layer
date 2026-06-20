"""User-controlled credential store (PTP §4).

A local, single-user store that holds one preference credential per category and
implements the three PTP operations against them:

* ``get_preference`` — scoped, selectively-disclosed, freshly-signed credential
* ``submit_outcome`` — enqueue + apply a DP update, then re-sign
* ``elicit``         — propose high-information-gain questions for weak nodes

Agents authenticate with opaque bearer tokens that carry a category scope and an
expiry, and can be revoked at any time. Credentials are encrypted at rest with a
NaCl secret box derived from the user's signing key.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import nacl.encoding
import nacl.secret
import nacl.utils
from nacl import signing
from nacl.hash import blake2b

from .credential import PreferenceCredential
from .update import BudgetExhausted, DPConfig, apply_outcome

# Coarse use-context -> attribute keyword mapping. A production system would learn
# this; for the prototype a keyword table is enough to route outcome signals.
_CONTEXT_KEYWORDS = {
    "performance": ["performance", "compute", "gaming", "ml", "render", "sustained"],
    "portability": ["travel", "portable", "light", "commute", "mobile"],
    "build_quality": ["premium", "build", "metal", "sturdy", "durable"],
    "durability": ["durable", "longevity", "reliable", "failure", "broke"],
    "price_sensitivity": ["budget", "cheap", "value", "affordable", "price"],
    "ergonomics": ["comfort", "ergonomic", "wrist", "typing", "long sessions"],
    "aesthetics": ["design", "looks", "aesthetic", "rgb", "color"],
    "brand_affinity": ["brand", "thinkpad", "apple", "premium brand"],
}


@dataclass
class AgentToken:
    token: str
    agent_id: str
    scope: list[str]
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class ElicitationQuestion:
    id: str
    text: str
    response_schema: dict
    target_attribute: str
    information_gain: float


def context_to_nodes(use_context: str, available: list[str]) -> list[str]:
    """Map a free-text use context to affected attribute node ids."""
    text = (use_context or "").lower()
    hits = [
        attr for attr, kws in _CONTEXT_KEYWORDS.items()
        if attr in available and any(kw in text for kw in kws)
    ]
    return hits or available[: min(3, len(available))]


class AuthError(RuntimeError):
    pass


class CredentialStore:
    def __init__(self, signing_key: signing.SigningKey, issuer_did: str, dp: DPConfig | None = None):
        self.signing_key = signing_key
        self.issuer_did = issuer_did
        self.dp = dp or DPConfig()
        self._creds: dict[str, PreferenceCredential] = {}   # category -> credential
        self._tokens: dict[str, AgentToken] = {}

    # --------------------------------------------------------------- credentials
    def put_credential(self, cred: PreferenceCredential) -> None:
        cred.sign(self.signing_key)
        self._creds[cred.graph.category] = cred

    def categories(self) -> list[str]:
        return list(self._creds)

    # --------------------------------------------------------------------- auth
    def authorize_agent(self, agent_id: str, scope: list[str], ttl_seconds: int = 86_400) -> str:
        token = "agt_" + secrets.token_urlsafe(24)
        self._tokens[token] = AgentToken(token, agent_id, scope, time.time() + ttl_seconds)
        return token

    def revoke_agent(self, agent_id: str) -> int:
        revoked = [t for t, a in self._tokens.items() if a.agent_id == agent_id]
        for t in revoked:
            del self._tokens[t]
        return len(revoked)

    def _auth(self, token: str, category: str) -> AgentToken:
        at = self._tokens.get(token)
        if at is None or at.expired:
            raise AuthError("invalid or expired token")
        if category not in at.scope and "*" not in at.scope:
            raise AuthError(f"token not scoped for category '{category}'")
        return at

    # ----------------------------------------------------------- PTP operations
    def get_preference(
        self,
        token: str,
        category: str,
        query_context: str = "",
        disclosure_scope: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> dict:
        self._auth(token, category)
        cred = self._creds.get(category)
        if cred is None:
            return {"status": 404, "detail": "no credential for category; call elicit to initialize"}

        scoped = cred.scoped(disclosure_scope, min_confidence).sign(self.signing_key)
        coverage = [n.id for n in scoped.graph.attributeNodes]
        requested = disclosure_scope or [n.id for n in cred.graph.attributeNodes]
        missing = [r for r in requested if r not in coverage]
        confidences = [n.confidence for n in scoped.graph.attributeNodes]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "status": 200,
            "credential": scoped.to_dict(),
            "confidence": round(mean_conf, 4),
            "coverage": coverage,
            "missing": missing,
            "elicitation_recommended": mean_conf < 0.4,
        }

    def submit_outcome(
        self,
        token: str,
        category: str,
        product_id: str,
        outcome_type: str,
        use_context: str = "",
        rating: float | None = None,
        elicitation_weights: dict[str, float] | None = None,
    ) -> dict:
        self._auth(token, category)
        cred = self._creds.get(category)
        if cred is None:
            return {"status": 404, "detail": "no credential for category"}

        available = [n.id for n in cred.graph.attributeNodes]
        affected = context_to_nodes(use_context, available)
        try:
            apply_outcome(
                cred,
                affected_nodes=affected,
                outcome_type=outcome_type,
                rating=rating,
                elicitation_weights=elicitation_weights,
                cfg=self.dp,
            )
        except BudgetExhausted as exc:
            # The DP budget is exhausted; the update is refused (no weight change,
            # no re-sign). Surface a 429 the agent can act on, and signal that
            # continuing requires explicit user consent (see reset_budget).
            return {
                "status": 429,
                "detail": str(exc),
                "consent_required": True,
                "privacy_budget_consumed": cred.graph.privacyBudgetConsumed,
            }
        cred.sign(self.signing_key)  # re-sign after update
        return {
            "status": 202,
            "signal_id": "sig_" + secrets.token_hex(8),
            "update_queued": True,
            "affected_nodes": affected,
            "privacy_budget_consumed": cred.graph.privacyBudgetConsumed,
        }

    def reset_budget(self, token: str, category: str, *, consent: bool = False) -> dict:
        """Reset the DP privacy budget for a category — STUB, requires user consent.

        Resetting ``privacyBudgetConsumed`` lets a credential accept further DP
        updates after exhaustion, but it spends a fresh privacy budget, so it must
        be a deliberate, *user*-authorized act. This is a stub: it enforces the
        consent gate and the re-sign, but the consent UX (how the user is asked,
        and any audit trail) is out of scope here (Phase 2). Without
        ``consent=True`` it refuses with a 403.
        """
        self._auth(token, category)
        cred = self._creds.get(category)
        if cred is None:
            return {"status": 404, "detail": "no credential for category"}
        if not consent:
            return {
                "status": 403,
                "detail": "budget reset requires explicit user consent",
                "consent_required": True,
            }
        cred.graph.privacyBudgetConsumed = 0.0
        cred.sign(self.signing_key)  # re-sign after the reset
        return {
            "status": 200,
            "budget_reset": True,
            "privacy_budget_consumed": cred.graph.privacyBudgetConsumed,
        }

    def elicit(
        self,
        token: str,
        category: str,
        attribute_focus: list[str] | None = None,
        max_questions: int = 3,
    ) -> dict:
        self._auth(token, category)
        cred = self._creds.get(category)
        if cred is None:
            return {"status": 404, "detail": "no credential for category"}

        nodes = cred.graph.attributeNodes
        if attribute_focus:
            nodes = [n for n in nodes if n.id in attribute_focus]
        # Greedy information gain: target the least-confident nodes first. Expected
        # IG is monotone in current uncertainty, approximated here by (1 - conf).
        ranked = sorted(nodes, key=lambda n: n.confidence)[: max(1, min(max_questions, 5))]
        questions = [
            ElicitationQuestion(
                id=f"q{i+1}",
                text=_question_for(n.id),
                response_schema={"type": "categorical", "options": ["not at all", "somewhat", "very", "critical"]},
                target_attribute=n.id,
                information_gain=round(1.0 - n.confidence, 4),
            )
            for i, n in enumerate(ranked)
        ]
        return {
            "status": 200,
            "session_id": "elc_" + secrets.token_hex(8),
            "questions": [q.__dict__ for q in questions],
        }

    # --------------------------------------------------------------- persistence
    def _secret_box(self) -> nacl.secret.SecretBox:
        key = blake2b(bytes(self.signing_key), digest_size=nacl.secret.SecretBox.KEY_SIZE, encoder=nacl.encoding.RawEncoder)
        return nacl.secret.SecretBox(key)

    def save(self, path: str | Path) -> None:
        payload = json.dumps({c: cred.to_dict() for c, cred in self._creds.items()}).encode()
        ciphertext = self._secret_box().encrypt(payload)
        Path(path).write_bytes(ciphertext)

    def load(self, path: str | Path) -> None:
        ciphertext = Path(path).read_bytes()
        payload = self._secret_box().decrypt(ciphertext)
        data = json.loads(payload.decode())
        self._creds = {c: PreferenceCredential.from_dict(d) for c, d in data.items()}


def _question_for(attribute: str) -> str:
    templates = {
        "price_sensitivity": "How important is staying within a tight budget for this purchase?",
        "performance": "How demanding are the tasks you'll run on this product?",
        "portability": "How often will you carry or travel with this product?",
        "build_quality": "How much do premium materials and construction matter to you?",
        "durability": "How important is long-term reliability and resistance to failure?",
        "ergonomics": "How important is comfort during long, sustained use?",
        "aesthetics": "How much does the product's design and appearance matter to you?",
        "brand_affinity": "How much do you prefer established premium brands?",
    }
    return templates.get(attribute, f"How important is {attribute.replace('_', ' ')} to you?")
