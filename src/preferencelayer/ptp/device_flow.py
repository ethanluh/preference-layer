"""OAuth 2.0 Device Authorization Grant (RFC 8628) for PTP agent auth.

PTP spec §4.1 specifies that agents authenticate via the device flow. The Phase 0
prototype minted opaque bearer tokens directly (``CredentialStore.authorize_agent``).
This module adds the RFC 8628 handshake *in front of* that call, so the token an
agent ultimately receives is a normal scoped store token — the existing
``_auth`` / scope / revoke path is reused unchanged.

The flow (RFC 8628 §3):

1. Agent -> ``request_device_code(client_id, scope)`` -> ``DeviceCodeResponse``
   with ``device_code``, ``user_code``, ``verification_uri``,
   ``verification_uri_complete``, ``expires_in``, ``interval``.
2. The user visits the verification URI, enters the ``user_code``, and the store
   owner ``approve(user_code)`` (explicit, matching the existing authorization
   model) or ``deny(user_code)``.
3. Agent polls ``poll_token(device_code)`` -> while pending raises
   ``authorization_pending`` / ``slow_down``; on approval returns the scoped
   bearer token; on denial/expiry raises ``access_denied`` / ``expired_token``.

The authority is in-memory (single-process) for v0.1, mirroring the prototype's
in-memory store. No raw behavioral data is involved; this is purely auth.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from .store import CredentialStore

# Human-friendly user-code alphabet (RFC 8628 §6.1 recommends an easily-typed set;
# excludes ambiguous chars 0/O/1/I).
_USER_CODE_ALPHABET = "BCDFGHJKLMNPQRSTVWXZ23456789"


class DeviceFlowError(RuntimeError):
    """Base for RFC 8628 token-endpoint errors; ``error`` is the OAuth code."""

    error = "invalid_request"


class AuthorizationPending(DeviceFlowError):
    error = "authorization_pending"


class SlowDown(DeviceFlowError):
    error = "slow_down"


class AccessDenied(DeviceFlowError):
    error = "access_denied"


class ExpiredToken(DeviceFlowError):
    error = "expired_token"


class InvalidDeviceCode(DeviceFlowError):
    error = "invalid_grant"


@dataclass
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass
class _PendingAuth:
    device_code: str
    user_code: str
    client_id: str
    scope: list[str]
    expires_at: float
    interval: int
    state: str = "pending"        # pending | approved | denied
    token: str | None = None      # set on approval
    last_poll: float = field(default_factory=float)


def _gen_user_code() -> str:
    raw = "".join(secrets.choice(_USER_CODE_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"  # e.g. "WDJB-MJHT"


class DeviceFlowAuthority:
    """In-memory RFC 8628 authority that issues store-backed scoped tokens.

    Bind it to a :class:`CredentialStore`; on approval it calls
    ``store.authorize_agent(client_id, scope, ttl_seconds)`` so the issued token
    is revocable and scope-checked exactly like any other agent token.
    """

    def __init__(
        self,
        store: CredentialStore,
        verification_uri: str = "https://preferencelayer.io/device",
        code_ttl: int = 600,
        interval: int = 5,
        token_ttl: int = 86_400,
    ):
        self.store = store
        self.verification_uri = verification_uri
        self.code_ttl = code_ttl
        self.interval = interval
        self.token_ttl = token_ttl
        self._by_device: dict[str, _PendingAuth] = {}
        self._by_user: dict[str, str] = {}  # user_code -> device_code

    # ----------------------------------------------------------- step 1: request
    def request_device_code(self, client_id: str, scope: list[str] | None = None) -> DeviceCodeResponse:
        scope = scope or ["*"]
        device_code = "dev_" + secrets.token_urlsafe(32)
        user_code = _gen_user_code()
        while user_code in self._by_user:  # avoid collision
            user_code = _gen_user_code()
        pending = _PendingAuth(
            device_code=device_code,
            user_code=user_code,
            client_id=client_id,
            scope=scope,
            expires_at=time.time() + self.code_ttl,
            interval=self.interval,
        )
        self._by_device[device_code] = pending
        self._by_user[user_code] = device_code
        return DeviceCodeResponse(
            device_code=device_code,
            user_code=user_code,
            verification_uri=self.verification_uri,
            verification_uri_complete=f"{self.verification_uri}?user_code={user_code}",
            expires_in=self.code_ttl,
            interval=self.interval,
        )

    # ------------------------------------------------ step 2: user-side decision
    def approve(self, user_code: str) -> None:
        """Owner approves a pending request, minting a scoped store token."""
        pending = self._lookup_user(user_code)
        if pending.state == "denied":
            raise AccessDenied("request was already denied")
        if self._expired(pending):
            raise ExpiredToken("user code expired")
        if pending.state == "approved":
            return
        pending.token = self.store.authorize_agent(
            pending.client_id, scope=pending.scope, ttl_seconds=self.token_ttl
        )
        pending.state = "approved"

    def deny(self, user_code: str) -> None:
        pending = self._lookup_user(user_code)
        pending.state = "denied"

    # --------------------------------------------------------- step 3: poll token
    def poll_token(self, device_code: str) -> str:
        """Agent polls for the token. Raises RFC 8628 errors until resolved."""
        pending = self._by_device.get(device_code)
        if pending is None:
            raise InvalidDeviceCode("unknown device_code")
        if self._expired(pending):
            self._forget(pending)
            raise ExpiredToken("device_code expired")

        now = time.time()
        # Enforce the polling interval (RFC 8628 §3.5): too-fast polling -> slow_down.
        if pending.last_poll and (now - pending.last_poll) < pending.interval:
            pending.last_poll = now
            raise SlowDown("polling too frequently")
        pending.last_poll = now

        if pending.state == "denied":
            self._forget(pending)
            raise AccessDenied("user denied the authorization request")
        if pending.state == "approved":
            token = pending.token
            self._forget(pending)  # device_code is single-use
            assert token is not None
            return token
        raise AuthorizationPending("authorization pending; keep polling")

    # ----------------------------------------------------------------- internals
    def _lookup_user(self, user_code: str) -> _PendingAuth:
        device_code = self._by_user.get(user_code.strip().upper())
        if device_code is None:
            raise InvalidDeviceCode(f"unknown user_code '{user_code}'")
        return self._by_device[device_code]

    def _expired(self, pending: _PendingAuth) -> bool:
        return time.time() > pending.expires_at

    def _forget(self, pending: _PendingAuth) -> None:
        self._by_device.pop(pending.device_code, None)
        self._by_user.pop(pending.user_code, None)
