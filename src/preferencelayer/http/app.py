"""FastAPI app exposing the PTP endpoints (spec §4).

``build_app(store)`` returns a FastAPI application that serves:

* ``GET  /preference`` — scoped, freshly re-signed credential (spec §4.2)
* ``POST /outcome``    — enqueue + apply a DP update, re-sign (spec §4.3)
* ``POST /elicit``     — propose high-information-gain questions (spec §4.4)

and the OAuth 2.0 Device Authorization Grant (RFC 8628, spec §4.1) used to
obtain an agent token in the first place:

* ``POST /device/code``     — agent requests a device + user code
* ``POST /token``           — agent polls for the scoped bearer token
* ``GET  /device``          — owner inspects a pending request's scope
* ``POST /device/decision`` — owner approves or denies a pending request

Auth is enforced at the boundary: every credential request must carry
``Authorization: Bearer <agent-token>``. Missing/expired tokens -> 401;
out-of-scope categories -> 403 (the store raises ``AuthError``); absent
credential -> 404. The store's own logic — and ``DeviceFlowAuthority`` — are
reused unchanged: this module adds no preference logic and persists no
behavioral data.
"""

from __future__ import annotations

from typing import Any, NoReturn

from ..ptp.device_flow import DeviceFlowAuthority, DeviceFlowError
from ..ptp.store import AuthError, CredentialStore

try:  # optional dependency, only needed to actually serve HTTP
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "HTTP transport requires the 'http' extra: pip install 'preferencelayer[http]'"
    ) from exc

# RFC 8628 device-code grant type for the token endpoint.
DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class PreferenceRequest(BaseModel):
    category: str
    query_context: str = ""
    disclosure_scope: list[str] | None = None
    min_confidence: float = 0.0


class OutcomeRequest(BaseModel):
    category: str
    product_id: str
    outcome_type: str = Field(..., description="purchase|return|dwell|rating|elicitation")
    use_context: str = ""
    timestamp: str | None = None
    rating: float | None = None
    elicitation_weights: dict[str, float] | None = None


class ElicitRequest(BaseModel):
    category: str
    attribute_focus: list[str] | None = None
    max_questions: int = 3


class DeviceCodeRequest(BaseModel):
    client_id: str
    scope: list[str] | None = None


class TokenRequest(BaseModel):
    grant_type: str
    device_code: str
    client_id: str | None = None


class DecisionRequest(BaseModel):
    user_code: str
    decision: str = Field(..., description="approve|deny")


def _bearer(authorization: str | None) -> str:
    """Extract the bearer token or raise 401 at the boundary."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty bearer token")
    return token


def build_app(
    store: CredentialStore,
    *,
    device_authority: "DeviceFlowAuthority | None" = None,
) -> "FastAPI":
    """Build the PTP HTTP app over an existing credential store.

    ``device_authority`` powers the RFC 8628 device-flow routes (spec §4.1). If
    omitted, one is constructed bound to ``store`` so an agent can obtain a token
    out of the box; pass an explicit instance to configure the verification URI.
    """
    app = FastAPI(title="PreferenceLayer PTP", version="0.1")
    authority = device_authority or DeviceFlowAuthority(store)

    def auth_header(authorization: str | None = Header(default=None)) -> str:
        return _bearer(authorization)

    def _auth_status(err: AuthError) -> int:
        """invalid/expired token -> 401; out-of-scope category -> 403 (spec §4.2)."""
        return 401 if "invalid or expired" in str(err) else 403

    def _handle(result: dict[str, Any]) -> dict[str, Any]:
        """Translate a store result dict into an HTTP response / error.

        The internal ``status`` key drives the HTTP status code only; it is
        off-spec in the response body (§4.2/§4.3), so strip it before returning.
        """
        status = result.get("status", 200)
        if status == 404:
            raise HTTPException(status_code=404, detail=result.get("detail", "not found"))
        if status == 403:
            raise HTTPException(status_code=403, detail=result.get("detail", "forbidden"))
        return {k: v for k, v in result.items() if k != "status"}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/preference")
    def get_preference(req: PreferenceRequest, token: str = Depends(auth_header)) -> dict[str, Any]:
        try:
            result = store.get_preference(
                token,
                category=req.category,
                query_context=req.query_context,
                disclosure_scope=req.disclosure_scope,
                min_confidence=req.min_confidence,
            )
        except AuthError as e:
            raise HTTPException(status_code=_auth_status(e), detail=str(e))
        return _handle(result)

    @app.post("/outcome", status_code=202)
    def post_outcome(req: OutcomeRequest, token: str = Depends(auth_header)) -> dict[str, Any]:
        try:
            result = store.submit_outcome(
                token,
                category=req.category,
                product_id=req.product_id,
                outcome_type=req.outcome_type,
                use_context=req.use_context,
                rating=req.rating,
                elicitation_weights=req.elicitation_weights,
            )
        except AuthError as e:
            raise HTTPException(status_code=_auth_status(e), detail=str(e))
        return _handle(result)

    @app.post("/elicit")
    def post_elicit(req: ElicitRequest, token: str = Depends(auth_header)) -> dict[str, Any]:
        try:
            result = store.elicit(
                token,
                category=req.category,
                attribute_focus=req.attribute_focus,
                max_questions=req.max_questions,
            )
        except AuthError as e:
            raise HTTPException(status_code=_auth_status(e), detail=str(e))
        return _handle(result)

    # ------------------------------------------------------- device flow (§4.1)
    @app.post("/device/code")
    def device_code(req: DeviceCodeRequest) -> dict[str, Any]:
        try:
            resp = authority.request_device_code(req.client_id, scope=req.scope)
        except DeviceFlowError as e:
            _raise_device_error(e)
        return {
            "device_code": resp.device_code,
            "user_code": resp.user_code,
            "verification_uri": resp.verification_uri,
            "verification_uri_complete": resp.verification_uri_complete,
            "expires_in": resp.expires_in,
            "interval": resp.interval,
        }

    @app.post("/token")
    def token(req: TokenRequest) -> dict[str, Any]:
        if req.grant_type != DEVICE_CODE_GRANT:
            # RFC 8628 reuses the OAuth 2.0 token endpoint; an unsupported grant
            # is a 400 invalid_request per RFC 6749 §5.2.
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unsupported_grant_type",
                    "error_description": f"grant_type must be '{DEVICE_CODE_GRANT}'",
                },
            )
        try:
            access_token = authority.poll_token(req.device_code)
        except DeviceFlowError as e:
            _raise_device_error(e)
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": authority.token_ttl,
        }

    @app.get("/device")
    def device_pending(user_code: str) -> dict[str, Any]:
        """Owner-side: surface the categories a pending request is asking for."""
        try:
            scope = authority.pending_scope(user_code)
        except DeviceFlowError as e:
            _raise_device_error(e)
        return {"user_code": user_code, "scope": scope}

    @app.post("/device/decision")
    def device_decision(req: DecisionRequest) -> dict[str, Any]:
        """Owner-side approval surface.

        NOTE: owner authentication on this route is a production follow-up — the
        v0.1 store is a local single-user daemon, so the owner is the local user.
        """
        decision = req.decision.strip().lower()
        if decision not in ("approve", "deny"):
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_request", "error_description": "decision must be 'approve' or 'deny'"},
            )
        try:
            if decision == "approve":
                authority.approve(req.user_code)
            else:
                authority.deny(req.user_code)
        except DeviceFlowError as e:
            _raise_device_error(e)
        return {"user_code": req.user_code, "decision": decision, "status": "ok"}

    return app


def _raise_device_error(e: "DeviceFlowError") -> "NoReturn":
    """Map an RFC 8628 ``DeviceFlowError`` to an HTTP 400 OAuth error body.

    The token endpoint signals pending/slow-down/denied/expired/invalid via a
    400 with an ``error`` code (RFC 8628 §3.5 / RFC 6749 §5.2), not an HTTP-level
    status, so clients parse the body to decide whether to keep polling.
    """
    raise HTTPException(status_code=400, detail={"error": e.error, "error_description": str(e)})
