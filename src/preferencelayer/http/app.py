"""FastAPI app exposing the three PTP endpoints (spec §4).

``build_app(store)`` returns a FastAPI application that serves:

* ``GET  /preference`` — scoped, freshly re-signed credential (spec §4.2)
* ``POST /outcome``    — enqueue + apply a DP update, re-sign (spec §4.3)
* ``POST /elicit``     — propose high-information-gain questions (spec §4.4)

Auth is enforced at the boundary: every request must carry
``Authorization: Bearer <agent-token>``. Missing/expired tokens -> 401;
out-of-scope categories -> 403 (the store raises ``AuthError``); absent
credential -> 404. The store's own logic is reused unchanged — this module adds
no preference logic and persists no behavioral data.
"""

from __future__ import annotations

from typing import Any

from ..ptp.store import AuthError, CredentialStore

try:  # optional dependency, only needed to actually serve HTTP
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "HTTP transport requires the 'http' extra: pip install 'preferencelayer[http]'"
    ) from exc


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


def _bearer(authorization: str | None) -> str:
    """Extract the bearer token or raise 401 at the boundary."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty bearer token")
    return token


def build_app(store: CredentialStore) -> "FastAPI":
    """Build the PTP HTTP app over an existing credential store."""
    app = FastAPI(title="PreferenceLayer PTP", version="0.1")

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

    return app
