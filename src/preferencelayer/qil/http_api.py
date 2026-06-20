"""QIL HTTP API: serve /quality and /compare over FastAPI (Work Stream B4).

Wraps the in-process :class:`~preferencelayer.qil.query.QualityService` behind
HTTP, mirroring the contract in ``docs/architecture.md`` (``POST /quality`` and
``POST /compare``). The HTTP layer adds NO scoring logic -- it maps the request
to a ``QualityService`` call and the result to JSON -- so the HTTP and MCP
surfaces stay byte-for-byte consistent.

Latency target: < 200ms p95. ``QualityService`` is an in-memory dict lookup
(O(#dimensions)), so request time is dominated by (de)serialization and
transport; there is no DB round-trip on the read path (posteriors are
precomputed by the nightly refit). ``fastapi``/``uvicorn`` are an optional
``[api]`` extra so the core package stays dependency-light; this module imports
them lazily inside :func:`build_app`.

Privacy: requests carry only ``product_id`` + a use profile -- never a user
identifier. The use profile is "how the product is used", consistent with the
QIL invariant.

Note: this module intentionally does NOT use ``from __future__ import
annotations``. FastAPI resolves endpoint parameter annotations at decoration
time; stringized annotations of function-local Pydantic models would not resolve
and FastAPI would mis-read the body as query params.
"""

from typing import Any, List, Optional

from .query import QualityService


def build_app(service: QualityService) -> Any:
    """Construct a FastAPI app serving /quality, /compare, /healthz over ``service``.

    Imported lazily: requires the ``[api]`` extra (``pip install
    'preferencelayer[api]'``).
    """
    try:
        from fastapi import FastAPI, Response
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "QIL HTTP API needs the optional [api] extra: pip install 'preferencelayer[api]'"
        ) from exc

    class QualityRequest(BaseModel):
        product_id: str
        # "How the product is used", e.g. 'gaming' | 'travel' | 'professional'.
        # NOT a user identifier (QIL holds none).
        use_profile: str
        dimensions: Optional[List[str]] = Field(
            default=None, description="Optional subset of quality dimensions."
        )

    class CompareRequest(BaseModel):
        product_id_a: str
        product_id_b: str
        use_profile: str

    app = FastAPI(
        title="PreferenceLayer QIL API",
        version="0.1",
        description="Use-profile-conditioned product quality intelligence.",
    )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/quality")
    def quality(req: QualityRequest, response: Response) -> dict:
        result = service.quality(req.product_id, req.use_profile, req.dimensions)
        # Surface the service's own status (e.g. 404 when no data) as the HTTP code.
        response.status_code = int(result.get("status", 200))
        return result

    @app.post("/compare")
    def compare(req: CompareRequest, response: Response) -> dict:
        result = service.compare(req.product_id_a, req.product_id_b, req.use_profile)
        response.status_code = int(result.get("status", 200))
        return result

    return app
