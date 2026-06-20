"""Tests for the QIL HTTP API (Work Stream B4).

Uses fastapi.testclient (no running server). Skips cleanly if the [api]/[dev]
extra (fastapi + httpx) is not installed.
"""

import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from preferencelayer.qil import QualityAggregator, QualityService, build_app  # noqa: E402
from preferencelayer.qil.extract import ExtractedSignal  # noqa: E402


def _perf(pid, dim, val, conf=0.9):
    return ExtractedSignal(pid, "laptops", "gaming", "performance", None, dim, val, conf)


def _client() -> TestClient:
    sigs = (
        [_perf("good", "thermal", 0.85) for _ in range(15)]
        + [_perf("bad", "thermal", 0.30) for _ in range(15)]
    )
    service = QualityService(QualityAggregator().fit(sigs))
    return TestClient(build_app(service))


def test_healthz():
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_quality_endpoint_returns_dimensions_and_ci():
    r = _client().post("/quality", json={"product_id": "good", "use_profile": "gaming"})
    assert r.status_code == 200
    body = r.json()
    th = body["dimensions"]["thermal"]
    lo, hi = th["credible_interval_90"]
    assert lo <= th["posterior_mean"] <= hi


def test_quality_missing_maps_to_http_404():
    r = _client().post("/quality", json={"product_id": "nope", "use_profile": "gaming"})
    assert r.status_code == 404


def test_quality_honors_dimension_subset():
    r = _client().post("/quality",
                       json={"product_id": "good", "use_profile": "gaming", "dimensions": ["thermal"]})
    assert r.status_code == 200
    assert set(r.json()["dimensions"]) == {"thermal"}


def test_compare_endpoint_prefers_better_product():
    r = _client().post("/compare",
                       json={"product_id_a": "good", "product_id_b": "bad", "use_profile": "gaming"})
    assert r.status_code == 200
    th = r.json()["dimensions"]["thermal"]
    assert th["p_a_better"] > 0.9 and th["difference"] > 0


def test_compare_insufficient_overlap_maps_to_404():
    r = _client().post("/compare",
                       json={"product_id_a": "good", "product_id_b": "nope", "use_profile": "gaming"})
    assert r.status_code == 404


def test_request_validation_rejects_missing_fields():
    r = _client().post("/quality", json={"use_profile": "gaming"})  # no product_id
    assert r.status_code == 422  # FastAPI/Pydantic validation


def test_quality_p95_under_200ms():
    """Smoke latency check: the read path is an in-memory lookup, so p95 << 200ms.

    Not a substitute for a real load test, but guards against an accidental
    per-request blow-up (e.g. refitting on every call).
    """
    client = _client()
    payload = {"product_id": "good", "use_profile": "gaming"}
    client.post("/quality", json=payload)  # warm up
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        client.post("/quality", json=payload)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies)) - 1]
    assert p95 < 200.0, f"p95={p95:.1f}ms"
