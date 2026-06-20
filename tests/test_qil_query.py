"""Tests for the QIL query service and MCP handler."""

from preferencelayer.qil import QILToolHandler, QualityAggregator, QualityService
from preferencelayer.qil.extract import ExtractedSignal


def _perf(pid, dim, val, conf=0.9):
    return ExtractedSignal(pid, "laptops", "gaming", "performance", None, dim, val, conf)


def _service():
    sigs = (
        [_perf("good", "thermal", 0.85) for _ in range(15)]
        + [_perf("bad", "thermal", 0.30) for _ in range(15)]
    )
    return QualityService(QualityAggregator().fit(sigs))


def test_quality_returns_dimensions_and_ci():
    svc = _service()
    res = svc.quality("good", "gaming")
    assert res["status"] == 200
    th = res["dimensions"]["thermal"]
    lo, hi = th["credible_interval_90"]
    assert lo <= th["posterior_mean"] <= hi


def test_quality_missing_returns_404():
    svc = _service()
    assert svc.quality("nonexistent", "gaming")["status"] == 404


def test_compare_prefers_better_product():
    svc = _service()
    res = svc.compare("good", "bad", "gaming")
    assert res["status"] == 200
    # 'good' has much higher thermal quality, so P(A>B) should be high.
    assert res["dimensions"]["thermal"]["p_a_better"] > 0.9
    assert res["dimensions"]["thermal"]["difference"] > 0


def test_mcp_handler_dispatch():
    handler = QILToolHandler(_service())
    assert handler.call("get_quality", {"product_id": "good", "use_profile": "gaming"})["status"] == 200
    cmp = handler.call("compare_quality",
                       {"product_id_a": "good", "product_id_b": "bad", "use_profile": "gaming"})
    assert cmp["status"] == 200
    assert handler.call("bogus_tool", {})["status"] == 400


def test_quality_rejects_empty_use_profile():
    svc = _service()
    for bad in ("", "   "):
        res = svc.quality("good", bad)
        assert res["status"] == 400
        assert "use_profile" in res["detail"]


def test_compare_rejects_empty_use_profile():
    svc = _service()
    assert svc.compare("good", "bad", "")["status"] == 400


def test_mcp_handler_rejects_empty_use_profile():
    # The MCP surface delegates to the service, so the guard covers it too.
    handler = QILToolHandler(_service())
    assert handler.call("get_quality", {"product_id": "good", "use_profile": ""})["status"] == 400
