"""Tests for Bayesian aggregation of extracted signals."""

from preferencelayer.qil import QualityAggregator
from preferencelayer.qil.extract import ExtractedSignal


def _perf(pid, prof, dim, val, conf=0.9):
    return ExtractedSignal(pid, "laptops", prof, "performance", None, dim, val, conf)


def _fail(pid, prof, conf=0.9):
    return ExtractedSignal(pid, "laptops", prof, "failure", "thermal_throttling", None, 0.2, conf)


def test_quality_posterior_brackets_mean():
    sigs = [_perf("p1", "gaming", "thermal", 0.8) for _ in range(20)]
    agg = QualityAggregator().fit(sigs)
    post = agg.quality[("p1", "gaming", "thermal")]
    assert post.credible_lo_90 <= post.posterior_mean <= post.credible_hi_90
    # With many high observations the posterior mean is pulled well above the 0.5 prior.
    assert post.posterior_mean > 0.65
    assert post.evidence_count == 20


def test_more_evidence_shrinks_interval():
    few = QualityAggregator().fit([_perf("p", "gaming", "thermal", 0.7) for _ in range(3)])
    many = QualityAggregator().fit([_perf("p", "gaming", "thermal", 0.7) for _ in range(50)])
    w_few = few.quality[("p", "gaming", "thermal")]
    w_many = many.quality[("p", "gaming", "thermal")]
    width_few = w_few.credible_hi_90 - w_few.credible_lo_90
    width_many = w_many.credible_hi_90 - w_many.credible_lo_90
    assert width_many < width_few


def test_failure_rate_increases_with_failures():
    mostly_ok = QualityAggregator().fit(
        [_perf("p", "gaming", "thermal", 0.6) for _ in range(20)] + [_fail("p", "gaming")]
    )
    mostly_fail = QualityAggregator().fit(
        [_fail("p", "gaming") for _ in range(20)] + [_perf("p", "gaming", "thermal", 0.6)]
    )
    assert mostly_fail.failure[("p", "gaming")].rate_mean > mostly_ok.failure[("p", "gaming")].rate_mean
