"""Tests for the GP quality model and the nightly refit job (Work Stream B3)."""

from datetime import datetime, timezone

from preferencelayer.qil import (
    GPHyperparams,
    InMemoryPosteriorSink,
    QualityAggregator,
    fit_gp_posterior,
    run_nightly_refit,
)
from preferencelayer.qil.extract import ExtractedSignal


def _perf(pid, prof, dim, val, conf=0.9, t=None):
    return ExtractedSignal(pid, "laptops", prof, "performance", None, dim, val, conf, observed_at=t)


# --- GP model ------------------------------------------------------------

def test_gp_no_observations_returns_prior():
    post = fit_gp_posterior([], [], [], query_time=0.0, hp=GPHyperparams(prior_mean=0.5))
    assert post.mean == 0.5
    assert post.evidence_count == 0
    assert post.credible_lo_90 < 0.5 < post.credible_hi_90


def test_gp_posterior_pulled_toward_observations():
    times = [0.0] * 20
    post = fit_gp_posterior(times, [0.85] * 20, [0.9] * 20, query_time=0.0)
    assert post.mean > 0.7
    assert post.credible_lo_90 <= post.mean <= post.credible_hi_90


def test_gp_more_evidence_shrinks_interval():
    few = fit_gp_posterior([0.0] * 3, [0.7] * 3, [0.9] * 3)
    many = fit_gp_posterior([0.0] * 50, [0.7] * 50, [0.9] * 50)
    assert (many.credible_hi_90 - many.credible_lo_90) < (few.credible_hi_90 - few.credible_lo_90)


def test_gp_temporal_kernel_tracks_recent_observations():
    # Quality drifts down over release time; querying near the recent (high-t)
    # observations should land below querying near the old (low-t) ones.
    times = [0.0, 30.0, 60.0, 300.0, 330.0, 360.0]
    values = [0.8, 0.8, 0.8, 0.4, 0.4, 0.4]
    hp = GPHyperparams(lengthscale_days=40.0)
    early = fit_gp_posterior(times, values, [0.9] * 6, query_time=30.0, hp=hp)
    late = fit_gp_posterior(times, values, [0.9] * 6, query_time=330.0, hp=hp)
    assert early.mean > late.mean  # the kernel localizes in time


def test_low_confidence_observation_moves_posterior_less():
    strong = fit_gp_posterior([0.0], [0.9], [0.95])
    weak = fit_gp_posterior([0.0], [0.9], [0.05])
    # Both pull above the 0.5 prior, but the low-confidence one pulls less.
    assert strong.mean > weak.mean > 0.5


# --- aggregator contract preserved ---------------------------------------

def test_aggregator_quality_contract_unchanged():
    agg = QualityAggregator().fit([_perf("p", "gaming", "thermal", 0.8) for _ in range(20)])
    post = agg.quality[("p", "gaming", "thermal")]
    assert post.credible_lo_90 <= post.posterior_mean <= post.credible_hi_90
    assert post.posterior_mean > 0.65
    assert post.evidence_count == 20


# --- nightly refit job ---------------------------------------------------

def test_refit_writes_parameters_only_and_decays_freshness():
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    sigs = (
        [_perf("fresh", "gaming", "thermal", 0.8, t=0.0) for _ in range(10)]
        + [_perf("stale", "gaming", "thermal", 0.8, t=365.0) for _ in range(10)]
    )
    sink = InMemoryPosteriorSink()
    n = run_nightly_refit(sigs, sink, now=now)
    assert n == 2
    fresh = sink.rows[("fresh", "gaming", "thermal")]
    stale = sink.rows[("stale", "gaming", "thermal")]
    # Parameters only: the row has no raw observation list, just posterior params.
    assert hasattr(fresh, "posterior_mean") and not hasattr(fresh, "observations")
    assert fresh.last_refit == now
    # Freshness decays with observation age (365d ~ one half-life).
    assert fresh.freshness_score > stale.freshness_score
    assert stale.freshness_score < 0.6


def test_refit_is_idempotent_upsert():
    sigs = [_perf("p", "gaming", "thermal", 0.7, t=0.0) for _ in range(5)]
    sink = InMemoryPosteriorSink()
    run_nightly_refit(sigs, sink)
    run_nightly_refit(sigs, sink)  # second pass upserts the same key
    assert len(sink.rows) == 1
