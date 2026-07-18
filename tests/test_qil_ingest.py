"""Tests for the QIL ingestion pipeline (Work Stream B1, scaffold)."""

from pathlib import Path

import pytest

from preferencelayer.qil import QILExtractor, generate
from preferencelayer.qil.ingest import (
    CanonicalProduct,
    FixtureConnector,
    InMemorySink,
    ProductRegistry,
    RateLimiter,
    RawDocument,
    RobotsPolicy,
    normalize_model_string,
    run_daily,
    schema_sql,
    signal_rows_from_json,
    signal_rows_to_json,
)
from preferencelayer.qil.ingest.connectors import RedditConnector, _assert_no_user_identifiers

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


# --- normalization --------------------------------------------------------

def test_normalize_strips_punctuation_and_case():
    assert normalize_model_string("ThinkPad X1 Carbon (Gen 12)!") == "thinkpad x1 carbon gen 12"


def test_registry_matches_surface_variants_to_canonical_id():
    reg = ProductRegistry(threshold=0.5).add(
        CanonicalProduct("lenovo-thinkpad-x1-carbon-gen12", "laptops",
                         "Lenovo ThinkPad X1 Carbon Gen 12")
    )
    hit = reg.match("loving my new thinkpad x1 carbon gen 12 for work", category="laptops")
    assert hit is not None
    assert hit[0].product_id == "lenovo-thinkpad-x1-carbon-gen12"


def test_registry_returns_none_below_threshold():
    reg = ProductRegistry(threshold=0.5).add(
        CanonicalProduct("dell-xps-15-9530", "laptops", "Dell XPS 15 9530")
    )
    assert reg.match("some unrelated gaming mouse review", category="laptops") is None


def test_registry_respects_category_filter():
    reg = ProductRegistry(threshold=0.5).add(
        CanonicalProduct("dell-xps-15-9530", "laptops", "Dell XPS 15 9530")
    )
    assert reg.match("dell xps 15 9530", category="keyboards") is None


# --- politeness -----------------------------------------------------------

def test_robots_disallow_and_allow_override():
    body = (FIXTURES / "robots_sample.txt").read_text()
    pol = RobotsPolicy(body, user_agent="genericbot")
    assert pol.can_fetch("https://x.com/public/page") is True
    assert pol.can_fetch("https://x.com/private/secret") is False
    assert pol.can_fetch("https://x.com/private/public/ok") is True  # Allow overrides
    assert pol.crawl_delay == 2.0


def test_robots_selects_named_user_agent_group():
    body = (FIXTURES / "robots_sample.txt").read_text()
    pol = RobotsPolicy(body, user_agent="PreferenceLayerBot")
    assert pol.can_fetch("https://x.com/admin/panel") is False
    assert pol.crawl_delay == 5.0


def test_robots_shared_ruleset_applies_to_grouped_user_agents():
    # Per RFC 9309, consecutive User-agent lines share the following ruleset.
    body = (
        "User-agent: botA\n"
        "User-agent: botB\n"
        "Disallow: /private\n"
        "Crawl-delay: 3\n"
        "\n"
        "User-agent: *\n"
        "Disallow: /\n"
    )
    for ua in ("botA", "botB"):
        pol = RobotsPolicy(body, user_agent=ua)
        assert pol.can_fetch("https://x.com/public/page") is True
        assert pol.can_fetch("https://x.com/private/secret") is False
        assert pol.crawl_delay == 3.0
    # A following group must not leak into the grouped agents.
    other = RobotsPolicy(body, user_agent="otherbot")
    assert other.can_fetch("https://x.com/public/page") is False


def test_rate_limiter_blocks_when_bucket_empty():
    slept: list[float] = []
    t = {"now": 0.0}
    rl = RateLimiter(rate=2.0, burst=1.0, clock=lambda: t["now"], sleep=lambda s: slept.append(s))
    rl.acquire()   # consumes the initial burst token, no sleep
    rl.acquire()   # bucket empty -> must sleep ~1/rate
    assert slept and slept[0] == pytest.approx(0.5, rel=1e-6)


# --- privacy invariant ----------------------------------------------------

def test_raw_document_rejects_user_identifier_fields():
    with pytest.raises(ValueError):
        _assert_no_user_identifiers({"source_local_id": "x", "text": "hi", "author": "alice"})


def test_content_hash_is_stable_and_source_scoped():
    a = RawDocument("reddit", "id1", "laptops", "same body")
    b = RawDocument("reddit", "id1", "laptops", "same body")
    c = RawDocument("ifixit", "id1", "laptops", "same body")
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash  # source-scoped dedup


# --- live connector scaffold boundary -------------------------------------

def test_live_connector_fetch_is_unconfigured_scaffold():
    conn = RedditConnector("laptops", subreddits=["thinkpad"])
    with pytest.raises(NotImplementedError):
        list(conn.documents())


# --- end-to-end against fixtures ------------------------------------------

def _extractor():
    return QILExtractor().fit(generate().train)


def _registry():
    return ProductRegistry(threshold=0.5).add(
        CanonicalProduct("lenovo-thinkpad-x1-carbon-gen12", "laptops",
                         "Lenovo ThinkPad X1 Carbon Gen 12",
                         aliases=("thinkpad x1 carbon gen 12",))
    ).add(
        CanonicalProduct("dell-xps-15-9530", "laptops", "Dell XPS 15 9530",
                         aliases=("dell xps 15 9530",))
    )


def test_run_daily_lands_normalized_rows_and_drops_unmatched():
    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    stats = run_daily([conn], _registry(), _extractor(), sink)

    assert stats.fetched == 3          # three fixture docs
    assert stats.matched == 2          # two mention canonical products
    assert stats.unmatched == 1        # the vague third post is dropped
    assert stats.written == 2
    ids = {r.product_id for r in sink.rows}
    assert ids == {"lenovo-thinkpad-x1-carbon-gen12", "dell-xps-15-9530"}
    # Use-profile is always set (a predicted, use-profile-conditioned signal).
    assert all(r.use_profile for r in sink.rows)


def test_run_daily_is_idempotent_on_rerun():
    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    ext, reg = _extractor(), _registry()
    run_daily([conn], reg, ext, sink)
    # Second pass over the same fixture writes nothing new (dedup by content_hash).
    again = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    stats2 = run_daily([again], reg, ext, sink)
    assert stats2.written == 0
    assert len(sink.rows) == 2


# --- quality_dim span tagger (closes the zero-GP-posteriors gap) -----------

def test_no_tagger_leaves_quality_dim_none():
    # Default path (no tagger): quality_dim stays None, so the aggregator forms
    # zero GP quality posteriors -- the gap the tagger exists to close.
    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    run_daily([conn], _registry(), _extractor(), sink)
    assert all(r.quality_dim is None for r in sink.rows)


def test_quality_tagger_populates_quality_dim():
    from preferencelayer.qil.quality_spans import QualityDimTagger

    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    run_daily([conn], _registry(), _extractor(), sink, quality_tagger=QualityDimTagger())
    assert any(r.quality_dim is not None for r in sink.rows)


def test_tagged_signals_form_gp_posteriors():
    # End-to-end claim: a tagged quality_dim flows into a GP quality posterior.
    # signal_type is fixed to a non-failure value to isolate the GP path from the
    # learned signal-type head (which is tested separately).
    from preferencelayer.qil.aggregate import QualityAggregator
    from preferencelayer.qil.extract import ExtractedSignal
    from preferencelayer.qil.quality_spans import QualityDimTagger

    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    run_daily([conn], _registry(), _extractor(), sink, quality_tagger=QualityDimTagger())

    signals = [
        ExtractedSignal(
            product_id=r.product_id, category=r.category, use_profile=r.use_profile,
            signal_type="performance", failure_mode=None, quality_dim=r.quality_dim,
            signal_value=r.signal_value, confidence=r.model_confidence,
        )
        for r in sink.rows if r.quality_dim is not None
    ]
    agg = QualityAggregator().fit(signals)
    assert len(agg.quality) > 0  # was 0 before the tagger populated quality_dim


# --- JSON persistence (stopgap before a real DB; see docs/whats-missing.md B4) --

def test_signal_rows_json_round_trip_preserves_fields():
    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    run_daily([conn], _registry(), _extractor(), sink)
    assert len(sink.rows) == 2  # sanity: fixture has two matched rows

    restored = signal_rows_from_json(signal_rows_to_json(sink.rows))
    assert restored == sink.rows


def test_signal_rows_from_json_rows_dedup_like_fresh_rows():
    # Rows reloaded via write() must key into InMemorySink's dedup set the same
    # way as freshly-ingested rows, so a preloaded row blocks a duplicate refetch.
    conn = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    sink = InMemorySink()
    run_daily([conn], _registry(), _extractor(), sink)

    reloaded_sink = InMemorySink()
    reloaded_sink.write(signal_rows_from_json(signal_rows_to_json(sink.rows)))
    again = FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")
    stats = run_daily([again], _registry(), _extractor(), reloaded_sink)
    assert stats.written == 0
    assert len(reloaded_sink.rows) == 2


def test_schema_sql_defines_both_tables_with_no_user_id():
    import re

    ddl = schema_sql().lower()
    assert "create table" in ddl
    assert "product_signal" in ddl and "quality_posterior" in ddl
    assert "use_profile" in ddl
    # No user-identifier columns leak into the QIL schema. Strip SQL comment lines
    # first so prose like "authoritative" can't trip a naive substring check, then
    # look for the forbidden token as a whole word (a would-be column name).
    code = "\n".join(ln for ln in ddl.splitlines() if not ln.strip().startswith("--"))
    for forbidden in ("user_id", "username", "author", "email", "account"):
        assert re.search(rf"\b{forbidden}\b", code) is None
