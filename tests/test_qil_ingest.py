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
