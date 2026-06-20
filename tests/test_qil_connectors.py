"""Tests for the QIL live-connector seam (Work Stream B1).

The only unplugged piece of a live connector is the network call, exposed as an
injectable ``fetch`` callable. These tests inject a fake fetcher returning a
captured-shape (synthetic, non-identifying) Reddit listing, exercising the REAL
``RedditConnector._parse`` end-to-end through ``documents()`` — and confirm that
an un-injected connector still fails loudly rather than silently doing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preferencelayer.qil.ingest.connectors import (
    IFixitConnector,
    NotebookcheckConnector,
    RawDocument,
    RedditConnector,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"
FIXTURE = FIXTURES / "reddit_listing.json"

# Identifier fields the listing payload carries that must NEVER reach a RawDocument.
_IDENTIFIER_FIELDS = ("author", "author_fullname", "username", "user", "user_id")


def _fetch_from(path: Path):
    def _fetch(_url: str):
        return json.loads(path.read_text())
    return _fetch


def _fake_fetch(_url: str) -> dict:
    return json.loads(FIXTURE.read_text())


def test_reddit_parse_yields_clean_documents():
    conn = RedditConnector("laptops", ["laptops"], fetch=_fake_fetch)
    docs = list(conn.documents())

    # Two valid children (post + comment); empty-text and id-less children dropped.
    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)
    ids = {d.source_local_id for d in docs}
    assert ids == {"abc123", "def456"}

    post = next(d for d in docs if d.source_local_id == "abc123")
    assert post.source_type == "reddit"
    assert post.category == "laptops"
    assert "throttle" in post.text.lower() or "throttling" in post.text.lower()
    assert post.text  # title + selftext joined
    assert post.source_url.startswith("https://www.reddit.com/r/laptops/")
    assert post.upvote_count == 142
    assert post.content_hash  # deterministic dedup key computed


def test_reddit_documents_carry_no_user_identifiers():
    conn = RedditConnector("laptops", ["laptops"], fetch=_fake_fetch)
    for doc in conn.documents():
        # RawDocument is a frozen dataclass with no identifier fields; assert the
        # parser did not smuggle any onto it.
        blob = json.dumps(doc.__dict__).lower()
        for field in _IDENTIFIER_FIELDS:
            assert field not in doc.__dict__
        assert "synthetic_user" not in blob  # author values never propagated


def test_uninjected_connector_raises_clear_scaffold_error():
    conn = RedditConnector("laptops", ["laptops"])  # no fetch injected
    with pytest.raises(NotImplementedError, match="network fetch not configured"):
        list(conn.documents())


@pytest.mark.parametrize("cls,kwargs,fixture,source_type,expected_ids", [
    (IFixitConnector, {"start_urls": ["https://www.ifixit.com/api/2.0/guides"]},
     "ifixit_guides.json", "ifixit", {"12345", "12346"}),
    (NotebookcheckConnector, {"start_urls": ["https://www.notebookcheck.net/laptops"]},
     "notebookcheck_reviews.json", "notebookcheck", {"nbc-001", "nbc-002"}),
])
def test_structured_connectors_parse_clean_documents(cls, kwargs, fixture, source_type, expected_ids):
    conn = cls("laptops", fetch=_fetch_from(FIXTURES / fixture), **kwargs)
    docs = list(conn.documents())

    # Records without an id or without text are dropped.
    assert {d.source_local_id for d in docs} == expected_ids
    for d in docs:
        assert isinstance(d, RawDocument)
        assert d.source_type == source_type
        assert d.category == "laptops"
        assert d.text                      # title + body joined, non-empty
        assert d.source_url                # per-record url present in fixtures
        assert d.content_hash
        # The fixtures seed an `author` on a record; the _emit key-whitelist must
        # never read it, so neither the field nor its value reaches the document.
        blob = json.dumps(d.__dict__).lower()
        for field in _IDENTIFIER_FIELDS:
            assert field not in d.__dict__
        assert "synthetic_user" not in blob  # seeded author value never propagated


def test_ifixit_favorites_map_to_upvote_count():
    conn = IFixitConnector("laptops", ["u"], fetch=_fetch_from(FIXTURES / "ifixit_guides.json"))
    by_id = {d.source_local_id: d for d in conn.documents()}
    assert by_id["12345"].upvote_count == 8  # 'favorites' -> upvote_count
    assert by_id["12346"].upvote_count == 3
