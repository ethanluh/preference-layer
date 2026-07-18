"""Tests for the live network fetch adapters (Work Stream B1).

Exercises the OAuth handshake, header assembly, token caching, and crawl-delay
pacing against a FAKE transport -- no network. The real network call is the only
unexercised line; everything around it is covered here.
"""

import pytest

from preferencelayer.qil.ingest.connectors import IFixitConnector, RedditConnector
from preferencelayer.qil.ingest.live_fetch import (
    make_arctic_shift_fetch,
    make_http_fetch,
    make_reddit_fetch,
)


class _FakeResponse:
    def __init__(self, *, json_body=None, text="", status=200):
        self._json = json_body
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class _FakeTransport:
    """Records calls and returns canned responses keyed by call order."""

    def __init__(self, *, token_body=None, get_body=None, get_text=""):
        self.token_body = token_body or {"access_token": "tok-123", "expires_in": 3600}
        self.get_body = get_body
        self.get_text = get_text
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    def post(self, url, *, data=None, headers=None, auth=None, timeout=None):
        self.posts.append({"url": url, "data": data, "headers": headers, "auth": auth})
        return _FakeResponse(json_body=self.token_body)

    def get(self, url, *, headers=None, timeout=None):
        self.gets.append({"url": url, "headers": headers})
        return _FakeResponse(json_body=self.get_body, text=self.get_text)


# --- Reddit -----------------------------------------------------------------

_REDDIT_LISTING = {"data": {"children": [
    {"data": {"id": "abc", "title": "X1 Carbon runs hot", "selftext": "throttles under load",
              "score": 12, "permalink": "/r/thinkpad/abc"}},
]}}


def test_reddit_fetch_does_oauth_then_authenticated_get():
    http = _FakeTransport(get_body=_REDDIT_LISTING)
    fetch = make_reddit_fetch("cid", "secret", "pref-bot/0.1", http=http)
    payload = fetch("https://oauth.reddit.com/r/thinkpad/new")

    # OAuth handshake: client-credentials grant with HTTP basic auth + UA.
    assert http.posts[0]["data"] == {"grant_type": "client_credentials"}
    assert http.posts[0]["auth"] == ("cid", "secret")
    assert http.posts[0]["headers"]["User-Agent"] == "pref-bot/0.1"
    # Authenticated GET carries the bearer token + UA.
    assert http.gets[0]["headers"]["Authorization"] == "bearer tok-123"
    assert http.gets[0]["headers"]["User-Agent"] == "pref-bot/0.1"
    assert payload == _REDDIT_LISTING


def test_reddit_token_is_cached_across_fetches():
    http = _FakeTransport(get_body=_REDDIT_LISTING)
    fetch = make_reddit_fetch("cid", "secret", "ua", http=http)
    fetch("https://oauth.reddit.com/r/a/new")
    fetch("https://oauth.reddit.com/r/b/new")
    assert len(http.posts) == 1   # token fetched once
    assert len(http.gets) == 2    # two listing GETs


def test_reddit_token_refreshes_after_expiry():
    clock = {"t": 0.0}
    http = _FakeTransport(get_body=_REDDIT_LISTING, token_body={"access_token": "t", "expires_in": 100})
    fetch = make_reddit_fetch("c", "s", "ua", http=http, clock=lambda: clock["t"])
    fetch("https://oauth.reddit.com/r/a/new")
    clock["t"] = 1000.0  # well past the 100s - 60s refresh window
    fetch("https://oauth.reddit.com/r/a/new")
    assert len(http.posts) == 2   # re-authenticated


def test_reddit_fetch_feeds_the_connector_parser():
    http = _FakeTransport(get_body=_REDDIT_LISTING)
    fetch = make_reddit_fetch("c", "s", "ua", http=http)
    conn = RedditConnector("laptops", subreddits=["thinkpad"], fetch=fetch)
    docs = list(conn.documents())
    assert len(docs) == 1
    assert docs[0].source_type == "reddit"
    assert "throttles" in docs[0].text
    assert docs[0].upvote_count == 12


def test_reddit_fetch_requires_credentials():
    with pytest.raises(ValueError):
        make_reddit_fetch("", "secret", "ua")


# --- Arctic Shift (Reddit fallback, no OAuth) --------------------------------

_ARCTIC_SHIFT_RESULTS = [
    {"id": "abc", "title": "X1 Carbon runs hot", "selftext": "throttles under load",
     "score": 12, "subreddit": "thinkpad"},
]


def test_arctic_shift_fetch_needs_no_oauth_and_hits_search_endpoint():
    http = _FakeTransport(get_body={"data": _ARCTIC_SHIFT_RESULTS})
    fetch = make_arctic_shift_fetch("pref-bot/0.1", http=http)
    payload = fetch("https://oauth.reddit.com/r/thinkpad/new")

    assert http.posts == []  # no OAuth handshake at all
    assert len(http.gets) == 1
    get = http.gets[0]
    assert get["headers"]["User-Agent"] == "pref-bot/0.1"
    assert "arctic-shift.photon-reddit.com/api/posts/search" in get["url"]
    assert "subreddit=thinkpad" in get["url"]
    assert payload == {"data": {"children": [{"data": _ARCTIC_SHIFT_RESULTS[0]}]}}


def test_arctic_shift_fetch_accepts_bare_list_response():
    http = _FakeTransport(get_body=_ARCTIC_SHIFT_RESULTS)
    fetch = make_arctic_shift_fetch("ua", http=http)
    payload = fetch("https://oauth.reddit.com/r/thinkpad/new")
    assert payload == {"data": {"children": [{"data": _ARCTIC_SHIFT_RESULTS[0]}]}}


def test_arctic_shift_fetch_feeds_the_connector_parser():
    http = _FakeTransport(get_body={"data": _ARCTIC_SHIFT_RESULTS})
    fetch = make_arctic_shift_fetch("ua", http=http)
    conn = RedditConnector("laptops", subreddits=["thinkpad"], fetch=fetch)
    docs = list(conn.documents())
    assert len(docs) == 1
    assert docs[0].source_type == "reddit"
    assert "throttles" in docs[0].text
    assert docs[0].upvote_count == 12


def test_arctic_shift_fetch_requires_user_agent():
    with pytest.raises(ValueError):
        make_arctic_shift_fetch("")


def test_arctic_shift_fetch_rejects_url_without_subreddit():
    fetch = make_arctic_shift_fetch("ua", http=_FakeTransport())
    with pytest.raises(ValueError):
        fetch("https://oauth.reddit.com/nope")


def test_arctic_shift_fetch_omits_after_param_with_no_watermark():
    http = _FakeTransport(get_body={"data": _ARCTIC_SHIFT_RESULTS})
    fetch = make_arctic_shift_fetch("ua", http=http, after_utc=lambda _sub: None)
    fetch("https://oauth.reddit.com/r/thinkpad/new")
    assert "after=" not in http.gets[0]["url"]


def test_arctic_shift_fetch_adds_after_param_from_watermark():
    http = _FakeTransport(get_body={"data": _ARCTIC_SHIFT_RESULTS})
    fetch = make_arctic_shift_fetch("ua", http=http, after_utc=lambda _sub: 1700000000)
    fetch("https://oauth.reddit.com/r/thinkpad/new")
    assert "after=1700000000" in http.gets[0]["url"]


def test_arctic_shift_fetch_reports_records_via_on_records():
    http = _FakeTransport(get_body={"data": _ARCTIC_SHIFT_RESULTS})
    seen = {}

    def on_records(subreddit, records):
        seen["subreddit"] = subreddit
        seen["records"] = records

    fetch = make_arctic_shift_fetch("ua", http=http, on_records=on_records)
    fetch("https://oauth.reddit.com/r/thinkpad/new")
    assert seen["subreddit"] == "thinkpad"
    assert seen["records"] == _ARCTIC_SHIFT_RESULTS


# --- HTTP (iFixit / Notebookcheck) ------------------------------------------

def test_http_fetch_returns_json_for_ifixit():
    body = {"guides": [{"guideid": 9, "title": "X1 fan", "introduction": "thermal fix",
                        "url": "https://ifixit/9", "favorites": 3}]}
    http = _FakeTransport(get_body=body)
    fetch = make_http_fetch(user_agent="ua", http=http)
    conn = IFixitConnector("laptops", start_urls=["https://ifixit/api"], fetch=fetch)
    docs = list(conn.documents())
    assert len(docs) == 1 and docs[0].source_type == "ifixit"


def test_http_fetch_applies_parser_for_html():
    http = _FakeTransport(get_text="<html>reviews here</html>")
    seen = {}

    def parser(html):
        seen["html"] = html
        return {"reviews": [{"id": "r1", "title": "t", "verdict": "good"}]}

    fetch = make_http_fetch(user_agent="ua", parser=parser, http=http)
    out = fetch("https://notebookcheck/x")
    assert seen["html"] == "<html>reviews here</html>"
    assert out["reviews"][0]["id"] == "r1"


def test_http_fetch_honors_crawl_delay():
    slept: list[float] = []
    http = _FakeTransport(get_body=[])
    fetch = make_http_fetch(user_agent="ua", crawl_delay=2.0, http=http,
                            sleep=lambda s: slept.append(s))
    fetch("https://ifixit/api")
    assert slept == [2.0]


def test_http_fetch_requires_user_agent():
    with pytest.raises(ValueError):
        make_http_fetch(user_agent="")
