"""Live network fetch adapters for the QIL connectors (Work Stream B1).

The connectors (``connectors.py``) already carry the polite-crawl wiring, the
parsers, and a single injectable network boundary: a ``fetch(url) -> payload``
callable. This module supplies the *real* ``fetch`` callables so an operator can
run live ingestion by injecting credentials -- no connector changes needed.

    Reddit:        OAuth2 client-credentials -> authenticated listing GET (JSON)
    iFixit:        polite HTTP GET -> guides JSON
    Notebookcheck: polite HTTP GET -> HTML, parsed to records by an injected parser

Data-source strategy (``docs/data-source-strategy.md``): Reddit runs on the
research/free tier and is the only source wired by default in
``cli.build_live_connectors``; iFixit and Notebookcheck are **parked** -- their
adapters below are retained and tested, but not crawled by default. They are a
later step, gated on a proven data gap after Reddit + retailer return data.

The HTTP transport is injectable (``http=``) so the OAuth handshake, header
assembly, and crawl-delay pacing are unit-tested against a fake transport with no
network. ``requests`` is an optional dependency (the ``[amazon]``/HTTP extras
already pull it in for other paths); it is imported lazily only when no transport
is injected.

Pacing note: steady-state rate limiting is the connector's job (it calls
``RateLimiter.acquire`` + ``RobotsPolicy.can_fetch`` before each fetch). The
optional ``crawl_delay`` here is a per-request floor honoring
``robots.txt``'s ``Crawl-delay`` for the HTTP sources, applied inside the fetch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol


class _Transport(Protocol):
    """Minimal ``requests``-shaped transport (duck-typed; ``requests`` satisfies it)."""

    def get(self, url: str, *, headers: dict | None = ..., timeout: float = ...) -> Any: ...
    def post(self, url: str, *, data: dict | None = ..., headers: dict | None = ...,
             auth: tuple[str, str] | None = ..., timeout: float = ...) -> Any: ...


def _default_transport() -> _Transport:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "live fetch needs `requests` (pip install requests), or inject a "
            "transport via the `http=` argument."
        ) from exc
    return requests


# --------------------------------------------------------------------------- #
# Reddit: OAuth2 client-credentials -> authenticated listing GET
# --------------------------------------------------------------------------- #

_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"


def make_reddit_fetch(
    client_id: str,
    client_secret: str,
    user_agent: str,
    *,
    http: _Transport | None = None,
    token_url: str = _REDDIT_TOKEN_URL,
    clock: Callable[[], float] = time.monotonic,
    timeout: float = 15.0,
) -> Callable[[str], Any]:
    """Build a ``fetch(url) -> listing JSON`` for ``RedditConnector``.

    Performs the OAuth2 client-credentials handshake on first use, caches the
    bearer token until ~60s before expiry, and GETs the listing URL with the
    ``Authorization`` + ``User-Agent`` headers Reddit requires. Returns the parsed
    JSON, which ``RedditConnector._parse`` already understands.
    """
    if not (client_id and client_secret and user_agent):
        raise ValueError("Reddit fetch needs client_id, client_secret, and user_agent")
    transport = http or _default_transport()
    token: dict[str, Any] = {"value": None, "expires_at": 0.0}

    def _ensure_token() -> str:
        if token["value"] is not None and clock() < token["expires_at"]:
            return token["value"]
        resp = transport.post(
            token_url,
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": user_agent},
            auth=(client_id, client_secret),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        token["value"] = payload["access_token"]
        # Refresh a minute early to avoid races on near-expiry tokens.
        token["expires_at"] = clock() + float(payload.get("expires_in", 3600)) - 60.0
        return token["value"]

    def fetch(url: str) -> Any:
        bearer = _ensure_token()
        resp = transport.get(
            url,
            headers={"Authorization": f"bearer {bearer}", "User-Agent": user_agent},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    return fetch


# --------------------------------------------------------------------------- #
# Arctic Shift: unauthenticated Reddit archive mirror (fallback when Reddit's
# own OAuth app-approval is unavailable/rejected -- see docs/data-source-strategy.md)
# --------------------------------------------------------------------------- #

_ARCTIC_SHIFT_BASE_URL = "https://arctic-shift.photon-reddit.com"


def make_arctic_shift_fetch(
    user_agent: str,
    *,
    base_url: str = _ARCTIC_SHIFT_BASE_URL,
    http: _Transport | None = None,
    limit: int = 100,
    sort: str = "desc",
    timeout: float = 15.0,
    after_utc: Callable[[str], int | None] | None = None,
    on_records: Callable[[str, list], None] | None = None,
) -> Callable[[str], Any]:
    """Build a ``fetch(url) -> listing JSON`` for ``RedditConnector`` backed by
    Arctic Shift (https://arctic-shift.photon-reddit.com), a community-run,
    unauthenticated mirror of Reddit's archived post data.

    Drop-in alternative to :func:`make_reddit_fetch` for when Reddit's own OAuth
    app approval is unavailable or rejected (as of 2026 Reddit requires manual,
    unpredictable review for new API clients). No client_id/secret/token
    handshake -- just an HTTP GET with a ``User-Agent``.

    ``RedditConnector._page_urls`` yields ``https://oauth.reddit.com/r/{sub}/new``
    URLs; this fetch extracts ``{sub}`` from that URL, queries Arctic Shift's
    ``/api/posts/search`` for that subreddit, and reshapes the result into the
    same ``{"data": {"children": [{"data": {...}}, ...]}}`` listing shape
    ``RedditConnector._parse`` already understands -- no connector changes needed.
    Verified against a live call: records use the same field names
    ``RedditConnector._parse`` reads (``id``/``title``/``selftext``/``score``/
    ``permalink``), so no reshaping beyond the envelope is needed. ``sort`` takes
    Arctic Shift's own vocabulary (``asc``/``desc`` by ``created_utc``), not
    Reddit's ``new``/``top`` -- ``"desc"`` (default) is newest-first.

    Incremental fetching (so a daily cron doesn't re-pull the same top-``limit``
    posts every run): pass ``after_utc(subreddit) -> created_utc | None`` to add
    Arctic Shift's ``after=`` cursor (confirmed live to filter by
    ``created_utc``), and ``on_records(subreddit, records)`` to observe the raw
    records for the caller to persist a new watermark. Both are optional --
    without them this behaves like a plain top-``limit`` fetch every call. State
    (where the watermark is stored) is the caller's concern, not this function's.

    Caveat (see docs/data-source-strategy.md): Arctic Shift is a volunteer-run
    third-party mirror, not a Reddit-sanctioned API -- fine for research-stage
    ingestion, but needs the same scrutiny as Reddit's own commercial licensing
    gate before any commercial use.
    """
    if not user_agent:
        raise ValueError("Arctic Shift fetch needs a user_agent")
    transport = http or _default_transport()

    def fetch(url: str) -> Any:
        subreddit = _subreddit_from_reddit_url(url)
        search_url = (
            f"{base_url}/api/posts/search"
            f"?subreddit={subreddit}&sort={sort}&limit={limit}"
        )
        since = after_utc(subreddit) if after_utc is not None else None
        if since is not None:
            search_url += f"&after={since}"
        resp = transport.get(search_url, headers={"User-Agent": user_agent}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if on_records is not None:
            on_records(subreddit, records)
        return {"data": {"children": [{"data": rec} for rec in records]}}

    return fetch


def _subreddit_from_reddit_url(url: str) -> str:
    """Extract the subreddit name from a ``.../r/{sub}/...`` listing URL."""
    parts = url.split("/r/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Arctic Shift fetch: cannot find '/r/{{subreddit}}/' in {url!r}")
    return parts[1].split("/", 1)[0]


def make_http_fetch(
    *,
    user_agent: str,
    parser: Callable[[str], Any] | None = None,
    crawl_delay: float = 0.0,
    http: _Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 15.0,
) -> Callable[[str], Any]:
    """Build a polite ``fetch(url) -> payload`` for an HTTP source.

    * iFixit returns JSON, so omit ``parser`` and the response is ``resp.json()``
      (a list or ``{"guides"|"results": [...]}``, which ``IFixitConnector`` parses).
    * Notebookcheck has no JSON API, so pass ``parser`` -- an HTML->records callable
      applied to ``resp.text`` -- returning the record list/dict
      ``NotebookcheckConnector`` parses.

    ``crawl_delay`` (seconds) is honored as a per-request floor before each GET.
    """
    if not user_agent:
        raise ValueError("HTTP fetch needs a user_agent (politeness)")
    transport = http or _default_transport()

    def fetch(url: str) -> Any:
        if crawl_delay > 0:
            sleep(crawl_delay)
        resp = transport.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
        resp.raise_for_status()
        return parser(resp.text) if parser is not None else resp.json()

    return fetch
