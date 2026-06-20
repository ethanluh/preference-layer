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
# iFixit / Notebookcheck: polite HTTP GET
# --------------------------------------------------------------------------- #

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
