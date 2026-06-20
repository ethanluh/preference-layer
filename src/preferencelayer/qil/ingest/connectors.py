"""Source connectors for QIL ingestion.

A connector turns one public source (Reddit / iFixit / Notebookcheck) into a
stream of :class:`RawDocument` -- normalized, *non-identifying* text records that
the extraction step (``qil.extract``) classifies. Connectors share a common
interface so the daily job (``pipeline.run_daily``) treats them uniformly.

SCAFFOLD STATUS (no API keys in this environment):
  * The live connectors (:class:`RedditConnector`, :class:`IFixitConnector`,
    :class:`NotebookcheckConnector`) carry the polite-crawl wiring, robots.txt
    handling, and the parse logic, but their network fetch is a single clearly
    marked method -- ``_fetch_pages`` -- that raises until credentials/HTTP are
    plugged in. See "PLUG API KEYS / HTTP HERE" below.
  * :class:`FixtureConnector` reads committed JSON fixtures so the whole pipeline
    is runnable and testable offline. Run the real connectors once keys land.

PRIVACY INVARIANT (architecture.md "QIL privacy"): a RawDocument carries NO user
identifier. Connectors must drop author/username/account fields at the source --
see ``RawDocument`` and the fixture loader, which assert this.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .politeness import RateLimiter, RobotsPolicy

# Fields that would identify a user; never allowed onto a RawDocument.
_FORBIDDEN_FIELDS = frozenset({"author", "username", "user", "user_id", "account", "email"})


@dataclass(frozen=True)
class RawDocument:
    """A single non-identifying text record from a source.

    Only the body text, a stable source-local id, the source type/url, the
    category hint, and an upvote count survive ingestion -- no author/user data.
    """

    source_type: str          # 'reddit' | 'ifixit' | 'notebookcheck'
    source_local_id: str      # stable id within the source (post id, guide id, ...)
    category: str             # 'laptops' | 'keyboards'
    text: str
    source_url: str | None = None
    upvote_count: int = 0

    @property
    def content_hash(self) -> str:
        """Dedup key: source + stable id + body, hashed (matches schema.sql)."""
        h = hashlib.sha256()
        h.update(self.source_type.encode())
        h.update(b"\x00")
        h.update(self.source_local_id.encode())
        h.update(b"\x00")
        h.update(self.text.encode())
        return h.hexdigest()


def _assert_no_user_identifiers(record: dict) -> None:
    bad = _FORBIDDEN_FIELDS & set(record)
    if bad:
        raise ValueError(
            f"QIL invariant violated: record carries user identifier(s) {sorted(bad)}; "
            "connectors must drop these at the source."
        )


class Connector(ABC):
    """Base connector: politeness wiring + a uniform ``documents()`` stream."""

    source_type: str = "abstract"

    def __init__(
        self,
        category: str,
        rate_limiter: RateLimiter | None = None,
        robots: RobotsPolicy | None = None,
    ):
        self.category = category
        self.rate_limiter = rate_limiter
        self.robots = robots

    def _check_polite(self, url: str | None) -> None:
        if self.robots is not None and url is not None and not self.robots.can_fetch(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

    @abstractmethod
    def documents(self) -> Iterator[RawDocument]:
        """Yield non-identifying documents from this source."""
        raise NotImplementedError


class _LiveConnector(Connector):
    """Shared base for the three live (network) connectors -- SCAFFOLD."""

    def documents(self) -> Iterator[RawDocument]:
        for url in self._page_urls():
            self._check_polite(url)
            payload = self._fetch_pages(url)  # <-- network boundary
            yield from self._parse(payload, url)

    def _page_urls(self) -> list[str]:  # pragma: no cover - overridden per source
        raise NotImplementedError

    def _fetch_pages(self, url: str) -> object:
        # ===================================================================
        # PLUG API KEYS / HTTP HERE.
        # Wire the source's authenticated client / HTTP GET in this one method.
        # Everything else (politeness, parsing, normalization, dedup, sink) is
        # implemented and tested via FixtureConnector. Suggested:
        #   Reddit:        PRAW / OAuth app creds from env (REDDIT_CLIENT_ID...)
        #   iFixit:        requests.get(url) honoring self.robots.crawl_delay
        #   Notebookcheck: requests.get(url) honoring self.robots.crawl_delay
        # ===================================================================
        raise NotImplementedError(
            f"{type(self).__name__}: network fetch not configured. "
            "Plug API keys / HTTP into _fetch_pages (see PLUG API KEYS HERE), "
            "or use FixtureConnector for offline runs."
        )

    def _parse(self, payload: object, url: str) -> Iterator[RawDocument]:  # pragma: no cover
        raise NotImplementedError


class RedditConnector(_LiveConnector):
    """Reddit (official API, rate-limited). SCAFFOLD -- see _fetch_pages."""

    source_type = "reddit"

    def __init__(self, category: str, subreddits: list[str], **kw):
        super().__init__(category, **kw)
        self.subreddits = subreddits

    def _page_urls(self) -> list[str]:
        return [f"https://oauth.reddit.com/r/{s}/new" for s in self.subreddits]


class IFixitConnector(_LiveConnector):
    """iFixit (polite crawl, robots.txt). SCAFFOLD -- see _fetch_pages."""

    source_type = "ifixit"

    def __init__(self, category: str, start_urls: list[str], **kw):
        super().__init__(category, **kw)
        self.start_urls = start_urls

    def _page_urls(self) -> list[str]:
        return list(self.start_urls)


class NotebookcheckConnector(_LiveConnector):
    """Notebookcheck (structured scrape). SCAFFOLD -- see _fetch_pages."""

    source_type = "notebookcheck"

    def __init__(self, category: str, start_urls: list[str], **kw):
        super().__init__(category, **kw)
        self.start_urls = start_urls

    def _page_urls(self) -> list[str]:
        return list(self.start_urls)


@dataclass
class FixtureConnector(Connector):
    """Offline connector that reads committed JSON fixtures.

    The fixture is a list of records, each ``{source_local_id, text, ...}`` with
    NO user identifier (enforced). Lets the whole pipeline -- normalization,
    dedup, extraction, sink -- run end-to-end with no network. ``source_type``
    is taken from the fixture so one fixture file can stand in for any source.
    """

    fixture_path: Path = field(default=None)  # type: ignore[assignment]
    source_type: str = "fixture"

    def __init__(self, category: str, fixture_path: str | Path, source_type: str = "fixture", **kw):
        super().__init__(category, **kw)
        self.fixture_path = Path(fixture_path)
        self.source_type = source_type

    def documents(self) -> Iterator[RawDocument]:
        records = json.loads(self.fixture_path.read_text())
        for rec in records:
            _assert_no_user_identifiers(rec)
            self._check_polite(rec.get("source_url"))
            yield RawDocument(
                source_type=rec.get("source_type", self.source_type),
                source_local_id=str(rec["source_local_id"]),
                category=rec.get("category", self.category),
                text=rec["text"],
                source_url=rec.get("source_url"),
                upvote_count=int(rec.get("upvote_count", 0)),
            )
