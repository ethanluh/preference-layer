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
    """Shared base for the three live (network) connectors.

    The ONLY unplugged piece is the network call. Everything else -- politeness,
    parsing, normalization, dedup, sink -- is implemented and tested. The network
    boundary is a single injectable ``fetch`` callable (``url -> payload``):

    * In production, pass ``fetch=`` the source's authenticated client / HTTP GET
      (e.g. PRAW for Reddit, ``requests.get`` honoring ``self.robots.crawl_delay``
      for iFixit / Notebookcheck).
    * In tests, pass a fake ``fetch`` returning canned payloads to exercise the
      real ``_parse`` without a network.
    * If no ``fetch`` is injected, ``_fetch_pages`` raises a clear error.
    """

    def __init__(self, category: str, *, fetch=None, **kw):
        super().__init__(category, **kw)
        self._fetch = fetch

    def documents(self) -> Iterator[RawDocument]:
        for url in self._page_urls():
            self._check_polite(url)
            payload = self._fetch_pages(url)  # <-- network boundary
            yield from self._parse(payload, url)

    def _page_urls(self) -> list[str]:  # pragma: no cover - overridden per source
        raise NotImplementedError

    def _fetch_pages(self, url: str) -> object:
        if self._fetch is not None:
            return self._fetch(url)
        # ===================================================================
        # PLUG API KEYS / HTTP HERE.
        # Inject a ``fetch`` callable (constructor arg) wrapping the source's
        # authenticated client / HTTP GET. Suggested:
        #   Reddit:        PRAW / OAuth app creds from env (REDDIT_CLIENT_ID...)
        #   iFixit:        requests.get(url) honoring self.robots.crawl_delay
        #   Notebookcheck: requests.get(url) honoring self.robots.crawl_delay
        # ===================================================================
        raise NotImplementedError(
            f"{type(self).__name__}: network fetch not configured. "
            "Inject a fetch callable (see PLUG API KEYS HERE), "
            "or use FixtureConnector for offline runs."
        )

    def _parse(self, payload: object, url: str) -> Iterator[RawDocument]:  # pragma: no cover
        raise NotImplementedError

    # -- shared parsing helpers (used by the structured-record connectors) -----
    @staticmethod
    def _records(payload: object, *container_keys: str) -> list:
        """Unwrap a payload that is either a list or ``{container_key: [...]}``."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in container_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(
            f"connector: expected a list or a dict with one of {container_keys!r}, "
            f"got {type(payload).__name__}"
        )

    def _emit(
        self,
        records: list,
        *,
        id_key: str,
        text_keys: tuple[str, ...],
        url_key: str | None = None,
        score_key: str | None = None,
        page_url: str | None = None,
    ) -> Iterator[RawDocument]:
        """Map structured source records to non-identifying RawDocuments.

        Reads only the id, the text field(s), an optional canonical url, and an
        optional score -- never author/username fields, so no user identifier
        reaches a RawDocument (architecture.md "QIL privacy").
        """
        for rec in records:
            if not isinstance(rec, dict):
                continue
            local_id = rec.get(id_key)
            if not local_id:
                continue
            text = "\n".join(str(rec[k]) for k in text_keys if rec.get(k)).strip()
            if not text:
                continue
            src_url = rec.get(url_key) if url_key else None
            yield RawDocument(
                source_type=self.source_type,
                source_local_id=str(local_id),
                category=self.category,
                text=text,
                source_url=src_url or page_url,
                upvote_count=int(rec.get(score_key, 0) or 0) if score_key else 0,
            )


class RedditConnector(_LiveConnector):
    """Reddit (official API, rate-limited).

    The network call is the injected ``fetch`` (e.g. a PRAW/OAuth listing GET);
    ``_parse`` handles the standard Reddit listing JSON shape
    (``{"data": {"children": [{"data": {...}}]}}``) and is fully tested.
    """

    source_type = "reddit"

    def __init__(self, category: str, subreddits: list[str], **kw):
        super().__init__(category, **kw)
        self.subreddits = subreddits

    def _page_urls(self) -> list[str]:
        return [f"https://oauth.reddit.com/r/{s}/new" for s in self.subreddits]

    def _parse(self, payload: object, url: str) -> Iterator[RawDocument]:
        """Yield non-identifying RawDocuments from a Reddit listing payload.

        Reads only the body text, the post id, score, and permalink -- the
        author/username and other identifier fields are deliberately NOT read, so
        no user identifier reaches a RawDocument (architecture.md "QIL privacy").
        """
        if not isinstance(payload, dict):
            raise ValueError("RedditConnector expects a listing dict payload")
        children = payload.get("data", {}).get("children", [])
        for child in children:
            data = child.get("data", {}) if isinstance(child, dict) else {}
            local_id = data.get("id")
            if not local_id:
                continue
            # Posts carry title + selftext; comments carry body. Use whatever text
            # is present, joined and stripped.
            text = "\n".join(
                part for part in (data.get("title"), data.get("selftext"), data.get("body"))
                if part
            ).strip()
            if not text:
                continue
            permalink = data.get("permalink")
            yield RawDocument(
                source_type=self.source_type,
                source_local_id=str(local_id),
                category=self.category,
                text=text,
                source_url=(f"https://www.reddit.com{permalink}" if permalink else url),
                upvote_count=int(data.get("score", 0) or 0),
            )


class IFixitConnector(_LiveConnector):
    """iFixit (polite crawl, robots.txt).

    The network call is the injected ``fetch`` (e.g. ``requests.get`` against the
    iFixit API 2.0 honoring ``self.robots.crawl_delay``); ``_parse`` handles the
    guides JSON shape (a list, or ``{"guides"|"results": [...]}``) and is tested.
    """

    source_type = "ifixit"

    def __init__(self, category: str, start_urls: list[str], **kw):
        super().__init__(category, **kw)
        self.start_urls = start_urls

    def _page_urls(self) -> list[str]:
        return list(self.start_urls)

    def _parse(self, payload: object, url: str) -> Iterator[RawDocument]:
        yield from self._emit(
            self._records(payload, "guides", "results"),
            id_key="guideid", text_keys=("title", "introduction", "conclusion"),
            url_key="url", score_key="favorites", page_url=url,
        )


class NotebookcheckConnector(_LiveConnector):
    """Notebookcheck (structured scrape).

    Notebookcheck has no public JSON API, so the HTML→records extraction lives in
    the injected ``fetch`` (which may use any parser it likes); ``_parse``
    normalizes the resulting review records (a list, or
    ``{"reviews"|"results": [...]}``) into RawDocuments and is tested.
    """

    source_type = "notebookcheck"

    def __init__(self, category: str, start_urls: list[str], **kw):
        super().__init__(category, **kw)
        self.start_urls = start_urls

    def _page_urls(self) -> list[str]:
        return list(self.start_urls)

    def _parse(self, payload: object, url: str) -> Iterator[RawDocument]:
        yield from self._emit(
            self._records(payload, "reviews", "results"),
            id_key="id", text_keys=("title", "verdict", "summary"),
            url_key="url", page_url=url,
        )


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
