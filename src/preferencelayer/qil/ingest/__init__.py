"""QIL ingestion pipeline (Work Stream B1).

SCAFFOLD: connectors, polite-crawl primitives, product-id normalization, the
PostgreSQL `product_signal` schema, and a daily-job entrypoint -- runnable
end-to-end against committed fixtures. The single network boundary
(``connectors._LiveConnector._fetch_pages``) is marked "PLUG API KEYS HERE";
everything else is implemented and tested offline.

Invariants: never persists raw scraped text (only normalized product_signal
rows); RawDocuments carry no user identifier.
"""

from __future__ import annotations

from pathlib import Path

from .connectors import (
    Connector,
    FixtureConnector,
    IFixitConnector,
    NotebookcheckConnector,
    RawDocument,
    RedditConnector,
)
from .live_fetch import make_http_fetch, make_reddit_fetch
from .normalize import CanonicalProduct, ProductRegistry, normalize_model_string
from .pipeline import (
    IngestionStats,
    InMemorySink,
    PostgresSink,
    ProductSignalRow,
    SignalSink,
    run_daily,
)
from .politeness import RateLimiter, RobotsPolicy

SCHEMA_SQL_PATH = Path(__file__).with_name("schema.sql")


def schema_sql() -> str:
    """Return the PostgreSQL DDL for product_signal + quality_posterior."""
    return SCHEMA_SQL_PATH.read_text()


__all__ = [
    "Connector",
    "RawDocument",
    "FixtureConnector",
    "RedditConnector",
    "IFixitConnector",
    "NotebookcheckConnector",
    "ProductRegistry",
    "CanonicalProduct",
    "normalize_model_string",
    "RateLimiter",
    "RobotsPolicy",
    "make_reddit_fetch",
    "make_http_fetch",
    "run_daily",
    "IngestionStats",
    "SignalSink",
    "InMemorySink",
    "PostgresSink",
    "ProductSignalRow",
    "schema_sql",
    "SCHEMA_SQL_PATH",
]
