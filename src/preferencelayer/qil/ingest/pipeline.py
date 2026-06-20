"""QIL daily ingestion job: connectors -> normalize -> extract -> product_signal.

Topology (docs/architecture.md "Ingestion Pipeline"):

    connectors (Reddit/iFixit/Notebookcheck)
        -> RawDocument stream
        -> ProductRegistry normalization (mention -> canonical product_id)
        -> QILExtractor (use_profile + signal_type heads)   [reused from Phase 0]
        -> dedup by (source_type, content_hash)
        -> SignalSink (product_signal rows)

``run_daily`` is the cron entrypoint. It is sink-agnostic: pass an
:class:`InMemorySink` for tests/fixtures or a :class:`PostgresSink` in
production. NEVER writes raw scraped text to disk -- only the structured,
normalized ``product_signal`` rows (the raw body lives only in memory during a
run). This honors the .gitignore / "never commit raw scraped data" rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..corpus import Sample
from ..extract import ExtractedSignal, QILExtractor
from .connectors import Connector, RawDocument
from .normalize import ProductRegistry


@dataclass
class ProductSignalRow:
    """One normalized row destined for the product_signal table."""

    product_id: str
    model_normalized: str
    category: str
    use_profile: str
    signal_type: str
    failure_mode: str | None
    quality_dim: str | None
    signal_value: float | None
    source_url: str | None
    source_type: str | None
    content_hash: str
    extracted_at: datetime
    model_confidence: float
    upvote_count: int


class SignalSink(ABC):
    """Where normalized signals land. Idempotent on (source_type, content_hash)."""

    @abstractmethod
    def write(self, rows: Iterable[ProductSignalRow]) -> int:
        """Insert rows, skipping dedup conflicts. Returns count actually written."""
        raise NotImplementedError


@dataclass
class InMemorySink(SignalSink):
    """Test/fixture sink: keeps rows in a list, dedups by (source_type, hash)."""

    rows: list[ProductSignalRow] = field(default_factory=list)
    _seen: set[tuple[str | None, str]] = field(default_factory=set)

    def write(self, rows: Iterable[ProductSignalRow]) -> int:
        written = 0
        for row in rows:
            key = (row.source_type, row.content_hash)
            if key in self._seen:
                continue
            self._seen.add(key)
            self.rows.append(row)
            written += 1
        return written


class PostgresSink(SignalSink):
    """Production sink: INSERT ... ON CONFLICT DO NOTHING into product_signal.

    Pass any DB-API 2.0 connection (e.g. psycopg). The dedup is handled by the
    uq_product_signal_dedup constraint in schema.sql, so re-running the daily job
    is idempotent. The SQL/param wiring is covered by a fake-DB-API test
    (tests/test_qil_postgres_sinks.py); only a live Postgres exercises the real
    constraint.
    """

    _INSERT = """
        INSERT INTO product_signal (
            product_id, model_normalized, category, failure_mode, quality_dim,
            use_profile, signal_type, signal_value, source_url, source_type,
            content_hash, extracted_at, model_confidence, upvote_count
        ) VALUES (
            %(product_id)s, %(model_normalized)s, %(category)s, %(failure_mode)s,
            %(quality_dim)s, %(use_profile)s, %(signal_type)s, %(signal_value)s,
            %(source_url)s, %(source_type)s, %(content_hash)s, %(extracted_at)s,
            %(model_confidence)s, %(upvote_count)s
        )
        ON CONFLICT (source_type, content_hash) DO NOTHING
    """

    def __init__(self, connection):
        self.connection = connection

    def write(self, rows: Iterable[ProductSignalRow]) -> int:
        written = 0
        with self.connection.cursor() as cur:
            for row in rows:
                cur.execute(self._INSERT, row.__dict__)
                written += cur.rowcount or 0
        self.connection.commit()
        return written


def _doc_to_sample(doc: RawDocument, product_id: str) -> Sample:
    """Adapt a RawDocument to the Sample shape the QILExtractor consumes.

    Gold-label fields are unknown at ingestion time (we are *predicting* them),
    so use_profile/signal_type are placeholders the extractor overwrites; the
    structured failure_mode/quality_dim/signal_value default to neutral and are a
    span-model's job in a fuller pipeline (see extract.py docstring).
    """
    return Sample(
        text=doc.text,
        category=doc.category,
        product_id=product_id,
        use_profile="light_use",      # placeholder; predicted by the extractor
        signal_type="performance",    # placeholder; predicted by the extractor
        failure_mode=None,
        quality_dim=None,
        signal_value=0.5,
        label_confidence=0.0,
    )


def _to_row(doc: RawDocument, product_id: str, model_normalized: str,
            sig: ExtractedSignal, extracted_at: datetime) -> ProductSignalRow:
    return ProductSignalRow(
        product_id=product_id,
        model_normalized=model_normalized,
        category=doc.category,
        use_profile=sig.use_profile,
        signal_type=sig.signal_type,
        failure_mode=sig.failure_mode,
        quality_dim=sig.quality_dim,
        signal_value=sig.signal_value,
        source_url=doc.source_url,
        source_type=doc.source_type,
        content_hash=doc.content_hash,
        extracted_at=extracted_at,
        model_confidence=sig.confidence,
        upvote_count=doc.upvote_count,
    )


@dataclass
class IngestionStats:
    fetched: int = 0
    matched: int = 0          # documents resolved to a canonical product
    unmatched: int = 0        # dropped: no confident product match
    written: int = 0          # rows actually persisted (post-dedup)


def run_daily(
    connectors: list[Connector],
    registry: ProductRegistry,
    extractor: QILExtractor,
    sink: SignalSink,
    now: datetime | None = None,
) -> IngestionStats:
    """Run one ingestion pass over all connectors. The cron entrypoint.

    Steps per document: politeness-gated fetch (inside the connector) ->
    resolve canonical product_id (drop if no confident match) -> extract
    use_profile/signal_type -> write to sink (dedup by content_hash).
    """
    extracted_at = now or datetime.now(timezone.utc)
    stats = IngestionStats()

    # Buffer matched docs so extraction runs in a single vectorized batch.
    pending: list[tuple[RawDocument, str, str]] = []
    for connector in connectors:
        for doc in connector.documents():
            stats.fetched += 1
            hit = registry.match(doc.text, category=doc.category)
            if hit is None:
                stats.unmatched += 1
                continue
            product, _score = hit
            from .normalize import normalize_model_string
            pending.append((doc, product.product_id, normalize_model_string(product.display_name)))
            stats.matched += 1

    if pending:
        samples = [_doc_to_sample(doc, pid) for doc, pid, _ in pending]
        signals = extractor.extract(samples)
        rows = [
            _to_row(doc, pid, model_norm, sig, extracted_at)
            for (doc, pid, model_norm), sig in zip(pending, signals)
        ]
        stats.written = sink.write(rows)

    return stats
