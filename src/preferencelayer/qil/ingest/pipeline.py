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

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..corpus import Sample
from ..extract import ExtractedSignal, QILExtractor
from ..quality_spans import QualityDimTagger
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


def signal_rows_to_json(rows: Iterable[ProductSignalRow]) -> str:
    """Serialize product_signal rows to JSON -- a stopgap store before a real DB.

    Used to persist accumulated rows across separate ``qil-ingest`` invocations
    (e.g. one GitHub Actions run per day) so evidence builds up over time instead
    of resetting with each run's in-memory sink. Not a replacement for
    :class:`PostgresSink`; see docs/whats-missing.md B4.
    """
    def _dump(row: ProductSignalRow) -> dict:
        d = asdict(row)
        d["extracted_at"] = row.extracted_at.isoformat()
        return d

    return json.dumps([_dump(r) for r in rows])


def signal_rows_from_json(text: str) -> list[ProductSignalRow]:
    """Inverse of :func:`signal_rows_to_json`."""
    rows = []
    for d in json.loads(text):
        d = dict(d)
        d["extracted_at"] = datetime.fromisoformat(d["extracted_at"])
        rows.append(ProductSignalRow(**d))
    return rows


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


def _doc_to_sample(
    doc: RawDocument, product_id: str, tagger: QualityDimTagger | None = None
) -> Sample:
    """Adapt a RawDocument to the Sample shape the QILExtractor consumes.

    Gold-label fields are unknown at ingestion time (we are *predicting* them),
    so use_profile/signal_type are placeholders the extractor overwrites. The
    structured failure_mode is still a fuller span-model's job; ``quality_dim``
    and ``signal_value`` are populated by ``tagger`` (the heuristic span tagger,
    ``quality_spans.QualityDimTagger``) when one is supplied, so non-failure
    signals carry a dimension and the aggregator forms GP posteriors. Without a
    tagger they default to neutral (``None`` / 0.5), preserving prior behavior.
    """
    if tagger is not None:
        quality_dim, signal_value = tagger.tag(doc.text)
    else:
        quality_dim, signal_value = None, 0.5
    return Sample(
        text=doc.text,
        category=doc.category,
        product_id=product_id,
        use_profile="light_use",      # placeholder; predicted by the extractor
        signal_type="performance",    # placeholder; predicted by the extractor
        failure_mode=None,
        quality_dim=quality_dim,
        signal_value=signal_value,
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
    quality_tagger: QualityDimTagger | None = None,
) -> IngestionStats:
    """Run one ingestion pass over all connectors. The cron entrypoint.

    Steps per document: politeness-gated fetch (inside the connector) ->
    resolve canonical product_id (drop if no confident match) -> extract
    use_profile/signal_type -> write to sink (dedup by content_hash).

    ``quality_tagger`` (when supplied) populates each sample's ``quality_dim`` /
    ``signal_value`` so non-failure signals form GP quality posteriors; without
    it those fields stay neutral and only failure rates aggregate (prior behavior).
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
        samples = [_doc_to_sample(doc, pid, quality_tagger) for doc, pid, _ in pending]
        signals = extractor.extract(samples)
        rows = [
            _to_row(doc, pid, model_norm, sig, extracted_at)
            for (doc, pid, model_norm), sig in zip(pending, signals)
        ]
        stats.written = sink.write(rows)

    return stats
