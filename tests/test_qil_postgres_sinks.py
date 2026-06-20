"""Cover the production Postgres sinks without a live DB (Work Stream B1/B3).

``PostgresSink`` (product_signal) and ``PostgresPosteriorSink``
(quality_posterior) are thin DB-API 2.0 adapters. A live Postgres is only needed
to exercise the real dedup/upsert *constraints*; the SQL text, the per-row
parameter binding, rowcount accounting, and the commit are all verifiable with a
fake DB-API connection — which is what this test does.
"""

from __future__ import annotations

from datetime import datetime, timezone

from preferencelayer.qil.ingest.pipeline import PostgresSink, ProductSignalRow
from preferencelayer.qil.refit import PostgresPosteriorSink, QualityPosteriorRow


class _FakeCursor:
    """Minimal DB-API 2.0 cursor double used as a context manager."""

    def __init__(self, rowcounts=None):
        self.calls: list[tuple[str, dict]] = []
        self._rowcounts = list(rowcounts) if rowcounts is not None else None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        idx = len(self.calls) - 1
        self.rowcount = self._rowcounts[idx] if self._rowcounts is not None else 1


class _FakeConnection:
    def __init__(self, rowcounts=None):
        self._cursor = _FakeCursor(rowcounts)
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1


def _signal_row(pid: str, content_hash: str) -> ProductSignalRow:
    return ProductSignalRow(
        product_id=pid, model_normalized=pid, category="laptops",
        use_profile="gaming", signal_type="performance", failure_mode=None,
        quality_dim="thermal", signal_value=0.7, source_url="https://example/x",
        source_type="reddit", content_hash=content_hash,
        extracted_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
        model_confidence=0.8, upvote_count=5,
    )


def test_postgres_signal_sink_binds_params_counts_rowcount_and_commits():
    # Second row is a dedup conflict (ON CONFLICT DO NOTHING -> rowcount 0).
    conn = _FakeConnection(rowcounts=[1, 0])
    sink = PostgresSink(conn)
    rows = [_signal_row("a", "h1"), _signal_row("b", "h2")]

    written = sink.write(rows)

    assert written == 1  # 1 + 0
    cur = conn.cursor()
    assert len(cur.calls) == 2
    for (sql, params), row in zip(cur.calls, rows):
        assert sql == PostgresSink._INSERT
        assert params == row.__dict__          # bound straight from the dataclass
    assert "ON CONFLICT (source_type, content_hash) DO NOTHING" in PostgresSink._INSERT
    assert conn.commits == 1


def _posterior_row(pid: str) -> QualityPosteriorRow:
    return QualityPosteriorRow(
        product_id=pid, use_profile="gaming", quality_dim="thermal",
        posterior_mean=0.6, posterior_std=0.1, credible_lo_90=0.4,
        credible_hi_90=0.8, evidence_count=12, freshness_score=0.9,
        last_refit=datetime(2026, 6, 20, tzinfo=timezone.utc),
    )


def test_postgres_posterior_sink_upserts_each_row_and_commits():
    conn = _FakeConnection()
    sink = PostgresPosteriorSink(conn)
    rows = [_posterior_row("a"), _posterior_row("b")]

    n = sink.upsert(rows)

    assert n == 2
    cur = conn.cursor()
    assert len(cur.calls) == 2
    for (sql, params), row in zip(cur.calls, rows):
        assert sql == PostgresPosteriorSink._UPSERT
        assert params == row.__dict__
    assert "ON CONFLICT (product_id, use_profile, quality_dim) DO UPDATE" in PostgresPosteriorSink._UPSERT
    assert conn.commits == 1
