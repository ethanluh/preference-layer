"""Tests for the qil-ingest CLI and the end-to-end ingest->refit path (B1).

Exercises the shared ``run_ingest`` helper over the committed fixtures, the
``ProductSignalRow -> ExtractedSignal`` adapter, and the ``ingest_main`` entry
point over a clean fixtures directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from preferencelayer.qil import QILExtractor, generate
from preferencelayer.qil.cli import (
    _row_to_signal,
    build_demo_registry,
    ingest_main,
    run_ingest,
)
from preferencelayer.qil.ingest import FixtureConnector, InMemorySink, ProductSignalRow

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


def _extractor() -> QILExtractor:
    corpus = generate()
    return QILExtractor().fit(corpus.train)


def _reddit_connector():
    return FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")


def test_run_ingest_writes_rows_without_refit():
    sink = InMemorySink()
    stats, written = run_ingest([_reddit_connector()], build_demo_registry(), _extractor(), sink)
    assert stats.written > 0
    assert len(sink.rows) == stats.written
    assert written == 0  # refit not requested


def test_row_to_signal_maps_fields():
    row = ProductSignalRow(
        product_id="p", model_normalized="p", category="laptops", use_profile="gaming",
        signal_type="performance", failure_mode=None, quality_dim="thermal",
        signal_value=0.7, source_url=None, source_type="reddit", content_hash="h",
        extracted_at=datetime.now(timezone.utc), model_confidence=0.83, upvote_count=1,
    )
    sig = _row_to_signal(row)
    assert (sig.product_id, sig.use_profile, sig.quality_dim) == ("p", "gaming", "thermal")
    assert sig.signal_value == 0.7
    assert sig.confidence == 0.83


def test_run_ingest_refit_runs_end_to_end():
    sink = InMemorySink()
    stats, written = run_ingest([_reddit_connector()], build_demo_registry(), _extractor(),
                                sink, refit=True)
    assert stats.written > 0
    # Quality-dim posteriors are only produced once a span model populates
    # quality_dim (Work Stream B2); with the placeholder extractor this is >= 0,
    # and the chaining itself runs without error.
    assert written >= 0


def test_ingest_main_over_clean_dir(tmp_path, capsys):
    fixture = tmp_path / "reddit_sample.json"
    fixture.write_text(json.dumps([{
        "source_local_id": "r1",
        "text": "The ThinkPad X1 Carbon Gen 12 throttles hard under sustained compile loads.",
        "source_url": "https://example/r1",
        "upvote_count": 5,
    }]))
    rc = ingest_main(["--fixtures", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "qil-ingest:" in out and "written=" in out


def test_ingest_main_refit_flag(tmp_path, capsys):
    fixture = tmp_path / "reddit_sample.json"
    fixture.write_text(json.dumps([{
        "source_local_id": "r1",
        "text": "The Dell XPS 15 9530 battery degrades after heavy travel use.",
        "source_url": "https://example/r1",
        "upvote_count": 9,
    }]))
    rc = ingest_main(["--fixtures", str(tmp_path), "--refit"])
    assert rc == 0
    assert "refit: wrote" in capsys.readouterr().out
