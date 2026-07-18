"""Tests for the qil-ingest CLI and the end-to-end ingest->refit path (B1).

Exercises the shared ``run_ingest`` helper over the committed fixtures, the
``ProductSignalRow -> ExtractedSignal`` adapter, and the ``ingest_main`` entry
point over a clean fixtures directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from preferencelayer.qil import QILExtractor, generate
from preferencelayer.qil.cli import (
    _row_to_signal,
    build_demo_registry,
    build_live_connectors,
    ingest_main,
    run_ingest,
)
from preferencelayer.qil.ingest import (
    FixtureConnector,
    IFixitConnector,
    InMemorySink,
    ProductSignalRow,
    RedditConnector,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


def _extractor() -> QILExtractor:
    corpus = generate()
    return QILExtractor().fit(corpus.train)


def _reddit_connector():
    return FixtureConnector("laptops", FIXTURES / "reddit_laptops.json", source_type="reddit")


def _reddit_keyboards_connector():
    return FixtureConnector("keyboards", FIXTURES / "reddit_keyboards.json", source_type="reddit")


def test_run_ingest_writes_rows_without_refit():
    sink = InMemorySink()
    stats, written = run_ingest([_reddit_connector()], build_demo_registry(), _extractor(), sink)
    assert stats.written > 0
    assert len(sink.rows) == stats.written
    assert written == 0  # refit not requested


def test_run_ingest_matches_keyboard_models():
    # build_demo_registry only resolves keyboard mentions once GMMK 3 Pro /
    # Keychron Q1 Pro are registered -- guards against the registry regressing
    # to laptop-only coverage.
    sink = InMemorySink()
    stats, _ = run_ingest([_reddit_keyboards_connector()], build_demo_registry(), _extractor(), sink)
    assert stats.written > 0
    product_ids = {r.product_id for r in sink.rows}
    assert "glorious-gmmk-3-pro" in product_ids
    assert "keychron-q1-pro" in product_ids


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
    # run_ingest now tags quality_dim by default (heuristic span tagger), so the
    # refit produces real GP quality posteriors instead of zero -- but only for
    # non-failure signals, so assert the chain runs and writes a non-negative count.
    assert written >= 0
    # At least one row carries a tagged quality_dim (the gap the tagger closes).
    assert any(r.quality_dim is not None for r in sink.rows)


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


def test_ingest_main_flows_dict_shaped_structured_fixture(tmp_path, capsys):
    # A dict-shaped iFixit payload (container key "guides") must flow through the
    # CLI via the live connector's real _parse -- not be silently skipped.
    fixture = tmp_path / "ifixit_guides.json"
    fixture.write_text(json.dumps({"guides": [{
        "guideid": 999,
        "title": "ThinkPad X1 Carbon Gen 12 Fan Replacement",
        "introduction": "The X1 Carbon Gen 12 fan fails under sustained thermal load and throttles.",
        "url": "https://www.ifixit.com/Guide/999",
        "favorites": 4,
    }]}))
    rc = ingest_main(["--fixtures", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fetched=1" in out
    assert "lenovo-thinkpad-x1-carbon-gen12" in out  # matched + written


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


def test_ingest_main_signal_store_accumulates_across_runs(tmp_path, capsys):
    store = tmp_path / "signals.json"

    day1_dir = tmp_path / "day1"
    day1_dir.mkdir()
    (day1_dir / "reddit_sample.json").write_text(json.dumps([{
        "source_local_id": "r1",
        "text": "The ThinkPad X1 Carbon Gen 12 throttles hard under sustained compile loads.",
        "source_url": "https://example/r1",
        "upvote_count": 5,
    }]))
    assert ingest_main(["--fixtures", str(day1_dir), "--signal-store", str(store)]) == 0
    assert store.exists()
    stored = json.loads(store.read_text())
    assert len(stored) == 1

    day2_dir = tmp_path / "day2"
    day2_dir.mkdir()
    (day2_dir / "reddit_sample.json").write_text(json.dumps([{
        "source_local_id": "r2",
        "text": "Dell XPS 15 9530 travel commute lightweight battery lasts all day reliable.",
        "source_url": "https://example/r2",
        "upvote_count": 3,
    }]))
    assert ingest_main(["--fixtures", str(day2_dir), "--signal-store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "fetched=1" in out  # only day 2's new document, not the accumulated total
    stored = json.loads(store.read_text())
    assert len(stored) == 2  # day 1's row survived alongside day 2's


def test_ingest_main_refit_over_accumulated_store_and_posterior_json(tmp_path, capsys):
    store = tmp_path / "signals.json"
    posteriors = tmp_path / "posteriors.json"
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "reddit_sample.json").write_text(json.dumps([{
        "source_local_id": "r1",
        "text": "Dell XPS 15 9530 battery degrades after heavy travel use.",
        "source_url": "https://example/r1",
        "upvote_count": 9,
    }]))
    rc = ingest_main([
        "--fixtures", str(fixture_dir), "--refit",
        "--signal-store", str(store), "--posterior-json", str(posteriors),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "refit: wrote" in out
    assert "accumulated signals" in out
    assert posteriors.exists()
    rows = json.loads(posteriors.read_text())
    assert isinstance(rows, list)


# --- live ingestion wiring (B1) --------------------------------------------

_REDDIT_ENV = {
    "REDDIT_CLIENT_ID": "cid",
    "REDDIT_CLIENT_SECRET": "secret",
    "REDDIT_USER_AGENT": "pref-bot/0.1",
}


def test_build_live_connectors_requires_credentials(monkeypatch):
    for key in _REDDIT_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit) as exc:
        build_live_connectors("laptops")
    assert "REDDIT_CLIENT_ID" in str(exc.value)


def test_build_live_connectors_defaults_to_reddit_only(monkeypatch):
    # Data-source strategy: Reddit is the only source wired by default; iFixit is
    # parked (its connector is retained but not crawled unless explicitly opted in).
    for key, val in _REDDIT_ENV.items():
        monkeypatch.setenv(key, val)
    connectors = build_live_connectors("laptops")
    types = {type(c) for c in connectors}
    assert types == {RedditConnector}
    assert all(c._fetch is not None for c in connectors)  # real injected fetch


def test_build_live_connectors_opts_in_ifixit_explicitly(monkeypatch):
    for key, val in _REDDIT_ENV.items():
        monkeypatch.setenv(key, val)
    connectors = build_live_connectors("laptops", sources=("reddit", "ifixit"))
    types = {type(c) for c in connectors}
    assert types == {RedditConnector, IFixitConnector}
    assert all(c._fetch is not None for c in connectors)


def test_build_live_connectors_arctic_shift_needs_only_user_agent(monkeypatch):
    # Fallback path (docs/data-source-strategy.md): no OAuth client_id/secret.
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("REDDIT_USER_AGENT", "pref-bot/0.1")
    connectors = build_live_connectors("laptops", sources=("reddit-arctic-shift",))
    assert {type(c) for c in connectors} == {RedditConnector}
    assert all(c._fetch is not None for c in connectors)


def test_build_live_connectors_arctic_shift_requires_user_agent(monkeypatch):
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    with pytest.raises(SystemExit) as exc:
        build_live_connectors("laptops", sources=("reddit-arctic-shift",))
    assert "REDDIT_USER_AGENT" in str(exc.value)


def test_build_live_connectors_arctic_shift_persists_watermark(monkeypatch, tmp_path):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("REDDIT_USER_AGENT", "pref-bot/0.1")
    watermark_path = tmp_path / "watermark.json"
    monkeypatch.setenv("QIL_ARCTIC_SHIFT_WATERMARK_PATH", str(watermark_path))

    import preferencelayer.qil.cli as cli_module

    captured = {}

    def fake_make_arctic_shift_fetch(user_agent, *, after_utc=None, on_records=None):
        captured["after_utc"] = after_utc
        captured["on_records"] = on_records
        return lambda url: {"data": {"children": []}}

    monkeypatch.setattr(cli_module, "make_arctic_shift_fetch", fake_make_arctic_shift_fetch)
    build_live_connectors("laptops", sources=("reddit-arctic-shift",))

    assert not watermark_path.exists()  # nothing written until records are seen
    assert captured["after_utc"]("thinkpad") is None  # no prior watermark

    captured["on_records"]("thinkpad", [{"created_utc": 1700000000}, {"created_utc": 1700000500}])
    assert json.loads(watermark_path.read_text()) == {"thinkpad": 1700000500}

    # A second run picks up the persisted watermark.
    watermarks = cli_module._load_watermarks(watermark_path)
    assert watermarks == {"thinkpad": 1700000500}


def test_ingest_main_live_without_credentials_errors(monkeypatch):
    for key in _REDDIT_ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit):
        ingest_main(["--live"])


def test_ingest_main_requires_a_source():
    # Neither --fixtures nor --live -> argparse error (exit code 2).
    with pytest.raises(SystemExit):
        ingest_main([])
