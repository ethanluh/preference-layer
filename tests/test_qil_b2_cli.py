"""Tests for the qil-b2 CLI (Work Stream B2 measurement harness)."""

import json

import pytest

from preferencelayer.qil.cli import b2_main


def test_b2_smoke_runs_and_reports_a_band(capsys):
    rc = b2_main(["--smoke"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SMOKE TEST" in out
    assert "checkpoint band:" in out
    # The smoke result must never be presented as verified real-text.
    assert "verified on REAL text: False" in out


def test_b2_smoke_writes_result_json(tmp_path, capsys):
    out_path = tmp_path / "result.json"
    rc = b2_main(["--smoke", "--out", str(out_path)])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["is_real_text"] is False
    assert "macro_precision" in payload and "band" in payload


def test_b2_requires_a_corpus_or_smoke():
    # Mutually exclusive group is required -> argparse exits (code 2).
    with pytest.raises(SystemExit):
        b2_main([])


def test_b2_real_corpus_missing_file_errors(tmp_path):
    # Pointing at a non-existent corpus raises the harness's clear FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        b2_main(["--corpus", str(tmp_path / "nope.jsonl")])
