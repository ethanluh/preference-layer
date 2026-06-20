"""Tests for the B2 real-text extraction harness (scaffold)."""

import json

import pytest

from preferencelayer.qil import (
    TfidfBaselineClassifier,
    TransformerClassifier,
    checkpoint_band,
    load_controlled_smoke,
    load_real_corpus,
    measure,
)
from preferencelayer.qil.harness import GATE_PASS, GATE_RECOVERABLE


def test_checkpoint_bands_match_kickoff_thresholds():
    assert checkpoint_band(0.72) == "pass"
    assert checkpoint_band(GATE_PASS) == "pass"
    assert checkpoint_band(0.65) == "recoverable"
    assert checkpoint_band(GATE_RECOVERABLE) == "recoverable"
    assert checkpoint_band(0.50) == "escalate"


def test_smoke_run_with_tfidf_is_runnable_and_not_real_text():
    split = load_controlled_smoke(n_train=600, n_test=200)
    assert split.is_real_text is False
    result = measure(TfidfBaselineClassifier(), split)
    # The plumbing works end-to-end and clears the gate ON THE CONTROLLED CORPUS...
    assert result.macro_precision >= GATE_PASS
    # ...but this is explicitly NOT a verified real-text result.
    assert result.verified_on_real_text is False
    assert result.is_real_text is False


def test_transformer_path_is_documented_scaffold():
    clf = TransformerClassifier()
    # Hyperparameters are recorded for reproducibility even though fit is a scaffold.
    hp = clf.hyperparameters()
    assert hp["model_name"] and hp["epochs"] > 0
    with pytest.raises(NotImplementedError):
        clf.fit(load_controlled_smoke(n_train=50, n_test=20).train)


def test_load_real_corpus_missing_path_raises_unverified():
    with pytest.raises(FileNotFoundError):
        load_real_corpus("/nonexistent/annotated.jsonl")


def test_load_real_corpus_reads_jsonl_and_marks_real(tmp_path):
    # A tiny stand-in for an annotated real corpus (NOT committed; tmp only).
    rows = [
        {"text": "thinkpad sustained rendering hours load workstation", "use_profile": "heavy_use"},
        {"text": "lightweight travel commute airport flights battery", "use_profile": "travel"},
        {"text": "competitive fps frames gpu esports refresh rig", "use_profile": "gaming"},
        {"text": "office productivity client meetings deadlines coding", "use_profile": "professional"},
    ]
    p = tmp_path / "annotated.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    split = load_real_corpus(str(p), test_frac=0.5)
    assert split.is_real_text is True
    assert split.source.startswith("real_annotated:")
    assert len(split.train) + len(split.test) == 4


def test_result_json_is_serializable_and_flags_real_text(tmp_path):
    result = measure(TfidfBaselineClassifier(), load_controlled_smoke(n_train=300, n_test=100))
    blob = result.to_json()
    json.dumps(blob)  # must not raise
    assert blob["is_real_text"] is False
    assert blob["verified_on_real_text"] is False
    assert "per_class" in blob and len(blob["per_class"]) >= 2
