"""Tests for the B2 annotation export / adjudication tooling."""

import json

from preferencelayer.qil.annotate import (
    adjudicate,
    cohen_kappa,
    export_for_annotation,
    stable_id,
)
from preferencelayer.qil.corpus import Sample
from preferencelayer.qil.schema import USE_PROFILES


def _sample(text, use_profile="gaming", product_id="p1", category="laptops"):
    return Sample(text=text, category=category, product_id=product_id,
                  use_profile=use_profile, signal_type="performance", failure_mode=None,
                  quality_dim="thermal", signal_value=0.6, label_confidence=1.0)


def _read(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_stable_id_is_deterministic_and_text_scoped():
    assert stable_id("hello") == stable_id("hello")
    assert stable_id("hello") != stable_id("world")


def test_export_writes_unlabeled_items_with_context(tmp_path):
    out = tmp_path / "to_annotate.jsonl"
    n = export_for_annotation([_sample("the gpu runs hot under load"),
                               _sample("battery lasts all day", use_profile="travel")], out)
    assert n == 2
    rows = _read(out)
    # Unlabeled (use_profile None), but carries id + context + the allowed label set.
    assert all(r["use_profile"] is None for r in rows)
    assert all(r["id"] and r["text"] for r in rows)
    assert rows[0]["product_id"] == "p1" and rows[0]["category"] == "laptops"
    assert list(USE_PROFILES) == rows[0]["allowed_use_profiles"]


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # All-same labels on both sides -> chance agreement is 1, kappa defined as 1.
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    # Swapped labels (same marginals, zero observed agreement) -> kappa negative.
    assert cohen_kappa(["a", "b"], ["b", "a"]) < 0


def _annotator_file(tmp_path, name, items):
    p = tmp_path / name
    with p.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    return p


def test_adjudicate_splits_gold_from_conflicts(tmp_path):
    a = _annotator_file(tmp_path, "a.jsonl", [
        {"id": "1", "text": "gpu hot", "use_profile": "gaming", "product_id": "p1"},
        {"id": "2", "text": "all day battery", "use_profile": "travel", "product_id": "p2"},
        {"id": "3", "text": "spreadsheets", "use_profile": "professional", "product_id": "p3"},
    ])
    b = _annotator_file(tmp_path, "b.jsonl", [
        {"id": "1", "text": "gpu hot", "use_profile": "gaming", "product_id": "p1"},
        {"id": "2", "text": "all day battery", "use_profile": "light_use", "product_id": "p2"},
        {"id": "3", "text": "spreadsheets", "use_profile": "professional", "product_id": "p3"},
    ])
    gold = tmp_path / "gold.jsonl"
    conflicts = tmp_path / "conflicts.jsonl"
    report = adjudicate(a, b, gold, conflicts_path=conflicts)

    assert report.n == 3
    assert report.n_agreed == 2  # ids 1 and 3 agree
    assert report.n_conflict == 1  # id 2 disagrees
    assert report.raw_agreement == 2 / 3

    gold_rows = _read(gold)
    assert {r["use_profile"] for r in gold_rows} == {"gaming", "professional"}
    # Gold rows are in the harness schema (text + use_profile) without the prompt-only field.
    assert all("allowed_use_profiles" not in r for r in gold_rows)

    conflict_rows = _read(conflicts)
    assert conflict_rows[0]["annotator_a"] == "travel"
    assert conflict_rows[0]["annotator_b"] == "light_use"


def test_adjudicated_gold_loads_in_the_harness(tmp_path):
    # The whole point: adjudicated gold feeds load_real_corpus directly.
    from preferencelayer.qil.harness import load_real_corpus

    a = _annotator_file(tmp_path, "a.jsonl", [
        {"id": str(i), "text": f"sample text number {i} gaming rig fps", "use_profile": "gaming",
         "category": "laptops", "product_id": "p"} for i in range(10)
    ])
    b = _annotator_file(tmp_path, "b.jsonl", [
        {"id": str(i), "text": f"sample text number {i} gaming rig fps", "use_profile": "gaming",
         "category": "laptops", "product_id": "p"} for i in range(10)
    ])
    gold = tmp_path / "gold.jsonl"
    report = adjudicate(a, b, gold)
    assert report.n_agreed == 10
    split = load_real_corpus(gold)
    assert split.is_real_text is True
    assert all(s.use_profile == "gaming" for s in split.train + split.test)
