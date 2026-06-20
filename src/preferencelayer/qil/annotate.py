"""Annotation export / adjudication for the B2 real-text corpus.

Work Stream B2 needs ~300 real scraped samples labeled by two annotators and
adjudicated (``docs/phase1-kickoff.md``). This module is the in-sandbox plumbing
for that loop so it runs the moment real data + annotators are available:

1. ``export_for_annotation`` — write the un-labeled items (text + a stable id +
   light context) as JSONL for annotators to label independently.
2. ``adjudicate`` — merge two annotators' labeled JSONL: matching labels become
   gold (written in the schema ``harness.load_real_corpus`` consumes), conflicts
   are set aside for a third pass, and inter-annotator agreement (raw + Cohen's
   κ) is reported.

JSONL throughout (one JSON object per line), matching the rest of the QIL tooling
(``harness.load_real_corpus``). Raw scraped text is ``.gitignore``'d, so these
files live outside version control.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .corpus import Sample
from .schema import USE_PROFILES


def stable_id(text: str) -> str:
    """Deterministic short id for a text item (aligns the two annotators' files)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _write_jsonl(rows: list[dict], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def export_for_annotation(
    samples: list[Sample], path: str | Path, *, context_fields: tuple[str, ...] = ("product_id", "category"),
) -> int:
    """Write un-labeled items for annotators as JSONL; returns the count written.

    Each line carries a stable ``id`` (so two annotators' files align on merge),
    the ``text`` to label, and light context — but NOT the ``use_profile`` (that is
    what the annotator supplies). ``USE_PROFILES`` is included once per row as a
    reminder of the allowed label set.
    """
    rows = []
    for s in samples:
        row = {"id": stable_id(s.text), "text": s.text, "use_profile": None,
               "allowed_use_profiles": list(USE_PROFILES)}
        for fld in context_fields:
            row[fld] = getattr(s, fld, None)
        rows.append(row)
    return _write_jsonl(rows, path)


@dataclass
class AdjudicationReport:
    """Inter-annotator agreement + the conflicts needing a third pass."""

    n: int
    n_agreed: int
    n_conflict: int
    raw_agreement: float
    cohen_kappa: float
    conflicts: list[dict]


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's κ for two annotators over the same items (paired, equal length).

    κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is the
    agreement expected by chance from each annotator's marginal label frequencies.
    Returns 1.0 when there is no chance-disagreement to correct for (p_e == 1).
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("annotator label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        return 0.0
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_e = sum((count_a[lbl] / n) * (count_b[lbl] / n) for lbl in set(count_a) | set(count_b))
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def adjudicate(
    path_a: str | Path,
    path_b: str | Path,
    out_path: str | Path,
    *,
    conflicts_path: str | Path | None = None,
    id_key: str = "id",
    label_key: str = "use_profile",
) -> AdjudicationReport:
    """Merge two annotators' labeled JSONL into a gold corpus + agreement report.

    Items present in both files (matched on ``id_key``) where the labels agree are
    written to ``out_path`` as gold in the ``harness.load_real_corpus`` schema
    (``text`` + ``use_profile`` + carried context). Conflicts are collected (and
    optionally written to ``conflicts_path``) for a third adjudication pass, never
    silently resolved. Items only one annotator labeled are ignored for agreement.
    """
    rows_a = {r[id_key]: r for r in _read_jsonl(path_a) if r.get(id_key) is not None}
    rows_b = {r[id_key]: r for r in _read_jsonl(path_b) if r.get(id_key) is not None}
    shared = [i for i in rows_a if i in rows_b]

    labels_a = [rows_a[i].get(label_key) for i in shared]
    labels_b = [rows_b[i].get(label_key) for i in shared]

    gold: list[dict] = []
    conflicts: list[dict] = []
    for i, la, lb in zip(shared, labels_a, labels_b):
        ra = rows_a[i]
        if la is not None and la == lb:
            row = {k: v for k, v in ra.items()
                   if k not in ("allowed_use_profiles",)}
            row[label_key] = la
            gold.append(row)
        else:
            conflicts.append({"id": i, "text": ra.get("text"),
                              "annotator_a": la, "annotator_b": lb})

    _write_jsonl(gold, out_path)
    if conflicts_path is not None:
        _write_jsonl(conflicts, conflicts_path)

    n = len(shared)
    return AdjudicationReport(
        n=n,
        n_agreed=len(gold),
        n_conflict=len(conflicts),
        raw_agreement=(len(gold) / n if n else 0.0),
        cohen_kappa=cohen_kappa(labels_a, labels_b) if n else 0.0,
        conflicts=conflicts,
    )
