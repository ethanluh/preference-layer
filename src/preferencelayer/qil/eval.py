"""Precision/recall evaluation for the QIL extraction gate.

The Phase 0 go/no-go criterion (``docs/implementation-plan.md``) is **>= 70%
precision on use-profile classification** on a held-out set. We report per-class
precision/recall/F1, macro- and micro-averages, and — to prove the classifier is
doing real work — the precision of a most-frequent-class baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GATE_PRECISION = 0.70


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationReport:
    per_class: list[ClassMetrics]
    macro_precision: float
    micro_precision: float        # == accuracy for single-label prediction
    baseline_precision: float     # most-frequent-class predictor
    n: int
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def gate_pass(self) -> bool:
        return self.macro_precision >= GATE_PRECISION


def evaluate(y_true: list[str], y_pred: list[str]) -> ClassificationReport:
    labels = sorted(set(y_true) | set(y_pred))
    tp = {c: 0 for c in labels}
    fp = {c: 0 for c in labels}
    fn = {c: 0 for c in labels}
    support = {c: 0 for c in labels}
    confusion = {t: {p: 0 for p in labels} for t in labels}

    for t, p in zip(y_true, y_pred):
        support[t] += 1
        confusion[t][p] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class: list[ClassMetrics] = []
    for c in labels:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append(ClassMetrics(c, prec, rec, f1, support[c]))

    macro_precision = sum(m.precision for m in per_class) / len(per_class)
    micro_precision = sum(tp.values()) / len(y_true) if y_true else 0.0

    # Most-frequent-class baseline: predict the majority label for everything.
    majority = max(support, key=support.get)
    baseline_precision = support[majority] / len(y_true) if y_true else 0.0

    return ClassificationReport(
        per_class=per_class,
        macro_precision=macro_precision,
        micro_precision=micro_precision,
        baseline_precision=baseline_precision,
        n=len(y_true),
        confusion=confusion,
    )
