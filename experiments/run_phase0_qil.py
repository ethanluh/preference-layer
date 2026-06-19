#!/usr/bin/env python3
"""Phase 0 Claim 2 experiment: can we extract use-profile-conditioned quality?

Runs the QIL feasibility pipeline end to end on the controlled labeled corpus and
checks the Phase 0 go/no-go gate (from the implementation plan):

    Use-profile classification reaches >= 70% precision on a held-out set.

It then aggregates extracted signals into Bayesian quality posteriors and shows
sample /quality and /compare responses.

Usage:
    python experiments/run_phase0_qil.py [--train N] [--test N] [--seed S] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preferencelayer.qil import (  # noqa: E402
    QILExtractor,
    QualityAggregator,
    QualityService,
    corpus as corpus_mod,
    evaluate,
)
from preferencelayer.qil.eval import GATE_PRECISION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("Generating controlled QIL extraction corpus...")
    cp = corpus_mod.generate(n_train=args.train, n_test=args.test, seed=args.seed)
    print(f"  train={len(cp.train)} test={len(cp.test)} "
          f"products={sum(len(v) for v in cp.products.values())}")

    print("Training extraction heads (TF-IDF + softmax)...")
    extractor = QILExtractor().fit(cp.train)

    y_true = [s.use_profile for s in cp.test]
    y_pred = extractor.predict_use_profiles(cp.test)
    report = evaluate(y_true, y_pred)

    print("\n=== Use-profile classification (held-out) ===")
    print(f"{'use_profile':<16}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    print("-" * 52)
    for m in sorted(report.per_class, key=lambda m: -m.support):
        print(f"{m.label:<16}{m.precision:>10.3f}{m.recall:>9.3f}{m.f1:>8.3f}{m.support:>9}")
    print("-" * 52)
    print(f"{'macro':<16}{report.macro_precision:>10.3f}")
    print(f"{'micro/accuracy':<16}{report.micro_precision:>10.3f}")
    print(f"{'baseline (mfc)':<16}{report.baseline_precision:>10.3f}")

    print(f"\nPhase 0 gate (macro precision >= {GATE_PRECISION:.0%}): ", end="")
    if report.gate_pass:
        print(f"PASS ({report.macro_precision:.1%}, baseline {report.baseline_precision:.1%})")
    else:
        print(f"FAIL ({report.macro_precision:.1%}, baseline {report.baseline_precision:.1%})")

    # --- aggregation + query demo --------------------------------------------
    print("\nAggregating extracted signals into quality posteriors...")
    signals = extractor.extract(cp.train + cp.test)
    agg = QualityAggregator().fit(signals)
    service = QualityService(agg)

    cat0 = list(cp.products)[0]
    pa, pb = cp.products[cat0][0], cp.products[cat0][1]
    print(f"\n/quality {pa} (use_profile=gaming):")
    print(json.dumps(service.quality(pa, "gaming"), indent=2))
    print(f"\n/compare {pa} vs {pb} (use_profile=gaming):")
    print(json.dumps(service.compare(pa, pb, "gaming"), indent=2))

    if args.json:
        payload = {
            "config": vars(args),
            "gate_precision": GATE_PRECISION,
            "macro_precision": report.macro_precision,
            "micro_precision": report.micro_precision,
            "baseline_precision": report.baseline_precision,
            "gate_pass": report.gate_pass,
            "per_class": [vars(m) for m in report.per_class],
            "n_test": report.n,
            "n_quality_posteriors": len(agg.quality),
            "n_failure_posteriors": len(agg.failure),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}")

    return 0 if report.gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
