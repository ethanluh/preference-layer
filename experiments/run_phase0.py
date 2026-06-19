#!/usr/bin/env python3
"""Phase 0 headline experiment: does a sparse preference graph beat flat baselines?

Runs within-category and cross-category transfer evaluation on the synthetic
benchmark and prints a results table. The Phase 0 go/no-go gate (from the
implementation plan) is:

    Sparse DAG outperforms the flat vector by >= 5% NDCG@10 on cross-category
    transfer.

Usage:
    python experiments/run_phase0.py [--users N] [--seed S] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preferencelayer.data import synthetic  # noqa: E402
from preferencelayer.eval import ExperimentHarness  # noqa: E402
from preferencelayer.models import (  # noqa: E402
    FlatAttributeRecommender,
    FlatItemEmbeddingRecommender,
    SparsePreferenceGraph,
)

GATE_THRESHOLD_PCT = 5.0


def factories():
    return {
        "flat_item_embedding": lambda: FlatItemEmbeddingRecommender(),
        "flat_attribute": lambda: FlatAttributeRecommender(),
        "preference_graph": lambda: SparsePreferenceGraph(),
    }


def _print_table(title: str, results) -> None:
    print(f"\n=== {title} ===")
    print(f"{'model':<22}{'NDCG@10':>10}{'±std':>8}{'Recall@10':>12}{'HitRate@10':>12}")
    print("-" * 64)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].ndcg):
        print(f"{name:<22}{r.ndcg:>10.4f}{r.ndcg_std:>8.3f}{r.recall:>12.4f}{r.hit_rate:>12.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("Generating synthetic multi-category preference dataset...")
    ds = synthetic.generate(
        n_users=args.users,
        categories=("laptops", "headphones"),
        items_per_category=args.items,
        seed=args.seed,
    )
    print(f"  users={len(ds.user_ids)} categories={list(ds.categories)} "
          f"planted_interactions={len(ds.phi_pairs)}")

    harness = ExperimentHarness(ds, k=10, n_candidates=100, seed=13)
    facs = factories()

    within = harness.run_within(facs, "laptops")
    _print_table("Within-category (laptops -> laptops)", within)

    transfer = harness.run_transfer(facs, source="laptops", target="headphones")
    _print_table("Cross-category transfer (laptops -> headphones)", transfer)

    print("\n=== Graph vs baselines on cross-category transfer ===")
    comparisons = harness.compare(transfer)
    gate_pass = True
    for c in comparisons:
        print(f"  vs {c.baseline:<22} "
              f"graph={c.graph_ndcg:.4f}  baseline={c.baseline_ndcg:.4f}  "
              f"rel_gain={c.rel_gain_pct:+.1f}%  p={c.p_value:.4f}")
    flat = next(c for c in comparisons if c.baseline == "flat_attribute")
    print(f"\nPhase 0 gate (>= {GATE_THRESHOLD_PCT}% over flat_attribute on transfer): ", end="")
    if flat.rel_gain_pct >= GATE_THRESHOLD_PCT and flat.p_value < 0.05:
        print(f"PASS (+{flat.rel_gain_pct:.1f}%, p={flat.p_value:.4f})")
    else:
        print(f"FAIL (+{flat.rel_gain_pct:.1f}%, p={flat.p_value:.4f})")
        gate_pass = False

    if args.json:
        payload = {
            "config": vars(args),
            "within": {k: vars(v) for k, v in within.items()},
            "transfer": {k: vars(v) for k, v in transfer.items()},
            "comparisons": [vars(c) for c in comparisons],
            "gate_pass": gate_pass,
        }
        # Drop per-user arrays from the JSON for brevity.
        for section in ("within", "transfer"):
            for v in payload[section].values():
                v.pop("per_user_ndcg", None)
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
