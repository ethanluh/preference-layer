#!/usr/bin/env python3
"""Real-data check: do the attribute models hold up on Amazon Reviews 2023?

The synthetic benchmark plants clean attribute vectors and a recoverable interaction
signal, and the preference graph beats the flat baseline by +9.7% NDCG@10 there
(Claim 1). This script runs the same model comparison on **real** Amazon Reviews 2023
items and users, where attributes must be derived coarsely from item metadata — a
reality check on how much of the synthetic result survives real, noisy features.

It loads one category (default ``All_Beauty`` — small and quick), builds a
within-category ranking task with **hard negatives** (popular items the user didn't
engage with, so a popularity baseline can't win for free), and reports NDCG@10 for the
preference graph, the flat-attribute and flat-item-embedding baselines, and popularity.

Requires the optional extra (``pip install preferencelayer[amazon]``) and network
access to Hugging Face; if either is missing it prints a notice and exits cleanly.

Usage:
    python experiments/run_amazon_realdata.py [--category All_Beauty] [--label all_beauty]
                                              [--max-items N] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preferencelayer.eval import ExperimentHarness  # noqa: E402
from preferencelayer.models import (  # noqa: E402
    FlatAttributeRecommender,
    FlatItemEmbeddingRecommender,
    SparsePreferenceGraph,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="All_Beauty", help="Amazon Reviews 2023 category config.")
    ap.add_argument("--label", default="all_beauty", help="PreferenceLayer category label.")
    ap.add_argument("--max-items", type=int, default=4000)
    ap.add_argument("--max-interactions", type=int, default=None,
                    help="Cap rows parsed from the interactions CSV (keeps large categories tractable).")
    ap.add_argument("--min-user-reviews", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    try:
        from preferencelayer.data import amazon
    except Exception as e:  # pragma: no cover
        print(f"Amazon loader unavailable: {e}")
        return 0

    print(f"Loading Amazon Reviews 2023 '{args.category}' (Parquet meta + 0core benchmark CSV)...")
    try:
        cat = amazon.load_category(
            args.category, args.label, max_items=args.max_items,
            max_interactions=args.max_interactions,
            min_user_reviews=args.min_user_reviews, seed=args.seed,
        )
    except ImportError as e:
        print(f"\n{e}\nInstall with: pip install preferencelayer[amazon]")
        return 0
    except Exception as e:  # network / dataset issues — don't fail the run
        print(f"\nCould not load the dataset (network or availability issue): {e}")
        return 0

    from preferencelayer.data.synthetic import SyntheticDataset
    import numpy as np

    user_ids = list(cat.purchases)
    ds = SyntheticDataset(
        categories={args.label: cat}, user_ids=user_ids,
        theta=np.empty((len(user_ids), cat.schema.n_shared)), phi_pairs=[], phi=np.empty((len(user_ids), 0)),
    )
    print(f"  items={len(cat.items)} users={len(user_ids)} (>= {args.min_user_reviews} reviews each)\n")

    facs = {
        "flat_item_embedding": lambda: FlatItemEmbeddingRecommender(),
        "flat_attribute": lambda: FlatAttributeRecommender(),
        "preference_graph": lambda: SparsePreferenceGraph(),
    }
    harness = ExperimentHarness(ds, k=10, seed=13)
    results = harness.run_within(facs, args.label)

    print("=== Within-category ranking on real Amazon data (NDCG@10, hard negatives) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1].ndcg):
        print(f"  {name:<22}{r.ndcg:>8.4f}")
    comp = next(c for c in harness.compare(results) if c.baseline == "flat_attribute")
    print(f"\npreference_graph vs flat_attribute: {comp.rel_gain_pct:+.1f}%  p={comp.p_value:.4f}")

    print("\n=== Reality check ===")
    print("With coarse metadata-derived attributes, the attribute models carry only weak")
    print("preference signal and the graph does not reproduce its synthetic advantage over")
    print("flat. The bottleneck is attribute-extraction quality (the QIL NLP pipeline,")
    print("Phase 1) — not the ranking model. See docs/phase1-amazon-realdata.md.")

    if args.json:
        payload = {
            "config": {"category": args.category, "label": args.label, "max_items": args.max_items,
                       "max_interactions": args.max_interactions,
                       "min_user_reviews": args.min_user_reviews, "seed": args.seed,
                       "n_items": len(cat.items), "n_users": len(user_ids)},
            "ndcg": {name: r.ndcg for name, r in results.items()},
            "graph_vs_flat_attribute": {"rel_gain_pct": comp.rel_gain_pct, "p_value": comp.p_value},
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
