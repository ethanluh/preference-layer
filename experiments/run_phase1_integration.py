#!/usr/bin/env python3
"""Phase 1 integration experiment: does fusing preference + quality beat either alone?

Phase 0 validated the two layers separately (Claim 1: the preference graph
transfers taste; Claim 2: the QIL extracts use-profile quality). This experiment
tests the *combination* — an agent that ranks products with the confidence-
adaptive α-blend from ``docs/architecture.md``:

    score = alpha * pref_score + (1 - alpha) * quality_score
    alpha = sigmoid(3.0 * (mean_confidence - 0.5))

It runs four conditions on the integrated benchmark and reports NDCG@10 with
paired-bootstrap significance:

    preference_only (a=1)  quality_only (a=0)  fixed_alpha (a=0.5)  adaptive_alpha

MILESTONE (gated): the α-blend beats *both* single-signal baselines — i.e.
combining the layers helps. A second question — whether *adaptive* α beats a
*fixed* α — is reported but not gated; in this uniform-evidence regime the answer
is honestly "no", and the script shows why via the per-cohort optimal-α curve.

Usage:
    python experiments/run_phase1_integration.py [--users N] [--seed S] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preferencelayer.agent import IntegrationHarness  # noqa: E402
from preferencelayer.agent.combine import alpha_from_confidence  # noqa: E402
from preferencelayer.data import integrated  # noqa: E402


def _print_conditions(rep) -> None:
    print("\n=== Ranking conditions (NDCG@10) ===")
    print(f"{'condition':<18}{'NDCG@10':>10}{'±std':>8}{'mean_alpha':>12}")
    print("-" * 48)
    for name in ("preference_only", "quality_only", "fixed_alpha", "adaptive_alpha"):
        r = rep.conditions[name]
        a = f"{r.mean_alpha:.3f}" if name == "adaptive_alpha" else "-"
        print(f"{name:<18}{r.ndcg:>10.4f}{r.ndcg_std:>8.3f}{a:>12}")


def _print_cohorts(rep) -> None:
    print("\n=== By history cohort (mean NDCG@10) ===")
    print(f"{'cohort':<8}{'n':>5}{'conf':>7}{'pref':>9}{'qual':>9}{'fixed':>9}{'adapt':>9}")
    print("-" * 56)
    for c in rep.cohorts:
        b = c.by_condition
        print(f"{c.cohort:<8}{c.n_users:>5}{c.mean_confidence:>7.2f}"
              f"{b['preference_only']:>9.3f}{b['quality_only']:>9.3f}"
              f"{b['fixed_alpha']:>9.3f}{b['adaptive_alpha']:>9.3f}")


def _print_optimal_alpha(rep) -> None:
    print("\n=== Empirically optimal α vs. the formula's α, by cohort ===")
    print(f"{'cohort':<8}{'conf':>7}{'formula_a':>11}{'optimal_a':>11}")
    print("-" * 37)
    for cohort, conf, opt in rep.optimal_alpha:
        print(f"{cohort:<8}{conf:>7.2f}{alpha_from_confidence(conf):>11.2f}{opt:>11.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("Generating integrated preference+quality benchmark...")
    scenario = integrated.generate(n_users=args.users, seed=args.seed)
    print(f"  users={len(scenario.users)} products={len(scenario.products)} "
          f"use_profiles={list(scenario.use_profiles)} "
          f"quality_weight={scenario.quality_weight} qil_signals={len(scenario.signals)}")

    rep = IntegrationHarness(scenario, k=10, seed=13).run(with_alpha_curve=True)
    _print_conditions(rep)
    _print_cohorts(rep)
    _print_optimal_alpha(rep)

    print("\n=== Adaptive α vs. the alternatives (paired bootstrap) ===")
    for name in ("preference_only", "quality_only", "fixed_alpha"):
        gain, p = rep.comparisons[name]
        print(f"  vs {name:<16} gain={gain:+.4f}  p={p:.4f}")

    print(f"\nMILESTONE — α-blend beats both single layers: "
          f"{'PASS' if rep.milestone_pass else 'FAIL'}")
    gain, p = rep.adaptive_vs_fixed
    verdict = "beats" if gain > 0 and p < 0.05 else ("trails" if gain < 0 else "ties")
    print(f"Secondary (not gated) — adaptive α {verdict} the fixed 0.5 blend "
          f"(gain={gain:+.4f}, p={p:.4f}).")
    print("  In this uniform-evidence regime the optimal α is ~constant, so a fixed")
    print("  balanced blend is hard to beat and the documented sigmoid overshoots.")

    if args.json:
        payload = {
            "config": {"users": args.users, "seed": args.seed,
                       "quality_weight": scenario.quality_weight},
            "conditions": {
                name: {"ndcg": r.ndcg, "ndcg_std": r.ndcg_std,
                       "mean_alpha": None if name != "adaptive_alpha" else r.mean_alpha}
                for name, r in rep.conditions.items()
            },
            "cohorts": [
                {"cohort": c.cohort, "n_users": c.n_users, "mean_confidence": c.mean_confidence,
                 "by_condition": c.by_condition}
                for c in rep.cohorts
            ],
            "optimal_alpha": [
                {"cohort": c, "mean_confidence": conf,
                 "formula_alpha": alpha_from_confidence(conf), "optimal_alpha": opt}
                for c, conf, opt in rep.optimal_alpha
            ],
            "comparisons": {name: {"abs_gain": g, "p_value": p}
                            for name, (g, p) in rep.comparisons.items()},
            "milestone_pass": rep.milestone_pass,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}")

    return 0 if rep.milestone_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
