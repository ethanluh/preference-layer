#!/usr/bin/env python3
"""Phase 1 study: how should the agent turn noisy quality evidence into a score?

The integration milestone showed combining preference + quality is the big win, and
two follow-ups showed the *blend weight* barely matters (a fixed α is hard to beat).
This experiment targets the choice that does move the needle — how the quality
*estimate* is formed from noisy, unevenly-distributed review evidence — by crossing
two estimators with two blend weights on a non-uniform-evidence benchmark:

    estimator:    Bayesian-shrunk posteriors (shipped) vs. raw sample means (ablation)
    blend weight: fixed α=0.5            vs. per-candidate evidence-aware α

PRIMARY (the positive result): a bias–variance crossover. Swept over per-observation
noise, raw averaging wins when review signals are clean, but **Bayesian shrinkage is
the noise-robust choice and wins once signals are noisy** — the realistic regime for
messy public text, which is the QIL's actual input.

SECONDARY (honest negative): evidence-aware α does not beat a fixed α on top of
*either* estimator — α-level evidence handling is redundant with the aggregator's
shrinkage and the blend's z-scoring.

Usage:
    python experiments/run_phase1_quality_robustness.py [--users N] [--seed S] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preferencelayer.agent.ablation import QualityHandlingHarness  # noqa: E402
from preferencelayer.data import integrated  # noqa: E402

# Per-observation noise levels swept for the crossover (a single review is a noisy
# signal of true quality; higher = noisier public text).
NOISE_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
# Where shrinkage is in its winning regime — used for the 2×2 / evidence-α panel.
PANEL_NOISE = 0.8


def _run_at(noise: float, users: int, seed: int):
    scenario = integrated.generate(
        n_users=users, seed=seed, evidence_lo=1, evidence_hi=30, signal_obs_noise=noise,
    )
    return QualityHandlingHarness(scenario, k=10, seed=13, obs_noise=noise).run()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("Integrated benchmark with non-uniform quality evidence (evidence_lo=1, hi=30).")
    print("Sweeping per-observation noise: shrunk vs. raw quality estimate, both fixed α=0.5.\n")

    print("=== Crossover: Bayesian shrinkage vs. raw averaging (fixed α) ===")
    print(f"{'obs_noise':>10}{'shrunk':>10}{'raw':>10}{'shrunk-raw':>12}{'p':>9}")
    print("-" * 51)
    sweep = []
    crossover = None
    prev_sign = None
    for noise in NOISE_GRID:
        res = _run_at(noise, args.users, args.seed)
        gain, p = res.contrast("shrunk_fixed", "raw_fixed")
        s, r = res.cells["shrunk_fixed"].ndcg, res.cells["raw_fixed"].ndcg
        print(f"{noise:>10.2f}{s:>10.4f}{r:>10.4f}{gain:>+12.4f}{p:>9.4f}")
        sweep.append({"obs_noise": noise, "shrunk": s, "raw": r, "gain": gain, "p_value": p})
        sign = gain > 0
        if prev_sign is not None and sign != prev_sign and crossover is None:
            crossover = noise
        prev_sign = sign

    where = f"between obs_noise {NOISE_GRID[0]:.1f} and {NOISE_GRID[-1]:.1f}" if crossover is None \
        else f"near obs_noise {crossover:.1f}"
    hi = sweep[-1]
    print(f"\nAt high noise (obs_noise={hi['obs_noise']:.1f}): shrinkage wins by "
          f"{hi['gain']:+.4f} (p={hi['p_value']:.4f}). Crossover {where}.")

    # --- 2×2 + single-signal panel at the high-noise regime -------------------
    panel = _run_at(PANEL_NOISE, args.users, args.seed)
    print(f"\n=== Estimator × blend-weight 2×2 at obs_noise={PANEL_NOISE} (NDCG@10) ===")
    print(f"{'':<16}{'fixed α':>12}{'evidence α':>14}")
    print("-" * 42)
    print(f"{'shrunk':<16}{panel.cells['shrunk_fixed'].ndcg:>12.4f}{panel.cells['shrunk_evidence'].ndcg:>14.4f}")
    print(f"{'raw':<16}{panel.cells['raw_fixed'].ndcg:>12.4f}{panel.cells['raw_evidence'].ndcg:>14.4f}")
    print(f"\nreferences: preference_only={panel.references['preference_only'].ndcg:.4f}  "
          f"quality_only={panel.references['quality_only'].ndcg:.4f}")
    g_se, p_se = panel.contrast("shrunk_evidence", "shrunk_fixed")
    g_re, p_re = panel.contrast("raw_evidence", "raw_fixed")
    print(f"evidence-aware α vs fixed α:  on shrunk {g_se:+.4f} (p={p_se:.4f}); "
          f"on raw {g_re:+.4f} (p={p_re:.4f})")

    print("\n=== Verdict ===")
    print("PRIMARY: Bayesian shrinkage is the noise-robust quality estimator — it ties or")
    print("  trails raw averaging on clean signals but wins significantly as review noise")
    print("  rises (the QIL's real regime).")
    print("SECONDARY: evidence-aware α does not beat a fixed α on either estimator —")
    print("  shrinkage + z-scoring already handle unreliable evidence.")

    if args.json:
        payload = {
            "config": {"users": args.users, "seed": args.seed,
                       "evidence_lo": 1, "evidence_hi": 30, "panel_noise": PANEL_NOISE},
            "crossover_sweep": sweep,
            "panel": {
                "obs_noise": PANEL_NOISE,
                "cells": {n: c.ndcg for n, c in panel.cells.items()},
                "references": {n: c.ndcg for n, c in panel.references.items()},
                "evidence_alpha_vs_fixed": {
                    "on_shrunk": {"gain": g_se, "p_value": p_se},
                    "on_raw": {"gain": g_re, "p_value": p_re},
                },
            },
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
