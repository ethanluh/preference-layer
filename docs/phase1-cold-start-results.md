# Phase 1: Zero-History Cold-Start — Adaptive α in Its Best-Case Regime

**The one regime the architecture's "new user → lean on quality" intuition targets**
**Status:** premise confirmed; documented adaptive α still only ties a fixed blend

---

## Summary

Three earlier increments found that confidence-adaptive α never beats a fixed balanced
α, and the integration report showed *why*: under the benchmark's conditions the
empirically-optimal α was nearly the same for every user (~0.5). But there was a
caveat — even the "cold" cohort (history 1–3) gets a usable population-prior
preference fit, so leaning on quality was never clearly optimal. The architecture's
intuition for adaptive α is really about a regime that benchmark never tested: a
**brand-new user with zero history**, whose preference fit collapses to the population
prior (credential confidence = 0) and who *must* fall back on community quality.

This study adds that cohort and asks, on the zero-history users specifically: does the
documented α = `sigmoid(3·(confidence − 0.5))` — which is exactly **0.18 at confidence
0** — beat a fixed α = 0.5? The answer is a clean, two-part honest result.

**The premise holds (new, positive).** For zero-history users — and *only* for them —
quality alone beats preference alone, and the empirically-optimal α collapses toward 0:

| cohort | mean conf | pref-only | quality-only | fixed α=0.5 | adaptive | **optimal α** |
|--------|----------:|----------:|-------------:|------------:|---------:|----------:|
| **new** (history 0) | 0.00 | 0.173 | **0.249** | 0.235 | 0.249 | **0.10** |
| cold (1–3) | 0.20 | 0.308 | 0.257 | 0.427 | 0.400 | 0.45 |
| warm (6–12) | 0.52 | 0.501 | 0.251 | 0.652 | 0.653 | 0.55 |
| rich (22–40) | 0.79 | 0.555 | 0.243 | 0.731 | 0.717 | 0.60 |

*(480 users, seed 23, `include_new_cohort=True`. Reproduce with
`python experiments/run_phase1_cold_start.py`.)*

This is the **first time the optimal α genuinely varies** across cohorts — from **0.10
for brand-new users up to 0.60 for rich-history users** — exactly the cold→quality /
rich→preference crossover the architecture predicts. The earlier "optimal α is
≈constant" finding was an artifact of never including true cold-start users.

**But the documented formula still only ties a fixed blend (honest verdict).** The
sigmoid gives α ≈ 0.18 at confidence 0 — directionally right (it leans on quality) and
enough to beat preference-only — yet on the zero-history cohort it is **statistically
indistinguishable from a fixed α = 0.5**: gain **+0.014 NDCG@10, p = 0.31** (and the
sign flips with seed/size — a wash). Two reasons:

- **Mis-calibration:** the optimal α for new users is ~0.10, but the sigmoid floors at
  0.18 at zero confidence — it does not lean far enough.
- **The fixed blend is already self-correcting:** z-scoring centers each stream, so a
  fixed α = 0.5 still leans on whichever signal is informative (here, quality, since
  the prior-only preference score is near-uninformative). Even the *optimal* α only
  edges fixed-0.5 by ~0.01.

So the adaptive mechanism's **premise is vindicated** in its ideal regime, but its
**practical payoff stays marginal** — the same conclusion as every prior increment,
now established even where adaptation should help most.

---

## Scope and honesty notes

- **Fairly set up for adaptive α.** Zero-history users genuinely have no personal
  signal (empty purchase history → the graph returns its population prior, the correct
  `λ = n/(n+pivot) = 0` limit), and the quality evidence is the same the QIL gives
  everyone. This is the mechanism's best case, not a strawman.
- **The optimal-α value for `new` is noisy at small n** (the cohort is ~¼ of users and
  its preference score is the near-uninformative prior); it is stable (~0–0.2, clearly
  below the other cohorts) only at n ≳ 400, which is why the headline uses 480 users.
  The *direction* (lowest for new) and the quality-beats-preference flip are robust at
  all sizes and are what the regression test asserts.
- **The adaptive-vs-fixed result is a tie, reported as such.** The gain is small and
  not significant (and its sign varies across seeds/sizes); we do not claim a win.
- Synthetic benchmark with planted ground truth, consistent with the other Phase-1
  studies; default benchmarks are byte-for-byte unchanged (`include_new_cohort`
  defaults off), so the integration and quality-handling results are unaffected.

---

## Method

- **`data/integrated.generate(include_new_cohort=True)`** prepends a `("new", 0, 0)`
  history cohort: zero purchases, `mean_confidence = 0/(0+pivot) = 0`, with relevant
  set and candidates still defined from the planted utility.
- **`models/graph.SparsePreferenceGraph.fit`** now returns the population prior
  directly for an empty history (the `λ=0` limit), instead of dividing by an empty
  training set.
- **`agent/evaluate.IntegrationHarness`** scores the four conditions per user as
  before; `CohortBreakdown` now carries per-condition per-user NDCG so the
  zero-history adaptive-vs-fixed contrast can be tested with the paired bootstrap
  (`eval/harness._paired_bootstrap_p`). The optimal-α-per-cohort sweep is reused
  unchanged.

---

## Reproducing

```bash
python experiments/run_phase1_cold_start.py     # cohort table + optimal-α curve + verdict
python -m pytest tests/test_cold_start_gate.py  # structural + premise regression
```

Raw metrics: [`experiments/phase1_cold_start_results.json`](../experiments/phase1_cold_start_results.json).

---

## Status

| Stage | Result |
|-------|--------|
| Integration — α-blend beats either layer alone | **+39% / +134%** ([report](phase1-integration-results.md)) |
| Integration — adaptive α beats fixed α | **No** — optimal α ≈ constant under the tested conditions ([report](phase1-integration-results.md)) |
| Quality handling — shrinkage vs. raw averaging | **Crossover**; shrinkage is noise-robust ([report](phase1-quality-robustness-results.md)) |
| Protocol — rank over the real PTP + QIL MCP tools | **Works** end-to-end ([report](phase1-protocol-integration.md)) |
| **Cold-start — adaptive α in its best-case (zero-history) regime** | **Premise confirmed** (optimal α 0.10→0.60 across cohorts; quality beats preference for new users) but adaptive **ties** a fixed blend (+0.014, p=0.31) (this report) |

The arc's conclusion: combining the two layers is the decisive win; the optimal blend
weight genuinely does shift toward quality for new users and toward preference for
established ones — but a fixed balanced blend over z-scored signals captures almost all
of that for free, so confidence-adaptive α is a sound idea with marginal practical
value. The natural production lever, if adaptation is wanted, is to calibrate α to fall
further at zero confidence (toward the measured optimum ~0.1) rather than the doc's 0.18.
