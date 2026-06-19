# Phase 1 Study: Handling Noisy Quality Evidence (Shrinkage vs. Raw)

**Where evidence-awareness belongs in the preference + quality agent**
**Status:** positive finding — Bayesian shrinkage is the noise-robust quality estimator; α-level evidence-adaptation is redundant

---

## Summary

The integration milestone showed that *combining* preference and quality is the
big win ([report](phase1-integration-results.md)), and that the *blend weight* α
barely matters — a fixed α≈0.5 is hard to beat, and a confidence-adaptive α does
not beat it. A natural follow-up asked whether making α **evidence-aware** (lean on
preference for products with thin quality evidence) would finally make adaptive
weighting pay off. It does not. But chasing that question surfaced the choice that
*does* move the needle: **how the quality estimate itself is formed from noisy,
unevenly-distributed review evidence.**

We cross two quality estimators with two blend weights on a benchmark with
**non-uniform evidence** (some products richly reviewed, others barely):

- **estimator:** Bayesian-shrunk posteriors (the shipped QIL aggregator) vs. raw
  confidence-weighted sample means (`prior_strength → 0` — the evidence-ignoring
  ablation);
- **blend weight:** fixed α=0.5 vs. per-candidate evidence-aware α.

**Headline (positive): a clean bias–variance crossover.** Sweeping per-observation
noise (a single review is a noisy signal of true quality), raw averaging wins when
signals are clean, but **Bayesian shrinkage is the noise-robust choice and wins
significantly once review signals are noisy** — the realistic regime for messy
public text, which is exactly the QIL's input.

| per-observation noise | shrunk (fixed α) | raw (fixed α) | shrunk − raw | p |
|---:|---:|---:|---:|---:|
| 0.20 | 0.6002 | **0.6171** | −0.0169 | 0.0002 |
| 0.40 | 0.5977 | **0.6078** | −0.0101 | 0.0042 |
| 0.60 | 0.5877 | 0.5909 | −0.0032 | 0.42 |
| 0.80 | **0.5740** | 0.5631 | +0.0109 | 0.022 |
| 1.00 | **0.5578** | 0.5359 | +0.0220 | 0.0002 |

The crossover sits near observation-noise 0.7; by noise 1.0 shrinkage wins by
**+0.022 NDCG@10 (p = 0.0002)**. *(300 users, seed 23, non-uniform evidence
`evidence_lo=1, evidence_hi=30`. Reproduce with
`python experiments/run_phase1_quality_robustness.py`.)*

**Secondary (honest negative): evidence-aware α does not help.** At noise 0.8, in
the estimator × blend-weight 2×2:

| | fixed α | evidence-aware α |
|---|---:|---:|
| **shrunk** | **0.5740** | 0.5308 (−0.043, p=0.0002) |
| **raw** | 0.5631 | 0.5429 (−0.020, p=0.006) |

*(reference: preference-only 0.4424, quality-only 0.2383)*

On *either* estimator the per-candidate evidence-aware α is **worse** than a flat
α=0.5. The reason closes the loop on the whole α investigation: **the aggregator's
shrinkage and the blend's z-scoring already handle unreliable evidence**, so
re-deriving that adjustment at the α layer is redundant and, when it over-corrects,
harmful.

---

## Scope and honesty notes (read this first)

- **The 2×2's first hypothesis was wrong, and we report it.** We initially expected
  Bayesian shrinkage to dominate raw means at *all* noise levels. It does not —
  raw averaging is significantly **better** on clean signals (shrinkage over-biases
  toward the prior there). The honest result is a *crossover*, not a clean win, and
  that is what the table shows.
- **The non-uniform-evidence regime is required and realistic.** With uniform, rich
  evidence every estimator is accurate and the comparison flattens; the effect lives
  precisely where evidence is thin and noisy for some products — the realistic case.
  The benchmark draws each product's evidence count from `[evidence_lo, evidence_hi]`
  independently of its true quality (a pure reliability signal, not a quality proxy).
- **The "raw" estimator is a faithful ablation, not a strawman** — it is the same
  `QualityAggregator` with `prior_strength → 0`, i.e. the confidence-weighted sample
  mean the Normal-Normal posterior reduces to without a prior.
- **Synthetic, mechanism-level.** As with Phase 0, this validates a *design
  principle* on a controlled benchmark with planted ground truth; it is not a
  live-data measurement and remains short of the architecture's design-partner
  Phase 1 gate.

---

## Why — the mechanism

Two layers already encode evidence-awareness, and they compound:

1. **Bayesian shrinkage** (`qil/aggregate.py`, Normal-Normal) pulls a thin-evidence
   posterior toward the neutral prior. This *trades variance for bias*: when a single
   review is very noisy (high observation noise), the raw mean of one or two reviews
   is worse than the prior, and shrinkage's bias is the better bet — so shrinkage
   wins at high noise. When reviews are clean, the same shrinkage just throws away
   usable signal — so raw means win at low noise. That tradeoff is the crossover.
2. **z-score normalization** in the blend (`agent/combine.py`) centers each score
   stream, so a shrunk (near-prior, near-mean) quality estimate maps to ≈0 and
   contributes ≈nothing to the ranking — the blend *automatically* falls back to
   preference for poorly-evidenced products.

Together these mean the pipeline is already self-correcting for unreliable quality.
An explicit evidence-aware α tries to do the same job a third time, at the weighting
layer, and mostly just adds miscalibration — which is why it loses to a flat α.

**Design takeaway:** keep the agent's blend weight **fixed and simple**; put
evidence-awareness in the **aggregation layer**, where Bayesian shrinkage already
provides a principled, noise-robust hedge. The QIL's choice of Bayesian aggregation
over naive averaging is what earns its keep precisely in the noisy regime it targets.

---

## Method

- **Benchmark** (`data/integrated.py`): the integrated preference+quality scenario,
  with per-product evidence drawn from `[evidence_lo, evidence_hi]` and a tunable
  `signal_obs_noise` (the per-review spread that is swept).
- **Estimators**: `QualityAggregator()` (shipped shrinkage, `prior_strength=4`) and
  `QualityAggregator(prior_strength≈0)` (raw means), both over the *same* extracted
  signals, served through `QualityService`.
- **Agent** (`agent/recommender.py`): `AgentRecommender.rank` with `alpha=0.5`
  (fixed) or `evidence_aware=True` (per-candidate α from credential confidence ×
  per-candidate `quality_reliability`, via `agent/combine.evidence_adaptive_alpha`).
- **Harness** (`agent/ablation.py`): `QualityHandlingHarness` evaluates the four
  cells plus preference-only / quality-only references on identical per-user
  candidate sets, with paired-bootstrap significance (`eval/harness._paired_bootstrap_p`)
  and NDCG@10 (`eval/metrics`).

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python experiments/run_phase1_quality_robustness.py    # crossover sweep + 2×2 + verdict
python -m pytest                                       # full suite (incl. this gate)
```

Raw metrics for the headline run are in
[`experiments/phase1_quality_robustness_results.json`](../experiments/phase1_quality_robustness_results.json).

---

## Status

| Stage | Result |
|-------|--------|
| Claim 1 — preference graph beats flat baseline on transfer | **+9.7%** NDCG@10 ([report](phase0-results.md)) |
| Claim 2 — use-profile quality extractable from public text | **88.3%** macro precision ([report](phase0-qil-results.md)) |
| Integration — α-blend beats either layer alone | **+39% / +134%** ([report](phase1-integration-results.md)) |
| Integration — adaptive α (confidence or evidence) beats fixed α | **No** — fixed α is robust ([report](phase1-integration-results.md), this report) |
| Quality handling — shrinkage vs. raw averaging | **Crossover**: shrinkage is the noise-robust estimator, winning as review noise rises (this report) |

The practical upshot for the agent: **combine both layers, weight them with a fixed
balanced α, and rely on the QIL's Bayesian aggregation — not a clever blend weight —
to absorb noisy, uneven evidence.**
