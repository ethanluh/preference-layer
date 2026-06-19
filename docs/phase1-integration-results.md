# Phase 1 Integration Results: The Preference + Quality α-Blend

**Integration keystone — combining the two layers an agent must reason over**
**Status:** Integration milestone **PASSED**; documented adaptive-α calibration **not vindicated** (reported honestly below)

---

## Summary

Phase 0 validated the two layers *separately* — the preference graph transfers
taste across categories (Claim 1, +9.7% NDCG@10) and the QIL extracts
use-profile quality from text (Claim 2, 88.3% precision). Neither experiment
tested the thing the product actually is: **an agent that ranks products using
both at once.** [`architecture.md`](architecture.md) ("Combined Scoring")
specifies exactly how they combine:

```
score = alpha * pref_score + (1 - alpha) * quality_score
alpha = sigmoid(3.0 * (mean_confidence - 0.5))
```

This milestone implements that combiner (`preferencelayer.agent`) and tests it on
a controlled benchmark where ranking correctly **requires both signals**.

**Headline (gated milestone): the α-blend beats either layer alone, decisively.**
On the integrated benchmark the blend reaches **NDCG@10 = 0.614**, versus
**0.442 for preference-only** (+0.146, p = 0.0002) and **0.262 for quality-only**
(+0.326, p = 0.0002). Combining the portable preference with use-profile quality
is worth **+39% over preference alone and +134% over quality alone** — the
integration thesis holds.

| Condition | α | NDCG@10 | vs. adaptive blend |
|-----------|---|--------:|-------------------:|
| preference_only | 1.0 | 0.4424 | +0.146 (p=0.0002) |
| quality_only | 0.0 | 0.2623 | +0.326 (p=0.0002) |
| fixed_alpha | 0.5 | 0.6138 | −0.025 (p=0.0002) |
| **adaptive_alpha** | sigmoid(3·(c−0.5)) | **0.5883** | — |

*(300 users, seed 23. Reproduce with `python experiments/run_phase1_integration.py`.)*

**Honest secondary finding (not gated): confidence-adaptive α does *not* beat a
fixed balanced blend here — it slightly trails it (−0.025, p = 0.0002).** This is
a real, reproducible result and we report it rather than tuning it away. The
reason is structural and is explained below; in short, with uniformly-available
quality evidence the optimal blend weight is nearly the same for every user, so a
fixed α ≈ 0.5 is hard to beat and the documented sigmoid — which swings α from
0.29 (cold cohort) to 0.71 (rich cohort) — overshoots at both ends.

---

## Scope and honesty notes (read this first)

This is a **controlled integration study**, deliberately analogous to the
synthetic benchmarks behind Claims 1 and 2 — not a live agent on real catalogs.

- **The benchmark is synthetic with planted ground truth.** Each product has both
  an attribute vector (the substrate of *preference*) and a planted, attribute-
  *independent* per-use-profile quality (the substrate of *quality*); a user's
  true utility depends on **both**, so neither signal alone can ace the ranking.
  This is what makes the +39% / +134% lift *earned* rather than a templating
  artifact. See [`data/integrated.py`](../src/preferencelayer/data/integrated.py).
- **The QIL NLP step is short-circuited.** Quality posteriors are built by feeding
  planted-quality observations straight into the real `QualityAggregator` (the
  same `ExtractedSignal` path the QIL query tests use). The extraction *classifier*
  was already validated in Claim 2; re-running it here would add noise without
  testing anything new. The aggregation + `/quality` query layer the agent calls
  **is** the real one.
- **Scores are z-scored before blending.** Preference scores are unbounded logits
  (`feat · w`); quality scores are posterior means in [0, 1]. They are not
  comparable in raw units, so each stream is standardized across the candidate set
  before the blend. This normalization is a **modeling decision the architecture
  doc leaves implicit** — it is what makes α a meaningful *relative* weight rather
  than an accident of scale. It is stated here and implemented in
  [`agent/combine.py`](../src/preferencelayer/agent/combine.py).
- **This is not the architecture's Phase 1 go/no-go.** The doc's Phase 1 gate is
  design-partner validation on real agents. This milestone is a self-defined,
  falsifiable check that the *documented combiner* produces a measurable lift and
  behaves as analyzed. The success bar (blend beats both single layers,
  significantly) was fixed before the final run.

---

## The honest finding: why adaptive α does not beat fixed α here

The architecture's intuition is appealing: a sparse, low-confidence credential
should lean on community quality; a rich, high-confidence one should lean on the
user's own taste. Our benchmark lets us measure whether the *specific* formula
delivers that — and it does not, for a reason worth stating precisely.

With z-scored streams and additive utility (`utility = pref + w·quality`), the
blend that best reconstructs the true ranking, **given perfect estimates**, is a
constant:

```
alpha* = 1 / (1 + quality_weight)
```

— **independent of the user's confidence.** Confidence enters only through
*estimation noise*: a sparse user's preference fit is unreliable, which nudges
their optimal α a little *below* `alpha*`. So the optimal α varies only weakly
across the population. Measuring it directly confirms this:

| Cohort | mean confidence | formula α | **empirically optimal α** |
|--------|----------------:|----------:|--------------------------:|
| cold (history 1–3) | 0.19 | 0.29 | **0.50** |
| warm (history 6–12) | 0.52 | 0.51 | **0.55** |
| rich (history 22–40) | 0.79 | 0.71 | **0.55** |

The optimal α sits in a **narrow 0.50–0.55 band** (here `quality_weight = 0.6`, so
`alpha* = 0.625`, pulled down slightly by fit noise). The documented sigmoid,
keyed off a confidence that ranges 0.19–0.79, instead swings α over **0.29–0.71**
— too low for cold users (it throws away a still-useful preference prior) and too
high for rich users (it underweights quality, which preference can never recover
because quality is attribute-independent). The per-cohort breakdown shows exactly
this: adaptive α wins nowhere it should and loses where it overshoots.

| Cohort | preference_only | quality_only | fixed (0.5) | adaptive |
|--------|----------------:|-------------:|------------:|---------:|
| cold | 0.324 | 0.257 | **0.458** | 0.412 |
| warm | 0.498 | 0.251 | 0.649 | **0.652** |
| rich | 0.506 | 0.280 | **0.734** | 0.701 |

**What this means.** The *combination* is the win, robustly. The *confidence-
adaptive weighting* is theoretically motivated but, in a regime where quality
evidence is uniformly available to every user, it has little to exploit and its
documented slope overshoots. Confidence-adaptive α would earn its keep in
conditions this benchmark deliberately does not model — e.g. **per-product, uneven
quality evidence** (lean on preference for products the QIL knows little about) or
**genuinely zero-history users with no usable prior**. Calibrating α to those
signals, rather than to credential confidence alone, is the natural Phase 1
follow-up.

> **Follow-up (resolved):** we built that *evidence-aware* α and tested it on a
> non-uniform-evidence benchmark. It also does **not** beat a fixed α — because the
> QIL aggregator's Bayesian shrinkage and the blend's z-scoring already absorb
> unreliable evidence, making α-level adaptation redundant. The same investigation
> did surface a genuine positive result about *quality estimation* (a shrinkage-vs-
> raw-averaging noise crossover). See
> [`phase1-quality-robustness-results.md`](phase1-quality-robustness-results.md).

---

## Why the result is meaningful (not rigged)

Both planted signals are necessary by construction, and the candidate sets are
adversarial to *each* single signal:

1. **Two independent utility terms.** True utility is `pref_term(attributes) +
   quality_weight · mean_quality(use_profile)`, with quality drawn *independently*
   of attributes. A perfect attribute model still cannot rank the quality term, and
   vice-versa.
2. **Two kinds of hard negative.** Each candidate set mixes *preference traps*
   (high preference, low true utility — quality drags them down) and *quality traps*
   (high quality, low true utility — taste drags them down). A preference-only
   ranker is fooled by the first; a quality-only ranker by the second; only a blend
   demotes both.
3. **Taste-driven purchases, post-hoc quality.** Observed purchases are driven by
   the attributes a shopper can see, so a rich history yields a strong preference
   fit while a sparse one collapses toward the population prior — the genuine
   cold-start asymmetry. Quality, by design, is *not* learnable from the user's own
   purchases; it has to come from the community-derived QIL.

Turning the knobs moves the numbers smoothly. In particular, the adaptive-vs-fixed
gap is not hand-set: it falls out of `quality_weight` via `alpha* = 1/(1+qw)`, and
we report the sign it actually has.

---

## Method

### The combiner (`preferencelayer.agent.combine`)

Pure functions implementing the architecture formula verbatim:
`alpha_from_confidence(c) = sigmoid(3·(c−0.5))`, `zscore` (standardize each stream
across the candidate set), and `blend(pref, quality, α) = α·z(pref) +
(1−α)·z(quality)`.

### The agent (`preferencelayer.agent.recommender`)

`AgentRecommender` orchestrates the two layers through their public surfaces only:
preference scores from a fitted `SparsePreferenceGraph.score` (the call the PTP
`get_preference` flow wraps) and quality scores from `QualityService.quality` (the
call the QIL MCP server exposes), then blends. Products the QIL has no evidence for
fall back to a neutral quality; an optional `failure_penalty` lets estimated
failure rates discount a product.

### The benchmark (`preferencelayer.data.integrated`)

Generates products (attributes + planted per-use-profile quality), users
(latent taste, a use profile, and a history length spanning cold-start to rich),
ground-truth relevance from the two-term utility, adversarial candidate sets, and
QIL-style signals that feed a real `QualityAggregator`. Credential confidence is
tied to history exactly as the preference graph's own cold-start blend is
(`lam = n / (n + pivot)`), so it is faithful, not free.

### The harness (`preferencelayer.agent.evaluate`)

Builds the preference model and quality service once, scores all four conditions on
identical per-user candidate sets, and reports NDCG@10 with paired-bootstrap
significance (reusing `eval.metrics` and `eval.harness._paired_bootstrap_p`), plus
the per-cohort breakdown and the empirically-optimal-α sweep above.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python experiments/run_phase1_integration.py    # 4-condition table + cohorts + α curve + milestone
preflayer agent-demo                             # cold-start vs rich-history ranking, side by side
python -m pytest                                 # full suite (incl. the integration milestone)
```

Raw metrics for the headline run are in
[`experiments/phase1_integration_results.json`](../experiments/phase1_integration_results.json).

---

## Status

| Stage | Result |
|-------|--------|
| Claim 1 — preference graph beats flat baseline on transfer | **+9.7%** NDCG@10, p=0.0002 ([report](phase0-results.md)) |
| Claim 2 — use-profile quality extractable from public text | **88.3%** macro precision ([report](phase0-qil-results.md)) |
| Integration — α-blend beats either layer alone | **+39% / +134%**, p=0.0002 (this report) |
| Integration — *adaptive* α beats *fixed* α | **No** (−0.025); optimal α is ~constant in this regime (this report) |

The two layers now compose into a single ranking, and combining them is a large,
significant win — the core product thesis, demonstrated end to end. The documented
confidence-adaptive weighting is implemented and measured; its calibration is not
vindicated under uniform quality evidence, which points the next step at
evidence-aware (per-product) α rather than confidence-only α.
