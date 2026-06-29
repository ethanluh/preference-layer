# Phase 1: Real-Text Attribute Extraction — Closing the Featurization Gap

**Wiring the QIL text extractor into the real-data ranking loop — the lever the Amazon reality check identified**
**Status:** design / proposal — not yet run

---

## Why this experiment exists

Two results bracket an untested hypothesis. The synthetic benchmark shows the
sparse-DAG preference graph beats flat baselines by **+9.7% NDCG@10** on transfer
*given clean attribute vectors* ([Claim 1](phase0-results.md)). The Amazon Reviews
2023 reality check shows the same graph **loses** to a flat baseline (−20.8% on
All_Beauty, −50.2% on Cell_Phones) when fed **coarse metadata features** — price
percentile, average rating, title keywords ([real-data report](phase1-amazon-realdata.md)).
Both reports localize the binding constraint identically: **attribute-extraction
quality**, not the ranking model.

But the two halves of the thesis have never actually met. The QIL text extractor
validated at [88.3% macro precision](phase0-qil-results.md) runs on a
*synthetic-but-realistic generated corpus*, not on real review text — and the
real-data ranking experiment used metadata features, **not the QIL extractor**. So
the central claim — *that text-derived attributes are rich enough for structured
preference to beat flat baselines on real data* — has not been tested. This document
designs the experiment that tests it, and defines a deliverable that holds **whether
or not the advantage replicates**.

---

## Hypothesis

**H1.** When item attribute vectors are extracted from review *text* (rather than
item metadata), the sparse-DAG preference graph beats flat baselines on real Amazon
data, and the graph's advantage increases monotonically with extraction quality.

**What would refute it.** If structured preference fails to beat flat baselines even
at near-gold extraction quality, the graph's interaction-modeling value does not
survive real attribute distributions, and the architecture's modeling premise — not
just its featurization — is wrong. That outcome is also a result (see *Outcomes*).

---

## Design

### 1. Attribute schema (kept fixed)

The text extractor must emit the **same** shared-attribute vocabulary the models
already consume, so the only thing that changes between the metadata run and this run
is the *feature source*. This mirrors how the real-data check reused the synthetic
harness unchanged and varied only the data — the comparison stays clean.

### 2. Extractor options and tradeoffs

The current TF-IDF + softmax extractor is the floor, not a candidate; it almost
certainly will not carry messy real text. Three options span the cost/ceiling curve:

| extractor | label need | reproducibility | ceiling | cost |
|-----------|-----------|-----------------|---------|------|
| TF-IDF + softmax (current) | low | high (deterministic) | low | trivial |
| fine-tuned encoder (BERT-class) | medium (real gold set) | high (pinned weights) | medium–high | moderate (training) |
| LLM structured extraction (few-shot) | low (eval only) | medium (model/version drift) | high | per-call \$ |

The recommended primary is the **fine-tuned encoder** for a reproducible, pinnable
result; the **LLM extractor** is the strong-ceiling comparison and the fast path to a
first data point. *This choice is a deliberate open question for the advisor* — it is
the single decision most worth their input before code is written.

### 3. The attribute-quality dial (the methodological core)

Rather than ship one extractor and report one number, produce attribute vectors at
**controlled quality levels** — e.g. by interpolating each extractor's output toward
the gold labels, or by degrading gold with calibrated noise — and trace the
**extraction-quality → transfer-gain curve**. This is what guarantees a result: even
if no available extractor closes the gap today, the curve locates the **quality
threshold Q\*** above which structured preference wins, and reports how far the best
current extractor (Q_now) sits below it. The gap becomes a measured quantity, not a
hand-wave.

### 4. Experiment matrix

| axis | values |
|------|--------|
| data | All_Beauty (within), Cell_Phones (within), Electronics ∩ Cell_Phones shared-user pair (cross-category transfer) |
| feature source | metadata baseline (existing) vs. text-extracted at quality levels {Q_now, …, gold} |
| models | preference_graph, flat_attribute, flat_item_embedding, popularity, **+ one established cross-domain recsys baseline** |
| task | within-category, then cross-category transfer (the headline setting) |
| metric | NDCG@10, paired bootstrap (existing `eval/harness`) |
| seeds | ≥5, report variance (per existing convention) |

The cross-domain baseline is non-negotiable: "beats flat vector" is not "beats the
cross-domain recommendation literature." Pinning the *current* standard baseline is a
literature task that should precede implementation — classic references (EMCDR,
CoNet) may be dated.

### 5. Harness reuse

`ExperimentHarness.run_within` / transfer and the paired bootstrap are reused
verbatim; only the feature source changes. Proposed entry point
`experiments/run_phase1_text_extraction.py`, parallel to `run_amazon_realdata.py`.

### 6. Extraction evaluation (dependency on the real gold set)

The extractor's own quality is measured against a **real-review gold set** — a few
hundred reviews labeled for use-profile and use-conditioned reliability, with a
documented protocol and inter-annotator agreement — reported as macro precision, the
same stricter metric as the synthetic gate. Without real labels, "88.3%" stays a
synthetic artifact. This gold set is the slowest build and the clearest place advisor
infrastructure (annotators, labeling tooling) is load-bearing.

---

## Outcomes (both are results)

- **A — advantage replicates.** Text-derived attributes let structured preference
  beat flat baselines on real data. Report the gain and the quality level at which it
  emerges. Claim: *portable structured preference beats flat baselines on real data
  given extraction quality ≥ Q\*.*
- **B — partial / no replication.** Report the extraction-quality → transfer-gain
  curve, the threshold Q\*, the best current extractor's Q_now, and the residual gap.
  Claim: *structured preference requires extraction quality ≥ Q\* to beat flat
  baselines; the binding open problem is extraction, quantified here.* If even
  near-gold extraction fails, report that the modeling premise does not survive real
  attribute distributions.

There is no run of this experiment that yields nothing publishable. That property is
the reason it is a research question rather than a bet.

---

## Scope and honesty notes (read this first)

- The synthetic gates were always scoped as *model given good features*. This tests
  the features. A and B are both faithful to that scoping; neither retroactively
  weakens Claim 1.
- Within-category is the easier setting and the place to start; cross-category
  transfer is the headline and the heavier offline run.
- The quality dial must be fixed **before** the final run, not tuned to produce a
  flattering threshold — same discipline as the synthetic corpus knobs.
- A real gold set is a hard prerequisite for the extraction-quality number; the curve
  can be traced with gold-interpolation in the interim, but the headline extractor
  claim waits on real labels.

## What this deliberately does not do

Product/protocol work (PTP store, MCP tooling), the confidence-adaptive α line
(settled: it ties a fixed z-scored blend across every increment), and the GP temporal
kernel are all out of scope. They do not move the binding constraint.

---

## Open questions for the advisor

1. Extractor architecture: fine-tuned encoder vs. LLM extraction as the primary —
   reproducibility vs. ceiling.
2. Whether to frame the contribution around the **quality → gain curve** itself (the
   characterization) rather than a single replication result.
3. The right *current* cross-domain recsys baseline to benchmark against.
4. Annotation resources for the real gold set.
5. Target venue and the minimal result that clears its bar.

---

## Proposed reproduction

```bash
# extraction-quality sweep on the existing within-category task
python experiments/run_phase1_text_extraction.py --category All_Beauty --quality-sweep
# cross-category transfer on a shared-user pair (heavier)
python experiments/run_phase1_text_extraction.py \
  --transfer Electronics:Cell_Phones_and_Accessories --quality gold
python -m pytest tests/test_text_extraction.py   # offline assembly + gate tests
```

---

## Status

| Stage | Result |
|-------|--------|
| Claim 1 — graph beats flat (synthetic transfer, clean features) | **+9.7%** NDCG@10 |
| Real-data check — same models, coarse *metadata* features | advantage does not replicate; extraction is the bottleneck |
| Claim 2 — use-profile extractable (synthetic-but-realistic corpus) | **88.3%** macro precision |
| **This design — text-extracted features in the real ranking loop** | **not yet run**; defines the experiment and the quality → gain deliverable |
