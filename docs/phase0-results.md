# Phase 0 Results: Preference Graph vs. Flat Baselines

**Work Stream A — Preference Graph Prototype**
**Status:** Go/no-go gate **PASSED**

---

## Summary

The central Phase 0 research question (from [`implementation-plan.md`](implementation-plan.md)):

> **Claim 1:** A compact preference graph outperforms cold-start baselines on
> cross-category recommendation tasks.
>
> **Go/no-go criterion:** Sparse DAG outperforms the flat vector by ≥ 5% NDCG@10
> on cross-category transfer.

**Result: the sparse preference graph beats the strong flat-attribute baseline by
+9.7% NDCG@10 on cross-category transfer (laptops → headphones), p = 0.0002.**
The lift is robust across random seeds (+9.5% to +17.5% over five seeds, every one
significant at p < 0.05). The gate is met.

| Model | Transfer NDCG@10 | Within-category NDCG@10 |
|-------|-----------------:|------------------------:|
| **preference_graph (sparse DAG)** | **0.7491** | **0.5392** |
| flat_attribute (mean attribute vector) | 0.6827 | 0.5260 |
| flat_item_embedding (mean item embedding) | 0.2865 | 0.2467 |
| popularity (non-personalized) | 0.2721 | 0.1488 |

*(500 users, seed 7, laptops → headphones transfer. Reproduce with
`python experiments/run_phase0.py --users 500`.)*

---

## Why the graph wins (and why the comparison is fair)

The flat-attribute baseline is **strong, not a strawman**: it represents the user
as the mean attribute vector of their purchases over the *shared* attribute
vocabulary, so it recovers the full **linear** component of taste and transfers it
across categories perfectly. On a task with purely linear preferences, it is hard
to beat — and indeed the graph only ties it there.

The graph's advantage comes from one structural capability the flat vector lacks:
**modeling attribute interactions** (the edges). An edge encodes a *conditional*
preference — e.g. "battery life matters to me only when portability is also high".
These are exactly the cases the design docs argue flat encodings cannot represent.

Three design choices make the demonstration honest rather than rigged:

1. **Both models see the same shared attributes.** The graph is literally the flat
   model *plus learned edges*: its linear weights are warm-started from the same
   robust mean estimator the flat baseline uses. Any lift is attributable to the
   edges alone.
2. **The benchmark plants a known interaction structure** and the evaluation uses
   **hard negatives** — items that score highly on linear attributes but are
   dispreferred once interactions are accounted for. This is the realistic,
   discriminating case; with easy (random) negatives both models saturate and the
   distinction disappears (a result we verified and report below).
3. **Edge discovery is unsupervised.** The graph does not see the planted pairs; it
   recovers ≥ 7 of 8 of them via the cross-user-correlation-variance heuristic
   (below), then learns their weights per user.

---

## Method

### Data (synthetic benchmark)

`preferencelayer.data.synthetic` generates users whose preferences transfer across
categories. Each user `u` has:

- `theta_u`: linear taste over the 8 shared attributes (identical across categories),
- `phi_u`: sparse interaction taste over 8 shared-attribute pairs (also shared).

Item utility is `theta_u·x + Σ phi_u·x_a·x_b + (category-local taste) + noise`. The
linear term transfers and is recoverable by a flat model; the interaction term is
recoverable only by a model with edges; the category-local term is deliberately
non-transferable.

The Amazon Reviews 2023 real-data path is implemented in
`preferencelayer.data.amazon` (requires `pip install preferencelayer[amazon]`); it
produces the same `CategoryData` objects so every model and metric runs unchanged.
The synthetic benchmark is used for the headline result because it provides a known
ground truth and full reproducibility offline.

### Models (`preferencelayer.models`)

- **popularity** — non-personalized; recommends globally popular items. The floor.
- **flat_item_embedding** — mean of per-category item embeddings (random projection
  of attributes). Strong within a category; transfers poorly because item-embedding
  bases do not align across categories (the real-world platform-silo problem).
- **flat_attribute** — mean shared-attribute vector, cosine scoring. Strong and
  fully transferable on the linear component.
- **preference_graph** — sparse DAG over shared attributes. Nodes = linear weights,
  edges = interaction weights. Topology discovered by PMI/correlation-variance;
  weights fit per user by regularized logistic ranking with a population cold-start
  prior and a history-size-dependent blend.

### Edge discovery

A genuine interaction pair shows a *conditional* dependence whose sign varies across
users (a complement for some, a substitute for others). Pooled co-occurrence cancels
this out, so plain corpus-wide PMI misses it. Instead we rank pairs by the **variance
across users of the within-user attribute correlation**. This recovers all 8 planted
pairs in the top 10 at 400+ users (7–8 of 8 at 200 users).

### Evaluation (`preferencelayer.eval`)

Cross-category transfer: fit each model on a user's *laptop* purchases, then rank a
*headphone* candidate set (relevant items + hard negatives + random fill). No
headphone history is shown to the model. Metric: NDCG@10, averaged over users, with a
paired bootstrap for significance.

---

## Ablations and honesty notes

- **Hard negatives are necessary to surface the effect.** With random negatives, both
  the graph and the flat model exceed 0.82 NDCG@10 and the gap is within noise
  (+0.2%, n.s.). Distinguishing high-utility items from random ones does not require
  interaction modeling; distinguishing them from linearly-attractive *traps* does.
- **The effect grows with history length.** At ~12 purchases/user the per-user
  interaction fit is high-variance and the lift is small/unstable; at 30+ purchases it
  is robustly +10–17%. This matches the design thesis that rich-history users benefit
  most from preference structure (and motivates the cold-start prior for sparse users).
- **Within-category, the graph only ties the flat baseline.** The transfer setting is
  where the graph's transferable interaction structure pays off; this is the claim the
  project actually makes, and the result is specific to it.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python experiments/run_phase0.py --users 500          # headline table + gate
python -m pytest                                       # full test suite (incl. the gate)
```

Raw metrics for the headline run are in
[`experiments/phase0_results.json`](../experiments/phase0_results.json).
