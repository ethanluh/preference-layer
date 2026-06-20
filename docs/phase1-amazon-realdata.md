# Phase 1: Real-Data Reality Check (Amazon Reviews 2023)

**Do the attribute models survive real, coarsely-featurized data?**
**Status:** honest negative — the synthetic advantage does *not* replicate; attribute extraction is the bottleneck

---

## Summary

The synthetic benchmark plants clean attribute vectors and a recoverable interaction
signal, and the preference graph beats the flat baseline by **+9.7% NDCG@10** there
(Claim 1). This study runs the *same* model comparison on **real Amazon Reviews 2023**
items and users, where attribute vectors must be derived coarsely from item metadata.
The result is a deliberately honest reality check.

On a within-category ranking task over the `All_Beauty` category (112,590 items, 681
users with ≥5 reviews), with **hard negatives** so a popularity baseline cannot win for
free:

| model | NDCG@10 |
|-------|--------:|
| flat_attribute | **0.0998** |
| flat_item_embedding | 0.0945 |
| preference_graph | 0.0791 |
| popularity | 0.0109 |

**preference_graph vs flat_attribute: −20.8% (p = 0.0004).**
*(Reproduce: `python experiments/run_amazon_realdata.py`.)*

Two honest findings:

1. **The graph's synthetic advantage does not replicate on real data** — it is
   *worse* than the flat-attribute baseline, not +9.7% better. The planted interaction
   structure the graph is built to exploit is not present (or not recoverable) in
   coarse metadata features, so its extra edge parameters only add variance.
2. **All attribute models are weak in absolute terms** (~0.08–0.10 NDCG@10). The
   bottleneck is **attribute-extraction quality**, not the ranking model: a vector
   derived from price percentile, average rating, and a handful of title keywords
   carries little of a user's actual preference. The popularity baseline is correctly
   neutralized by the hard negatives (0.011), confirming the comparison is fair.

This is exactly the caveat the loader documents: *production-grade attribute extraction
is Phase 1 work (the QIL NLP pipeline)*. The real-data check makes that concrete — the
synthetic Claim 1 result is a statement about the *model given good features*, and real
data shows the features are the gating factor.

### Larger-category check (Cell_Phones_and_Accessories)

To rule out that the negative is a small-category artifact, the same within-category task
was rerun on a category ~29× larger by user count: `Cell_Phones_and_Accessories`
(184,070 items, **19,498** users with ≥5 reviews; interactions parsed up to the
`--max-interactions 3000000` cap).

> Note: `--max-items` is checked *per metadata shard*, not per row, so `--max-items 6000`
> here loads the whole first shard (184,070 items) rather than stopping at 6,000 — which
> is what the JSON config (`max_items: 6000`, `n_items: 184070`) honestly records. The
> larger load is harmless (more candidates), but the flag is coarser than it looks; making
> the cap row-precise is a tracked follow-up.

| model | NDCG@10 |
|-------|--------:|
| flat_item_embedding | **0.0961** |
| flat_attribute | 0.0870 |
| preference_graph | 0.0433 |
| popularity | 0.0189 |

**preference_graph vs flat_attribute: −50.2% (p = 0.0002).**
*(Reproduce: `python experiments/run_amazon_realdata.py --category Cell_Phones_and_Accessories --label cell_phones --max-items 6000 --max-interactions 3000000`.)*

The conclusion **holds and sharpens at scale**: with ~29× more users (so very tight
statistics) the graph is roughly *half* as good as the flat baseline, not better. Two
further observations: `flat_item_embedding` now edges out `flat_attribute` — at scale the
purchase-history embedding carries more signal than the coarse keyword-derived attribute
vector — and popularity stays correctly neutralized by hard negatives (0.019). Both
reinforce the same takeaway: the bottleneck is attribute/feature quality, and the graph's
extra edge parameters only add variance when fed weak features.

---

## Scope and honesty notes

- **This is a reality check, not a refutation of Claim 1.** The synthetic result was
  always scoped as "controlled, planted signal." It says: *if* attributes are extracted
  well, the graph's interaction modeling helps. Real data says: *with coarse metadata
  features, attributes barely carry preference*, so neither attribute model is strong
  and the graph's advantage vanishes. Both can be true; together they locate the work.
- **Within-category, not cross-category transfer.** The synthetic headline is transfer;
  here we report within-category, which is the stronger (easier) setting — and the
  attribute models already fail to beat each other meaningfully there, so transfer
  would not rescue them. Cross-category transfer *is* feasible on this corpus
  (e.g. Electronics ∩ Cell_Phones_and_Accessories ≈ 961k shared users), but it is a
  heavier offline run and would not change the featurization-bottleneck conclusion.
- **Coarse featurization is the point of the caveat, deliberately not improved here.**
  Improving it (the QIL NLP attribute pipeline) is the actual Phase 1 lever; this study
  exists to *measure the gap*, honestly.
- **Hard negatives matter.** Without them a popularity baseline scores ~0.86 (a user's
  favorites are also broadly popular); the candidate sets therefore mix relevant items
  with popular-but-not-for-this-user hard negatives, mirroring the synthetic benchmark,
  so the model comparison is meaningful.

---

## Method

- **Loader (`data/amazon.py`), rebuilt.** The original loader used the dataset's legacy
  loading script via `trust_remote_code`, which modern `datasets` no longer supports —
  it could not run at all. It now reads item metadata from the `raw_meta_<category>`
  **Parquet** shards and interactions from the `0core/last_out` benchmark **CSV** via
  `huggingface_hub` + `pandas`/`pyarrow` (the lighter `[amazon]` extra). The assembly
  (`build_category_data`) is split from the network fetch and unit-tested offline.
- **Featurization** is unchanged: a coarse shared-attribute vector per item from price
  percentile, average rating, rating volume, and title/feature keyword cues.
- **Evaluation** reuses `ExperimentHarness.run_within` and the paired bootstrap, the
  same machinery as the synthetic Claim 1 experiment, so the models and metrics are
  identical — only the data differs.

---

## Reproducing

```bash
bash scripts/setup-amazon.sh                   # opt-in: installs the [amazon] extra
python experiments/run_amazon_realdata.py      # loads All_Beauty, runs the comparison
# larger category (cap interaction parsing with --max-interactions for tractability):
python experiments/run_amazon_realdata.py \
  --category Cell_Phones_and_Accessories --label cell_phones \
  --max-items 6000 --max-interactions 3000000
python -m pytest tests/test_amazon_loader.py   # offline assembly tests (no network)
```

Raw metrics: [`experiments/amazon_realdata_results.json`](../experiments/amazon_realdata_results.json)
(All_Beauty), [`experiments/amazon_realdata_cell_phones_results.json`](../experiments/amazon_realdata_cell_phones_results.json)
(Cell_Phones_and_Accessories).

---

## Status

| Stage | Result |
|-------|--------|
| Claim 1 — preference graph beats flat baseline (synthetic transfer) | **+9.7%** NDCG@10 ([report](phase0-results.md)) |
| Real-data check — same models on Amazon Reviews 2023 (coarse features) | **Advantage does not replicate** (graph −20.8% vs flat on All_Beauty; **−50.2%** on Cell_Phones at 19.5k users); attribute extraction is the bottleneck (this report) |

The takeaway sharpens the roadmap: the integration, blending, and protocol results
stand on the *modeling* side, but turning them into real-world performance depends on
the **QIL attribute/quality extraction pipeline** — the documented Phase 1 investment —
which this real-data check shows is the binding constraint.
