# Results Ledger

**Recorded:** 2026-06-20T07:48:54Z
**Repo state:** `HEAD = 6ed019bb` (2026-06-20T06:14:09Z), branch `claude/amazing-hamilton-5od112`

A provenance-anchored index of every experimental result to date. Each entry
records the headline number, the date/commit it landed, the reproduction command,
the raw-metrics file, and a one-line note on what it does and does **not** establish.
"Landed" timestamps are the git author date of the commit that introduced the
report doc (UTC unless noted).

> Convention for future updates: **append**, never rewrite. Add a new dated entry
> when a result lands or is superseded; note supersession explicitly.

---

## R1 — Claim 1: preference graph beats flat baseline (synthetic transfer)

- **Headline:** sparse-DAG preference graph **+9.7% NDCG@10** over the flat-attribute
  baseline on cross-category transfer (laptops → headphones); p = 0.0002; robust
  +9.5%–17.5% across 5 seeds.
- **Landed:** 2026-06-19 07:09 UTC · commit `f2bb965`
- **Report:** [`../phase0-results.md`](../phase0-results.md)
- **Reproduce:** `python experiments/run_phase0.py --users 500`
- **Raw metrics:** `experiments/phase0_results.json`
- **Establishes:** *given clean attribute vectors and a recoverable planted
  interaction*, edges (attribute interactions) add transferable signal a flat
  vector cannot represent. Lift requires hard negatives; grows with history length;
  ties flat within-category.
- **Does NOT establish:** any advantage on real data (see R3). The benchmark plants
  the interaction structure the model exploits.

## R2 — Claim 2: use-profile quality extractable from public text (synthetic corpus)

- **Headline:** use-profile classifier **88.3% macro precision** on held-out set
  (vs 24.2% most-frequent-class baseline); 88.3%–91.7% across 5 seeds.
- **Landed:** 2026-06-19 08:52 UTC · commit `0e367e4`
- **Report:** [`../phase0-qil-results.md`](../phase0-qil-results.md)
- **Reproduce:** `python experiments/run_phase0_qil.py`
- **Raw metrics:** `experiments/phase0_qil_results.json`
- **Establishes:** use-profile labels are learnable from surface text at
  feasibility scale, with deliberately injected ambiguity (overlapping cue
  lexicons, filler-dominated weak posts, distractor phrasing, class imbalance).
  Classical TF-IDF + softmax, not a transformer.
- **Does NOT establish:** production extraction quality on live, messy, scraped
  Reddit/iFixit/Notebookcheck text. Corpus is synthetic-but-realistic.

## R3 — Real-data reality check (Amazon Reviews 2023) — HONEST NEGATIVE

- **Headline:** on real data the graph's synthetic advantage **does not replicate** —
  it is **worse** than the flat baseline. All_Beauty (112,590 items, 681 users):
  graph **−20.8%** vs flat (p = 0.0004). Cell_Phones (184,070 items, 19,498 users):
  graph **−50.2%** vs flat (p = 0.0002). All attribute models weak in absolute
  terms (~0.08–0.10 NDCG@10).
- **Landed:** 2026-06-20 02:38 UTC · commit `4c3114e`
- **Report:** [`../phase1-amazon-realdata.md`](../phase1-amazon-realdata.md)
- **Reproduce:**
  - `python experiments/run_amazon_realdata.py --max-items 112590` (All_Beauty)
  - `python experiments/run_amazon_realdata.py --category Cell_Phones_and_Accessories --label cell_phones --max-items 184070 --max-interactions 3000000`
- **Raw metrics:** `experiments/amazon_realdata_results.json`,
  `experiments/amazon_realdata_cell_phones_results.json`
- **Establishes (the key finding):** the bottleneck is **attribute-extraction
  quality, not the ranking model**. Coarse metadata features (price percentile,
  avg rating, title keywords) carry little real preference; extra edge parameters
  only add variance. Conclusion holds and sharpens with ~29× more users. Popularity
  correctly neutralized by hard negatives — comparison is fair.
- **Note:** this is the most paper-relevant result. It converts R1 into a scoped
  claim ("model helps *given good features*") and names the gating problem.

## R4 — Integration: the preference + quality α-blend (synthetic)

- **Headline:** α-blend reaches **NDCG@10 = 0.614** vs 0.442 preference-only
  (+39%, p=0.0002) and 0.262 quality-only (+134%, p=0.0002). Combining the two
  layers is the decisive win.
- **Secondary (honest negative):** confidence-adaptive α does **not** beat a fixed
  balanced α — it slightly trails (−0.025, p=0.0002). Optimal α ≈ constant
  (0.50–0.55 band) under uniform quality evidence.
- **Landed:** 2026-06-19 22:17 UTC · commit `8a8288d`
- **Report:** [`../phase1-integration-results.md`](../phase1-integration-results.md)
- **Reproduce:** `python experiments/run_phase1_integration.py`
- **Raw metrics:** `experiments/phase1_integration_results.json`
- **Establishes:** the core product thesis (blend both signals) end-to-end on a
  benchmark where ranking requires both; z-scoring of streams before blending
  (a modeling decision the architecture doc leaves implicit).
- **Does NOT establish:** value of the *documented* adaptive-α calibration.

## R5 — Quality robustness: shrinkage vs. raw averaging (synthetic) — POSITIVE MECHANISM

- **Headline:** clean **bias–variance crossover**. Raw confidence-weighted means win
  on clean signals; **Bayesian shrinkage is noise-robust and wins once review
  signals are noisy** (+0.022 NDCG@10 at obs-noise 1.0, p=0.0002; crossover ≈ 0.7).
- **Secondary (honest negative):** evidence-aware α is **worse** than fixed α on
  either estimator — the aggregator's shrinkage + the blend's z-scoring already
  absorb unreliable evidence, so α-level adaptation is redundant.
- **Landed:** 2026-06-19 17:23 UTC · commit `3a9f7cb`
- **Report:** [`../phase1-quality-robustness-results.md`](../phase1-quality-robustness-results.md)
- **Reproduce:** `python experiments/run_phase1_quality_robustness.py`
- **Raw metrics:** `experiments/phase1_quality_robustness_results.json`
- **Design takeaway:** keep the blend weight fixed/simple; put evidence-awareness in
  the aggregation layer (Bayesian shrinkage). Good supporting material for a paper.

## R6 — Cold-start: adaptive α in its best-case (zero-history) regime (synthetic)

- **Headline:** **premise confirmed** — for zero-history users quality-only beats
  preference-only, and optimal α genuinely varies across cohorts (**0.10 for new →
  0.60 for rich**), the predicted cold→quality / rich→preference crossover. But the
  **documented sigmoid formula still only ties** a fixed α=0.5 even here
  (+0.014 NDCG@10, p=0.31; sign flips with seed).
- **Landed:** 2026-06-19 22:17 UTC · commit `8a8288d`
- **Report:** [`../phase1-cold-start-results.md`](../phase1-cold-start-results.md)
- **Reproduce:** `python experiments/run_phase1_cold_start.py`
- **Raw metrics:** `experiments/phase1_cold_start_results.json`
- **Establishes:** the architectural intuition for adaptive α is directionally
  correct in its ideal regime; the specific calibration (floors at 0.18 at zero
  confidence vs measured optimum ~0.10) underperforms. Practical payoff marginal.

## R7 — Protocol-level integration (PTP + QIL over MCP) (synthetic)

- **Headline:** an agent ranking **purely over the two MCP tools** (PTP
  `get_preference` + QIL `get_quality`) **works end-to-end**: protocol blend
  NDCG@10 = 0.81 vs 0.63 preference-only (quality adds +0.19). Auth, selective
  disclosure, Ed25519 re-signing, and **revocation (403 enforced)** all exercised.
- **Landed:** 2026-06-19 18:58 UTC · commit `1249e7b`
- **Report:** [`../phase1-protocol-integration.md`](../phase1-protocol-integration.md)
- **Reproduce:** `preflayer protocol-demo` ; `python -m pytest tests/test_agent_protocol.py`
- **Establishes:** the PTP credential is a sufficient portable carrier of
  preference (no model object crosses the boundary) and the two tools compose into
  a correct ranking over real protocol surfaces.
- **Note:** credential here carries *true* weights (perfectly-learned), so the 0.81
  is NOT comparable to R4's fit-from-sparse-history numbers. Validates transport +
  composition, not learning.

---

## Cross-cutting caveats (apply to all synthetic results R1, R2, R4–R7)

- All use synthetic benchmarks with planted ground truth and paired-bootstrap
  significance. They validate *mechanisms and design principles*, not live-data
  performance. None meets the architecture's stated Phase 1 go/no-go gate
  (design-partner validation on real agents).
- The only real-data measurement is R3, and it is a negative for the win-claim /
  positive for the diagnostic.
- `.gitignore` forbids committing raw datasets/weights, so the Amazon data is
  fetched on demand (`scripts/setup-amazon.sh`, the `[amazon]` extra); only the
  derived JSON metrics are in-repo.

## Additional experiment scripts present (no dedicated report doc as of this date)

- `experiments/run_qil_api.py`, `experiments/run_qil_ingest.py`,
  `experiments/run_qil_realtext_harness.py` — QIL serving/ingestion/real-text
  harness. If these produce results worth citing, add ledger entries when they land.
</content>
