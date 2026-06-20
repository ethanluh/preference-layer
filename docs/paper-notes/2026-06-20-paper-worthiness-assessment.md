# Paper-Worthiness Assessment

**Recorded:** 2026-06-20T07:48:54Z
**Repo state:** `HEAD = 6ed019bb` (2026-06-20T06:14:09Z), branch `claude/amazing-hamilton-5od112`
**Author of assessment:** Claude Code (research review at user request)
**Status:** point-in-time judgment — revisit when the real-data featurization gap is addressed

---

## Verdict (one paragraph)

The work is **not yet** a defensible empirical "our method beats baselines" paper,
because every positive headline is measured on synthetic benchmarks that plant the
exact structure the model is built to exploit, and the single real-data check
inverts the result. It **is**, however, the basis for two legitimate publications:
(1) an honest negative/diagnostic result — *attribute-extraction quality, not
ranking-model expressiveness, is the binding constraint for cross-category
cold-start preference transfer* — publishable as a workshop paper today; and (2) a
systems/position paper on the PTP+QIL architecture (portable, signed, DP-updated
preference credentials for the agentic/MCP web), which is a design contribution
that does not yet have deployment evidence. The research is unusually honest about
its own limits; it is simply earlier-stage than "write it up" implies.

## Why the obvious paper does not hold (yet)

1. **The positive results are on planted-signal synthetic data.**
   - Claim 1 (+9.7% NDCG@10, cross-category transfer): the benchmark *plants* an
     interaction structure (`phi_u` over attribute pairs); the model with edges
     then recovers it. The lift is close to circular — "a model with interaction
     terms wins on data generated with interaction terms." (`phase0-results.md` is
     candid about this.)
   - Claim 2 (88.3% macro precision): on a synthetic-but-realistic corpus
     generated to be learnable from surface text. Validates *feasibility*, not
     production extraction.
   - Integration (+39% / +134%), cold-start, quality-robustness, and protocol
     composition results: all on synthetic benchmarks with planted ground truth.

2. **The one real-data check inverts the headline.** On Amazon Reviews 2023 the
   preference graph is **−20.8%** vs. the flat baseline (All_Beauty) and **−50.2%**
   at scale (Cell_Phones, 19.5k users, p=0.0002). With coarse metadata features the
   graph's extra edge parameters only add variance. A reviewer will read
   "synthetic win + real-data loss" as a fatal combination *for the win-claim*.

3. **The protocol is undeployed.** No latency numbers, no design-partner data
   (that is the architecture's Phase 1 go/no-go gate, not yet run). The protocol
   can only support a *design* paper today, and `protocol-spec.md` §8 still lists
   open design questions (DID method, cross-category merge, household credentials,
   export format, trust tiers).

## What is genuinely publishable

- **The negative/diagnostic result (strongest near-term option).** There is a
  clean, useful, slightly counterintuitive finding: a controlled benchmark where
  the graph helps by +9.7%, *and* real data where the same model loses by 20–50%
  because features are coarse — with the cause isolated (edge params add variance
  under weak features). The field under-reports this. Workshop-ready as-is; a full
  paper needs the featurization lever actually pulled (extract attributes from
  review *text*) to show whether the advantage returns.

- **A systems/position paper on the architecture.** PTP+QIL decoupling, user-owned
  signed VC preference credentials (Ed25519), on-device DP updates (ε=2, δ=1e-5),
  MCP-native transport, confidence/evidence-aware blending. The protocol-level
  integration result (rank purely over the two MCP tools, with auth + selective
  disclosure + revocation enforced) is real supporting evidence that the design
  composes. Novelty bar is "well-reasoned, useful design," which it clears.

- **A secondary methodological nugget:** the shrinkage-vs-raw bias–variance
  crossover (Bayesian shrinkage is the noise-robust quality estimator once review
  signals are noisy) plus the repeated, robust finding that *adaptive α buys almost
  nothing over a fixed balanced blend on z-scored streams*. These are honest,
  reproducible mechanism results — supporting material, not a standalone paper.

## Recommendation

The user is **not** publishing now; they are recording for a future write-up.
Given that, the priority is preservation of provenance and a clear path forward,
captured in the companion files:
- [`2026-06-20-results-ledger.md`](2026-06-20-results-ledger.md) — what we have, with timestamps and repro.
- [`2026-06-20-publication-roadmap.md`](2026-06-20-publication-roadmap.md) — the two framings and what each still needs.

The single most valuable future step for paper #1 is the **QIL text-extraction
pipeline on real Amazon/Reddit data**: if good features resurrect the graph
advantage on real data, the narrative arc becomes complete (synthetic win →
real-data failure → fix the features → real-data win), which is a real empirical
contribution rather than a diagnostic.

## Caveats on this assessment

- It is a point-in-time judgment at `HEAD = 6ed019bb`. New results may change it.
- "Publishable" is judged against ML/RecSys norms (positive results on real data,
  fair baselines, ablations). A different venue (HCI, security/privacy, systems)
  weights the protocol contribution more heavily and the empirical bar less.
- The honesty of the existing reports is a genuine asset for any of these papers —
  the negative results are documented, not buried.
</content>
