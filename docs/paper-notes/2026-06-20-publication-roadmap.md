# Publication Roadmap

**Recorded:** 2026-06-20T07:48:54Z
**Repo state:** `HEAD = 6ed019bb` (2026-06-20T06:14:09Z), branch `claude/amazing-hamilton-5od112`
**Status:** forward-looking plan — not a commitment to a venue or timeline

Two viable paper framings, with the concrete work each needs before it is
submittable. See [`2026-06-20-paper-worthiness-assessment.md`](2026-06-20-paper-worthiness-assessment.md)
for the verdict behind these and [`2026-06-20-results-ledger.md`](2026-06-20-results-ledger.md)
for the underlying results (referenced below as R1–R7).

---

## Paper A — Empirical: "Featurization, not model expressiveness, gates cross-category preference transfer"

**Thesis:** on cross-category cold-start recommendation, the binding constraint is
attribute-extraction quality, not the ranking model's expressiveness. A model with
interaction structure helps substantially given good features (R1) but loses to a
simpler baseline on real data under coarse features (R3); closing the feature gap is
what unlocks the gain.

**Strongest evidence we already have:** R1 (synthetic +9.7%), R3 (real-data
−20.8% / −50.2% with cause isolated). This is the most defensible and least common
contribution — a documented, reproducible negative-with-diagnosis.

**Workshop-paper readiness:** ~now. The synthetic-win/real-loss/diagnosis arc is
self-contained, honest, and reproducible.

**To upgrade to a full paper (the high-value work):**
1. **Pull the featurization lever.** Build the QIL text-extraction pipeline on real
   review text (Amazon Reviews 2023 + Reddit) and re-derive attribute vectors from
   text, not metadata. Re-run R3's comparison.
2. **Show whether the graph advantage returns** with good features. Either outcome is
   publishable: it returns → "fix the features, recover the gain" (complete arc);
   it doesn't → stronger negative ("interaction modeling is not the lever").
3. **Add real cross-category transfer** on Amazon (e.g. Electronics ∩ Cell_Phones,
   ~961k shared users — noted feasible in R3 but not yet run).
4. **Baseline breadth:** include MemRerank (reproduce 2603.29247) and BM25-over-
   stated-preferences, which Phase 0's plan specified but the reports compare
   against flat/popularity only.

**Candidate venues:** RecSys / NeurIPS / KDD workshop now; RecSys or a recommender
track full paper after step 1–4.

**Risks:** real text extraction may stay weak (then the paper is "the wall is
real," still publishable but less satisfying); compute/data-handling effort is the
main cost.

---

## Paper B — Systems / position: "Portable, user-owned preference credentials for the agentic web"

**Thesis:** preference data should be a user-owned, portable, signed credential read
and updated by any authorized agent over MCP — decoupled from a server-side quality
layer and blended at query time — instead of siloed per platform.

**Contribution type:** architecture + protocol design (not an empirical win).
Components: PTP credential schema (W3C VC 2.0 + `did:key` + Ed25519), on-device DP
update (ε=2, δ=1e-5, budget tracking), QIL use-profile-conditioned Bayesian
posteriors, MCP tool bindings, confidence/evidence-aware blending.

**Strongest supporting evidence:** R7 (ranking purely over the two MCP tools, with
auth + selective disclosure + revocation enforced) shows the design composes; R4–R6
characterize the blending design space (and honestly report that adaptive α is
marginal and where evidence-awareness belongs).

**To make it submittable:**
1. **Resolve or scope the open design questions** in `protocol-spec.md` §8 (DID
   method/rotation, cross-category merge, household credentials, export format,
   trust tiers) — a position paper can state positions, but they must be argued.
2. **Add deployment/latency evidence.** Measure VC verification and credential
   round-trip latency (the spec targets <100ms p95; Phase 0 Work Stream C flagged
   the ">50ms p99 → use lighter signed JSON" decision — report the actual number).
3. **Threat model / privacy analysis.** Make the DP guarantee and the
   "raw behavioral data never leaves the device" claim precise and adversarial.
4. **At least one external integration** (a design partner agent, even minimal) to
   move from "designed" toward "validated" — this is the architecture's own Phase 1
   gate and would lift the paper from position to systems.

**Candidate venues:** an agents/LLM-tooling or decentralized-identity workshop, or a
systems/security venue if the privacy + deployment analysis is substantial. A pure
position paper is possible now; a systems paper needs steps 2–4.

**Risks:** without deployment data it reads as vision; reviewers at empirical venues
will discount it. Best paired with at least latency + one integration.

---

## Sequencing recommendation

1. **Preserve provenance** (this directory) — done.
2. If publishing opportunistically: **Paper A as a workshop paper** is the lowest-
   effort, most-honest output from what already exists.
3. **Invest in the QIL real-text extraction pipeline** — it is simultaneously the
   project's Phase 1 critical path *and* the unlock for Paper A (full) *and* a step
   toward Paper B's validation. Highest leverage single piece of work.
4. Decide A-vs-B (or both, A empirical + B systems) once real-text results are in.

## Material to reuse when drafting

- Method/honesty sections in the existing reports are written to near-paper quality;
  the "why the result is meaningful (not rigged)" sections map directly to a paper's
  validity discussion.
- `experiments/*.json` are the raw-metrics tables; `experiments/run_*.py` are the
  reproduction harness; paired-bootstrap significance is already implemented
  (`eval/harness._paired_bootstrap_p`).
- The synthetic benchmark generators (`data/synthetic.py`, `data/integrated.py`,
  `qil/corpus.py`) are documented with their planted ground truth — useful for a
  paper's "controlled benchmark" appendix.
</content>
