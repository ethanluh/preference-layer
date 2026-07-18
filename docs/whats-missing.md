# What's missing — Phase 1 → Phase 2 status & gap map

**As of 2026-06-20.** This is a status snapshot: what is done, what remains to
clear the next gate, and — for each remaining item — whether it can be done in a
code sandbox or needs an external resource / a design decision. It is descriptive,
not a spec change. Read it alongside `implementation-plan.md` (the authoritative
roadmap and go/no-go gates) and `phase1-kickoff.md` (Work Stream A/B/C breakdown).

## Where we are

Phase 0 → Phase 1 gate **passed** (both falsifiable claims validated), and most
code that can be written without external resources is written and tested.

- **Phase 0 — both gates passed.**
  - Claim 1: the sparse-DAG preference graph beats the flat baseline by
    **+9.7% NDCG@10** on cross-category transfer (p=0.0002) — `phase0-results.md`.
  - Claim 2: use-profile extraction reaches **88.3% macro precision** on the
    controlled corpus (gate ≥70%) — `phase0-qil-results.md`.
  - Honest real-data check (`phase1-amazon-realdata.md`): the graph's advantage
    does **not** replicate on coarse Amazon metadata-derived attributes — the
    bottleneck is **attribute/extraction quality**. That is exactly what Phase 1
    Work Stream B2 exists to validate on real text.
- **PTP v0.1 (Work Stream A) — complete and tested.** Credential schema, `did:key`
  issuer, W3C VC 2.0 envelope, Ed25519 sign/re-sign, selective disclosure,
  on-device differentially-private update (ε=2, δ=1e-5, with `privacyBudgetConsumed`
  enforcement), client-side-encrypted cloud sync, the three HTTP endpoints, the
  OAuth 2.0 device flow (RFC 8628), and MCP tools tested against both LangChain and
  the Claude SDK. (`src/preferencelayer/ptp/*`, `http/app.py`, `mcp/*`.)
- **QIL — all in-sandbox parts hardened.** Use-profile + signal-type extraction;
  the controlled corpus; hierarchical Beta-Binomial failure rates and a Gaussian
  Process over release time for quality dimensions (`qil/aggregate.py`, `qil/gp.py`);
  the nightly refit with a `qil-refit` scheduler entry point; Postgres sinks
  covered by a fake-DB-API test; connectors with real `_parse` for all three
  sources (Reddit/iFixit/Notebookcheck) behind an injectable `fetch` seam;
  `run_daily`, the `qil-ingest` CLI, and the end-to-end ingest→refit; the
  `/quality` + `/compare` HTTP and MCP surface; and the combined α-blend scoring
  (`agent/combine.py`).

## The gate that governs everything

**Phase 1 go/no-go (end of Month 9):** at least **2 of 5 design partners report a
measurable improvement in recommendation relevance** with a PreferenceLayer
credential attached, versus running without it. If not met, Phase 2 pivots to
elicitation-first and protocol scaling is deferred. Everything below is on the
path to being able to run that test honestly.

## What's missing — by blocker type

### A. External-resource-gated (cannot be done in a code sandbox)

The code seams exist; these need a resource only the operator can supply.

1. **B1 — live ingestion fetch.** `_LiveConnector._fetch_pages` raises
   `NotImplementedError` ("PLUG API KEYS HERE", `qil/ingest/connectors.py`). The
   parsers and politeness layer are implemented and tested against injected fake
   fetchers. Real `fetch` callables now exist (`qil/ingest/live_fetch.py`); per the
   [data-source strategy](data-source-strategy.md) the default `--live` wiring is
   **Reddit-only (research/free tier)** and **iFixit/Notebookcheck are parked**
   (retained + tested, not crawled by default). *Needs:* Reddit API credentials
   (`REDDIT_CLIENT_ID/SECRET/USER_AGENT`) + a host to run the daily job. Reddit
   commercial licensing must be priced before any sale (#47).
   **Update (2026-07-17):** the official Reddit OAuth application was
   **rejected** (Reddit closed self-service registration in 2026). A fallback,
   `make_arctic_shift_fetch` (unauthenticated, via the Arctic Shift community
   mirror), is now wired and verified against live traffic —
   `qil-ingest --live --source reddit-arctic-shift` needs only
   `REDDIT_USER_AGENT`, no client_id/secret. This unblocks running live ingestion
   now; it does not resolve Reddit's commercial-licensing question (see
   [data-source-strategy.md](data-source-strategy.md)), and a host to run the
   daily job is still needed either way.
2. **B2 — precision on real text (the gate-behind-the-gate).** Must reach **≥70%
   use-profile precision on real scraped text** (the 88.3% was a synthetic corpus).
   The transformer fine-tune and the real-annotated-corpus loader are scaffolded in
   `qil/harness.py` ("PLUG FINE-TUNE HERE") but unrun. *Needs:* ~300 real annotated
   samples (two annotators + adjudication; ~$600 per the plan), a real scraped
   corpus, and fine-tuning compute. **Related gap:** `quality_dim`, `failure_mode`,
   and `signal_value` are currently read from structured corpus fields, **not
   extracted** from text (`qil/extract.py`). Until a span model populates
   `quality_dim`, a *real* ingest produces no GP quality posteriors (the
   `qil-ingest --refit` path writes 0).
3. **B4 — coverage at scale.** 500 laptop + 300 keyboard models with non-trivial
   evidence counts. The `/quality` + `/compare` endpoints and latency targets are
   done; coverage *needs* a live Postgres, sustained ingestion, and B2 passing.
4. **Work Stream C — design partners (the formal Phase 1 → Phase 2 gate).** Recruit
   3–5 developers building MCP-native agents, support integration, and measure
   before/after relevance. The onboarding doc and the `eval/partner.py`
   measurement harness are ready. *Needs:* human recruitment/outreach and a
   deployed, stable v0.1 API.
5. **Retailer outreach — STARTED now (Month 6); Phase 2 critical path.** The long
   sales cycle gates the Phase 2 milestone, so outreach is **kicked off now** per
   the [data-source strategy](data-source-strategy.md) (return data is the data
   foundation a commercial product rests on). Templates are ready
   (`retailer-data-sharing-agreement.md`, `retailer-return-signal-schema.md`);
   schema ratification is in #14 (recommendations posted), outreach tracked in #48.
   *Needs:* business development + legal, not code.
6. **A1 ops — public `@context` hosting.** The JSON-LD context exists in-repo
   (`contexts/`); it still needs to resolve at `https://preferencelayer.io/context/v1`.
   *Needs:* DNS/hosting. Does not block the Phase 1 definition of done.

### B. In-sandbox work that can still be done now (no external dependency)

1. **`quality_dim` heuristic span tagger (stand-in).** A lexicon/rule extractor
   that populates `quality_dim` from post text, so `qil-ingest --refit` produces
   real GP quality posteriors end-to-end instead of zero — and a clean seam for the
   eventual B2 span model. Fully testable here. **Highest in-sandbox value.**
2. **B2 enablement scaffolding.** Flesh out `qil/harness.py`: an annotation
   export/import tool (JSONL schema + adjudication), a real-text CLI, and the
   `TransformerClassifier.fit` wiring, so B2 runs the moment real data lands.
3. **Smaller invariant/UX gaps.** Reject an empty `use_profile` at the `/quality`
   API (semantic non-emptiness of the use-profile-conditioning invariant), and a
   budget-reset consent stub for DP-budget exhaustion.

### C. Open design decisions — must be decided, not silently resolved

From `protocol-spec.md` §8, all still open: (1) **DID method** (`did:key` vs
`did:web`/`did:ion` for key rotation); (2) **cross-category credential merge**
mapping; (3) **multi-user household** credentials; (4) **export/portability
format** (Q4 left unfrozen for v0.1); (5) **agent trust tiers**. Each needs an
issue thread and an explicit decision, not a code default.

### D. Described-but-not-built (Phase 2+, by design)

Context-conditioner activation driven by `query_context`; a **learned** per-user
blend weight α (today a fixed `sigmoid(3·(c−0.5))`); cross-agent credential
merge with conflict resolution; automated freshness scoring (currently partial);
an editorial flag for high-stakes categories; per-agent-token rate limiting /
DDoS protection; and credential/key rotation & refresh.

## Recommended sequence

1. **Now (sandbox):** build the `quality_dim` span-tagger stand-in (B-1 above) so
   the QIL ingest→refit story is genuinely end-to-end.
2. **Now (operator, in parallel, non-code):** retailer outreach is **underway**
   (#48; the data foundation, per [data-source strategy](data-source-strategy.md));
   continue design-partner outreach (long lead times) and open issue threads for the
   five §8 decisions. iFixit/Notebookcheck outreach is **parked** (#47 context).
3. **When resources land:** inject the B1 `fetch` → run B2 annotation + fine-tune →
   if ≥70% on real text, scale B4 coverage → recruit and measure Work Stream C →
   Phase 1 go/no-go.

## Sources

`implementation-plan.md` (phases & gates), `phase1-kickoff.md` (WS-A/B/C status),
`phase0-results.md` / `phase0-qil-results.md` / `phase1-amazon-realdata.md`
(Phase 0 + real-data check), `protocol-spec.md` §8 (open questions),
`architecture.md` (described-but-not-built), and the scaffold markers in
`qil/ingest/connectors.py`, `qil/extract.py`, and `qil/harness.py`.
