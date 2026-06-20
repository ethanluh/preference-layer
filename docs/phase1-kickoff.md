# Phase 1 Kickoff Plan

**Phase 1 — Core Protocol & Minimal Viable Data Product**
**Months 4–9 · 2–3 people · $50–100k**
**Status:** in progress (entered after the Phase 0 gate decision — see
[`implementation-plan.md`](implementation-plan.md))

This document operationalizes the Phase 1 section of
[`implementation-plan.md`](implementation-plan.md) into concrete milestones,
sequencing, and definitions-of-done. The plan itself is the source of truth for
*what* Phase 1 is; this is the *how and in what order*, plus an honest accounting
of what already exists from the Phase 0 prototype.

---

## Phase 1 goal (restated)

Ship a working PTP implementation and a QIL covering two product categories,
onboard 3–5 agent developers as design partners, and validate that the credential
provides **measurable** value in a real agent workflow.

**Go/no-go gate (Work Stream C):** at least 2 of 5 design partners report a
measurable improvement in recommendation relevance. If not, the cold-start problem
needs more investment before scaling — pivot Phase 2 to elicitation-first.

---

## What Phase 0 already gives us

Phase 1 does **not** start from zero. The Phase 0 prototype already implements
reference versions of much of Work Streams A and B; Phase 1 is about hardening,
persistence, real data, and partners. Honest inventory:

| Area | Exists (Phase 0 prototype) | Phase 1 gap |
|------|----------------------------|-------------|
| Credential schema | W3C-VC-shaped, Ed25519-signed, selective disclosure (`ptp/credential.py`) | JSON-LD `@context` published at a resolvable URI; schema validator; freeze v0.1 |
| Three endpoints | `get_preference` / `submit_outcome` / `elicit` in `ptp/store.py` | HTTP/MCP transport with latency targets; auth hardening |
| DP update | clipped + Gaussian-noised on-device update, ε=2 (`ptp/update.py`) | budget accounting reviewed; per-node sensitivity audit |
| Credential store | **persistent, SQLite-backed, encrypted at rest, CLI** (`ptp/persistence.py`, `cli.py`) ✅ **done** | OAuth 2.0 device flow; optional cloud-sync of ciphertext |
| MCP server | tool handlers + descriptions (`mcp/server.py`, `qil/mcp_server.py`) | tested against live Claude agent SDK + LangChain |
| QIL extraction | TF-IDF + softmax classifier, 88.3% on a controlled corpus (`qil/extract.py`) | real scraped corpus; fine-tuned transformer; precision on live text |
| QIL aggregation | Beta-Binomial failure rate + Normal-Normal quality dims (`qil/aggregate.py`) | Gaussian-process temporal kernel; nightly refit job |
| QIL API | `/quality` + `/compare` over posteriors (`qil/query.py`) | served behind HTTP with latency targets; coverage at scale |
| Integration | α-blend agent, fixed vs. adaptive α studied (`agent/`) | calibrate α on real design-partner data |

The **real-data reality check** ([`phase1-amazon-realdata.md`](phase1-amazon-realdata.md))
is the load-bearing caveat: on real Amazon data the graph advantage did not
replicate, and the bottleneck was attribute-extraction quality. **This makes QIL
extraction on real text (Work Stream B) the highest-risk item in Phase 1** and
drives the sequencing below.

---

## Sequencing

The three work streams are largely parallelizable (the protocol and the QIL
pipeline are independent until the Phase 2 scoring integration). The two
non-parallelizable dependencies, per the plan's critical path:

1. **Design-partner recruiting needs a stable v0.1 API.** Do not start partner
   outreach (WS-C) until the three PTP endpoints + QIL `/quality` are deployed and
   stable. Shipping a broken API to partners early is worse than shipping late.
2. **Retailer outreach has a long sales cycle.** It is a Phase 2 deliverable but
   must *begin* in Phase 1 (~Month 6) — see the plan's critical path. Tracked here
   as a Phase 1 kickoff item even though it lands later.

```
Month:        4         5         6         7         8         9
WS-A (PTP):   schema    store ✅  MCP+OAuth  ──────── harden/stabilize ────────
WS-B (QIL):   ingest    aggregate  API       ──────── coverage 500/300 ────────
WS-C (part.): ─────────────────────────────  recruit   integrate   measure→gate
(retailer outreach begins ~Month 6, completes in Phase 2)
```

---

## Work Stream A — PTP v0.1

### A1. Freeze the credential schema (Month 4)
- **Do:** publish the JSON-LD `@context` at a resolvable URI; pin required fields
  (`attributeNodes`, `edgeWeights`, `contextConditioners`, `updateMetadata`,
  `issuer`, `proof`); write a schema validator; tag `protocol-spec.md` as v0.1.
- **Done when:** an external party can validate a credential against a published
  context without reading our code; the validator is in CI.
- **Note:** §8 of `protocol-spec.md` lists unresolved open questions — resolve the
  ones that block schema freeze via a `protocol`-labeled issue thread first (per
  `CONTRIBUTING.MD`); do not silently decide them.

### A2. Three endpoints over real transport (Months 4–5)
- **Do:** expose `GET /preference`, `POST /outcome`, `POST /elicit` over HTTP and
  MCP; meet the latency targets (`/preference` < 100 ms p95).
- **Done when:** a load test shows the p95 targets met on representative
  credentials; auth (agent token / scope / revoke) enforced at the boundary.

### A3. Credential store (Months 5–6) — ✅ first deliverable landed
- **Done:** persistent, SQLite-backed store encrypted at rest, persistent Ed25519
  identity (optionally passphrase-locked via Argon2id), persistent agent tokens,
  and the `preflayer init / view / authorize / revoke / export / delete` CLI
  (`ptp/persistence.py`, `cli.py`; tests in `tests/test_persistence.py`,
  `tests/test_cli_store.py`).
  - *Encryption-at-rest threat model:* the SQLite DB holds **only** ciphertext plus
    opaque keyed-hash indexes (category / agent id are never stored in plaintext),
    so the DB is safe to cloud-sync; the identity key stays on device. This matches
    the architecture's "cloud sync stores client-side-encrypted ciphertext only"
    invariant.
- **Remaining for A3:** OAuth 2.0 device flow for agent authentication (current
  tokens are opaque bearer tokens, not a device-flow grant); optional cloud-sync of
  the encrypted DB.

### A4. MCP server wrapper (Month 6)
- **Do:** wrap all three endpoints as MCP tools with self-selection-optimized
  descriptions; test against the Claude agent SDK and LangChain.
- **Done when:** an unprompted agent in both frameworks invokes the right tool for
  rank / post-purchase / low-confidence situations in a scripted eval.

---

## Work Stream B — QIL v0.1 (two categories) — highest risk

### B1. Real ingestion pipeline (Months 4–5)
- **Do:** productionize ingestion from Reddit (official API, rate-limited), iFixit
  (polite crawl, respect robots.txt), Notebookcheck (structured scrape); run daily;
  land records in the `product_signal` PostgreSQL schema.
- **Done when:** the pipeline runs unattended for a week and lands deduplicated,
  normalized signals for laptops + keyboards.

### B2. Precision on **real** text (Months 4–5) — the gate-behind-the-gate
- **Do:** annotate ~300 real samples (2 annotators, adjudicate); fine-tune a small
  transformer; measure use-profile precision on a held-out set of *live* text.
- **Done when:** ≥ 70% precision **on real scraped text** (the Phase 0 88.3% was on
  a controlled corpus). If 60–70%, assess whether more annotation recovers it; below
  60%, the automation story doesn't hold — escalate before scaling coverage.
- **Why first:** the real-data reality check located the bottleneck here. Validate
  extraction precision on real text **before** investing in coverage (B4).

### B3. Bayesian aggregation, productionized (Months 5–6)
- **Do:** keep Beta-Binomial failure rates; upgrade quality dimensions from the
  Phase 0 Normal-Normal stand-in to the Gaussian process over release time
  specified in `architecture.md`; refit posteriors nightly; store parameters only.
- **Done when:** nightly refit job is scheduled; `/quality` returns GP-backed
  posteriors with the same contract (mean + 90% CI + failure rate + evidence count).

### B4. Coverage + API (Month 6)
- **Do:** serve `/quality` and `/compare` over HTTP (< 200 ms p95); reach the
  launch coverage target.
- **Done when:** **500 laptop models + 300 keyboard models** covered with non-trivial
  evidence counts, served behind the API at the latency target.

---

## Work Stream C — Design partner program (Months 7–9)

### C1. Recruit (Month 7)
- **Do:** recruit 3–5 developers building MCP-native agents (open-source projects,
  small startups, hobbyists — *not* platform incumbents); set up a private feedback
  channel; commit to prioritizing their bugs / API friction.
- **Done when:** ≥ 3 partners have working credentials against the live v0.1 API.

### C2. Integrate + measure (Months 8–9)
- **Do:** support integration; measure whether recommendation quality improves
  (NDCG or user-reported satisfaction), where cold-start dominates, and the
  integration friction per framework.
- **Done when:** each active partner has a before/after relevance measurement on
  their own task.

### C3. Phase 1 go/no-go gate
- **Gate:** ≥ 2 of 5 partners report a **measurable** improvement in recommendation
  relevance.
- **If not met:** pivot Phase 2 to elicitation-first; defer protocol scaling.

---

## Cross-cutting: process & guardrails

- **Issue-first for protocol/schema/API changes** (per `CONTRIBUTING.MD`): open a
  labeled issue (`research` / `protocol` / `infra` / `data`) before the PR.
- **Reproducibility:** every experiment documents dataset version, hyperparameters,
  and metric. Results that can't be reproduced don't count.
- **Invariants that must not regress:** raw behavioral data never leaves the device;
  credentials are re-signed on every update; QIL holds no user identifiers; quality
  signals are always use-profile-conditioned, never population-level aggregates.
- **`.gitignore`:** never commit raw datasets, model weights, credentials, or the
  local store (`identity.key`, `store.db`, `.preflayer/`).

---

## Definition of done for Phase 1

1. PTP v0.1: schema frozen + published, three endpoints stable behind HTTP/MCP at
   their latency targets, persistent encrypted credential store with CLI ✅, MCP
   wrapper tested against two agent frameworks.
2. QIL v0.1: real ingestion running daily; ≥ 70% extraction precision **on real
   text**; GP-backed nightly posteriors; 500 laptops + 300 keyboards served.
3. Design partners: ≥ 3 integrated, ≥ 2 reporting measurable improvement → **gate
   passed → proceed to Phase 2.**
