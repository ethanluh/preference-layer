# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **pre-prototype, documentation-only repository**. It contains the research foundation, protocol design, and phased implementation plan for PreferenceLayer — there is no application source code, build system, or test suite yet. The current active work is **Phase 0** (research validation).

Because there is no code, there are no build/lint/test commands. When the first code lands, the intended conventions (from `CONTRIBUTING.MD`) are:
- **Python:** PEP 8, type hints required on public functions, docstrings for non-obvious code. Tests required for all protocol logic.
- **TypeScript (when applicable):** strict mode, no `any`.
- **Hard constraint:** no dependencies on platform-specific APIs. The entire premise is platform-agnostic infrastructure.

## What PreferenceLayer is

Portable preference infrastructure for the open agent web. It exists to break the platform lock-in of preference data: today every AI shopping agent silos a user's preferences inside its own platform, so cross-platform agents start cold every time. PreferenceLayer is delivered as two **independently deployable** components that combine only at agent query time:

- **PTP (Preference Transport Protocol):** a user-owned, cryptographically signed preference credential plus the REST/MCP protocol agents use to read and update it. Privacy-preserving and platform-agnostic.
- **QIL (Quality Intelligence Layer):** a server-side database of *use-profile-conditioned* product quality signals (failure rates, longitudinal performance, counterfactual comparisons), exposed as an agent-queryable API with Bayesian posteriors and confidence intervals.

The key architectural insight: PTP and QIL are decoupled. PTP is stateless server-side with a **client-side, user-controlled credential store**; QIL is proprietary and server-side. An agent calls both as MCP tools and blends the results with a learned weight `α` (lean on quality when the credential is sparse/low-confidence, lean on preference when it is rich). This blending is defined in `docs/architecture.md` ("Combined Scoring").

## Architecture you must read across files to understand

The design is spread across three docs that should be read together before touching anything protocol- or data-related:

- **`docs/protocol-spec.md`** — the authoritative PTP draft spec (v0.1): credential schema (W3C Verifiable Credentials 2.0 envelope, `did:key` issuer, Ed25519 proof), the preference-graph payload (sparse DAG of attribute nodes + conditional edges + context conditioners), the three API endpoints (`GET /preference`, `POST /outcome`, `POST /elicit`), the on-device differentially-private update protocol, and MCP tool bindings. **Section 8 lists unresolved open design questions** — do not silently resolve these; they require discussion.
- **`docs/architecture.md`** — system topology for *both* components, including QIL internals not in the protocol spec: the `product_signal` / `quality_posterior` SQL schema, the NLP ingestion pipeline, the Bayesian aggregation (hierarchical Beta-Binomial for failure rates, Gaussian Process for continuously-varying quality dimensions), QIL's `POST /quality` and `POST /compare` endpoints, and deployment/security considerations.
- **`docs/implementation-plan.md`** — the phased roadmap (Phases 0–3 over 24 months) with explicit **go/no-go gates**. This governs *what should be built and in what order*. Phase 0 is solo-executable research; do not advance a phase without meeting its gate.

Core design invariants that cut across the system:
- **Raw behavioral data never leaves the user's control.** Preference updates are computed on-device using a clipped + Gaussian-noised gradient (ε=2, δ=1e-5), and the `privacyBudgetConsumed` field tracks the DP budget per credential. Cloud sync stores client-side-encrypted ciphertext only.
- **Credentials are signed and re-signed on every update** (Ed25519); `updateCount` and `lastUpdated` increment with each.
- **QIL holds no user identifiers** — only product + use-profile signals.
- **Quality signals are conditioned on use profile, never reported as population-level aggregates** — this is the core product differentiator.

## Phase 0 work (current)

Two falsifiable claims gate the whole project (see `implementation-plan.md` Phase 0 and `CONTRIBUTING.MD`):
1. A sparse-DAG preference graph beats cold-start baselines (flat vector, MemRerank, BM25) by ≥5% NDCG@10 on cross-category transfer, evaluated on the Amazon Reviews 2023 dataset.
2. Use-profile classification from public unstructured sources (Reddit, iFixit, Notebookcheck) reaches ≥70% precision on a held-out set.

If you write Phase 0 experiment code, document dataset version, hyperparameters, and evaluation metric — results that can't be reproduced are not useful.

## Conventions

- Per `.gitignore`, never commit raw datasets or model weights (`data/raw/`, `data/processed/`, `*.csv`, `*.parquet`, `*.pkl`, `*.pt`, etc.) or credentials (`*.pem`, `*.key`, `*.cred`, `.env`).
- Protocol/schema/API changes require a prior GitHub issue thread before a PR (per `CONTRIBUTING.MD`). Label issues: `research`, `protocol`, `infra`, `data`.
- **Branch naming:** name branches `<type>/<short-description>`, where `<type>` is one of `docs`, `feature`, `bug`, `fix`, etc., followed by a slash and a brief kebab-case description (e.g. `feature/new-sign-in`, `docs/update-protocol-spec`, `fix/credential-resign-bug`).
</content>
</invoke>
