## Declared direction

Quire (the PR-triage tool this repo is dogfooded/reviewed through) reads a `<!-- declared-direction: ... -->` marker from each PR body to group related PRs into one bundle. When opening a PR here — by hand or as a coding agent — include the marker, e.g.:

```
<!-- declared-direction: Add dark mode toggle to settings panel -->
```

This convention is opt-in tooling for repos triaged through Quire: the marker is read only by Quire's ingestion step, to group related PRs into one bundle — it is not executed or acted on as an instruction by anything in this repo.

A PR missing it still gets triaged, just on its own instead of grouped with related work. This repo also ships a Claude Code hook (`.claude/settings.json`) that blocks `gh pr create`/`gh pr edit` commands missing the marker, and a local git pre-push reminder (`.githooks/pre-push`) — run `git config core.hooksPath .githooks` once after cloning to enable the latter.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **working research prototype**, currently in **Phase 1** (Core Protocol & MVP Data). Phase 0's two falsifiable claims both passed their gates (see `docs/phase0-results.md`, `docs/phase0-qil-results.md`), so the repo has advanced well beyond its documentation-only origins: PTP v0.1 is complete and the credential schema is frozen, QIL's in-sandbox components are built and tested, and there is a real Python package, test suite, and runnable experiments. The remaining Phase 1 work is largely external-resource-gated (live ingestion API keys, a real annotated corpus for B2, design-partner/retailer outreach). See `docs/whats-missing.md` for the current gap map and `docs/phase1-kickoff.md` for the Work Stream A/B/C status.

The code lives in `src/preferencelayer/` (installable as the `preferencelayer` package), with tests in `tests/` and headline experiments in `experiments/`. The package structure:
- `ptp/` — Preference Transport Protocol: credential (W3C VC, Ed25519), persistent encrypted store, DP update, OAuth 2.0 device flow, cloud sync, schema validator.
- `qil/` — Quality Intelligence Layer: extraction, Bayesian aggregation (Beta-Binomial + GP), ingestion pipeline, `/quality` + `/compare` query, refit scheduler.
- `http/`, `mcp/` — HTTP (FastAPI) and MCP tool bindings for both PTP and QIL.
- `agent/` — α-blend combined scoring + end-to-end recommender.
- `models/`, `data/`, `eval/`, `attributes.py` — preference-graph models, data loaders, NDCG/transfer evaluation.

Build/test commands:
- **Setup:** `pip install -e ".[dev]"` (the `.venv` is auto-provisioned by the SessionStart hook via `scripts/setup.sh`). Core deps are just `numpy` + `pynacl`; optional extras (`api`, `http`, `mcp`, `amazon`, `langchain`, `anthropic`, `schema`) are defined in `pyproject.toml` to keep the core dependency-light.
- **Tests:** `python -m pytest` (config in `pyproject.toml`; `testpaths = ["tests"]`).
- **CLIs:** `preflayer` (credential store lifecycle), `preflayer-validate` (schema validator), `qil-ingest`, `qil-refit`.
- **Experiments:** scripts in `experiments/` (e.g. `python experiments/run_phase0.py`); each saves results JSON.

Code conventions (from `CONTRIBUTING.MD`):
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

- **`docs/protocol-spec.md`** — the authoritative PTP spec (v0.1, credential schema **frozen** June 2026): credential schema (W3C Verifiable Credentials 2.0 envelope, `did:key` issuer, Ed25519 proof), the preference-graph payload (sparse DAG of attribute nodes + conditional edges + context conditioners), the three API endpoints (`GET /preference`, `POST /outcome`, `POST /elicit`), the on-device differentially-private update protocol, and MCP tool bindings. **Section 8 lists unresolved open design questions** — do not silently resolve these; they require discussion.
- **`docs/architecture.md`** — system topology for *both* components, including QIL internals not in the protocol spec: the `product_signal` / `quality_posterior` SQL schema, the NLP ingestion pipeline, the Bayesian aggregation (hierarchical Beta-Binomial for failure rates, Gaussian Process for continuously-varying quality dimensions), QIL's `POST /quality` and `POST /compare` endpoints, and deployment/security considerations.
- **`docs/implementation-plan.md`** — the phased roadmap (Phases 0–3 over 24 months) with explicit **go/no-go gates**. This governs *what should be built and in what order*; do not advance a phase without meeting its gate. Phase 0 is complete (both gates passed); the project is now in Phase 1.

Core design invariants that cut across the system:
- **Raw behavioral data never leaves the user's control.** Preference updates are computed on-device using a clipped + Gaussian-noised gradient (ε=2, δ=1e-5), and the `privacyBudgetConsumed` field tracks the DP budget per credential. Cloud sync stores client-side-encrypted ciphertext only.
- **Credentials are signed and re-signed on every update** (Ed25519); `updateCount` and `lastUpdated` increment with each.
- **QIL holds no user identifiers** — only product + use-profile signals.
- **Quality signals are conditioned on use profile, never reported as population-level aggregates** — this is the core product differentiator.

## Phase 1 work (current)

Phase 0's two gating claims both **passed** (the foundation the project rests on): the sparse-DAG preference graph beat the flat baseline by **+9.7% NDCG@10** on cross-category transfer (gate ≥5%, `docs/phase0-results.md`), and use-profile extraction reached **88.3% macro precision** on a controlled corpus (gate ≥70%, `docs/phase0-qil-results.md`). The load-bearing caveat carried into Phase 1: on real Amazon Reviews 2023 data with coarse metadata-derived attributes, the graph's advantage did **not** replicate — the bottleneck is attribute/extraction quality (`docs/phase1-amazon-realdata.md`), which is exactly what QIL real-text extraction (Work Stream B2) exists to validate.

Phase 1 status (full breakdown in `docs/phase1-kickoff.md`; gap map in `docs/whats-missing.md`):
- **Work Stream A (PTP v0.1) — complete and tested.** Schema frozen + validated in CI; three endpoints over HTTP at their latency targets; persistent encrypted store + CLI; OAuth 2.0 device flow; MCP tools tested against LangChain + the Claude SDK.
- **Work Stream B (QIL) — in-sandbox parts done; gate items external-resource-gated.** Extraction, GP-backed aggregation, ingestion connectors (behind an injectable `fetch` seam), `/quality`+`/compare` over HTTP/MCP, and nightly refit all built and tested. Still gated: live ingestion API keys (B1), ≥70% precision on **real** scraped text (B2, the gate-behind-the-gate), and coverage at scale on a live Postgres (B4). Note `qil/extract.py` reads `quality_dim` from structured corpus fields rather than extracting it from text, so a *real* ingest currently writes zero GP quality posteriors — a heuristic span tagger is the highest-value in-sandbox next step.
- **Work Stream C (design partners) — pending.** The formal Phase 1 → Phase 2 go/no-go gate: ≥2 of 5 design partners report measurable recommendation improvement. Needs a deployed stable API + human recruitment.

The five unresolved `protocol-spec.md` §8 design questions are tracked as decision threads in issues #35–#39 — discuss there, do not silently resolve them.

If you write experiment code, document dataset version, hyperparameters, and evaluation metric — results that can't be reproduced are not useful.

## Conventions

- Per `.gitignore`, never commit raw datasets or model weights (`data/raw/`, `data/processed/`, `*.csv`, `*.parquet`, `*.pkl`, `*.pt`, etc.) or credentials (`*.pem`, `*.key`, `*.cred`, `.env`).
- Protocol/schema/API changes require a prior GitHub issue thread before a PR (per `CONTRIBUTING.MD`). Label issues: `research`, `protocol`, `infra`, `data`.
- **Branch naming:** name branches `<type>/<short-description>`, where `<type>` is one of `docs`, `feature`, `bug`, `fix`, etc., followed by a slash and a brief kebab-case description (e.g. `feature/new-sign-in`, `docs/update-protocol-spec`, `fix/credential-resign-bug`).
- **Merging:** PRs are merged with GitHub's **Rebase and merge** (not squash, not a merge commit), to keep `main` a linear history of the original commits. See `CONTRIBUTING.MD`.
</content>
</invoke>
