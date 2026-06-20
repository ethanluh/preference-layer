# PreferenceLayer

**Portable preference infrastructure for the open agent web.**

PreferenceLayer is a protocol and data product that gives AI shopping agents access to rich, user-controlled preference signals — regardless of which platform they run on.

It has two components:

- **Preference Transport Protocol (PTP):** A user-owned, cryptographically signed preference credential and the protocol agents use to read and update it. Platform-agnostic, MCP-native, privacy-preserving.
- **Quality Intelligence Layer (QIL):** A continuously maintained database of use-profile-conditioned product quality signals — failure rates, longitudinal performance, and counterfactual comparisons — exposed as an agent-queryable API.

---

## The Problem

Every AI shopping agent today locks your preference data inside its own platform. Perplexity knows what you like when you use Perplexity. That knowledge disappears the moment you use a different agent. Agents operating outside any single platform's ecosystem must infer your preferences from scratch at every interaction.

This is not an oversight. It is the intended design. Platform-locked preference data is a retention mechanism.

The structural consequence: any agent operating across platforms — the growth vector in the MCP/A2A ecosystem — starts cold every time.

PreferenceLayer is the layer that fixes this.

---

## Repository Structure

```
preferencelayer/
├── README.md                  # This file
├── DESCRIPTION.md             # Project summary (one page)
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE.md                 # License
├── pyproject.toml             # Installable package (preferencelayer)
│
├── src/preferencelayer/       # Phase 0 prototype (working code)
│   ├── attributes.py          # Shared cross-category attribute vocabulary
│   ├── data/                  # Synthetic benchmark + Amazon Reviews 2023 loader
│   ├── models/                # Baselines + the sparse DAG preference graph
│   ├── eval/                  # NDCG metrics + transfer evaluation harness
│   ├── ptp/                   # PTP: credential, store, DP update (reference impl)
│   ├── mcp/                   # PTP MCP server (agent tool bindings)
│   ├── qil/                   # QIL: extraction, Bayesian aggregation, query + MCP
│   ├── agent/                 # Integration: the preference+quality α-blend agent
│   └── cli.py                 # `preflayer` command-line interface
│
├── experiments/
│   ├── run_phase0.py          # Claim 1: cross-category transfer experiment
│   ├── run_phase0_qil.py      # Claim 2: QIL extraction feasibility study
│   ├── run_phase1_integration.py  # Integration: preference+quality α-blend benchmark
│   ├── run_phase1_quality_robustness.py  # Quality handling: shrinkage vs. raw averaging
│   └── *.json                 # Saved metrics for the headline runs
│
├── tests/                     # Test suite (incl. the Phase 0 go/no-go gate)
│
├── proposals/                 # Technical + investor proposals
└── docs/
    ├── implementation-plan.md # Full phased build plan
    ├── architecture.md        # System architecture reference
    ├── protocol-spec.md       # PTP protocol specification (draft)
    ├── phase0-results.md      # Phase 0 Claim 1 report (preference graph)
    ├── phase0-qil-results.md  # Phase 0 Claim 2 report (QIL extraction)
    ├── phase1-integration-results.md  # Integration report (α-blend)
    ├── phase1-quality-robustness-results.md  # Quality handling (shrinkage vs. raw)
    ├── phase1-protocol-integration.md  # End-to-end over the PTP + QIL MCP tools
    ├── phase1-cold-start-results.md  # Adaptive α in the zero-history regime
    └── phase1-amazon-realdata.md  # Real-data reality check (Amazon Reviews 2023)
```

---

## Status

**Phase 1 in progress.** Phase 0 is complete — both research gates passed (gate
decision recorded in [`docs/implementation-plan.md`](docs/implementation-plan.md))
— and the project has formally advanced to Phase 1 (Core Protocol & Minimal Viable
Data Product). The milestone breakdown, sequencing, and definitions-of-done are in
[`docs/phase1-kickoff.md`](docs/phase1-kickoff.md). The repository contains a
working, tested implementation of the research prototype alongside the design docs.

**First Phase 1 deliverable landed:** the user-controlled **persistent credential
store** (Work Stream A, Months 5–6) — SQLite-backed and encrypted at rest, with a
persistent Ed25519 identity (optionally passphrase-locked) and persistent agent
tokens, driven by the `preflayer init / view / authorize / revoke / export /
delete` CLI.

- **Claim 1 (preference graph):** the **sparse DAG preference graph beats the strong
  flat-vector baseline by +9.7% NDCG@10 on cross-category transfer** (laptops →
  headphones, p = 0.0002), robust across seeds — clearing the ≥ 5% gate. See
  [`docs/phase0-results.md`](docs/phase0-results.md).
- **Claim 2 (QIL extraction):** use-profile-conditioned quality signals are
  extracted from unstructured text at **88.3% macro precision** (vs. a 24.2%
  baseline), clearing the ≥ 70% gate. See
  [`docs/phase0-qil-results.md`](docs/phase0-qil-results.md).
- **Integration (the α-blend):** an agent that fuses portable preference with
  use-profile quality — the documented `α·pref + (1−α)·quality` scoring — **beats
  either layer alone by +39% / +134% NDCG@10** (p = 0.0002) on a benchmark where
  both signals are required. The *confidence-adaptive* α from the architecture is
  implemented and measured; honestly, it does **not** beat a fixed balanced blend
  in this uniform-evidence regime (the optimal α is ~constant). See
  [`docs/phase1-integration-results.md`](docs/phase1-integration-results.md).
- **Quality handling (shrinkage vs. raw):** following up on *why* adaptive α didn't
  help, an estimator×blend ablation finds a clean **bias–variance crossover** —
  raw averaging wins on clean review signals, but **Bayesian shrinkage is the
  noise-robust choice and wins as signals get noisy** (the QIL's real regime), while
  evidence-aware α stays redundant. Takeaway: combine both layers, use a fixed α,
  and let the QIL's Bayesian aggregation absorb noisy evidence. See
  [`docs/phase1-quality-robustness-results.md`](docs/phase1-quality-robustness-results.md).
- **Protocol-level end-to-end:** an agent that ranks products using **only the real
  PTP `get_preference` and QIL `get_quality` MCP tools** — reconstructing preference
  from the disclosed credential, blending with confidence-adaptive α. On the
  benchmark a credential round-trip ranks at **0.81 NDCG@10** (quality adds +0.19 over
  preference alone), and revoking the agent's token is honored at the boundary (403,
  no ranking). See
  [`docs/phase1-protocol-integration.md`](docs/phase1-protocol-integration.md).
- **Zero-history cold-start:** adding a brand-new-user cohort (no history, confidence
  0) finally makes the optimal α vary — **0.10 for new users up to 0.60 for
  rich-history users**, and quality alone beats preference alone *only* for new users,
  vindicating the architecture's "new user → lean on quality" premise. But the
  documented sigmoid α still only **ties** a fixed blend even here (+0.014, p=0.31) —
  z-scoring already lets the fixed blend lean on the informative signal. See
  [`docs/phase1-cold-start-results.md`](docs/phase1-cold-start-results.md).
- **Real-data reality check (Amazon Reviews 2023):** running the same models on real
  items/users with **coarse metadata-derived attributes**, the graph's synthetic
  advantage **does not replicate** (−20.8% vs flat) and all attribute models are weak —
  locating the bottleneck at **attribute-extraction quality** (the QIL NLP pipeline,
  Phase 1), not the ranking model. An honest negative. See
  [`docs/phase1-amazon-realdata.md`](docs/phase1-amazon-realdata.md).

Also implemented: the **PTP credential** (W3C-VC-shaped, Ed25519-signed, selective
disclosure), an **on-device differentially private update** mechanism, a
user-controlled **credential store** (agent auth, scoping, elicitation, encrypted at
rest), a **PTP MCP server**, the **QIL pipeline** (TF-IDF + softmax extraction,
Beta-Binomial / Normal-Normal aggregation, `/quality` + `/compare`, and a QIL MCP
server), and the **agent integration layer** that combines them.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python experiments/run_phase0.py --users 500   # Claim 1: transfer experiment + gate
python experiments/run_phase0_qil.py           # Claim 2: QIL extraction + gate
python experiments/run_phase1_integration.py   # Integration: α-blend benchmark + milestone
python experiments/run_phase1_quality_robustness.py  # Quality handling: shrinkage vs. raw
python -m preferencelayer.cli demo             # end-to-end PTP credential lifecycle
python -m preferencelayer.cli agent-demo       # preference+quality α-blend ranking
python -m preferencelayer.cli protocol-demo    # rank over the real PTP + QIL MCP tools
python experiments/run_phase1_cold_start.py    # adaptive α in the zero-history regime
python -m pytest                               # full test suite
```

The persistent credential store (Phase 1) is driven through the `preflayer` CLI;
it writes to `$PREFLAYER_HOME` (default `~/.preflayer`):

```bash
preflayer init --seed-demo                     # create identity + store, seed a credential
preflayer authorize agent.shop --scope laptops # mint a scoped, expiring agent token
preflayer view                                 # identity, credentials, active tokens
preflayer revoke agent.shop                    # revoke an agent's tokens
preflayer export --out bundle.json             # export the signed credential(s)
preflayer delete --yes                         # irreversibly wipe the store
```

The Amazon Reviews 2023 real-data path needs the optional extra and network access:
`pip install -e ".[amazon]"`, then `python experiments/run_amazon_realdata.py` (see
[`docs/phase1-amazon-realdata.md`](docs/phase1-amazon-realdata.md)).

---

## Quick Links

- [Phase 1 Kickoff Plan — Milestones & Sequencing](docs/phase1-kickoff.md)
- [Phase 0 Results — Claim 1 (Preference Graph)](docs/phase0-results.md)
- [Phase 0 Results — Claim 2 (QIL Extraction)](docs/phase0-qil-results.md)
- [Phase 1 Integration Results — the α-Blend](docs/phase1-integration-results.md)
- [Phase 1 Quality Handling — Shrinkage vs. Raw Averaging](docs/phase1-quality-robustness-results.md)
- [Phase 1 Protocol Integration — PTP + QIL over MCP](docs/phase1-protocol-integration.md)
- [Phase 1 Cold-Start — Adaptive α in the Zero-History Regime](docs/phase1-cold-start-results.md)
- [Phase 1 Real-Data Reality Check — Amazon Reviews 2023](docs/phase1-amazon-realdata.md)
- [Technical Proposal](proposals/technical.md)
- [Investor Proposal](proposals/investor.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Architecture](docs/architecture.md)
- [Protocol Spec (Draft)](docs/protocol-spec.md)
