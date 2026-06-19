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
│   └── cli.py                 # `preflayer` command-line interface
│
├── experiments/
│   ├── run_phase0.py          # Claim 1: cross-category transfer experiment
│   ├── run_phase0_qil.py      # Claim 2: QIL extraction feasibility study
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
    └── phase0-qil-results.md  # Phase 0 Claim 2 report (QIL extraction)
```

---

## Status

**Phase 0 prototype — both research gates passed.** The repository contains a
working, tested implementation of the Phase 0 research prototype alongside the
design docs.

- **Claim 1 (preference graph):** the **sparse DAG preference graph beats the strong
  flat-vector baseline by +9.7% NDCG@10 on cross-category transfer** (laptops →
  headphones, p = 0.0002), robust across seeds — clearing the ≥ 5% gate. See
  [`docs/phase0-results.md`](docs/phase0-results.md).
- **Claim 2 (QIL extraction):** use-profile-conditioned quality signals are
  extracted from unstructured text at **88.3% macro precision** (vs. a 24.2%
  baseline), clearing the ≥ 70% gate. See
  [`docs/phase0-qil-results.md`](docs/phase0-qil-results.md).

Also implemented: the **PTP credential** (W3C-VC-shaped, Ed25519-signed, selective
disclosure), an **on-device differentially private update** mechanism, a
user-controlled **credential store** (agent auth, scoping, elicitation, encrypted at
rest), a **PTP MCP server**, and the **QIL pipeline** (TF-IDF + softmax extraction,
Beta-Binomial / Normal-Normal aggregation, `/quality` + `/compare`, and a QIL MCP
server).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python experiments/run_phase0.py --users 500   # Claim 1: transfer experiment + gate
python experiments/run_phase0_qil.py           # Claim 2: QIL extraction + gate
python -m preferencelayer.cli demo             # end-to-end PTP credential lifecycle
python -m pytest                               # full test suite
```

The Amazon Reviews 2023 real-data path needs the optional extra:
`pip install -e ".[amazon]"` (see `src/preferencelayer/data/amazon.py`).

---

## Quick Links

- [Phase 0 Results — Claim 1 (Preference Graph)](docs/phase0-results.md)
- [Phase 0 Results — Claim 2 (QIL Extraction)](docs/phase0-qil-results.md)
- [Technical Proposal](proposals/technical.md)
- [Investor Proposal](proposals/investor.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Architecture](docs/architecture.md)
- [Protocol Spec (Draft)](docs/protocol-spec.md)
