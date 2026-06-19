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
│   └── cli.py                 # `preflayer` command-line interface
│
├── experiments/
│   ├── run_phase0.py          # Headline cross-category transfer experiment
│   └── phase0_results.json    # Saved metrics for the headline run
│
├── tests/                     # Test suite (incl. the Phase 0 go/no-go gate)
│
├── proposals/                 # Technical + investor proposals
└── docs/
    ├── implementation-plan.md # Full phased build plan
    ├── architecture.md        # System architecture reference
    ├── protocol-spec.md       # PTP protocol specification (draft)
    └── phase0-results.md      # Phase 0 research report (results below)
```

---

## Status

**Phase 0 prototype — core claim validated.** The repository now contains a working,
tested implementation of the Phase 0 research prototype alongside the design docs.

The headline result: the **sparse DAG preference graph beats the strong flat-vector
baseline by +9.7% NDCG@10 on cross-category transfer** (laptops → headphones,
p = 0.0002), robust across seeds — clearing the Phase 0 go/no-go gate (≥ 5%). Full
methodology, ablations, and honesty notes are in
[`docs/phase0-results.md`](docs/phase0-results.md).

Also implemented: the **PTP credential** (W3C-VC-shaped, Ed25519-signed, selective
disclosure), an **on-device differentially private update** mechanism, a
user-controlled **credential store** (agent auth, scoping, elicitation, encrypted at
rest), and a **PTP MCP server** exposing the three operations as agent tools.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python experiments/run_phase0.py --users 500   # run the headline experiment + gate
python -m preferencelayer.cli demo             # end-to-end PTP credential lifecycle
python -m pytest                               # full test suite
```

The Amazon Reviews 2023 real-data path needs the optional extra:
`pip install -e ".[amazon]"` (see `src/preferencelayer/data/amazon.py`).

---

## Quick Links

- [Phase 0 Results](docs/phase0-results.md)
- [Technical Proposal](proposals/technical.md)
- [Investor Proposal](proposals/investor.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Architecture](docs/architecture.md)
- [Protocol Spec (Draft)](docs/protocol-spec.md)
