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
├── .gitignore
│
├── proposals/
│   ├── technical.md           # Technical proposal
│   └── investor.md            # Investor proposal
│
└── docs/
    ├── implementation-plan.md # Full phased build plan
    ├── architecture.md        # System architecture reference
    └── protocol-spec.md       # PTP protocol specification (draft)
```

---

## Status

**Pre-prototype.** This repository contains the research foundation, protocol design, and implementation plan. Phase 0 (validation research) is the current active work.

See [`docs/implementation-plan.md`](docs/implementation-plan.md) for the full roadmap.

---

## Quick Links

- [Technical Proposal](proposals/technical.md)
- [Investor Proposal](proposals/investor.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Architecture](docs/architecture.md)
- [Protocol Spec (Draft)](docs/protocol-spec.md)
