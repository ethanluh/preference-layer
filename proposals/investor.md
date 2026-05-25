# Investor Proposal

## PreferenceLayer

**The infrastructure layer AI shopping agents are missing.**

---

## The Problem

AI shopping agents are changing how people buy things. Instead of spending an hour on Amazon, you tell an agent what you need and it handles the rest. The market is real and moving fast — McKinsey estimates agentic commerce could redirect $3–5 trillion in global retail spend by 2030.

But every AI shopping agent today makes the same architectural mistake.

They lock your preference data inside their own platform.

Perplexity knows what you like when you're on Perplexity. Amazon knows your history when you're on Amazon. OpenAI's shopping agent starts from scratch when you use it for the first time. The moment you use a different agent — or when a developer builds a new agent that isn't owned by one of these platforms — all of that learned preference history disappears.

This isn't a bug. It's intentional. Platform-locked preference data keeps users from leaving.

The result: agents operating outside a single platform's ecosystem have to guess what you want every time. They make worse recommendations. Users get worse outcomes. And the open-agent ecosystem — which is growing fast, backed by open standards from Anthropic, Google, and over a hundred technology companies — is operating with one hand tied behind its back.

---

## The Insight

There is a structural gap between what platform-native agents can do and what users actually need.

Platform-native preference models are optimized for platform-side objectives — conversion rate, margin, ad revenue — not user welfare. An agent operating on a user's behalf has adversarial interests with the platform it's shopping on. This creates permanent demand for a neutral, user-aligned preference layer that platform incumbents cannot provide without undermining their own business model.

This gap doesn't close as AI gets better. It's structural.

---

## What We're Building

PreferenceLayer is two things:

### 1. The Preference Credential

A portable, user-owned profile of what you want — built from your purchase history, returns, and feedback, stored on your device, and shared selectively with agents you authorize.

When any agent needs to make a buying decision on your behalf, it queries your credential and immediately has a richer starting point than it could build on its own. Your preferences follow you across every agent you use. The more agents you use, the richer your credential gets.

Users own their data. No raw behavioral data leaves their device. Agents pay per-query to access the credential.

### 2. The Quality Intelligence Layer

A continuously updated database of product quality intelligence that doesn't exist anywhere publicly: how products actually perform under specific use conditions, not averaged across all buyers.

A laptop with a 4.3-star rating is useless information to someone who runs sustained compute jobs. That laptop might throttle aggressively under load. That signal exists — in repair forums, return data, teardown databases — but nobody has structured it and exposed it as a machine-readable API.

We do. Agents query this layer and get quality scores calibrated to what you actually do with a product.

---

## Why This Works Now

The agent layer is fragmenting, not consolidating.

The Model Context Protocol (MCP) and Agent2Agent (A2A) — backed by Anthropic, Google, and over 100 technology companies — are building the infrastructure for agents that operate across platforms. Thousands of developers are building agents that aren't owned by Amazon or Google. As this ecosystem grows, the demand for portable user context grows with it.

A user whose preferences only live inside one platform is a user who's locked in. Most users won't accept that indefinitely. Regulation is pushing the same direction — EU data portability requirements are expanding, and U.S. regulatory pressure on platform interoperability is stronger than it was during the mobile era.

The transition window is 18–36 months. Either one platform wins the agent runtime (reduces our distribution surface) or the open ecosystem establishes itself and portable preference infrastructure becomes table stakes. We're building for the second scenario — and our plan is viable even under partial consolidation, because the Quality Intelligence Layer is a standalone business independent of the protocol adoption story.

---

## Business Model

**Agent API access.** Per-query fees for preference credential reads and quality intelligence lookups. This is the primary revenue stream. Agents that want to make good recommendations pay to access better data.

**Outcome signal fees.** Agents pay a smaller fee to submit outcome signals (purchases, returns) that improve user credentials. This aligns incentives: agents that provide feedback get better data in return.

**B2B data feed.** The Quality Intelligence database sold directly to retailers (merchandising decisions), insurers (product warranty pricing), and warranty providers (risk assessment). This is a separate revenue stream that doesn't depend on agent adoption.

**Performance warranty.** For high-confidence recommendations, we offer a satisfaction guarantee. Cheap to underwrite when our model is accurate; a strong trust signal that drives agent adoption.

---

## The Moat

Two compounding advantages:

**The Quality Intelligence database.** It requires years of data collection and continuous maintenance. Every new signal improves coverage and accuracy. A competitor starting today faces a data gap that widens over time. This is the same dynamic that made Wirecutter worth acquiring and Consumer Reports worth subscribing to — except we're structured for machine consumption and we're use-profile-conditioned, not aggregate.

**Credential network effects.** As more agents adopt the protocol, each user's credential gets more valuable — more outcome signals, more cross-domain preference inference. Switching costs for agents increase as their users' credentials accumulate. The protocol becomes infrastructure.

Neither advantage is replicable by a platform-native competitor without dismantling their own lock-in strategy.

---

## Traction & Validation

This is a pre-prototype project. The immediate work is validating two core technical claims:

1. A portable preference credential meaningfully outperforms cold-start recommendation baselines.
2. Use-profile-conditioned quality signals are extractable at useful precision from public data sources.

Both claims are testable research questions with explicit go/no-go criteria. We are not asking for capital to build on unvalidated assumptions. Phase 0 exists to answer these questions before Phase 1 begins.

See the [Implementation Plan](../docs/implementation-plan.md) for the full phased roadmap.

---

## The Ask

Phase 0–1 is executable as a research project with minimal external resources. Pre-seed capital of $50–100k enables dedicated engineering time and data annotation at Phase 1. Seed capital of $1–2M enables the team and infrastructure for Phase 2–3.

We are currently seeking research collaborators and design partners (agent developers in the MCP ecosystem) before seeking external capital.

---

## Risk Summary

| Risk | Why It Matters | Our View |
|------|---------------|----------|
| Platform consolidation | Reduces distribution for the protocol | Regulatory environment more hostile to consolidation now than in 2008; MCP ecosystem breadth limits single-platform capture |
| Foundation models ship "good enough" preference modeling | Reduces demand for the protocol layer | Platform-native doesn't solve cross-platform portability; QIL is independent of this risk |
| Cold-start quality too low for adoption | First-user experience is poor | Measurable and addressable; managed via onboarding elicitation and expectation-setting |
| QIL precision insufficient | Core data product doesn't work | Phase 0 validation gate; go/no-go at 70% precision before building the product |
