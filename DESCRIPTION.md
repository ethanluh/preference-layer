# Project Description

## PreferenceLayer

**Type:** Infrastructure protocol + data product  
**Stage:** Pre-prototype  
**Domain:** Agentic commerce, AI infrastructure, data portability

---

## Summary

PreferenceLayer is portable preference infrastructure for the open agent web. It enables AI shopping agents to access rich, user-controlled preference signals across platform boundaries — and to query use-profile-conditioned product quality intelligence as an API.

The project is motivated by a structural gap in the current agentic commerce landscape: every major AI shopping agent (Amazon, Perplexity, OpenAI, Daydream) implements preference memory as a platform-resident silo. Agents operating outside a single platform's ecosystem have no access to this state and must bootstrap preference inference from scratch.

---

## Components

### Preference Transport Protocol (PTP)
A signed, user-controlled preference credential encoding a sparse preference graph over product attribute space. Agents read and update credentials via a lightweight REST protocol with MCP-native bindings. Updates are processed on-device using a differentially private gradient protocol; no raw behavioral data leaves the user's control.

### Quality Intelligence Layer (QIL)
A continuously maintained knowledge base of use-profile-conditioned product quality signals. Failure rates, longitudinal performance estimates, and counterfactual comparisons are segmented by use profile — not reported as population-level aggregates. Exposed as an agent-queryable API with Bayesian posteriors and confidence intervals.

---

## Key Claims

1. A portable preference credential meaningfully improves agent recommendation quality relative to cold-start baselines, even with minimal elicitation.
2. Use-profile-conditioned quality signals are extractable at useful precision from public data sources (repair forums, return signal, teardown databases).
3. The combination of (1) and (2), delivered as MCP-native APIs, is a viable infrastructure layer that no platform-native system can replicate without undermining its own retention incentives.

Claims 1 and 2 are testable research questions. Phase 0 exists to validate or falsify them.

---

## Differentiation from Existing Work

| System | Preference Memory | Portable | QIL | Agent API |
|--------|------------------|----------|-----|-----------|
| Amazon / Perplexity / OpenAI | ✓ | ✗ | ✗ | ✗ |
| Daydream | ✓ (fashion only) | ✗ | ✗ | ✗ |
| MemRerank (research) | ✓ | ✗ | ✗ | ✗ |
| **PreferenceLayer** | **✓** | **✓** | **✓** | **✓** |

---

## Timeline

- **Phase 0** (Months 1–3): Research validation
- **Phase 1** (Months 4–9): Core protocol + minimal data product
- **Phase 2** (Months 10–18): Scale + network effects
- **Phase 3** (Months 19–24): Protocol standardization + ecosystem

See [`docs/implementation-plan.md`](docs/implementation-plan.md) for details.
