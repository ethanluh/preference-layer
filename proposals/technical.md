# Technical Proposal

## PreferenceLayer: A Portable Preference Transport Protocol for Cross-Platform Agentic Commerce

---

## Abstract

Agentic shopping systems optimize over a user-defined objective function. In practice, this function is severely underspecified: preferences are latent, high-dimensional, and context-dependent, and there is currently no portable mechanism for a user to supply a richer preference signal across agent-platform boundaries. We propose PreferenceLayer — a user-controlled preference credential and transport protocol that makes this possible — paired with a Quality Intelligence Layer (QIL) providing use-profile-conditioned product quality signals as an agent-queryable API. Both components are designed to be MCP-native and platform-agnostic.

---

## 1. Problem Statement

### 1.1 The Underspecified Objective Function

Autonomous shopping agents optimize over a proxy of user preference, not the latent preference distribution itself. The gap between proxy and latent distribution is the core failure mode. An agent asked to "find the best laptop for ML work" must resolve several layers of ambiguity: What is the user's workload profile? What tradeoffs do they make between cost and performance? What have past purchases taught them about which attributes matter?

Existing agents resolve this via platform-specific behavioral priors. Amazon conditions on purchase history within Amazon. Perplexity conditions on search history within Perplexity. Daydream builds a "Style Passport" scoped to the Daydream platform. In each case, the preference model is trained on signals collected within the platform, stored in platform-controlled infrastructure, and consumed exclusively by that platform's agent.

This is not an oversight. Platform-locked preference data is a retention mechanism. The structural consequence is that any agent operating outside a single platform's ecosystem — the growth vector in the MCP/A2A ecosystem — must bootstrap preference inference from scratch at each interaction.

### 1.2 The Missing Quality Signal

The second gap is orthogonal. Even agents with rich preference models lack access to the quality signal most relevant to heterogeneous users: how a product actually performs under a specific use profile. Aggregate review scores are a coarse proxy. A laptop with 4.3 stars averaged across casual users and power users tells a sustained-compute user nothing useful about thermal behavior under load. This signal exists — in repair forums, return data, manufacturer service bulletins, long-form teardown databases — but it is unstructured, scattered, and not exposed in any machine-readable form calibrated to use profile.

### 1.3 What Doesn't Exist

Neither a portable preference credential nor a use-profile-conditioned quality API exists in any current system. Recent academic work (MemRerank, 2026) addresses compact preference memory for single-platform reranking but does not address portability or cross-agent update protocols. The infrastructure gap is real and buildable.

---

## 2. Proposed System

### 2.1 The Preference Credential

The core primitive is a signed, user-controlled preference credential: a structured document encoding a compact representation of the user's latent preference distribution over a product attribute space.

**Schema.** The credential is modeled as a W3C Verifiable Credential (VC Data Model 2.0), enabling cryptographic binding to user identity, selective disclosure, and third-party verification without exposing raw behavioral data. Internally, it stores a sparse preference graph:

```
G = (A, E, W, C)
```

where:
- `A` is the attribute node set (price, durability, form factor, brand affinity, etc.)
- `E` encodes conditional preference dependencies between attributes
- `W` are learned edge weights
- `C` is a context-conditioning function mapping query contexts to subgraph activations

**Update protocol.** Outcome signals (purchase, return, dwell time, explicit rating) are processed on-device. Gradient updates to the graph are computed locally and applied using a differentially private mechanism (Gaussian noise, ε = 2) before any state leaves the device. The cloud sync option stores only the encrypted credential; the server holds ciphertext and cannot read the graph.

**Cold-start handling.** The credential must be useful at initialization, before any interaction history. Population-level priors conditioned on coarse stated preferences (product category, approximate budget range, a brief elicitation sequence) provide a starting distribution. The elicitation protocol is modeled as adaptive information-gain maximization over low-confidence attribute subgraphs.

### 2.2 The Preference Transport Protocol (PTP)

PTP is a lightweight REST protocol defining three operations against a credential store:

**`GET /preference`**
Agent requests a user's preference credential, scoped by product category. Returns a signed credential with selective attribute disclosure based on query context. The agent does not receive the full graph — only the subgraph relevant to the current query.

**`POST /outcome`**
Agent submits a structured outcome signal post-transaction. Fields: `product_id`, `outcome_type` (purchase / return / dwell / rating), `use_context`, `timestamp`. Triggers an asynchronous, privacy-preserving update to the user's preference graph.

**`POST /elicit`**
Agent requests active preference elicitation for a specific attribute vector where the credential has low confidence. Returns a structured question sequence. User responses update the credential with higher-confidence signal than behavioral inference alone.

PTP is designed to be MCP-native. A PTP MCP server can be instantiated against any credential store, allowing agents built on any MCP-compatible framework to consume preference signals without direct API integration work.

### 2.3 The Quality Intelligence Layer (QIL)

The QIL is a continuously maintained, proprietary knowledge base of use-profile-conditioned product quality signals.

**Ingestion pipeline.** Automated extraction from Reddit (r/laptops, r/MechanicalKeyboards, r/homelab, category-specific subreddits), iFixit repair databases, Notebookcheck and similar review corpora, manufacturer service bulletins, and return data sourced via retailer partnerships. An NLP pipeline performs entity recognition (product model, failure mode, use context) and use-profile classification.

**Aggregation model.** Per-product, per-profile posteriors are maintained using:
- Hierarchical Beta-Binomial model for failure rate estimation
- Gaussian process for continuous quality dimensions (thermal performance, build quality degradation, battery cycle behavior)

Posteriors are updated continuously as new signals arrive. Each product record carries a data quality score: coverage (evidence count), recency (signal freshness), and profile specificity (how many distinct use profiles have signal).

**API.** `POST /quality` accepts a `product_id` and a `use_profile_vector`; returns a structured quality report with posterior mean, credible intervals, evidence count, and evidence pointers. A counterfactual endpoint (`POST /compare`) returns a posterior difference estimate between two products conditioned on a shared use profile.

### 2.4 Combined Scoring

An agent using both components scores candidates as:

```
s(i, u, c) = α · pref_score(i, G_u, c) + (1−α) · quality_score(i, profile_u)
```

where `α` is a learned trust weight calibrated per user: low-confidence credential (sparse graph, new user) → lean on quality score; high-confidence credential (rich history) → lean on preference score. As QIL evidence thins for niche products, the blend shifts back toward preference.

---

## 3. Research Questions

Three open problems are pursued as parallel research tracks. All are testable against existing datasets and baselines.

**Preference graph topology.** Does a sparse DAG with attention-gated edge weights outperform flat preference vectors on cross-category recommendation tasks? Hypothesis: yes, because cross-category preference dependencies (e.g., durability preference transferring from laptop to headphone purchasing) are not representable in flat encodings. Baseline: MemRerank on Amazon Reviews 2023.

**Privacy-utility tradeoff.** What is the Pareto frontier between differential privacy budget (ε) and recommendation quality degradation (NDCG@10 regression) as a function of graph sparsity and update frequency? Target operating point: ε = 2 with < 5% NDCG regression versus the non-private baseline.

**Cold-start elicitation efficiency.** What is the minimum elicitation query sequence that produces a useful preference initialization? Modeled as an adaptive information-gain maximization problem over the attribute space. Target: ≥ 80% of full-history recommendation quality achieved in ≤ 5 elicitation questions.

---

## 4. Competitive Position

The differentiating claim is not that PreferenceLayer produces better recommendations than Perplexity's in-platform model — it won't, for users who only shop on Perplexity. The claim is narrower and more durable:

1. PreferenceLayer is the only system whose recommendations improve across agent-platform boundaries.
2. PreferenceLayer is the only system that provides use-profile-conditioned quality intelligence as an agent-consumable API.

Neither property is achievable by a platform-native system without undermining the platform's own retention incentives. This is a structural constraint, not a temporary technical gap.

The moat compounds over time: the QIL database deepens with every ingestion cycle, and the preference credential network becomes more valuable as more agents adopt the protocol and contribute outcome signals.

---

## 5. Failure Modes

| Risk | Assessment | Mitigation |
|------|------------|------------|
| Platform consolidation captures agent runtime | Moderate probability; reduces distribution surface | MCP/A2A ecosystem breadth; regulatory tailwind on interoperability |
| Foundation models ship native preference elicitation | High probability; does not solve cross-platform portability | Platform-native ≠ portable; QIL moat is independent |
| Cold-start quality too poor for adoption | Moderate; measurable in Phase 0 | Population priors + elicitation; expectation-setting at onboarding |
| Users unwilling to share behavioral data | Moderate; trust is a real cost | On-device processing; selective disclosure; full deletion rights |
| QIL staleness as product lines evolve | Ongoing operational risk | Continuous ingestion; automated freshness scoring; editorial layer |
| Extraction precision too low for QIL utility | Phase 0 falsifiable | Annotated validation set; go/no-go gate at 70% precision |

---

## 6. Related Work

- **MemRerank (2026):** Compact preference memory for single-platform product reranking. Demonstrates that structured preference representations outperform raw history injection. Does not address portability or cross-agent update protocols.
- **W3C Verifiable Credentials Data Model 2.0:** Cryptographic credential framework used as the schema basis for the preference credential.
- **Model Context Protocol (MCP, Anthropic 2024):** Open protocol for agent-tool integration. PTP is designed to be MCP-native from the ground up.
- **Agent2Agent (A2A, Google 2025):** Agent-to-agent communication protocol. PTP credential reads/writes can be mediated via A2A in multi-agent pipelines.
- **Differential Privacy (Dwork et al.):** Gaussian mechanism used in the on-device credential update protocol.
