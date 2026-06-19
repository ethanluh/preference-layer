# Implementation Plan

## PreferenceLayer: Phased Build Roadmap

**Total timeline:** 24 months across four phases  
**Current phase:** Phase 0

Each phase has explicit go/no-go gates. Do not advance without meeting them.

---

## Overview

| Phase | Name | Timeline | Team | Gate |
|-------|------|----------|------|------|
| 0 | Research Foundation | Months 1–3 | 1–2 people | Preference graph beats baseline; QIL extraction ≥70% precision |
| 1 | Core Protocol & MVP Data | Months 4–9 | 2–3 people | 2+ design partners validate credential utility |
| 2 | Scale & Network Effects | Months 10–18 | 4–6 people | 1+ retailer data partnership; self-sustaining update loop |
| 3 | Standardization & Ecosystem | Months 19–24 | 6–8 people | PTP referenced in major agent framework |

---

## Phase 0: Research Foundation & Scope Validation

**Months 1–3 | 1–2 people | ~$3k external costs**

### Goal

Validate the two core technical claims before building anything. If either fails, the project's scope and architecture need to change. Do not skip this phase.

**Claim 1:** A compact preference graph outperforms cold-start baselines on cross-category recommendation tasks.  
**Claim 2:** Use-profile-conditioned quality signals are extractable at ≥70% precision from public unstructured sources.

---

### Work Stream A: Preference Graph Prototype

**Objective:** Determine whether the graph-structured preference representation is worth building into a protocol, or whether simpler representations suffice.

**Dataset:** Amazon Reviews 2023 (publicly available). Use purchase sequences as implicit preference signal. Hold out 20% of users for evaluation.

**Implementation steps:**

1. Define the attribute node set for a single product category (laptops recommended — rich signal, well-structured). Start with 15–20 attributes: price tier, weight, battery life, display quality, thermal performance, keyboard quality, build quality, brand affinity, etc.

2. Implement three baselines:
   - Flat preference vector (mean of item embeddings in purchase history)
   - MemRerank (reproduce the 2603.29247 paper's approach)
   - BM25 over stated preference strings

3. Implement the sparse DAG preference graph:
   - Nodes: attribute embeddings (use a pretrained product encoder)
   - Edges: learned via co-occurrence in purchase sequences (PMI-based initialization, gradient-refined)
   - Context conditioning: attention mask over subgraph nodes, conditioned on query context embedding

4. Measure NDCG@10 on the held-out set for all four systems. Also measure cross-category transfer: train on laptop preferences, evaluate on headphone recommendations for the same users.

**Go/no-go criterion:** Sparse DAG outperforms flat vector by ≥5% NDCG@10 on cross-category transfer. If not, revisit graph topology before proceeding.

**Deliverable:** Research report, open-source prototype, reproducible experiment scripts.

---

### Work Stream B: QIL Extraction Feasibility

**Objective:** Determine whether the NLP extraction pipeline can produce signal of sufficient precision to build a quality database worth paying for.

**Target categories:** Laptops and mechanical keyboards. Both have rich, structured community discourse with explicit failure and use-context reporting.

**Data sources:**
- Reddit: r/laptops, r/thinkpad, r/MechanicalKeyboards via Pushshift or Reddit API
- iFixit: repair guides and teardown pages (publicly crawlable)
- Notebookcheck: review corpus for laptops (thermal, battery, display benchmarks)

**Implementation steps:**

1. Collect 2,000 Reddit posts/comments per category mentioning product failures, performance issues, or use-context comparisons. Filter to posts with ≥3 upvotes to reduce noise.

2. Define the annotation schema:
   - `product_id`: normalized model name
   - `failure_mode`: categorical (thermal throttling, battery degradation, structural failure, etc.)
   - `use_profile`: categorical (light use, heavy use, gaming, professional/sustained compute, travel, etc.)
   - `signal_type`: failure report / performance observation / comparative judgment
   - `confidence`: annotator confidence score

3. Annotate 300 samples manually (use Mechanical Turk or Label Studio with 2 annotators per sample, adjudicate disagreements). Budget: ~$600 at $2/sample.

4. Fine-tune a classification model (start with a small BERT variant) on the annotated set. Evaluate on a 100-sample held-out set.

**Go/no-go criterion:** ≥70% precision on use-profile classification on the held-out set. If 60–70%, assess whether more annotation data recovers precision before deciding to continue. Below 60%, the automation story doesn't hold and manual annotation costs make the QIL economically unviable at current scale.

**Deliverable:** Annotated dataset (300 samples), trained classifier, precision/recall report.

---

### Work Stream C: Protocol & Infrastructure Landscape

**Objective:** De-risk the credential schema and MCP server architecture before implementation begins.

**Tasks:**

1. Audit W3C VC Data Model 2.0 and DIF Presentation Exchange specs. Specifically assess: Is the VC overhead acceptable for low-latency agent queries? If round-trip credential verification adds >50ms p99, consider a lighter signed JSON schema instead.

2. Build a minimal MCP server (TypeScript, using the official MCP SDK) wrapping a mock preference credential store. Verify that agent frameworks (LangChain, Claude agent SDK) can call it without friction.

3. Audit the MCP and A2A protocol landscape for relevant prior art on credential exchange. Document any conflicts with the PTP design.

**Deliverable:** Protocol feasibility memo, mock MCP server stub.

---

## Phase 1: Core Protocol & Minimal Viable Data Product

**Months 4–9 | 2–3 people | $50–100k**

### Goal

Ship a working PTP implementation and a QIL covering two product categories. Onboard 3–5 agent developers as design partners. Validate that the credential provides measurable value in a real agent workflow.

---

### Work Stream A: PTP v0.1 Implementation

**Credential schema (Month 4):**

Define the JSON-LD context for the preference credential. Required fields:
- `@context`: PreferenceLayer VC context URI
- `attributeNodes`: sparse list of (node_id, embedding, confidence) tuples
- `edgeWeights`: list of (source, target, weight, context_key) tuples
- `contextConditioners`: mapping from context embeddings to subgraph activation masks
- `updateMetadata`: last update timestamp, update count, privacy budget consumed
- `issuer`: user DID (W3C DID method; `did:key` is simplest for v0)
- `proof`: linked data proof (Ed25519Signature2020)

**Three endpoints (Months 4–5):**

`GET /preference`
- Input: `category` (product category string), `query_context` (embedding or text), `agent_id` (authenticated agent identifier)
- Processing: retrieve credential, apply context conditioner to produce query-scoped subgraph, apply selective disclosure (redact low-relevance nodes)
- Output: signed, scoped credential fragment + confidence metadata
- Latency target: <100ms p95

`POST /outcome`
- Input: `product_id`, `outcome_type`, `use_context`, `timestamp`, `agent_id`
- Processing: enqueue for async on-device update; return 202 immediately
- Update pipeline (async): compute gradient update to relevant subgraph nodes; apply Gaussian noise (ε=2 DP); apply update; re-sign credential
- No raw behavioral data stored server-side

`POST /elicit`
- Input: `attribute_focus` (list of attribute node IDs with low confidence), `agent_id`
- Processing: select highest information-gain questions from the attribute subgraph using a greedy IG policy
- Output: ordered list of ≤5 structured questions with response schemas
- User responses submitted as `POST /outcome` with `outcome_type: elicitation`

**Credential store (Month 5–6):**
- Local daemon (Python, runs on user's machine or mobile device)
- SQLite-backed, AES-256 encrypted at rest
- OAuth 2.0 device flow for agent authentication
- Optional cloud sync: credential encrypted with user's Ed25519 key before leaving device; server stores ciphertext only
- CLI: `preflayer view`, `preflayer revoke <agent_id>`, `preflayer export`, `preflayer delete`

**MCP server wrapper (Month 6):**
- Wrap all three endpoints as MCP tools
- Write tool descriptions optimized for agent self-selection (agents should correctly invoke the right tool without explicit instruction)
- Test against Claude agent SDK and LangChain

---

### Work Stream B: QIL v0.1 — Two Categories

**Ingestion pipeline (Months 4–5):**

Productionize the Phase 0 NLP pipeline. Automate ingestion from:
- Reddit (Pushshift API or official API with rate limiting)
- iFixit (crawler with polite rate limiting, respect robots.txt)
- Notebookcheck (laptop benchmarks, structured scrape)

Run daily. Store extracted records in a structured database (PostgreSQL). Schema:
```
product_signal(
  product_id TEXT,
  model_normalized TEXT,
  failure_mode TEXT,
  use_profile TEXT,
  signal_type TEXT,
  source_url TEXT,
  extracted_at TIMESTAMP,
  annotator_confidence FLOAT,
  upvote_count INT
)
```

**Bayesian aggregation layer (Months 5–6):**

For each (product_id, use_profile) pair:
- **Failure rate:** Hierarchical Beta-Binomial. Prior parameters estimated from category-level failure rates. Update with per-product failure counts.
- **Quality dimensions** (thermal, build quality, battery longevity, display): Gaussian process with squared-exponential kernel over product release time. Handles temporal degradation of component quality.

Refit posteriors nightly. Store posterior parameters (not raw samples) for fast query response.

**API (Month 6):**

`POST /quality`
- Input: `product_id`, `use_profile_vector` (embedding or categorical), `dimensions` (optional filter)
- Output: posterior mean + 90% credible interval per quality dimension, failure rate estimate, evidence count, data quality score, evidence pointers (source URLs)
- Latency target: <200ms p95

`POST /compare`
- Input: `product_id_a`, `product_id_b`, `use_profile_vector`
- Output: posterior difference estimate with credible interval; probability that A > B on each dimension

**Coverage target at Phase 1 launch:** 500 laptop models, 300 keyboard models.

---

### Work Stream C: Design Partner Program

**Months 7–9:**

Recruit 3–5 developers building MCP-native agents. Target profile: open-source agent projects, small startups, hobbyist agents — not platform incumbents (they won't integrate a competitor's preference layer).

Provide:
- Direct access to PTP v0.1 and QIL v0.1
- Private Slack channel for feedback
- Commitment to prioritize bugs and API friction they report

Measure:
- Does recommendation quality improve measurably on their tasks? (NDCG or user-reported satisfaction)
- Where does the cold-start problem dominate?
- What is the integration friction in their agent framework?

**Go/no-go gate:** At least 2 of 5 design partners report measurable improvement in recommendation relevance. If not, the cold-start problem requires more investment before scaling — pivot Phase 2 to elicitation-first, defer protocol scaling.

---

## Phase 2: Scale, Coverage & Network Effects

**Months 10–18 | 4–6 people | $1–2M seed**

### Goal

Expand QIL to 10 product categories. Grow PTP adoption to 50+ agent integrations. Establish the self-sustaining update loop (outcome signals from agents feeding back to improve credentials). Reach operational sustainability on the QIL data feed.

---

### Work Stream A: QIL Expansion

**8 new categories (Months 10–14):**
Consumer electronics (smartphones, headphones, monitors), home appliances (washing machines, refrigerators), power tools, cycling components, audio equipment, gaming peripherals.

**Generalized extraction pipeline:**
Retire per-category classifiers. Train a cross-category NLP model (fine-tuned on accumulated annotated data from Phases 0–1). Target: a new category is onboardable with <40 hours of annotation effort. This requires the model to generalize failure mode and use-profile detection across domains.

**Counterfactual comparison endpoint:**
`POST /compare` matured into a primary product. Users (via agents) can ask: "Given that I run sustained compute workloads, is the ThinkPad X1 Carbon better than the Dell XPS 15?" The response is a posterior difference with uncertainty, not a point estimate.

**Data quality layer:**
Add automated freshness scoring: if a product model has no new signal in 90 days, decay its confidence scores and surface this in API responses. Add an editorial flag for high-stakes categories (medical devices, safety equipment) requiring manual review.

---

### Work Stream B: PTP v1.0 — Cross-Agent Update Loop

**Cross-agent credential merging (Months 10–12):**

When a user authorizes two agents to update their credential, both contribute outcome signals to the same graph. Conflict resolution protocol:
- Timestamp-weighted: more recent signals have higher weight
- Preference-stability-regularized: large deviations from prior weights are penalized (prevents a single unusual purchase from destabilizing the graph)
- Async merge with user-visible changelog

**Learned alpha (Month 12):**
The blending weight `α` between preference score and quality score is currently a fixed hyperparameter. In v1.0, it becomes a per-user learned weight: users with sparse credentials (new, or low-engagement) automatically lean on quality score; users with rich histories lean on preference score. Calibrated against design partner data.

**Publish PTP v1.0 as open standard (Month 13):**
- Formal spec (RFC-style Markdown document)
- Reference implementations in Python and TypeScript
- Submit to W3C Credentials Community Group and DIF for feedback

---

### Work Stream C: Go-to-Market

**Developer portal (Month 10):**
- Public documentation site
- SDK packages (`pip install preferencelayer`, `npm install @preferencelayer/sdk`)
- Sandbox environment with synthetic user credentials for agent testing
- API key management, usage dashboard, billing

**Retailer data partnerships (Months 10–14):**
Begin outreach in Phase 1 (not Phase 2). Target: 3–5 mid-size online retailers willing to share anonymized return signal in exchange for QIL API access. Return data is the highest-quality outcome signal and is not available from public sources. This is the critical path item for QIL quality; delay here delays the Phase 3 gate.

Return signal data sharing agreement requirements:
- Anonymized at source (no PII)
- Product ID + return reason + use context only
- Retailer receives QIL API credits in exchange

**Pricing (Month 10):**
- Free tier: <1,000 queries/month (for developers and researchers)
- Starter: $99/month, 50k queries
- Growth: $499/month, 500k queries
- Enterprise: custom, includes QIL data feed

---

### Phase 2 Go/No-Go Gate

Advance to Phase 3 when:
1. At least one retailer data partnership is live and generating return signal
2. Self-sustaining update loop demonstrated: outcome signals from agents improve credential quality measurably (NDCG improvement on held-out set after 60 days of agent feedback)
3. 50+ agent integrations active

---

## Phase 3: Protocol Standardization & Ecosystem Moat

**Months 19–24 | 6–8 people**

### Goal

PTP becomes the default portable preference infrastructure for the MCP ecosystem. QIL covers all major consumer product categories. The update loop is self-sustaining.

---

### Work Stream A: Standards & Ecosystem

**Protocol standardization:**
PTP v1.0 submitted to a recognized standards body. Goal: referenced in at least one major agent framework (LangChain, AutoGPT, or a major platform SDK) as the default preference integration path. This requires active engagement with framework maintainers — assign a dedicated developer relations role.

**Open-source strategy:**
Open-source the preference graph training code, the credential schema validator, the reference credential store, and the MCP server implementation. Keep proprietary: the QIL database, the Bayesian aggregation models, the cloud sync infrastructure, and the API serving layer.

The open-source core lowers friction for agent developers and makes the protocol harder to fork away from (network effects accrue to the protocol, not just the implementation).

**Ecosystem programs:**
- Developer grants for novel PTP integrations
- Hackathon sponsorship focused on cross-platform agent use cases
- Integrations team: 1–2 people dedicated to onboarding enterprise agent developers

---

### Work Stream B: QIL Full Coverage & B2B

**Coverage expansion (Months 19–22):**
Expand to 25+ product categories covering >80% of consumer electronics and home goods spend by value. Prioritize by category spend volume and data availability.

**B2B product line:**
Sell structured QIL data as a feed to:
- **Retailers:** Merchandising teams use failure rate data by use profile to make stocking decisions and write more accurate product descriptions
- **Insurers:** Product warranty pricing calibrated to use-profile-conditioned failure rates
- **Warranty providers:** Risk assessment for extended warranty products

This revenue stream is independent of agent API adoption and provides a financial floor that doesn't depend on protocol success.

**Performance warranty (Month 20):**
For recommendations above a confidence threshold (posterior mean > 0.85, evidence count > 50), offer a satisfaction guarantee: if the user returns the product, PreferenceLayer refunds the query fee. Track warranty claim rates as a model accuracy proxy — high claim rate indicates model miscalibration, triggering a review of the relevant product-profile posteriors.

---

### Success Metrics at 24 Months

**Protocol:**
- 200+ agent integrations
- 100k+ active user credentials with cross-agent update history
- PTP referenced in at least one major open agent framework

**Quality Intelligence Layer:**
- 25+ product categories
- 10,000+ product models covered
- 5+ retailer data partnerships
- B2B feed revenue covering QIL operational costs

**Research:**
- At least one publication from Phase 0–1 research (preference graph topology or cold-start elicitation)
- PTP v1.0 spec adopted or under review by W3C CCG or DIF

---

## Resource Requirements

### Phase 0–1 (Months 1–9)

**Team:** 1–2 engineers. Phase 0 is solo-executable.

**External costs:**
- Annotation budget: ~$600–1,200 (300–600 samples at $2/sample)
- Cloud compute: ~$500/month (NLP training, Bayesian model fitting, API serving in Phase 1)
- Reddit API access: free tier sufficient for Phase 0; $100/month for Phase 1 volumes
- Total Phase 0: ~$3k

**Phase 1 funding need:** $50–100k pre-seed. Primarily covers dedicated engineering time. Infrastructure costs are minimal.

### Phase 2–3 (Months 10–24)

**Team (Phase 2):** 4–6 people
- 2 ML/data engineers (QIL pipeline, preference graph, Bayesian models)
- 2 backend/infrastructure engineers (PTP serving, credential store, API)
- 1 developer experience (SDK, documentation, developer portal)
- 1 partnerships/BD (retailer data partnerships, enterprise QIL)

**Team (Phase 3):** Add 2 people
- 1 developer relations (agent framework integrations, ecosystem)
- 1 standards/protocol (W3C CCG, DIF engagement, spec maintenance)

**Infrastructure (Phase 2):**
- API serving: ~$3–5k/month at 50+ agent integration scale
- Ingestion pipeline: ~$1–2k/month
- Database: ~$500/month
- Total: ~$5–8k/month

**Funding:** $1–2M seed. Covers 18 months of team + infrastructure at Phase 2 scale.

---

## Critical Path

The longest dependency chain is:

**Retailer data partnerships → QIL quality plateau → Phase 3 gate**

Retailer outreach requires business development effort that cannot be parallelized with technical work. It also has a long sales cycle (legal review, data sharing agreements, procurement). Start in Month 6 (Phase 1), not Month 10 (Phase 2). Every month of delay here delays the Phase 3 gate.

Everything else on the technical side can be parallelized. The protocol and the QIL pipeline are independent until the scoring function integration in Phase 2. The credential store and the MCP server can be built concurrently.

The one other non-parallelizable dependency: design partner recruiting requires a working v0.1 API. Don't start partner outreach until the three endpoints are deployed and stable. Shipping a broken API to design partners early is worse than shipping late.
