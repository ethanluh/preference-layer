# Design Partner Onboarding (Phase 1 Work Stream C)

This guide is for developers building **MCP-native agents** who have agreed to be
PreferenceLayer design partners. It walks you from zero to a **before/after
relevance measurement** on *your own* task, which is the evidence the Phase 1
go/no-go gate is decided on:

> **Phase 1 WS-C gate:** at least **2 of 5** design partners report a
> **measurable improvement** in recommendation relevance when the user's
> PreferenceLayer credential is attached, versus running without it.

There are three steps:

1. [Get a credential against the PTP v0.1 API](#1-get-a-credential)
2. [Integrate the PTP + QIL MCP tools into your agent](#2-integrate-the-mcp-tools)
3. [Run the measurement harness on your task](#3-run-the-measurement-harness)

> **What is automated vs. what is a human step.** Everything in this document is
> something *you* (the partner) run. Recruiting partners, granting API access,
> and the private feedback channel are handled by the PreferenceLayer team — see
> [Getting set up with the team](#getting-set-up-with-the-team-human-steps).

---

## Concepts in one minute

PreferenceLayer is two independently deployable pieces that your agent combines
at query time (full design in [`architecture.md`](architecture.md)):

- **PTP** — the user's **portable preference credential** and the protocol to
  read/update it. The user owns it; it is signed and travels across platforms.
  Your agent reads it with `get_preference`.
- **QIL** — a server-side database of **use-profile-conditioned product quality**
  signals (failure rates, longitudinal performance). Your agent reads it with
  `get_quality`. The QIL holds **no user identifiers**.

Your agent ranks candidate products by **blending** a preference score (from the
credential) and a quality score (from the QIL):

```
score = alpha * preference_score + (1 - alpha) * quality_score
alpha = sigmoid(3.0 * (mean_confidence - 0.5))
```

`alpha` is confidence-adaptive: a sparse/new credential leans on community
quality; a rich credential leans on the user's own taste. You do **not** have to
implement the blend yourself — `AgentRecommender` does it.

---

## 1. Get a credential

You need (a) a running credential store holding a test user's credential, and (b)
an **agent token** authorizing your agent to read/update it.

### Option A — local store via the CLI (recommended for partners)

The reference credential store is a local, SQLite-backed, encrypted-at-rest
daemon shipped in this repo. Install and initialize it:

```bash
pip install preferencelayer            # or: pip install -e . from a checkout
preflayer init                         # creates the encrypted local store (~/.preflayer)
preflayer authorize my-agent           # mint an agent token; prints the token once
preflayer view                         # inspect the credential (no secrets printed)
```

`preflayer authorize <agent_id>` prints a bearer **agent token** — copy it; this
is what your agent presents on every PTP call. Revoke any time with
`preflayer revoke my-agent`. Other commands: `preflayer export`, `preflayer
delete`.

> The store database (`store.db`), the identity key (`identity.key`), and the
> `.preflayer/` home are **git-ignored** — never commit them, and never paste a
> credential or token into an issue or PR.

### Option B — hosted v0.1 API

If the team has provisioned you a hosted endpoint, you will receive a base URL
and an agent token out of band. The three PTP endpoints are:

| Endpoint | Use |
|----------|-----|
| `GET /preference` | read the user's query-scoped preference subgraph + confidence (call **before** ranking) |
| `POST /outcome` | report a purchase/return/rating so the credential improves (call **after** a transaction) |
| `POST /elicit` | get ≤5 high-information-gain questions when confidence is low |

Full request/response shapes are in [`protocol-spec.md`](protocol-spec.md).

### Seeding a credential with some history

A brand-new credential is cold (low confidence) — which is a valid condition to
measure, but to see the credential *help* you usually want it to carry some
signal. Replay a few of the user's past outcomes:

```jsonc
// POST /outcome (or submit_outcome MCP tool), once per past interaction
{
  "category": "laptops",
  "product_id": "thinkpad-x1-carbon-g11",
  "outcome_type": "purchase",      // purchase | return | dwell | rating | elicitation
  "use_context": "professional"
}
```

Updates are differentially private and computed on-device; raw behavioral data
never leaves the store.

---

## 2. Integrate the MCP tools

Both layers are exposed as MCP tools with descriptions written for agent
self-selection. Register both servers with your agent framework (Claude Agent
SDK, LangChain, etc.).

**PTP tools** (`preferencelayer.mcp.server`):

| Tool | When the agent should call it |
|------|-------------------------------|
| `get_preference` | **before** ranking/recommending, to load the user's preferences |
| `submit_outcome` | **after** a transaction or significant interaction |
| `request_elicitation` | when `get_preference` reports low confidence |

**QIL tools** (`preferencelayer.qil.mcp_server`):

| Tool | When the agent should call it |
|------|-------------------------------|
| `get_quality` | to fetch use-profile-conditioned quality + failure rate for a product |
| `compare_quality` | to compare two products for a given use profile |

`get_preference` returns, among other fields, a **`confidence`** (the credential's
mean node confidence) — keep it; it drives the blend weight α in step 3. The
harness parameter is named `mean_confidence`, so map it explicitly:

```python
response = get_preference(...)        # PTP get_preference / GET /preference
mean_confidence = response["confidence"]
```

### Wiring it to the ranking code

The repo ships the orchestration so you do not have to re-derive the blend.
`AgentRecommender` (`preferencelayer.agent.recommender`) calls a fitted
preference model and a `QualityService`, then ranks by the α-blend:

```python
from preferencelayer.agent.recommender import AgentRecommender

agent = AgentRecommender(
    pref_model,        # a fitted Recommender built from the credential's graph
    pref_state,        # the per-user state for this credential
    quality_service,   # a QualityService wrapping the QIL /quality endpoint
    n_shared,          # width of the shared attribute block
)

result = agent.rank(
    candidate_ids, candidate_attrs, use_profile="professional",
    mean_confidence=mean_confidence,   # = get_preference response["confidence"]
)
ranking = [candidate_ids[i] for i in result.order]
```

Converting a `get_preference` response into `pref_model` + `pref_state`: the
credential's `attributeNodes` / `edgeWeights` are exactly the graph the
preference model scores with. See `preferencelayer.agent.protocol`
(`credential_from_arrays`, `score_from_credential`) for the round-trip used in
the integration tests — that is the canonical example to copy.

> If converting the credential into a preference model is more wiring than you
> want for a first pass, you can still run the measurement: supply a
> `before_ranker` that reflects your *current* production ranking and let the
> harness measure the after-credential blend against it (see step 3).

---

## 3. Run the measurement harness

The harness lives at `preferencelayer.eval.partner`. You describe your task as a
list of `PartnerQuery` cases — **one per ranking decision your agent makes** —
and it runs two conditions on the *same* candidate sets:

- **before** — no credential: a credential-less, quality-only cold-start ranking
  (or your own production baseline, if you pass `before_ranker`);
- **after** — the credential-blended ranking.

It reports **NDCG@10**, recall@10, and MRR for each condition, the per-query
deltas, and a paired-bootstrap p-value.

### Build your task

```python
import numpy as np
from preferencelayer.eval.partner import (
    PartnerQuery, measure_partner, partner_improved,
)

queries = [
    PartnerQuery(
        query_id="session-001",
        candidate_ids=["lap-a", "lap-b", "lap-c", "lap-d"],
        candidate_attrs=np.array([...]),     # (n_candidates, dim), same schema as the credential
        relevant_ids=["lap-c"],              # what the user actually wanted/bought/clicked
        use_profile="professional",          # the use profile you query the QIL with
        # relevance_map={"lap-c": 3.0, "lap-a": 1.0},  # optional graded relevance
    ),
    # ... one PartnerQuery per ranking decision; more queries => a more reliable p-value
]
```

**Where does ground truth come from?** Use a held-out slice of your users'
real behavior: rank the candidate set as it was at decision time, and treat the
item the user actually chose (purchased / clicked / kept) as relevant. Do **not**
include the chosen item's outcome in the credential you measure with, or you
leak the label.

### Measure and check the gate

```python
result = measure_partner(
    partner_id="acme-agent",
    agent=agent,                 # the AgentRecommender from step 2
    queries=queries,
    mean_confidence=mean_confidence,   # = get_preference response["confidence"]
    k=10,
)

print(f"before NDCG@10: {result.before.ndcg:.3f}")
print(f"after  NDCG@10: {result.after.ndcg:.3f}")
print(f"abs gain: {result.abs_gain:+.3f}  ({result.rel_gain_pct:+.1f}%)  p={result.p_value:.3f}")
print("measurable improvement?", partner_improved(result))
```

`partner_improved(result)` is the per-partner verdict that feeds the gate:
**`True`** when the after-credential NDCG@10 beats the before condition **and**
the improvement is significant (paired bootstrap, p < 0.05). The default counts
*any* significant positive gain; pass `min_abs_gain=0.02` to require a
practically meaningful lift instead.

### Using your own production baseline

If "no credential" for you does not mean "quality-only", pass a `before_ranker`
that returns candidate indices best-first under your current system:

```python
def my_baseline(candidate_ids, candidate_attrs, use_profile):
    # return indices into candidate_ids, best first
    return my_production_ranker(candidate_ids)

result = measure_partner(..., before_ranker=my_baseline)
```

### How the cohort gate is computed

The team aggregates every partner's `PartnerResult` and calls `gate_passed`:

```python
from preferencelayer.eval.partner import gate_passed

report = gate_passed([result_partner_1, result_partner_2, ...])  # required=2 by default
print(report.passed, report.n_improved, "of", report.n_partners)
print("improved:", report.improved_partners)
```

`gate_passed(..., required=2)` implements the Phase 1 WS-C criterion verbatim:
**≥ 2 partners show a measurable improvement → gate passed → proceed to Phase 2.**

---

## Reproducibility

Per [`CONTRIBUTING.MD`](../CONTRIBUTING.MD), when you report a measurement,
record:

- the credential's `confidence` (the `mean_confidence` you passed) and how it was
  seeded (which outcomes);
- the number of queries and how ground truth was derived;
- the metric (`k`) and the `partner_improved` thresholds you used;
- the `bootstrap_seed` (default `0`) so the p-value is reproducible.

The harness is deterministic given the same inputs and seed.

---

## What we measure from you (and what we do not)

- We collect your **aggregate `PartnerResult`** (NDCG/recall/MRR per condition,
  the deltas, the p-value) and qualitative notes on integration friction.
- We do **not** want your users' raw behavioral data, candidate attributes, or
  credentials. Run the harness on your side; share the summary numbers.

---

## Getting set up with the team (human steps)

These steps are **not** automated and are handled by the PreferenceLayer team,
not by this tooling:

- **Recruiting** you into the program and confirming fit (open-source / startup /
  hobbyist MCP agents — not platform incumbents).
- **Granting** hosted v0.1 API access / provisioning a base URL and token, if you
  are not running the local store.
- **The private feedback channel** (Slack) and the commitment to prioritize the
  bugs and API friction you report.
- **Reviewing** your reported measurement and folding it into the cohort gate.

Reach out through the channel the team shared with you to start any of the above.
