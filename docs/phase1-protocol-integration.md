# Phase 1: Protocol-Level Integration (PTP + QIL over MCP)

**An agent that ranks products using only the two MCP tools — no in-process model**
**Status:** demonstrated end-to-end, including auth / selective disclosure / revocation

---

## Summary

Every result so far ranked with an *in-process* preference model — the harness holds
the fitted `SparsePreferenceGraph` object and calls its `score` method directly. A
real shopping agent never has that object. It has two **MCP tools**: PTP
`get_preference` (returns a signed, selectively-disclosed *preference credential*)
and QIL `get_quality` (returns use-profile quality posteriors). This milestone builds
the agent that lives on the far side of those tools and shows the integration works
over the real protocol surfaces.

The agent (`preferencelayer.agent.protocol.ProtocolAgent`):

1. calls **PTP `get_preference`** → receives a credential graph (attribute weights +
   interaction edges) and a confidence score;
2. **reconstructs a preference score** for each candidate *from the disclosed
   credential* — `Σ wᵢ·xᵢ + Σ w_edge·xₐ·x_b` — proving the credential is a sufficient,
   portable carrier of preference (no model object crosses the boundary);
3. calls **QIL `get_quality`** per candidate → a quality score;
4. blends them with the documented confidence-adaptive α =
   `sigmoid(3·(confidence − 0.5))` and ranks.

**It works.** On the integrated benchmark, building each user's credential from their
preference, serving it through the real `CredentialStore`/`PTPToolHandler` (signed,
scoped, re-signed on disclosure) and querying `QILToolHandler`, the agent ranks at:

| Path (300 users, seed 23) | NDCG@10 |
|---|---:|
| protocol blend (credential preference + QIL quality) | **0.81** |
| protocol preference-only (credential, no quality) | 0.63 |

Quality adds **+0.19 NDCG@10** over preference alone — the same "combining wins"
result as the in-process integration, now reproduced purely over the tool handlers.
And the protocol boundary is real: revoking the agent's token makes the next
`get_preference` return **403**, and the agent surfaces that (no ranking) rather than
guessing.

```
$ preflayer protocol-demo
get_preference -> confidence=0.70 coverage=['performance', 'portability', 'price_sensitivity']
blended with alpha = sigmoid(3*(confidence-0.5)) = 0.65
product         pref  quality  blended
workhorse       0.73     0.76     0.97
ultrabook       0.68     0.54     0.36
budget          0.21     0.35    -1.33
Top recommendation: workhorse
After revocation: get_preference denied -> status 403, no ranking produced.
```

---

## Scope and honesty notes

- **This validates transport + composition, not learning.** The credential here
  carries the user's *true* preference weights (a perfectly-learned credential), so
  the absolute NDCG (0.81) is **not** comparable to the in-process integration
  (~0.61), which ranks from a model *fit* on sparse history. The point being proven is
  that the PTP credential is a sufficient carrier and that the two MCP tools compose
  into a correct ranking — not that the protocol path is "better."
- **The credential→score reconstruction is the agent's own.** It applies the
  published node/edge weights to the candidate attribute vectors directly (raw, not
  standardized) — the same linear+interaction form the graph model uses. A node or
  edge naming an attribute outside the agent's schema is skipped (credentials may
  disclose a subset).
- **Real protocol behavior is exercised**, not stubbed: bearer-token auth + category
  scoping, selective disclosure (`disclosure_scope`), and Ed25519 re-signing of the
  scoped credential all run on every call via `CredentialStore`. Only the MCP
  *transport* (stdio/JSON-RPC) is bypassed — the agent calls `PTPToolHandler.call`
  / `QILToolHandler.call` directly, which is the same logic `build_server` wraps.
- Synthetic benchmark with planted ground truth, consistent with the other Phase 1
  studies; not a live-data measurement.

---

## Method

`preferencelayer.agent.protocol`:

- `score_from_credential(credential, candidate_attrs, schema)` — reconstructs the
  preference score from a disclosed `PreferenceCredential` graph.
- `quality_from_response(response, ...)` — collapses a `get_quality` response to one
  score (mean posterior, optional failure discount, neutral fallback on 404).
- `ProtocolAgent(ptp_handler, qil_handler, schema).recommend(...)` — orchestrates the
  two tool calls and the blend; returns the ranking plus the component scores, α,
  disclosed coverage, and the PTP status (e.g. 403/404 when preference is
  unavailable).
- `credential_from_arrays(schema, theta, phi_pairs, phi, ...)` — the bridge a client
  uses to *export* a fitted/planted preference into a portable PTP credential; lets
  the protocol path be driven from the same parameters the benchmark plants.

Reuses `PTPToolHandler` (`mcp/server.py`), `QILToolHandler` (`qil/mcp_server.py`),
`CredentialStore` (`ptp/store.py`), and the blend math in `agent/combine.py`.

---

## Reproducing

```bash
preflayer protocol-demo                       # end-to-end ranking + revocation, in ~20 lines of output
python -m pytest tests/test_agent_protocol.py # unit + credential-roundtrip regression
```

---

## Status

| Stage | Result |
|-------|--------|
| Integration — α-blend beats either layer alone (in-process) | **+39% / +134%** ([report](phase1-integration-results.md)) |
| Integration — adaptive α beats fixed α | **No** — fixed α is robust ([report](phase1-integration-results.md)) |
| Quality handling — shrinkage vs. raw averaging | **Crossover**; shrinkage is noise-robust ([report](phase1-quality-robustness-results.md)) |
| **Protocol — rank over the real PTP + QIL MCP tools** | **Works** end-to-end; quality adds +0.19 NDCG@10; auth/revocation enforced (this report) |

The pieces compose not just as Python calls but **over the agent-facing protocol**:
a credential read with `get_preference` carries enough to rank, `get_quality` adds the
quality signal, and the user's revocation is honored at the boundary.
