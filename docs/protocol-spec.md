# Preference Transport Protocol (PTP)

## Specification v0.1 (credential schema frozen)

**Status:** v0.1 — credential schema **frozen** (June 2026). The per-credential
schema in §3 is published as a JSON-LD `@context` (`contexts/ptp-v1.jsonld`) and a
JSON Schema (`contexts/ptp-credential-v0.1.schema.json`), validated in CI. The API
(§4), auth (§4.1), and §8 open questions remain in flux. See issue #10 for the
freeze rationale and the §8 impact assessment (Q4 export-bundle format is left
unfrozen for v0.1).  
**Authors:** PreferenceLayer Contributors  
**Last Updated:** June 2026

---

## Abstract

The Preference Transport Protocol (PTP) defines a schema and API for portable, user-controlled preference credentials. A PTP credential encodes a compact representation of a user's latent preference distribution over a product attribute space. Agents read credentials to personalize recommendations; they write outcome signals to update credentials over time. The protocol is designed to be MCP-native and platform-agnostic.

---

## 1. Motivation

AI shopping agents require preference signals to make useful recommendations. Current approaches store these signals in platform-controlled infrastructure, making them inaccessible to agents operating outside the originating platform. PTP defines a portable credential format and transport protocol that allows any authorized agent to access a user's preference state, regardless of where the user's previous interactions occurred.

---

## 2. Terminology

- **Credential:** A signed document encoding a user's preference graph for one or more product categories.
- **Credential Store:** A user-controlled service that holds, updates, and selectively discloses the credential.
- **Agent:** Any software system that requests a preference credential or submits outcome signals.
- **Outcome Signal:** A structured record of a user's purchasing decision, return, or explicit preference response.
- **Attribute Node:** A node in the preference graph representing a product attribute dimension.
- **Context Conditioner:** A function that activates a subgraph of the preference graph given a query context.
- **Selective Disclosure:** The process of exposing only a subset of the credential relevant to a given query.

---

## 3. Credential Schema

### 3.1 Outer Envelope

PTP credentials follow the W3C Verifiable Credentials Data Model 2.0. The `credentialSubject` carries the preference graph payload.

Required outer fields:

| Field | Type | Description |
|-------|------|-------------|
| `@context` | array | Must include W3C VC context and PTP context URI |
| `type` | array | Must include `"VerifiableCredential"` and `"PreferenceCredential"` |
| `issuer` | string | User's DID (RECOMMENDED: `did:key`) |
| `issuanceDate` | string | ISO 8601 datetime |
| `credentialSubject` | object | Preference graph payload (see 3.2) |
| `proof` | object | Linked data proof (REQUIRED: Ed25519Signature2020) |

### 3.2 Credential Subject

The `credentialSubject` object contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | User DID (matches `issuer`) |
| `preferenceGraph` | object | Yes | The preference graph (see 3.3) |

### 3.3 Preference Graph Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `category` | string | Yes | Product category (e.g., `"laptops"`, `"headphones"`) |
| `version` | string | Yes | PTP schema version (currently `"0.1"`) |
| `attributeNodes` | array | Yes | List of attribute node objects (see 3.4) |
| `edges` | array | Yes | List of edge objects (see 3.5) |
| `contextConditioners` | array | No | Context-to-subgraph mappings (see 3.6) |
| `updateCount` | integer | Yes | Number of outcome signals applied |
| `privacyBudgetConsumed` | float | Yes | Cumulative DP budget consumed (ε) |
| `lastUpdated` | string | Yes | ISO 8601 datetime of last update |
| `coldStartPrior` | string | No | Identifier of population prior used at initialization |

### 3.4 Attribute Node Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier within the graph (e.g., `"price_sensitivity"`) |
| `weight` | float | Yes | Preference weight in [-1, 1]. Positive = higher is better. |
| `confidence` | float | Yes | Confidence in the weight estimate, in [0, 1] |
| `embedding` | array | No | Float array; latent representation for similarity computation |

### 3.5 Edge Object

Edges encode conditional dependencies between attribute preferences.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Source attribute node ID |
| `target` | string | Yes | Target attribute node ID |
| `weight` | float | Yes | Edge weight in [-1, 1]. Negative = tradeoff. |
| `contextKey` | string | No | If present, edge is only active when this context key is matched |

### 3.6 Context Conditioner Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `contextKey` | string | Yes | Context label (e.g., `"professional_use"`, `"travel"`) |
| `activeNodes` | array | Yes | List of attribute node IDs activated by this context |
| `suppressedNodes` | array | No | List of attribute node IDs suppressed by this context |

---

## 4. API Specification

All endpoints use HTTPS. All requests require a valid agent access token in the `Authorization` header.

### 4.1 Authentication

Agents authenticate via OAuth 2.0 Device Authorization Grant (RFC 8628). The credential store issues scoped access tokens. Tokens carry a `scope` claim indicating which credential categories the agent is authorized to access.

Token expiry: 24 hours. Refresh tokens are not issued; agents must re-authenticate after expiry.

### 4.2 GET /preference

Retrieve a scoped preference credential for the authenticated agent.

**Request:**

```
GET /preference
Authorization: Bearer <agent-token>
Content-Type: application/json

{
  "category": string,            // Required. Product category.
  "query_context": string,       // Recommended. Free-text or embedding of current query.
  "disclosure_scope": [string],  // Optional. Limit returned nodes to this list.
  "min_confidence": float        // Optional. Only return nodes above this confidence threshold.
}
```

**Response 200:**

```json
{
  "credential": { ... },          // Signed, scoped PTP credential
  "confidence": float,            // Mean confidence across returned nodes
  "coverage": [string],           // Attribute node IDs included in response
  "missing": [string],            // Requested nodes not available (below confidence or absent)
  "elicitation_recommended": bool // True if confidence is below useful threshold
}
```

**Response 404:** No credential exists for this category. Agent should call `POST /elicit` to initialize.

**Response 403:** Agent token does not have scope for this category.

### 4.3 POST /outcome

Submit an outcome signal for asynchronous processing.

**Request:**

```
POST /outcome
Authorization: Bearer <agent-token>
Content-Type: application/json

{
  "product_id": string,           // Required. Canonical product identifier.
  "outcome_type": string,         // Required. One of: purchase | return | dwell | rating | elicitation
  "use_context": string,          // Recommended. Description of use context.
  "timestamp": string,            // Required. ISO 8601.
  "rating": float,                // Optional. Explicit rating in [0, 1]. Only for outcome_type=rating.
  "elicitation_response": object  // Optional. Response to /elicit question. Only for outcome_type=elicitation.
}
```

**Response 202:** Signal accepted for async processing.

```json
{
  "signal_id": string,
  "update_queued": true,
  "estimated_processing": "async"
}
```

Signals are processed asynchronously. There is no guarantee of processing order for concurrent signals.

### 4.4 POST /elicit

Request an active elicitation sequence for low-confidence attribute nodes.

**Request:**

```
POST /elicit
Authorization: Bearer <agent-token>
Content-Type: application/json

{
  "category": string,             // Required.
  "attribute_focus": [string],    // Optional. Specific attribute IDs to target.
  "max_questions": integer        // Optional. Default 3, maximum 5.
}
```

**Response 200:**

```json
{
  "session_id": string,
  "questions": [
    {
      "id": string,
      "text": string,
      "response_schema": {
        "type": "categorical" | "ordinal" | "boolean" | "free_text",
        "options": [string],       // For categorical/ordinal
        "scale": [float, float]    // For ordinal: [min, max]
      },
      "target_attribute": string,
      "information_gain": float    // Expected IG in bits; questions ordered descending
    }
  ]
}
```

Elicitation responses are submitted via `POST /outcome` with `outcome_type: elicitation` and the `elicitation_response` field containing `{"session_id": ..., "question_id": ..., "response": ...}`.

---

## 5. Update Protocol

Updates are computed on-device to ensure raw behavioral data does not leave user control.

### 5.1 Update Trigger

An update is triggered when the credential store receives a `POST /outcome` signal. The update is queued and processed asynchronously.

### 5.2 Gradient Computation

1. Map `product_id` and `use_context` to a set of affected attribute nodes using the category's attribute mapping function.
2. For each affected node, compute a gradient signal from the outcome:
   - `purchase`: positive gradient on high-weight nodes for matching attributes
   - `return`: negative gradient on high-weight nodes for matching attributes
   - `dwell`: small positive gradient (lower magnitude than purchase)
   - `rating`: gradient proportional to (rating - 0.5), both directions
   - `elicitation`: direct weight update from stated preference

### 5.3 Differential Privacy Mechanism

```
For each affected node weight w_i:
  1. Compute gradient g_i
  2. Clip: g_i = g_i / max(1, ||g_i|| / C)    where C = 1.0 (sensitivity bound)
  3. Add noise: g_i += N(0, σ²I)
     where σ = C * sqrt(2 * ln(1.25/δ)) / ε    (ε = 2.0, δ = 1e-5)
  4. Apply: w_i += η * g_i    (η = 0.01 learning rate)
  5. Clip weight to [-1, 1]
  6. Update confidence: c_i = c_i + η_c * (1 - c_i)    (η_c = 0.02)
```

### 5.4 Budget Tracking

`privacyBudgetConsumed` is incremented by ε_step per update. When the budget exceeds a configurable maximum (default: ε_max = 20.0), further updates are paused and the user is prompted to consent to a budget reset. This is a soft limit; users may override.

### 5.5 Re-signing

After each update, the credential is re-signed with the user's private key. The `lastUpdated` and `updateCount` fields are incremented. If cloud sync is enabled, the updated credential is encrypted with the user's public key and the ciphertext is pushed to the sync endpoint.

---

## 6. MCP Integration

PTP exposes the three API endpoints as MCP tools. The MCP server descriptor:

```json
{
  "name": "preferencelayer-ptp",
  "version": "0.1",
  "tools": [
    {
      "name": "get_preference",
      "description": "Retrieve a user's preference credential for a product category. Use this before ranking or recommending products to personalize results based on the user's known preferences.",
      "inputSchema": { ... }
    },
    {
      "name": "submit_outcome",
      "description": "Submit a purchase, return, or interaction signal to update the user's preference model. Call this after a transaction or significant interaction.",
      "inputSchema": { ... }
    },
    {
      "name": "request_elicitation",
      "description": "Request clarifying questions to improve preference model confidence for specific attributes. Use when the preference credential has low confidence and the user is available for a brief interaction.",
      "inputSchema": { ... }
    }
  ]
}
```

---

## 7. Versioning & Compatibility

- Breaking changes to the credential schema increment the major version.
- Additive changes (new optional fields) increment the minor version.
- Credential stores must reject credentials with a schema version they do not support.
- Agents should degrade gracefully when a credential is absent or below confidence threshold (fall back to cold-start population prior or explicit elicitation).

---

## 8. Open Issues

The following design questions are not yet resolved. Contributions and discussion welcome via GitHub Issues.

1. **DID method selection.** `did:key` is simplest for v0 but may not be suitable for credential rotation. `did:web` or `did:ion` offer rotation but add infrastructure complexity. What is the right tradeoff?

2. **Cross-category credential merge.** When a user has credentials for multiple categories, can preference signals from one category inform another (e.g., durability preference learned from laptop purchasing informs headphone recommendations)? This requires a cross-category attribute mapping scheme not yet defined.

3. **Multi-user household credentials.** How should shared purchasing accounts (families, businesses) be handled? Separate credentials per user, or a shared credential with contributor tagging?

4. **Credential portability format.** Should users be able to export their credential in a human-readable format? If so, what schema? This affects the privacy-portability tradeoff.

5. **Agent trust tiers.** Should all authorized agents receive the same scoped credential, or should there be trust tiers (e.g., a highly trusted agent gets more attribute nodes, a less trusted agent gets fewer)? How would trust be established?
