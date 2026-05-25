# Architecture Reference

## PreferenceLayer System Design

---

## Overview

PreferenceLayer has two independently deployable components that combine at query time:

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Runtime                         │
│   (LangChain / Claude SDK / AutoGPT / custom MCP agent)     │
└──────────────┬─────────────────────────┬────────────────────┘
               │ MCP tool call           │ MCP tool call
               ▼                         ▼
┌──────────────────────┐   ┌─────────────────────────────────┐
│   PTP MCP Server     │   │        QIL MCP Server           │
│  (preference layer)  │   │   (quality intelligence layer)  │
└──────────┬───────────┘   └──────────────┬──────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐   ┌─────────────────────────────────┐
│  Credential Store    │   │       QIL Database              │
│  (user-controlled,   │   │  (proprietary, server-side,     │
│   on-device/cloud)   │   │   continuously updated)         │
└──────────────────────┘   └─────────────────────────────────┘
```

The agent calls both MCP servers, receives preference scores and quality scores, and combines them using the learned blending weight α.

---

## Component 1: Preference Transport Protocol (PTP)

### Credential Schema

The preference credential is a W3C Verifiable Credential. Core fields:

```json
{
  "@context": [
    "https://www.w3.org/ns/credentials/v2",
    "https://preferencelayer.io/context/v1"
  ],
  "type": ["VerifiableCredential", "PreferenceCredential"],
  "issuer": "did:key:<user-public-key>",
  "issuanceDate": "2026-01-15T00:00:00Z",
  "credentialSubject": {
    "id": "did:key:<user-public-key>",
    "preferenceGraph": {
      "category": "laptops",
      "version": "0.1",
      "attributeNodes": [
        {
          "id": "price_sensitivity",
          "weight": 0.72,
          "confidence": 0.85,
          "embedding": [0.12, -0.34, ...]
        }
      ],
      "edges": [
        {
          "source": "price_sensitivity",
          "target": "build_quality",
          "weight": -0.31,
          "contextKey": "professional_use"
        }
      ],
      "updateCount": 47,
      "privacyBudgetConsumed": 1.4,
      "lastUpdated": "2026-05-10T14:22:00Z"
    }
  },
  "proof": {
    "type": "Ed25519Signature2020",
    "verificationMethod": "did:key:<user-public-key>#key-1",
    "proofValue": "<base64url-signature>"
  }
}
```

### Credential Store

The credential store runs as a local daemon on the user's device. It handles:
- Credential storage (SQLite, AES-256 encrypted at rest)
- Agent authentication (OAuth 2.0 device flow)
- Selective disclosure (scoping the credential to a query context)
- On-device update computation (DP gradient updates)
- Optional cloud sync (client-side encrypted)

**Agent authorization model:** Agents are identified by a registered `agent_id`. Users authorize agents explicitly (via CLI or web UI) and can revoke at any time. Each agent gets a scoped access token that expires.

### API Endpoints

#### `GET /preference`

```
Request:
  Authorization: Bearer <agent-token>
  Body: {
    "category": "laptops",
    "query_context": "sustained ML workload, portability important",
    "disclosure_scope": ["price_sensitivity", "thermal_tolerance", "weight_preference"]
  }

Response 200:
  {
    "credential": { ...signed scoped VC... },
    "confidence": 0.78,
    "coverage": ["price_sensitivity", "thermal_tolerance"],
    "missing": ["weight_preference"],
    "elicitation_recommended": false
  }
```

#### `POST /outcome`

```
Request:
  Authorization: Bearer <agent-token>
  Body: {
    "product_id": "lenovo-thinkpad-x1-carbon-gen12",
    "outcome_type": "purchase",
    "use_context": "software development, frequent travel",
    "timestamp": "2026-05-15T10:30:00Z"
  }

Response 202:
  { "update_queued": true, "estimated_processing": "async" }
```

#### `POST /elicit`

```
Request:
  Authorization: Bearer <agent-token>
  Body: {
    "attribute_focus": ["thermal_tolerance", "keyboard_preference"],
    "max_questions": 3
  }

Response 200:
  {
    "questions": [
      {
        "id": "q1",
        "text": "How often do you run tasks that stress the CPU for more than 30 minutes?",
        "response_schema": {
          "type": "categorical",
          "options": ["rarely", "sometimes", "frequently", "constantly"]
        },
        "target_attribute": "thermal_tolerance",
        "information_gain": 0.41
      }
    ]
  }
```

### On-Device Update Protocol

When an outcome signal arrives:

1. Identify affected attribute nodes (use context → attribute mapping)
2. Compute gradient: ∂loss/∂W for the affected subgraph
3. Clip gradient to norm bound C (default C = 1.0)
4. Add Gaussian noise: N(0, σ²) where σ = C·√(2ln(1.25/δ)) / ε (ε=2, δ=1e-5)
5. Apply clipped + noised gradient to node weights
6. Update `privacyBudgetConsumed += ε_step`
7. Re-sign credential with user's private key
8. If cloud sync enabled, re-encrypt and push ciphertext

---

## Component 2: Quality Intelligence Layer (QIL)

### Data Model

```sql
-- Core signal table
CREATE TABLE product_signal (
  id              BIGSERIAL PRIMARY KEY,
  product_id      TEXT NOT NULL,
  model_normalized TEXT NOT NULL,
  category        TEXT NOT NULL,
  failure_mode    TEXT,
  quality_dim     TEXT,
  use_profile     TEXT NOT NULL,
  signal_type     TEXT NOT NULL,  -- 'failure' | 'performance' | 'comparison'
  signal_value    FLOAT,          -- normalized quality score if quantifiable
  source_url      TEXT,
  source_type     TEXT,           -- 'reddit' | 'ifixit' | 'notebookcheck' | 'return_data'
  extracted_at    TIMESTAMPTZ,
  model_confidence FLOAT,
  upvote_count    INT DEFAULT 0
);

-- Posterior parameters (refit nightly)
CREATE TABLE quality_posterior (
  product_id      TEXT NOT NULL,
  use_profile     TEXT NOT NULL,
  quality_dim     TEXT NOT NULL,
  posterior_mean  FLOAT NOT NULL,
  posterior_std   FLOAT NOT NULL,
  credible_lo_90  FLOAT NOT NULL,
  credible_hi_90  FLOAT NOT NULL,
  evidence_count  INT NOT NULL,
  freshness_score FLOAT NOT NULL,  -- decays with signal age
  last_refit      TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (product_id, use_profile, quality_dim)
);
```

### Ingestion Pipeline

```
Reddit API ──────────┐
iFixit crawler ───────┤──► Raw text ──► NLP extraction ──► Structured records
Notebookcheck ────────┤                    │
Retailer return data ─┘                    │
                                           ▼
                                    Normalization
                                    (product ID, use profile, failure mode)
                                           │
                                           ▼
                                    product_signal table
                                           │
                               (nightly batch)
                                           ▼
                                    Bayesian aggregation
                                           │
                                           ▼
                                    quality_posterior table
```

**NLP extraction pipeline steps:**
1. Entity recognition: product model name (fuzzy-matched to canonical model list)
2. Failure mode classification: categorical classifier (trained on annotated corpus)
3. Use profile classification: categorical classifier
4. Signal type classification: failure report / performance observation / comparison
5. Confidence scoring: model confidence × source reliability weight

### Bayesian Aggregation

**Failure rate (Beta-Binomial):**

```
Prior: Beta(α₀, β₀) — estimated from category-level base rates
Update: for each failure signal, increment α; for each non-failure signal, increment β
Posterior: Beta(α₀ + failures, β₀ + non-failures)
Point estimate: posterior mean = α / (α + β)
Credible interval: posterior quantiles
```

Hierarchical: category-level prior pools information across products with sparse data.

**Quality dimensions (Gaussian Process):**

```
Kernel: squared-exponential over (release_date, use_profile_embedding)
Observations: normalized quality scores from NLP-extracted signals
Posterior: GP posterior mean and variance at query point
```

Used for quality dimensions that vary continuously (thermal performance degrades with thermal paste age; battery capacity degrades with cycle count).

### API Endpoints

#### `POST /quality`

```
Request:
  Body: {
    "product_id": "lenovo-thinkpad-x1-carbon-gen12",
    "use_profile_vector": {
      "workload_intensity": "high",
      "primary_use": "software_development",
      "portability_priority": "high"
    },
    "dimensions": ["thermal", "battery_longevity", "build_quality"]
  }

Response 200:
  {
    "product_id": "lenovo-thinkpad-x1-carbon-gen12",
    "use_profile": "high_intensity_portable_professional",
    "quality_scores": {
      "thermal": {
        "posterior_mean": 0.61,
        "credible_interval_90": [0.48, 0.74],
        "interpretation": "Moderate thermal performance under sustained load; throttling reported in 38% of high-intensity use cases"
      },
      "battery_longevity": {
        "posterior_mean": 0.79,
        "credible_interval_90": [0.68, 0.88],
        "interpretation": "Strong battery longevity; minimal degradation at 18-month mark across observed units"
      },
      "build_quality": {
        "posterior_mean": 0.84,
        "credible_interval_90": [0.76, 0.91],
        "interpretation": "High build quality; low structural failure rate across use profiles"
      }
    },
    "failure_rate": {
      "posterior_mean": 0.07,
      "credible_interval_90": [0.03, 0.13]
    },
    "data_quality": {
      "evidence_count": 847,
      "freshness_score": 0.91,
      "profile_specificity": 0.73
    },
    "evidence_pointers": [
      "https://reddit.com/r/thinkpad/comments/...",
      "https://www.notebookcheck.net/..."
    ]
  }
```

#### `POST /compare`

```
Request:
  Body: {
    "product_id_a": "lenovo-thinkpad-x1-carbon-gen12",
    "product_id_b": "dell-xps-15-9530",
    "use_profile_vector": { "workload_intensity": "high", ... }
  }

Response 200:
  {
    "comparison": {
      "thermal": {
        "posterior_difference": -0.18,
        "credible_interval_90": [-0.31, -0.05],
        "probability_a_better": 0.12,
        "interpretation": "Dell XPS 15 substantially better thermal performance under high-intensity workloads"
      },
      "battery_longevity": {
        "posterior_difference": 0.11,
        "credible_interval_90": [-0.02, 0.24],
        "probability_a_better": 0.83,
        "interpretation": "ThinkPad X1 likely better battery longevity; moderate confidence"
      }
    }
  }
```

---

## Combined Scoring

The scoring function used by agents integrating both components:

```python
def score_candidate(
    item: Product,
    preference_credential: PreferenceCredential,
    qil_report: QualityReport,
    alpha: float  # learned per-user; low for sparse credentials
) -> float:
    pref_score = compute_preference_score(item, preference_credential)
    quality_score = compute_quality_score(item, qil_report)
    return alpha * pref_score + (1 - alpha) * quality_score

def compute_alpha(credential: PreferenceCredential) -> float:
    # Blend toward quality when credential confidence is low
    mean_confidence = mean([n.confidence for n in credential.attribute_nodes])
    return sigmoid(3.0 * (mean_confidence - 0.5))  # 0.5 at confidence=0.5
```

---

## Deployment Topology

**Phase 1 (minimal):**
- PTP server: single VPS or small cloud instance. Stateless; credential store is client-side.
- QIL API: single server + PostgreSQL. Ingestion pipeline as cron jobs.
- MCP servers: same host as API servers.

**Phase 2+ (production):**
- PTP: stateless API behind load balancer; credential store remains client-side by design.
- QIL API: horizontally scaled read replicas; write path (ingestion + nightly refit) on separate compute.
- Ingestion pipeline: managed workflow (Airflow or Prefect); separate from API serving.
- CDN for MCP server discovery documents.

---

## Security Considerations

- **Credential integrity:** Ed25519 signatures; tampering detectable at verification.
- **Agent access control:** Per-agent scoped tokens; revocable; expiring.
- **Cloud sync:** Client-side encryption; server cannot read plaintext credential.
- **QIL privacy:** No user identifiers in the QIL database; only product and use profile signals.
- **DP budget tracking:** `privacyBudgetConsumed` tracked per credential; when budget exhausted, updates pause until user consents to budget reset.
- **Rate limiting:** Per-agent-token rate limits on all endpoints; DDoS protection at the CDN layer.
