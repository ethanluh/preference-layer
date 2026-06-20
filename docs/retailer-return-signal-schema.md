# Retailer Return-Signal Schema (v0.1 draft)

**Status:** draft for ratification. Tracking issue:
[#14](https://github.com/ethanluh/preference-layer/issues/14) (`data`). Per
`CONTRIBUTING.MD`, this schema is being discussed in that issue thread before it
is treated as frozen.

This is the **anonymized return-signal schema** a retailer shares under the
[data-sharing agreement template](retailer-data-sharing-agreement.md). It is the
field-level companion (Exhibit A) to that agreement and operationalizes the
Phase 1 / critical-path retailer requirement in
[`implementation-plan.md`](implementation-plan.md):

> Return signal data sharing agreement requirements:
> - Anonymized at source (no PII)
> - Product ID + return reason + use context only
> - Retailer receives QIL API credits in exchange

---

## Design constraints (non-negotiable)

These come from the architecture invariants in
[`CLAUDE.md`](../CLAUDE.md) / [`architecture.md`](architecture.md):

1. **No PII, anonymized at source.** The retailer strips all PII before any
   record leaves their systems. No customer id, order id, name, email, address
   (below country), IP, device id, payment data, exact timestamp, or any field
   that could re-identify a person directly or by join.
2. **Minimal columns.** Only the fields below. Nothing order- or customer-level.
3. **QIL holds no user identifiers.** The schema makes it structurally impossible
   to land one — there is no per-person, per-order, or per-session field.
4. **Use-profile-conditioned, never population aggregates.** `use_context` maps
   onto the QIL `use_profile` vocabulary so returns feed conditioned posteriors.

---

## Schema

One flat, append-only record per **anonymized, k-anonymized return cell**. The
record is *not* per individual return event — see [k-anonymity](#k-anonymity-floor):
the smallest shareable unit is a count over a (`product_id`, `return_reason`,
`use_context`, `category`, `return_window_bucket`) group.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `product_id` | TEXT | yes | Canonical model id, normalized to the QIL `product_signal.product_id` keyspace (e.g. `lenovo-thinkpad-x1-carbon-gen12`). Retailer-internal SKUs must be mapped to this canonical id before sharing. |
| `return_reason` | TEXT (controlled vocab) | yes | Why the item was returned. Controlled vocabulary below. |
| `use_context` | TEXT (controlled vocab) | yes | How the product was used. Maps onto the QIL `use_profile` vocabulary; `unknown` is first-class. |
| `category` | TEXT (controlled vocab) | yes | `laptops` or `keyboards` for v0.1 (the QIL `CATEGORIES`). |
| `return_window_bucket` | TEXT (enum) | no | Coarse bucket of days-to-return: `0-7` / `8-30` / `31-90` / `90+`. **Never an exact date or day count.** |
| `cell_count` | INT | yes | Number of returns aggregated into this cell. Must be `>= k` (see below). |

**Explicitly excluded** (must never appear): any timestamp finer than the coarse
bucket; any geography below country; any free-text the customer wrote; any
quantity, price, SKU, order id, or customer id; anything per-individual.

---

## Controlled vocabularies

### `return_reason`

A **superset** of the QIL `failure_mode` vocabulary plus return-specific reasons.
Quality-relevant reasons map onto QIL failure modes; non-quality reasons are
retained for transparency but **excluded from failure-rate aggregation** (see
`is_quality_signal` below).

| `return_reason` | Maps to QIL `failure_mode` | Quality signal? |
|-----------------|----------------------------|-----------------|
| `thermal_throttling` | `thermal_throttling` | yes |
| `battery_degradation` | `battery_degradation` | yes |
| `structural_failure` | `structural_failure` | yes |
| `connectivity_issue` | `connectivity_issue` | yes |
| `switch_failure` | `switch_failure` (keyboards) | yes |
| `display_defect` | `display_defect` | yes |
| `dead_on_arrival` | (general defect) | yes |
| `performance_below_expectation` | (no single mode; quality-dim signal) | yes |
| `not_as_described` | — | no |
| `changed_mind` | — | no |
| `wrong_item` | — | no |
| `better_price_elsewhere` | — | no |
| `other` | — | no |

`is_quality_signal` is **derived** by PreferenceLayer from the table above (not a
field the retailer sends), so non-quality returns (`changed_mind`, `wrong_item`,
…) never bias the Beta-Binomial failure posteriors.

> **Open question (issue #14, Q1):** ratify the superset approach vs. restricting
> the retailer to QIL `failure_mode` values only. Proposed: superset, with the
> derived `is_quality_signal` split. Do not silently decide.

### `use_context`

Maps onto the QIL `use_profile` vocabulary
(`src/preferencelayer/qil/schema.py`):

| `use_context` | QIL `use_profile` |
|---------------|-------------------|
| `light_use` | `light_use` |
| `heavy_use` | `heavy_use` |
| `gaming` | `gaming` |
| `professional` | `professional` |
| `travel` | `travel` |
| `unknown` | (none — profile-agnostic prior) |

Retailers rarely know the true use profile. `unknown` is **first-class**: such
returns are routed to a profile-agnostic prior and are **never** silently bucketed
into a specific profile.

> **Open question (issue #14, Q2):** confirm `unknown` handling and whether
> retailers may supply a best-effort profile derived from product configuration
> (e.g. a gaming GPU SKU → `gaming`). Proposed: allow, but flag provenance.

### `category`

`laptops`, `keyboards` (the QIL `CATEGORIES` for v0.1).

---

## k-anonymity floor

To prevent re-identification of rare returns, the retailer **suppresses at source**
any (`product_id`, `return_reason`, `use_context`, `category`,
`return_window_bucket`) cell whose `cell_count` is below `k`.

- **Proposed `k = 5`** (issue #14, Q3 — ratify before freeze).
- Suppression happens in the retailer's systems; sub-`k` cells are simply not
  transmitted (not sent as zero, not merged into `other` in a way that re-creates
  identifiability).

This is why the shareable unit is an aggregated cell with a `cell_count`, not a
per-event row: a per-event feed cannot satisfy a k-anonymity floor.

---

## How it lands in the QIL

The return cells are ingested into the existing QIL `product_signal` keyspace
(see [`architecture.md`](architecture.md)) as `signal_type = 'failure'` rows for
quality-signal reasons:

- `product_id` → `product_signal.product_id`
- `use_context` → `product_signal.use_profile` (or profile-agnostic prior for `unknown`)
- `return_reason` → `product_signal.failure_mode` (when `is_quality_signal`)
- `cell_count` → contributes to the Beta-Binomial failure-rate evidence count
- `category` → `product_signal` category scoping

Non-quality reasons are stored for auditability but excluded from failure-rate
aggregation. Return signal carries **no** user identifier, so nothing about the
QIL's "no user identifiers" invariant changes.

This schema does **not** add fields to `product_signal`; it defines a conformant
*source* that maps onto the existing columns. If aggregation later needs a new
`product_signal` column, that is a separate schema change requiring its own
`data`-labeled issue and PR.

---

## Example records (NDJSON)

```jsonc
{"product_id": "lenovo-thinkpad-x1-carbon-gen12", "return_reason": "thermal_throttling", "use_context": "professional", "category": "laptops", "return_window_bucket": "8-30", "cell_count": 7}
{"product_id": "keychron-q1", "return_reason": "switch_failure", "use_context": "heavy_use", "category": "keyboards", "return_window_bucket": "31-90", "cell_count": 5}
{"product_id": "dell-xps-15-9530", "return_reason": "changed_mind", "use_context": "unknown", "category": "laptops", "return_window_bucket": "0-7", "cell_count": 12}
```

Note every record is an aggregated cell with `cell_count >= 5`, carries no PII, no
exact date, and no order/customer reference.

---

## Open questions (tracked in issue #14)

1. `return_reason` vocabulary: superset (proposed) vs. QIL `failure_mode` only.
2. `use_context` derivation and `unknown` handling.
3. k-anonymity floor value (`k = 5` proposed).
4. Transport (CSV drop vs. ingestion endpoint) — out of scope here; see ingestion
   issue [#11](https://github.com/ethanluh/preference-layer/issues/11).

Do not freeze this schema until these are ratified in the issue thread.
