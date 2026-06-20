# Retailer Return-Signal Data-Sharing Agreement (Template)

**Status:** template / draft for negotiation — **NOT legal advice and NOT
executable as-is.** Every `[BRACKETED]` field is a placeholder, and every clause
marked **[legal review required]** must be reviewed by qualified counsel for both
parties before signature.

This template operationalizes the Phase 1 / critical-path retailer requirement in
[`implementation-plan.md`](implementation-plan.md) (Work Stream C and the Critical
Path section) and [`phase1-kickoff.md`](phase1-kickoff.md): begin retailer
outreach in Phase 1 because the sales cycle (legal review, data-sharing
agreement, procurement) is the longest non-parallelizable dependency on the path
to the Phase 3 gate.

The companion field-level schema is
[`retailer-return-signal-schema.md`](retailer-return-signal-schema.md).

---

## 0. Why this exists (non-binding context)

Returns are the highest-quality outcome signal for the Quality Intelligence Layer
(QIL) and are not available from public sources. A return with a structured
reason is a near-ground-truth dissatisfaction/failure event. In exchange for a
**minimal, anonymized** return feed, the retailer receives QIL API credits.

The agreement is built around four hard constraints that are non-negotiable on
PreferenceLayer's side because they are product and architectural invariants:

1. **Anonymized at source** — PII is removed by the Retailer before any data
   leaves the Retailer's systems.
2. **Minimal columns only** — `product_id`, `return_reason`, `use_context` (plus
   the coarse, non-identifying fields in the schema doc). Nothing order- or
   customer-level.
3. **No user identifiers reach PreferenceLayer** — the QIL holds no user
   identifiers, by design.
4. **Use-profile-conditioned use** — signals feed use-profile-conditioned quality
   posteriors, never population-level aggregates reported as such.

---

## 1. Parties

This Data-Sharing Agreement ("**Agreement**") is entered into as of
`[EFFECTIVE DATE]` by and between:

- **`[RETAILER LEGAL NAME]`**, `[entity type and jurisdiction]` ("**Retailer**"); and
- **`[PREFERENCELAYER LEGAL ENTITY]`**, `[entity type and jurisdiction]` ("**PreferenceLayer**").

Each a "Party" and together the "Parties."

---

## 2. Definitions

- **"Return Signal"** — the anonymized, structured records described in
  [`retailer-return-signal-schema.md`](retailer-return-signal-schema.md), and
  nothing else.
- **"PII"** — any information that identifies or could reasonably be used,
  alone or in combination, to identify a natural person, directly or indirectly,
  including but not limited to name, email, postal address, phone number,
  customer id, order id, payment token, IP address, device identifier, and exact
  timestamps. **[legal review required: align this definition with the controlling
  privacy law(s) — e.g. GDPR "personal data", CCPA/CPRA "personal information".]**
- **"Anonymized at Source"** — irreversibly stripped of PII within the Retailer's
  systems, before transmission, such that no record can be re-identified by
  PreferenceLayer alone or by reasonably foreseeable combination with other data
  PreferenceLayer holds.
- **"QIL Credits"** — usage credits against the PreferenceLayer Quality
  Intelligence Layer API, as specified in Exhibit B.
- **"Permitted Purpose"** — Section 5.

---

## 3. Data shared and explicitly excluded

3.1 **In scope.** Retailer will share only Return Signal records conforming to
the schema in [`retailer-return-signal-schema.md`](retailer-return-signal-schema.md):
`product_id`, `return_reason`, `use_context`, and the coarse non-identifying
fields enumerated there (`category`, `return_window_bucket`).

3.2 **Out of scope (must never be transmitted).** Customer identifiers; order
identifiers; names; contact details; addresses (below country level); payment
data; IP/device identifiers; exact timestamps; quantities, prices, or SKUs that
could fingerprint an individual order; and any free-text written by a customer.

3.3 **Anonymization at source.** Retailer performs all PII removal before
transmission. PreferenceLayer never receives raw or pre-anonymization data.
**[legal review required.]**

3.4 **k-anonymity floor.** Retailer suppresses any (`product_id`,
`return_reason`, `use_context`, `category`, `return_window_bucket`) cell with a
count below `[k = 5]` before transmission, to prevent re-identification of rare
returns. See the schema doc.

---

## 4. Anonymization, security, and no re-identification

4.1 Retailer represents that the Return Signal is Anonymized at Source.

4.2 **No re-identification.** PreferenceLayer will not attempt to re-identify any
natural person from the Return Signal, and will not combine it with other data
for that purpose.

4.3 **Security.** Each Party maintains commercially reasonable technical and
organizational safeguards. Transmission is encrypted in transit; PreferenceLayer
stores Return Signal in the QIL `product_signal` keyspace, which holds no user
identifiers. **[legal review required: specify breach-notification SLA, sub-
processor terms, audit rights, hosting region.]**

4.4 **No PII obligation triggered.** Because the Return Signal is Anonymized at
Source and contains no PII, this Agreement is not intended to constitute a
controller-processor relationship. **[legal review required — this conclusion
depends on jurisdiction and on the anonymization being legally sufficient; do not
rely on it without counsel.]**

---

## 5. Permitted purpose and use restrictions

5.1 **Permitted Purpose.** PreferenceLayer may use the Return Signal solely to
compute and serve **use-profile-conditioned** product quality signals (failure
rates, quality-dimension posteriors) via the QIL, and to improve the QIL models.

5.2 **Use-profile-conditioned only.** PreferenceLayer will not publish or expose
the Retailer's Return Signal as a raw, retailer-attributable, or population-level
aggregate. Outputs are conditioned on use profile and blended across sources.

5.3 **No resale of raw signal.** PreferenceLayer will not resell or redistribute
the Retailer's raw Return Signal. Derived, multi-source QIL posteriors are
PreferenceLayer's product and are not "raw signal."

5.4 **No competitive disclosure.** PreferenceLayer will not disclose to any third
party that a specific return originated from the Retailer, nor any
retailer-identifying volume that could reveal the Retailer's sales/returns.

---

## 6. Consideration: QIL credits

6.1 In exchange for the Return Signal, PreferenceLayer grants Retailer QIL
Credits per **Exhibit B** (`[volume, tier, rate]`). No monetary fee is exchanged
unless stated in Exhibit B.

6.2 **[legal review required: tax treatment of in-kind exchange.]**

---

## 7. Ownership

7.1 Retailer retains ownership of its underlying business data. This Agreement
grants PreferenceLayer a license to the Return Signal for the Permitted Purpose
only.

7.2 PreferenceLayer owns the QIL, its models, and all derived posteriors.

---

## 8. Term, termination, and deletion

8.1 **Term.** `[INITIAL TERM]`, auto-renewing for `[RENEWAL TERM]` unless either
Party gives `[NOTICE PERIOD]` written notice.

8.2 **Termination.** Either Party may terminate for material breach uncured
within `[CURE PERIOD]` days of notice.

8.3 **On termination.** Retailer stops sending Return Signal. Because the Return
Signal is anonymized and irreversibly blended into QIL posteriors, the Parties
agree that already-ingested anonymized records need not be deleted from derived
models; any un-ingested batch in transit is deleted. **[legal review required:
confirm this is acceptable and lawful given the anonymized nature of the data.]**

---

## 9. Warranties and liability

9.1 Retailer warrants it has the right to share the Return Signal and that it is
Anonymized at Source.

9.2 **[legal review required: limitation of liability, indemnification, warranty
disclaimers, governing law, dispute resolution, insurance — all standard clauses
to be drafted by counsel for `[GOVERNING JURISDICTION]`.]**

---

## Exhibit A — Return-Signal schema

Incorporated by reference:
[`retailer-return-signal-schema.md`](retailer-return-signal-schema.md). The
schema, controlled vocabularies, k-anonymity floor, and exclusions there are part
of this Agreement.

## Exhibit B — QIL credit schedule

| Item | Value |
|------|-------|
| Credit grant | `[e.g. N QIL queries / month]` |
| Tier | `[Free / Starter / Growth / Enterprise — see implementation-plan pricing]` |
| Delivery cadence of Return Signal | `[e.g. weekly batch]` |
| Minimum volume (if any) | `[…]` |
| Review / true-up cadence | `[…]` |

## Exhibit C — Technical delivery

| Item | Value |
|------|-------|
| Transport | `[secure batch drop / ingestion endpoint — see ingestion issue #11]` |
| Format | `[CSV / NDJSON conforming to the schema doc]` |
| Encryption in transit | `[required]` |
| Contact (Retailer data eng) | `[…]` |
| Contact (PreferenceLayer) | `[…]` |

---

## What this template does NOT do (human steps)

This is **scaffold only**. The following are human, long-cycle steps and are
**not** automated by this document:

- **Business development outreach** — identifying and contacting the 3–5 mid-size
  online retailers (per the plan), pitching the exchange, and qualifying fit.
- **Legal review** — every `[legal review required]` clause, the PII/anonymization
  legal sufficiency analysis, governing law, liability, and the final executable
  contract must be drafted/reviewed by counsel for both Parties.
- **Procurement and signature** — the Retailer's vendor onboarding, security
  review, and signature workflow.

Start these in Phase 1 (~Month 6), per the critical path, even though the
partnership lands in Phase 2.
