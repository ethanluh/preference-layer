# PTP published contexts & schema

This directory holds the **published, freezable** artifacts for the Preference
Transport Protocol credential, so an external party can validate a credential
without reading our implementation.

| File | Purpose |
|------|---------|
| `ptp-v1.jsonld` | The JSON-LD `@context` referenced by every PTP credential as `https://preferencelayer.io/context/v1` (see `PTP_CONTEXT` in `src/preferencelayer/ptp/credential.py`). Maps the preference-graph fields to the `ptp:` vocabulary. |
| `ptp-credential-v0.1.schema.json` | JSON Schema (Draft 2020-12) for the **frozen v0.1** credential. Authoritative for required fields, types, and value ranges. A byte-equivalent copy ships as package data at `src/preferencelayer/ptp/` (kept in sync by a test). |

## Validating a credential (no PreferenceLayer source needed)

```bash
pip install jsonschema           # optional; the validator has a built-in fallback
python -m preferencelayer.ptp.schema_validate path/to/credential.json
# or, after install: preflayer-validate path/to/credential.json
```

Exit code `0` = valid, `1` = invalid (violations printed). CI validates a
known-good fixture and asserts a known-bad fixture is rejected.

## Field-name reconciliation (v0.1 freeze)

The Phase 1 kickoff A1 checklist named three fields differently from the
authoritative spec (`docs/protocol-spec.md` §3). The **spec wording is
canonical**; the JSON-LD context defines the kickoff names as aliases so both
resolve:

| Kickoff checklist name | Canonical (spec §3) | Resolution |
|------------------------|---------------------|------------|
| `edgeWeights` | `edges` (each edge object carries a `weight`) | `edgeWeights` aliased to `ptp:edges` in the context |
| `updateMetadata` | flat `updateCount` / `privacyBudgetConsumed` / `lastUpdated` | flat fields are required by the schema; `updateMetadata` kept as an optional wrapper alias |
| `contextConditioners` | `contextConditioners` | identical |

See issue #10 for the freeze discussion and the §8 open-question impact
assessment (Q1/Q2/Q3/Q5 do not block the freeze; Q4 — the export-bundle wrapper —
is explicitly left **unfrozen** for v0.1; only the per-credential VC is frozen).
```
