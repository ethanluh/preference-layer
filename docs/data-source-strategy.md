# QIL Data-Source Strategy

**Status:** active decision (Month 6, Phase 1). This records *which* product-quality
data sources QIL pursues now, which are parked, and why. Read alongside
[`implementation-plan.md`](implementation-plan.md) (critical path) and
[`phase1-kickoff.md`](phase1-kickoff.md) (Work Stream B).

QIL's value rests on the *quality* of its conditioned signals, and the Phase 0
real-data check ([`phase1-amazon-realdata.md`](phase1-amazon-realdata.md)) showed
data/extraction quality is the binding constraint. So source selection is a
deliberate sequencing decision, not "wire everything we can."

## Decision

### Reddit — proceed now, research/free tier, research-stage
- The **only** live ingestion source wired by default (`build_live_connectors`,
  `src/preferencelayer/qil/ingest/live_fetch.py` + `qil/cli.py`). Run it on the
  **research/free tier**, scoped honestly as **research-stage** prototype use.
- **Before any commercial sale**, the Reddit commercial API license must be
  reviewed and priced — the free/research tier does not cover commercial
  redistribution of derived data. `implementation-plan.md:409` notes the *volume*
  pricing ("free tier sufficient for Phase 0; $100/month for Phase 1 volumes"),
  which is **not** a commercial-use license. Tracked in
  [#47](https://github.com/ethanluh/preference-layer/issues/47).

### Reddit access path — Arctic Shift fallback (added 2026-07-17)
- Reddit closed self-service OAuth app registration in 2026 (manual "Responsible
  Builder Policy" review); a legitimate research-use application was **rejected**
  with no actionable reason given. Reapplying with a tighter use-case description
  remains worth trying in parallel, but has no reliable timeline.
- **Wired now, and the default access path:** `make_arctic_shift_fetch`
  (`src/preferencelayer/qil/ingest/live_fetch.py`) hits
  [Arctic Shift](https://arctic-shift.photon-reddit.com/), a community-run,
  unauthenticated mirror of Reddit's archived post data. No client_id/secret —
  only a `User-Agent`. Verified against a live call: same field names
  (`id`/`title`/`selftext`/`score`/`permalink`) as the official API, so it drops
  into `RedditConnector` unchanged. `build_live_connectors`/`qil-ingest --live`
  now default to `reddit-arctic-shift` (no `--source` flag needed); the official
  OAuth path is still selectable with `--source reddit` for accounts with an
  approved app.
- **Caveat:** Arctic Shift is a volunteer-run third-party mirror, not
  Reddit-sanctioned — its own legality rests on Reddit tolerating it. Fine for
  research-stage prototyping now; needs the **same commercial-licensing scrutiny
  as the official API** (probably more, since it has no license of its own)
  before any commercial use — do not treat it as a way to route around Reddit's
  commercial terms.
- This is an access-path decision, not a change to source selection: Reddit
  (via whichever access path is live) remains the only source wired by default.

### iFixit + Notebookcheck — parked
- Their connectors and parsers are **retained and tested** (`qil/ingest/connectors.py`),
  but **not crawled by default**: `build_live_connectors` defaults to Reddit
  (via Arctic Shift) only, and iFixit is explicit opt-in
  (`qil-ingest --live --source ifixit`).
- **Do not** send permission/key/ToS outreach now. They are only worth the
  negotiation if **Reddit + retailer return data leave a specific, proven gap**.
  Revisiting them is a deliberate later step, not a default.

### Retailer return data — pulled forward to now
- Return data is the highest-quality outcome signal and is **not available from
  public sources**. Per the critical path (`implementation-plan.md:440-446`),
  "Retailer data partnerships → QIL quality plateau → Phase 3 gate ... Start in
  Month 6 (Phase 1), not Month 10. Every month of delay here delays the Phase 3
  gate." The long sales cycle (legal review, data-sharing agreements, procurement)
  is the binding constraint, so **outreach begins now**.
- It is the data foundation a commercial product should actually rest on — the
  reason to start here rather than over-investing in scraping breadth.
- Templates are ready: [`retailer-data-sharing-agreement.md`](retailer-data-sharing-agreement.md)
  and [`retailer-return-signal-schema.md`](retailer-return-signal-schema.md)
  (schema ratification tracked in [#14](https://github.com/ethanluh/preference-layer/issues/14)).
  Outreach tracked in [#48](https://github.com/ethanluh/preference-layer/issues/48).

## Rationale (sequencing)

1. **Reddit first** is cheap, immediately available, and enough to exercise the
   real ingest → extraction → posterior path end-to-end (the B1/B2 risk lives in
   *extraction quality*, not source count).
2. **Retailer data is the foundation**, but has a long lead time — so it must start
   in parallel *now* even though it lands later.
3. **iFixit/Notebookcheck add breadth, not foundation.** Negotiating access (and,
   for Notebookcheck, building a site-specific HTML parser) is only justified once
   we can point at a concrete gap Reddit + retailer data did not cover. Until then,
   parking them avoids spending scarce BD/legal effort on speculative sources.

## Operator actions (not code)

- Reddit: confirm research-tier terms for current use; get a commercial-license
  quote before the first paid deployment (#47). Arctic Shift is the default
  access path (see above) while the official OAuth app reapplication (if
  pursued) has no reliable timeline; `--source reddit` remains available once
  approved.
- Retailer: begin outreach to 3–5 mid-size retailers; drive the agreement through
  legal and ratify the schema (#48, #14).
- iFixit/Notebookcheck: **no action** — parked.
