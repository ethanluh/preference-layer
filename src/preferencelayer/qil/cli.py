"""QIL operational CLI — nightly posterior refit scheduling (Work Stream B3).

``run_nightly_refit`` (``qil/refit.py``) is the refit *job*; this module is the
*scheduling* wrapper the kickoff calls for. The recommended production setup is a
cron entry or systemd timer invoking ``qil-refit`` once a day:

    # crontab: refit at 03:30 UTC daily
    30 3 * * *  qil-refit

    # or a systemd timer: OnCalendar=*-*-* 03:30:00  ->  ExecStart=qil-refit

``--loop --interval-hours N`` is a dependency-free in-process alternative for
environments without cron (it sleeps to the next interval boundary and re-runs).

Production wiring: load signals from the ``product_signal`` table (B1 output) and
upsert posteriors to ``quality_posterior`` via ``PostgresPosteriorSink``. This
sandbox has no DB, so the CLI refits over the controlled Phase 0 corpus — the
same stand-in ``experiments/run_qil_ingest.py`` uses — into an in-memory sink,
exercising the full GP-refit path end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .corpus import generate
from .extract import ExtractedSignal, QILExtractor
from .quality_spans import QualityDimTagger
from .ingest import (
    CanonicalProduct,
    FixtureConnector,
    IFixitConnector,
    InMemorySink,
    IngestionStats,
    NotebookcheckConnector,
    ProductRegistry,
    RateLimiter,
    ProductSignalRow,
    RedditConnector,
    make_arctic_shift_fetch,
    make_http_fetch,
    make_reddit_fetch,
    run_daily,
    signal_rows_from_json,
    signal_rows_to_json,
)
from .harness import (
    TfidfBaselineClassifier,
    TransformerClassifier,
    load_controlled_smoke,
    load_real_corpus,
    measure,
)
from .refit import (
    InMemoryPosteriorSink,
    PosteriorSink,
    posterior_rows_to_json,
    run_nightly_refit,
)

_B2_BANNER = {
    "pass": ">= 70% -> PROCEED to coverage (B4).",
    "recoverable": "60-70% -> assess whether MORE ANNOTATION recovers it.",
    "escalate": "< 60% -> automation story does NOT hold; ESCALATE before scaling.",
}


def seconds_until_next_run(now: datetime, interval_hours: float) -> float:
    """Seconds from ``now`` to the next run boundary, aligned to UTC midnight.

    Aligning to fixed offsets from midnight (rather than now+interval) keeps a
    daily job landing at a stable wall-clock time across restarts. Always returns
    a value in ``(0, interval]``.
    """
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    interval = interval_hours * 3600.0
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - midnight).total_seconds()
    next_boundary = (math.floor(elapsed / interval) + 1) * interval
    return next_boundary - elapsed


def _sandbox_signals() -> list[ExtractedSignal]:
    """Controlled-corpus stand-in for the production ``product_signal`` read."""
    corpus = generate()
    extractor = QILExtractor().fit(corpus.train)
    return extractor.extract(corpus.train + corpus.test)


def _run_once(sink: PosteriorSink) -> int:
    written = run_nightly_refit(_sandbox_signals(), sink)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{now}] qil-refit: wrote {written} posteriors (parameters only)")
    sample = next(iter(sink.rows.values()), None) if hasattr(sink, "rows") else None
    if sample is not None:
        print(f"  e.g. {sample.product_id} / {sample.use_profile} / {sample.quality_dim}: "
              f"mean={sample.posterior_mean:.3f} "
              f"90% CI=[{sample.credible_lo_90:.3f}, {sample.credible_hi_90:.3f}] "
              f"n={sample.evidence_count} freshness={sample.freshness_score:.3f}")
    return written


def build_demo_registry() -> ProductRegistry:
    """A tiny canonical model list for the offline demo.

    Production loads this from the product catalog; here it is just enough for the
    committed fixtures to resolve mentions to a canonical product_id.
    """
    return ProductRegistry(threshold=0.5).add(
        CanonicalProduct("lenovo-thinkpad-x1-carbon-gen12", "laptops",
                         "Lenovo ThinkPad X1 Carbon Gen 12",
                         aliases=("thinkpad x1 carbon gen 12", "x1c g12", "x1 carbon gen 12"))
    ).add(
        CanonicalProduct("dell-xps-15-9530", "laptops", "Dell XPS 15 9530",
                         aliases=("dell xps 15 9530", "xps 15 9530", "xps 15"))
    ).add(
        CanonicalProduct("glorious-gmmk-3-pro", "keyboards", "Glorious GMMK 3 Pro",
                         aliases=("gmmk 3 pro", "gmmk3 pro", "gmmk 3", "gmmk3"))
    ).add(
        CanonicalProduct("keychron-q1-pro", "keyboards", "Keychron Q1 Pro",
                         aliases=("keychron q1", "keychron q1 pro", "q1 pro"))
    )


def _row_to_signal(row: ProductSignalRow) -> ExtractedSignal:
    """Adapt a persisted product_signal row to the ExtractedSignal refit consumes.

    In production the nightly refit reads product_signal straight from Postgres;
    this bridges the in-memory ingest output to the refit input for an
    end-to-end demo.
    """
    return ExtractedSignal(
        product_id=row.product_id,
        category=row.category,
        use_profile=row.use_profile,
        signal_type=row.signal_type,
        failure_mode=row.failure_mode,
        quality_dim=row.quality_dim,
        signal_value=row.signal_value if row.signal_value is not None else 0.5,
        confidence=row.model_confidence,
    )


def run_ingest(
    connectors,
    registry: ProductRegistry,
    extractor: QILExtractor,
    sink: InMemorySink,
    *,
    refit: bool = False,
    posterior_sink: PosteriorSink | None = None,
    quality_tagger: QualityDimTagger | None = None,
) -> tuple[IngestionStats, int]:
    """Run one ingest pass; optionally chain straight into a posterior refit.

    Returns ``(stats, posteriors_written)`` -- ``posteriors_written`` is 0 unless
    ``refit`` is set. A ``QualityDimTagger`` is used by default (so the refit
    produces real GP quality posteriors); pass one explicitly to override.
    """
    stats = run_daily(connectors, registry, extractor, sink,
                      quality_tagger=quality_tagger or QualityDimTagger())
    written = 0
    if refit:
        signals = [_row_to_signal(r) for r in sink.rows]
        written = run_nightly_refit(signals, posterior_sink or InMemoryPosteriorSink())
    return stats, written


# Live connectors keyed by filename-stem source (reddit_*.json -> reddit).
_LIVE_CONNECTORS = {
    "reddit": RedditConnector,
    "ifixit": IFixitConnector,
    "notebookcheck": NotebookcheckConnector,
}


def _connectors_from_dir(fixtures_dir: Path, category: str) -> list:
    """Build one connector per ``*.json`` fixture in a directory.

    The source is the filename stem (``reddit_*.json`` -> reddit). Two shapes flow:

    * **list-shaped** files are ``FixtureConnector`` record-lists
      (``[{source_local_id, text, ...}]``);
    * **dict-shaped** files are treated as a captured live-source payload and
      replayed through the matching live connector's real ``_parse`` via an
      injected ``fetch`` -- so the iFixit / Notebookcheck / Reddit-listing parsers
      flow through the CLI offline, not only the unit tests.

    Files with an unrecognized source or shape are skipped, so a mixed directory
    is tolerated.
    """
    connectors = []
    for p in sorted(fixtures_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        source = p.stem.split("_")[0]
        if isinstance(data, list):
            connectors.append(FixtureConnector(category, p, source_type=source))
        elif isinstance(data, dict) and source in _LIVE_CONNECTORS:
            connectors.append(
                _LIVE_CONNECTORS[source](
                    category, [f"fixture://{p.name}"], fetch=lambda _url, _d=data: _d
                )
            )
    if not connectors:
        raise SystemExit(f"qil-ingest: no usable *.json fixtures found in {fixtures_dir}")
    return connectors


# Default live sources per category. Operators can extend these; the point is a
# working out-of-the-box --live wiring, not an exhaustive source list.
_LIVE_SUBREDDITS = {
    "laptops": ("laptops", "thinkpad", "Dell"),
    "keyboards": ("MechanicalKeyboards",),
}
_LIVE_IFIXIT_URLS = {
    "laptops": ("https://www.ifixit.com/api/2.0/guides?category=Laptop",),
    "keyboards": ("https://www.ifixit.com/api/2.0/guides?category=Keyboard",),
}

# Per-subreddit created_utc watermark for the Arctic Shift fallback, so a daily
# run only pulls posts newer than the last run instead of re-fetching the same
# top-`limit` page every time (dedup via content_hash makes this an efficiency
# fix, not a correctness one).
_DEFAULT_WATERMARK_PATH = Path(".qil_arctic_shift_watermark.json")


def _watermark_path() -> Path:
    return Path(os.environ.get("QIL_ARCTIC_SHIFT_WATERMARK_PATH", _DEFAULT_WATERMARK_PATH))


def _load_watermarks(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _bump_watermark(subreddit: str, records: list, watermarks: dict[str, int]) -> None:
    seen = [rec.get("created_utc") for rec in records if isinstance(rec, dict)]
    seen = [ts for ts in seen if isinstance(ts, (int, float))]
    if not seen:
        return
    watermarks[subreddit] = max(watermarks.get(subreddit, 0), int(max(seen)))
    _watermark_path().write_text(json.dumps(watermarks))


def build_live_connectors(category: str, *, sources: tuple[str, ...] = ("reddit",),
                          rate: float = 1.0) -> list:
    """Assemble live connectors from environment credentials.

    Reads ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` / ``REDDIT_USER_AGENT``;
    injects the real ``fetch`` callables (``live_fetch``) and a token-bucket
    ``RateLimiter`` into the existing connectors. Raises ``SystemExit`` with a
    clear message when Reddit credentials are absent.

    ``sources`` selects which sources to wire. **Default: Reddit only** -- per the
    data-source strategy (``docs/data-source-strategy.md``), Reddit runs on the
    research/free tier (research-stage) and is the only source wired by default;
    iFixit and Notebookcheck are **parked** (their connectors/parsers are retained
    and tested, but not crawled by default). Opt iFixit in explicitly with
    ``sources=("reddit", "ifixit")``.

    ``"reddit-arctic-shift"`` is a fallback wired the same way as ``"reddit"`` but
    via ``make_arctic_shift_fetch`` (no OAuth client_id/secret needed) -- for when
    Reddit's own app-approval process is unavailable or rejects the application
    (see docs/data-source-strategy.md). Only needs ``REDDIT_USER_AGENT``.

    Notebookcheck is never auto-wired: it has no JSON API and needs a site-specific
    HTML->records parser (inject one via ``make_http_fetch(parser=...)`` +
    ``NotebookcheckConnector``).
    """
    user_agent = os.environ.get("REDDIT_USER_AGENT")
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

    connectors: list = []
    if "reddit" in sources:
        if not (user_agent and client_id and client_secret):
            raise SystemExit(
                "qil-ingest --live: missing Reddit credentials. Set REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in the environment, or use "
                "--source reddit-arctic-shift if Reddit's OAuth app approval is unavailable."
            )
        reddit_fetch = make_reddit_fetch(client_id, client_secret, user_agent)
        connectors.append(RedditConnector(
            category, list(_LIVE_SUBREDDITS.get(category, ())),
            fetch=reddit_fetch, rate_limiter=RateLimiter(rate=rate),
        ))
    if "reddit-arctic-shift" in sources:
        if not user_agent:
            raise SystemExit(
                "qil-ingest --live: missing REDDIT_USER_AGENT (Arctic Shift still "
                "expects a descriptive User-Agent, even without OAuth credentials)."
            )
        watermarks = _load_watermarks(_watermark_path())
        arctic_shift_fetch = make_arctic_shift_fetch(
            user_agent,
            after_utc=watermarks.get,
            on_records=lambda sub, records: _bump_watermark(sub, records, watermarks),
        )
        connectors.append(RedditConnector(
            category, list(_LIVE_SUBREDDITS.get(category, ())),
            fetch=arctic_shift_fetch, rate_limiter=RateLimiter(rate=rate),
        ))
    if "ifixit" in sources:
        # Parked by default; only wired when explicitly requested.
        ifixit_fetch = make_http_fetch(user_agent=user_agent, crawl_delay=1.0)
        connectors.append(IFixitConnector(
            category, list(_LIVE_IFIXIT_URLS.get(category, ())),
            fetch=ifixit_fetch, rate_limiter=RateLimiter(rate=rate),
        ))
    return connectors


def ingest_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="qil-ingest",
        description="Run the QIL daily ingestion over fixture sources (Work Stream B1).",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fixtures", type=Path,
                     help="directory of *.json source fixtures (one connector per file)")
    src.add_argument("--live", action="store_true",
                     help="ingest from live sources using credentials in the environment "
                          "(REDDIT_CLIENT_ID/SECRET/USER_AGENT); see build_live_connectors")
    ap.add_argument("--source", action="append",
                    choices=["reddit", "reddit-arctic-shift", "ifixit"], dest="sources",
                    help="live source to wire (repeatable; default: reddit only). iFixit is "
                         "parked by default -- pass --source ifixit to opt in. Use "
                         "reddit-arctic-shift instead of reddit if Reddit's OAuth app "
                         "approval is unavailable/rejected (no client_id/secret needed). "
                         "Only used with --live.")
    ap.add_argument("--category", default="laptops", help="product category (default: laptops)")
    ap.add_argument("--refit", action="store_true",
                    help="after ingest, run the posterior refit end-to-end and report counts")
    ap.add_argument("--signal-store", type=Path, default=None,
                    help="JSON file of accumulated product_signal rows: loaded before the run "
                         "and rewritten after, so evidence builds up across separate "
                         "invocations instead of resetting each run. A stopgap before a real "
                         "DB (see docs/whats-missing.md B4) -- not a replacement for it.")
    ap.add_argument("--posterior-json", type=Path, default=None,
                    help="if --refit is set, write the resulting quality_posterior rows to "
                         "this JSON path (write-only snapshot; posteriors are always "
                         "recomputed fresh from the full accumulated signal set)")
    args = ap.parse_args(argv)

    if args.fixtures is not None and not args.fixtures.is_dir():
        raise SystemExit(f"qil-ingest: --fixtures {args.fixtures} is not a directory")

    # Train the extractor on the controlled Phase 0 corpus (stand-in for the
    # production fine-tuned model -- see Work Stream B2 for real-text precision).
    corpus = generate()
    extractor = QILExtractor().fit(corpus.train)

    if args.live:
        connectors = build_live_connectors(args.category, sources=tuple(args.sources or ("reddit",)))
    else:
        connectors = _connectors_from_dir(args.fixtures, args.category)

    sink = InMemorySink()
    if args.signal_store is not None and args.signal_store.exists():
        # Preload via write() (not a raw list assignment) so its dedup _seen set
        # stays consistent with the rows already on disk.
        sink.write(signal_rows_from_json(args.signal_store.read_text()))

    posterior_sink = InMemoryPosteriorSink()
    stats, written = run_ingest(connectors, build_demo_registry(), extractor, sink,
                                refit=args.refit, posterior_sink=posterior_sink)

    print(f"qil-ingest: fetched={stats.fetched} matched={stats.matched} "
          f"unmatched={stats.unmatched} written={stats.written}")
    for row in sink.rows:
        print(f"  {row.product_id:35s} use_profile={row.use_profile:12s} "
              f"signal_type={row.signal_type:11s} quality_dim={str(row.quality_dim):17s} "
              f"conf={row.model_confidence:.2f}")
    if args.refit:
        print(f"refit: wrote {written} posteriors (parameters only, over "
              f"{len(sink.rows)} accumulated signals)")
    print("NOTE: no raw scraped text persisted -- only normalized product_signal rows.")

    if args.signal_store is not None:
        args.signal_store.write_text(signal_rows_to_json(sink.rows))
    if args.refit and args.posterior_json is not None:
        args.posterior_json.write_text(posterior_rows_to_json(posterior_sink.rows.values()))
    return 0


def refit_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="qil-refit",
        description="Run the nightly QIL posterior refit (Work Stream B3).",
    )
    ap.add_argument("--loop", action="store_true",
                    help="run continuously, refitting every --interval-hours (else run once and exit)")
    ap.add_argument("--interval-hours", type=float, default=24.0,
                    help="hours between refits when --loop is set (default: 24)")
    args = ap.parse_args(argv)

    sink = InMemoryPosteriorSink()
    _run_once(sink)
    if not args.loop:
        return 0

    while True:  # pragma: no cover - exercised manually; the boundary math is unit-tested
        delay = seconds_until_next_run(datetime.now(timezone.utc), args.interval_hours)
        print(f"  next refit in {delay / 3600.0:.2f}h (Ctrl-C to stop)")
        time.sleep(delay)
        _run_once(sink)


def b2_main(argv: list[str] | None = None) -> int:
    """Measure use-profile extraction precision and map it to the B2 decision band.

    ``--smoke`` runs on the controlled corpus (plumbing check; NOT a real-text
    result); ``--corpus PATH`` runs on an annotated real-text JSONL corpus (the
    one ``annotate.adjudicate`` produces). This is the console-script twin of
    ``experiments/run_qil_realtext_harness.py``.
    """
    ap = argparse.ArgumentParser(
        prog="qil-b2",
        description="Measure use-profile extraction precision on real text (Work Stream B2).",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", type=str, help="path to an annotated real-text JSONL corpus")
    src.add_argument("--smoke", action="store_true",
                     help="run on the controlled corpus (plumbing check; NOT a real-text result)")
    ap.add_argument("--model", choices=["tfidf", "transformer"], default="tfidf")
    ap.add_argument("--out", type=str, default=None, help="optional path to write the result JSON")
    args = ap.parse_args(argv)

    if args.smoke:
        split = load_controlled_smoke()
        print("!! SMOKE TEST on the CONTROLLED corpus -- NOT a real-text result.")
        print("!! Real-text precision is UNVERIFIED until --corpus points at an")
        print("!! annotated scraped corpus (~300 adjudicated samples).\n")
    else:
        split = load_real_corpus(args.corpus)

    classifier = TransformerClassifier() if args.model == "transformer" else TfidfBaselineClassifier()
    result = measure(classifier, split)

    print(f"model:            {result.model}")
    print(f"corpus:           {result.source} (real_text={result.is_real_text})")
    print(f"macro precision:  {result.macro_precision:.4f}  (baseline {result.baseline_precision:.4f})")
    print(f"checkpoint band:  {result.band.upper()} -- {_B2_BANNER[result.band]}")
    print(f"verified on REAL text: {result.verified_on_real_text}")
    if not result.is_real_text:
        print("\n(reminder: band/precision above are controlled-corpus; do not quote as real-text.)")

    if args.out:
        Path(args.out).write_text(json.dumps(result.to_json(), indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(refit_main())
