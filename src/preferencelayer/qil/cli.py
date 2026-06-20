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
import time
from datetime import datetime, timezone
from pathlib import Path

from .corpus import generate
from .extract import ExtractedSignal, QILExtractor
from .ingest import (
    CanonicalProduct,
    FixtureConnector,
    InMemorySink,
    IngestionStats,
    ProductRegistry,
    ProductSignalRow,
    run_daily,
)
from .refit import InMemoryPosteriorSink, PosteriorSink, run_nightly_refit


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
) -> tuple[IngestionStats, int]:
    """Run one ingest pass; optionally chain straight into a posterior refit.

    Returns ``(stats, posteriors_written)`` -- ``posteriors_written`` is 0 unless
    ``refit`` is set.
    """
    stats = run_daily(connectors, registry, extractor, sink)
    written = 0
    if refit:
        signals = [_row_to_signal(r) for r in sink.rows]
        written = run_nightly_refit(signals, posterior_sink or InMemoryPosteriorSink())
    return stats, written


def _connectors_from_dir(fixtures_dir: Path, category: str) -> list:
    """One FixtureConnector per ``*.json`` record-list in a directory.

    The source_type is taken from the filename stem (``reddit_*.json`` -> reddit).
    Files whose top-level JSON is not a list are skipped (e.g. live-connector API
    payloads, which are a different shape) so a mixed directory is tolerated.
    """
    connectors = []
    for p in sorted(fixtures_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        connectors.append(FixtureConnector(category, p, source_type=p.stem.split("_")[0]))
    if not connectors:
        raise SystemExit(f"qil-ingest: no FixtureConnector record-list *.json found in {fixtures_dir}")
    return connectors


def ingest_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="qil-ingest",
        description="Run the QIL daily ingestion over fixture sources (Work Stream B1).",
    )
    ap.add_argument("--fixtures", type=Path, required=True,
                    help="directory of *.json source fixtures (one connector per file)")
    ap.add_argument("--category", default="laptops", help="product category (default: laptops)")
    ap.add_argument("--refit", action="store_true",
                    help="after ingest, run the posterior refit end-to-end and report counts")
    args = ap.parse_args(argv)

    if not args.fixtures.is_dir():
        raise SystemExit(f"qil-ingest: --fixtures {args.fixtures} is not a directory")

    # Train the extractor on the controlled Phase 0 corpus (stand-in for the
    # production fine-tuned model -- see Work Stream B2 for real-text precision).
    corpus = generate()
    extractor = QILExtractor().fit(corpus.train)

    connectors = _connectors_from_dir(args.fixtures, args.category)
    sink = InMemorySink()
    stats, written = run_ingest(connectors, build_demo_registry(), extractor, sink, refit=args.refit)

    print(f"qil-ingest: fetched={stats.fetched} matched={stats.matched} "
          f"unmatched={stats.unmatched} written={stats.written}")
    for row in sink.rows:
        print(f"  {row.product_id:35s} use_profile={row.use_profile:12s} "
              f"signal_type={row.signal_type:11s} conf={row.model_confidence:.2f}")
    if args.refit:
        print(f"refit: wrote {written} posteriors (parameters only)")
    print("NOTE: no raw scraped text persisted -- only normalized product_signal rows.")
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(refit_main())
