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
import math
import time
from datetime import datetime, timezone

from .corpus import generate
from .extract import ExtractedSignal, QILExtractor
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
