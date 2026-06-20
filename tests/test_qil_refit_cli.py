"""Tests for the qil-refit scheduling CLI (Work Stream B3).

Covers that a single refit pass writes GP-backed posteriors over the sandbox
stand-in corpus, that the CLI entry point runs once and exits 0, and that the
dependency-free scheduler's next-run math is correct (pure function — no real
sleeping in tests).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from preferencelayer.qil.cli import _run_once, refit_main, seconds_until_next_run
from preferencelayer.qil.refit import InMemoryPosteriorSink

UTC = timezone.utc


def test_run_once_writes_posteriors():
    sink = InMemoryPosteriorSink()
    n = _run_once(sink)
    assert n > 0
    assert len(sink.rows) == n
    # Posteriors carry GP parameters only (no raw observations).
    row = next(iter(sink.rows.values()))
    assert row.credible_lo_90 <= row.posterior_mean <= row.credible_hi_90
    assert 0.0 < row.freshness_score <= 1.0


def test_refit_main_runs_once_and_exits_zero(capsys):
    assert refit_main([]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and "posteriors" in out


@pytest.mark.parametrize("hour,interval,expected_h", [
    (0, 24.0, 24.0),   # at midnight, next daily run is 24h out
    (1, 24.0, 23.0),   # one hour past midnight
    (5, 6.0, 1.0),     # 6h cadence: 05:00 -> next boundary 06:00
    (6, 6.0, 6.0),     # exactly on a boundary -> the NEXT one, never 0
])
def test_seconds_until_next_run(hour, interval, expected_h):
    now = datetime(2026, 6, 20, hour, 0, 0, tzinfo=UTC)
    assert seconds_until_next_run(now, interval) == pytest.approx(expected_h * 3600.0)


def test_seconds_until_next_run_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        seconds_until_next_run(datetime(2026, 6, 20, tzinfo=UTC), 0.0)
