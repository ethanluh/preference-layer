"""Nightly posterior refit job (Work Stream B3).

``docs/architecture.md`` runs Bayesian aggregation as a nightly batch that writes
the ``quality_posterior`` table -- **posterior PARAMETERS only**, never the raw
observations (keeps the served table small and PII-free; QIL holds no user
identifiers). This module is that job, decoupled from any storage backend.

Flow: extracted signals -> :class:`QualityAggregator` (GP over release time) ->
``QualityPosteriorRow`` per (product, use_profile, quality_dim) -> a
:class:`PosteriorSink`. ``freshness_score`` decays with the age of the newest
observation backing each posterior, matching the schema's "decays with signal
age" note.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .aggregate import QualityAggregator
from .extract import ExtractedSignal

# Freshness halves every this-many days of observation age (exponential decay).
_FRESHNESS_HALFLIFE_DAYS = 365.0


@dataclass
class QualityPosteriorRow:
    """One row of the quality_posterior table -- parameters only, no raw data."""

    product_id: str
    use_profile: str
    quality_dim: str
    posterior_mean: float
    posterior_std: float
    credible_lo_90: float
    credible_hi_90: float
    evidence_count: int
    freshness_score: float
    last_refit: datetime


class PosteriorSink(ABC):
    """Where refit posteriors land (upsert by the table's primary key)."""

    @abstractmethod
    def upsert(self, rows: Iterable[QualityPosteriorRow]) -> int:
        raise NotImplementedError


@dataclass
class InMemoryPosteriorSink(PosteriorSink):
    rows: dict[tuple[str, str, str], QualityPosteriorRow] = field(default_factory=dict)

    def upsert(self, rows: Iterable[QualityPosteriorRow]) -> int:
        n = 0
        for row in rows:
            self.rows[(row.product_id, row.use_profile, row.quality_dim)] = row
            n += 1
        return n


class PostgresPosteriorSink(PosteriorSink):
    """INSERT ... ON CONFLICT DO UPDATE into quality_posterior. Parameters only.

    Pass any DB-API 2.0 connection. The SQL/param wiring is covered by a
    fake-DB-API test (tests/test_qil_postgres_sinks.py); only a live Postgres
    exercises the real upsert constraint.
    """

    _UPSERT = """
        INSERT INTO quality_posterior (
            product_id, use_profile, quality_dim, posterior_mean, posterior_std,
            credible_lo_90, credible_hi_90, evidence_count, freshness_score, last_refit
        ) VALUES (
            %(product_id)s, %(use_profile)s, %(quality_dim)s, %(posterior_mean)s,
            %(posterior_std)s, %(credible_lo_90)s, %(credible_hi_90)s,
            %(evidence_count)s, %(freshness_score)s, %(last_refit)s
        )
        ON CONFLICT (product_id, use_profile, quality_dim) DO UPDATE SET
            posterior_mean = EXCLUDED.posterior_mean,
            posterior_std  = EXCLUDED.posterior_std,
            credible_lo_90 = EXCLUDED.credible_lo_90,
            credible_hi_90 = EXCLUDED.credible_hi_90,
            evidence_count = EXCLUDED.evidence_count,
            freshness_score = EXCLUDED.freshness_score,
            last_refit     = EXCLUDED.last_refit
    """

    def __init__(self, connection):
        self.connection = connection

    def upsert(self, rows: Iterable[QualityPosteriorRow]) -> int:
        n = 0
        with self.connection.cursor() as cur:
            for row in rows:
                cur.execute(self._UPSERT, row.__dict__)
                n += 1
        self.connection.commit()
        return n


def _freshness(newest_age_days: float) -> float:
    """Exponential freshness in (0, 1]; 1.0 for a brand-new observation."""
    return float(math.exp(-math.log(2.0) * max(newest_age_days, 0.0) / _FRESHNESS_HALFLIFE_DAYS))


def run_nightly_refit(
    signals: list[ExtractedSignal],
    sink: PosteriorSink,
    aggregator: QualityAggregator | None = None,
    now: datetime | None = None,
) -> int:
    """Refit GP-backed quality posteriors and upsert PARAMETERS to the sink.

    Returns the number of (product, use_profile, quality_dim) posteriors written.
    """
    refit_at = now or datetime.now(timezone.utc)
    agg = (aggregator or QualityAggregator()).fit(signals)

    # Newest observation age per posterior key, for the freshness score.
    newest_age: dict[tuple[str, str, str], float] = defaultdict(lambda: float("inf"))
    for s in signals:
        if s.signal_type == "failure" or s.quality_dim is None:
            continue
        key = (s.product_id, s.use_profile, s.quality_dim)
        age = 0.0 if s.observed_at is None else float(s.observed_at)
        newest_age[key] = min(newest_age[key], age)

    rows = [
        QualityPosteriorRow(
            product_id=post.product_id, use_profile=post.use_profile,
            quality_dim=post.quality_dim, posterior_mean=post.posterior_mean,
            posterior_std=post.posterior_std, credible_lo_90=post.credible_lo_90,
            credible_hi_90=post.credible_hi_90, evidence_count=post.evidence_count,
            freshness_score=_freshness(newest_age.get(key, 0.0)),
            last_refit=refit_at,
        )
        for key, post in agg.quality.items()
    ]
    return sink.upsert(rows)
