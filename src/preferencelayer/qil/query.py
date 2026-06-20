"""QIL query service: the ``/quality`` and ``/compare`` operations.

Wraps a fitted :class:`~preferencelayer.qil.aggregate.QualityAggregator` and
serves the two endpoints from ``docs/architecture.md``:

* ``quality(product_id, use_profile)`` -> posterior mean + 90% credible interval
  per quality dimension, plus the failure-rate estimate and evidence count.
* ``compare(a, b, use_profile)`` -> per-dimension posterior difference and
  ``P(A > B)`` under a normal approximation of the two posteriors.
"""

from __future__ import annotations

import math

from .aggregate import QualityAggregator
from .schema import QUALITY_DIMS


# A query MUST be conditioned on a non-empty use profile -- quality signals are
# never reported as population-level aggregates (the core QIL invariant). Reject
# an empty/whitespace use_profile rather than silently keying a degenerate lookup.
_EMPTY_USE_PROFILE = "use_profile must be a non-empty string (quality is use-profile-conditioned)"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class QualityService:
    def __init__(self, aggregator: QualityAggregator):
        self.agg = aggregator

    def quality(self, product_id: str, use_profile: str, dimensions: list[str] | None = None) -> dict:
        if not (use_profile and use_profile.strip()):
            return {"status": 400, "detail": _EMPTY_USE_PROFILE}
        dims = dimensions or list(QUALITY_DIMS)
        out_dims = {}
        for dim in dims:
            post = self.agg.quality.get((product_id, use_profile, dim))
            if post is None:
                continue
            out_dims[dim] = {
                "posterior_mean": round(post.posterior_mean, 4),
                "credible_interval_90": [round(post.credible_lo_90, 4), round(post.credible_hi_90, 4)],
                "evidence_count": post.evidence_count,
            }
        fail = self.agg.failure.get((product_id, use_profile))
        if not out_dims and fail is None:
            return {"status": 404, "detail": f"no quality data for {product_id} / {use_profile}"}
        return {
            "status": 200,
            "product_id": product_id,
            "use_profile": use_profile,
            "dimensions": out_dims,
            "failure_rate": round(fail.rate_mean, 4) if fail else None,
            "evidence_count": fail.evidence_count if fail else 0,
        }

    def compare(self, product_id_a: str, product_id_b: str, use_profile: str) -> dict:
        """Compare two products' per-dimension quality posteriors.

        Both products are evaluated at the aggregator's single ``query_time``
        (``self.agg.query_time``, days since release): the GP ``posterior_mean``
        and ``posterior_std`` read here are the values at that shared point in
        time. The A-vs-B difference and ``P(A > B)`` are therefore only
        meaningful when both products are compared at the *same* ``query_time``
        — which is always the case for a given aggregator instance, since it
        holds one ``query_time`` for all products.

        Callers who need to compare products at *different* ages (e.g. a
        6-month-old A against a brand-new B) must fit/evaluate against separate
        aggregator instances and reconcile the results themselves; this method
        does not mix query times.
        """
        if not (use_profile and use_profile.strip()):
            return {"status": 400, "detail": _EMPTY_USE_PROFILE}
        dims = {}
        for dim in QUALITY_DIMS:
            pa = self.agg.quality.get((product_id_a, use_profile, dim))
            pb = self.agg.quality.get((product_id_b, use_profile, dim))
            if pa is None or pb is None:
                continue
            diff = pa.posterior_mean - pb.posterior_mean
            denom = math.sqrt(pa.posterior_std ** 2 + pb.posterior_std ** 2)
            p_a_gt_b = _normal_cdf(diff / denom) if denom > 0 else (1.0 if diff > 0 else 0.0)
            dims[dim] = {
                "difference": round(diff, 4),
                "p_a_better": round(p_a_gt_b, 4),
            }
        if not dims:
            return {"status": 404, "detail": "insufficient overlapping evidence to compare"}
        return {
            "status": 200,
            "product_id_a": product_id_a,
            "product_id_b": product_id_b,
            "use_profile": use_profile,
            "dimensions": dims,
        }
