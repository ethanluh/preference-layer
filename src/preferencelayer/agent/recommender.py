"""The integration keystone: an agent that ranks products with both layers.

``AgentRecommender`` is the piece that was missing — it is the shopping agent
that holds a user's portable preference credential *and* calls the Quality
Intelligence Layer, then fuses the two into one ranking. It deliberately depends
only on the public surfaces of each layer:

* preference scores come from a fitted :class:`~preferencelayer.models.base.Recommender`
  (the :class:`~preferencelayer.models.graph.SparsePreferenceGraph` in practice),
  via its ``score`` method — the same call the PTP ``get_preference`` flow wraps;
* quality scores come from a :class:`~preferencelayer.qil.query.QualityService`,
  via its ``quality`` endpoint — the same call the QIL MCP server exposes.

The blend itself lives in :mod:`preferencelayer.agent.combine`. Keeping the
orchestration here and the math there means the formula can be unit-tested in
isolation and this class can be re-pointed at the real MCP handlers without
touching the combination logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.base import Recommender
from ..qil.query import QualityService
from ..qil.schema import QUALITY_DIMS
from . import combine


@dataclass
class BlendResult:
    """One ranked recommendation, with the pieces that produced it.

    Carrying the component scores (not just the final order) keeps the blend
    auditable — an agent can explain *why* an item ranked where it did, and the
    evaluation harness can compare conditions on identical candidate sets.
    """

    order: list[int]            # candidate indices, best first
    blended: np.ndarray         # final blended score per candidate
    pref: np.ndarray            # raw preference score per candidate
    quality: np.ndarray         # raw quality score per candidate
    alpha: float                # blend weight actually used


class AgentRecommender:
    """Ranks candidate products by fusing preference and quality signals.

    Parameters
    ----------
    pref_model, pref_state:
        A fitted recommender and the per-user state returned by its ``fit``.
    quality_service:
        A :class:`QualityService` wrapping aggregated QIL posteriors.
    n_shared:
        Width of the shared attribute block (for the preference model).
    failure_penalty:
        How strongly an estimated failure rate discounts the quality score.
        Defaults to 0 (quality = mean dimension posterior only); raise it to let
        reliability problems pull a product down.
    neutral_quality:
        Quality score assigned to products the QIL has no evidence for (a 404
        from ``/quality``) — the neutral midpoint, so unknown products are
        neither rewarded nor punished before z-scoring.
    """

    def __init__(
        self,
        pref_model: Recommender,
        pref_state: Any,
        quality_service: QualityService,
        n_shared: int,
        *,
        failure_penalty: float = 0.0,
        neutral_quality: float = 0.5,
    ):
        self.pref_model = pref_model
        self.pref_state = pref_state
        self.quality_service = quality_service
        self.n_shared = n_shared
        self.failure_penalty = failure_penalty
        self.neutral_quality = neutral_quality

    # ------------------------------------------------------------- the two heads
    def preference_scores(self, candidate_attrs: np.ndarray) -> np.ndarray:
        """Preference utility per candidate, from the fitted preference model."""
        return self.pref_model.score(self.pref_state, candidate_attrs, self.n_shared)

    def quality_scores(self, candidate_ids: list[str], use_profile: str) -> np.ndarray:
        """Aggregate quality per candidate by querying the QIL ``/quality`` endpoint.

        Each product's quality is the mean posterior over its known quality
        dimensions, optionally discounted by ``failure_penalty * failure_rate``.
        Products with no evidence fall back to ``neutral_quality``.
        """
        out = np.empty(len(candidate_ids))
        for i, pid in enumerate(candidate_ids):
            res = self.quality_service.quality(pid, use_profile, dimensions=list(QUALITY_DIMS))
            if res.get("status") != 200 or not res.get("dimensions"):
                out[i] = self.neutral_quality
                continue
            means = [d["posterior_mean"] for d in res["dimensions"].values()]
            score = float(np.mean(means))
            fail = res.get("failure_rate")
            if fail is not None and self.failure_penalty:
                score -= self.failure_penalty * fail
            out[i] = score
        return out

    # ------------------------------------------------------------------- ranking
    def rank(
        self,
        candidate_ids: list[str],
        candidate_attrs: np.ndarray,
        use_profile: str,
        mean_confidence: float,
        *,
        alpha: float | None = None,
    ) -> BlendResult:
        """Rank candidates by the confidence-adaptive α-blend.

        ``alpha`` defaults to the documented confidence-adaptive value; pass an
        explicit value to force a fixed blend (the evaluation harness uses
        ``alpha=1`` for preference-only, ``0`` for quality-only, ``0.5`` for a
        fixed blend).
        """
        pref = self.preference_scores(candidate_attrs)
        quality = self.quality_scores(candidate_ids, use_profile)
        a = combine.alpha_from_confidence(mean_confidence) if alpha is None else float(alpha)
        blended = combine.blend(pref, quality, a)
        order = list(np.argsort(-blended))
        return BlendResult(order=order, blended=blended, pref=pref, quality=quality, alpha=a)
