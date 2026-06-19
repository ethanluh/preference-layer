"""Tests for the agent orchestration (``agent/recommender.py``).

Uses a tiny hand-built preference model and quality service so the assertions
are about wiring and blending, not about any learned signal.
"""

import numpy as np

from preferencelayer.agent import combine
from preferencelayer.agent.recommender import AgentRecommender
from preferencelayer.models.base import Recommender
from preferencelayer.qil.aggregate import QualityAggregator
from preferencelayer.qil.extract import ExtractedSignal
from preferencelayer.qil.query import QualityService


class _DotModel(Recommender):
    """Scores a candidate as the dot product of its attributes with a fixed weight."""

    name = "dot"

    def fit(self, purchased, catalog, n_shared, population=None):
        return {"w": np.ones(catalog.shape[1])}

    def score(self, state, candidates, n_shared):
        return candidates @ state["w"]


def _service(values: dict[str, float]) -> QualityService:
    sigs = []
    for pid, v in values.items():
        for _ in range(12):
            for dim in ("thermal", "build_quality"):
                sigs.append(ExtractedSignal(pid, "laptops", "gaming", "performance", None, dim, v, 0.9))
    return QualityService(QualityAggregator().fit(sigs))


def _agent(values, **kw):
    model = _DotModel()
    state = model.fit(None, np.zeros((1, 3)), 3)
    return AgentRecommender(model, state, _service(values), n_shared=3, **kw)


def test_one_score_per_candidate_and_finite():
    agent = _agent({"a": 0.8, "b": 0.2, "c": 0.5})
    attrs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    res = agent.rank(["a", "b", "c"], attrs, "gaming", mean_confidence=0.5)
    assert res.blended.shape == (3,)
    assert len(res.order) == 3
    assert np.all(np.isfinite(res.blended))


def test_quality_only_ranks_by_quality():
    agent = _agent({"a": 0.2, "b": 0.9, "c": 0.5})
    attrs = np.array([[3.0, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])  # 'a' wins on preference
    res = agent.rank(["a", "b", "c"], attrs, "gaming", mean_confidence=0.5, alpha=0.0)
    # alpha=0 ignores preference; 'b' has the best quality so it ranks first.
    assert res.order[0] == 1


def test_missing_quality_falls_back_to_neutral():
    agent = _agent({"a": 0.9})  # only 'a' has evidence
    q = agent.quality_scores(["a", "unknown"], "gaming")
    assert q[1] == agent.neutral_quality
    assert q[0] > q[1]  # evidenced product scores above the neutral fallback


def test_adaptive_alpha_used_when_not_overridden():
    agent = _agent({"a": 0.5, "b": 0.5})
    attrs = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    res = agent.rank(["a", "b"], attrs, "gaming", mean_confidence=0.8)
    assert np.isclose(res.alpha, combine.alpha_from_confidence(0.8))


def test_failure_penalty_discounts_quality():
    sigs = []
    for _ in range(12):
        sigs.append(ExtractedSignal("p", "laptops", "gaming", "performance", None, "thermal", 0.8, 0.9))
    for _ in range(12):  # heavy failure evidence
        sigs.append(ExtractedSignal("p", "laptops", "gaming", "failure", "thermal_throttling", None, 0.2, 0.9))
    svc = QualityService(QualityAggregator().fit(sigs))
    model = _DotModel()
    state = model.fit(None, np.zeros((1, 3)), 3)
    base = AgentRecommender(model, state, svc, 3, failure_penalty=0.0)
    penalized = AgentRecommender(model, state, svc, 3, failure_penalty=1.0)
    assert penalized.quality_scores(["p"], "gaming")[0] < base.quality_scores(["p"], "gaming")[0]
