"""Tests for the design-partner before/after measurement harness.

Builds tiny hand-controlled tasks so the assertions are about the measurement
machinery and the Phase 1 WS-C gate logic, not about any learned signal. The
"preference model" is a dot product against a fixed weight (the same stand-in
``tests/test_agent_recommender.py`` uses), and the QIL is loaded with chosen
posteriors so that the credential-blended ("after") ranking can be made to beat
or tie the credential-less quality-only ("before") ranking deterministically.
"""

import numpy as np
import pytest

from preferencelayer.agent.recommender import AgentRecommender
from preferencelayer.eval.partner import (
    PartnerQuery,
    PartnerResult,
    gate_passed,
    measure_partner,
    partner_improved,
)
from preferencelayer.models.base import Recommender
from preferencelayer.qil.aggregate import QualityAggregator
from preferencelayer.qil.extract import ExtractedSignal
from preferencelayer.qil.query import QualityService


class _DotModel(Recommender):
    """Scores a candidate as the dot product of its attributes with a weight."""

    name = "dot"

    def __init__(self, weight):
        self._w = np.asarray(weight, dtype=float)

    def fit(self, purchased, catalog, n_shared, population=None):
        return {"w": self._w}

    def score(self, state, candidates, n_shared):
        return np.asarray(candidates, dtype=float) @ state["w"]


def _service(quality_by_pid: dict[str, float], n_obs: int = 12) -> QualityService:
    """A QIL where each product gets ``n_obs`` observations at a chosen quality."""
    sigs = []
    for pid, v in quality_by_pid.items():
        for _ in range(n_obs):
            for dim in ("thermal", "build_quality"):
                sigs.append(ExtractedSignal(pid, "laptops", "gaming", "performance", None, dim, v, 0.9))
    return QualityService(QualityAggregator().fit(sigs))


def _agent(weight, quality_by_pid, **kw) -> AgentRecommender:
    model = _DotModel(weight)
    state = model.fit(None, np.zeros((1, len(weight))), len(weight))
    return AgentRecommender(model, state, _service(quality_by_pid), n_shared=len(weight), **kw)


# --------------------------------------------------------------------------- task builders
def _improving_queries(n: int = 12) -> list[PartnerQuery]:
    """Tasks where preference and quality *disagree*, and preference is right.

    The relevant item is the one the user's preference weight loves, but it has
    the *lowest* community quality, so a credential-less quality-only "before"
    ranking puts it last. The credential ("after") leans on preference (high
    confidence) and surfaces it. This makes after > before deterministically,
    without relying on argsort tie-break order.
    """
    queries = []
    for i in range(n):
        # 'good' wins on preference (strong rewarded attribute) but is rated worst.
        ids = [f"q{i}_good", f"q{i}_mid", f"q{i}_bad"]
        attrs = np.array([[3.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
        queries.append(PartnerQuery(
            query_id=f"q{i}",
            candidate_ids=ids,
            candidate_attrs=attrs,
            relevant_ids=[ids[0]],
            use_profile="gaming",
        ))
    return queries


def _flat_queries(n: int = 12) -> list[PartnerQuery]:
    """Tasks where preference is uninformative (zero weight) and quality is flat.

    With identical quality and a zero preference contribution, neither condition
    can order the candidates better than chance, so after does not beat before.
    """
    queries = []
    for i in range(n):
        ids = [f"f{i}_a", f"f{i}_b", f"f{i}_c"]
        attrs = np.zeros((3, 2))
        queries.append(PartnerQuery(
            query_id=f"f{i}",
            candidate_ids=ids,
            candidate_attrs=attrs,
            relevant_ids=[ids[1]],
            use_profile="gaming",
        ))
    return queries


def _improving_result(partner_id="p_improve") -> PartnerResult:
    qs = _improving_queries()
    # The relevant 'good' item is rated *worst* by the community, so quality-only
    # ranks it last; preference must rescue it.
    quality = {}
    for q in qs:
        quality[q.candidate_ids[0]] = 0.1   # good: loved by user, poorly reviewed
        quality[q.candidate_ids[1]] = 0.9   # mid:  best reviewed
        quality[q.candidate_ids[2]] = 0.5   # bad
    agent = _agent([1.0, 1.0], quality)
    return measure_partner(partner_id, agent, qs, mean_confidence=0.9)


def _flat_result(partner_id="p_flat") -> PartnerResult:
    qs = _flat_queries()
    quality = {cid: 0.5 for q in qs for cid in q.candidate_ids}
    agent = _agent([0.0, 0.0], quality)
    return measure_partner(partner_id, agent, qs, mean_confidence=0.5)


# --------------------------------------------------------------------------- shape / wiring
def test_result_shape_and_per_query_counts():
    res = _improving_result()
    assert res.n_queries == 12
    assert res.k == 10
    assert len(res.before.per_query_ndcg) == 12
    assert len(res.after.per_query_ndcg) == 12
    assert 0.0 <= res.before.ndcg <= 1.0
    assert 0.0 <= res.after.ndcg <= 1.0


def test_empty_queries_rejected():
    agent = _agent([1.0, 1.0], {"x": 0.5})
    with pytest.raises(ValueError):
        measure_partner("p", agent, [], mean_confidence=0.5)


def test_attr_row_mismatch_rejected():
    agent = _agent([1.0, 1.0], {"a": 0.5, "b": 0.5})
    bad = PartnerQuery("bad", ["a", "b"], np.zeros((1, 2)), ["a"], "gaming")
    with pytest.raises(ValueError):
        measure_partner("p", agent, [bad], mean_confidence=0.5)


# --------------------------------------------------------------------------- the measurement
def test_credential_improves_when_preference_is_the_signal():
    """After-credential ranking beats the credential-less quality-only baseline."""
    res = _improving_result()
    assert res.after.ndcg > res.before.ndcg
    assert res.abs_gain > 0.0
    assert res.rel_gain_pct > 0.0


def test_no_improvement_when_no_signal():
    """When preference carries no signal, the credential does not help."""
    res = _flat_result()
    assert res.abs_gain == pytest.approx(0.0, abs=1e-9)


def test_custom_before_ranker_is_used():
    """A partner can supply their own baseline; it is what 'before' measures."""
    qs = _improving_queries()
    quality = {cid: 0.5 for q in qs for cid in q.candidate_ids}
    agent = _agent([1.0, 1.0], quality)

    # An oracle baseline that already puts the relevant item first => no headroom.
    def oracle(candidate_ids, candidate_attrs, use_profile):
        # 'good' is index 0 by construction.
        return [0, 1, 2]

    res = measure_partner("p", agent, qs, mean_confidence=0.9, before_ranker=oracle)
    assert res.before.ndcg == pytest.approx(1.0)
    assert res.abs_gain <= 0.0


# --------------------------------------------------------------------------- gate logic
def test_partner_improved_true_for_significant_gain():
    res = _improving_result()
    assert partner_improved(res) is True
    assert res.improved is True


def test_partner_improved_false_without_gain():
    res = _flat_result()
    assert partner_improved(res) is False


def test_partner_improved_respects_min_abs_gain():
    res = _improving_result()
    # An absurdly high practical-significance bar is not cleared.
    assert partner_improved(res, min_abs_gain=0.99) is False


def test_gate_passes_with_two_of_five_improving():
    """Phase 1 WS-C gate: ≥2 of 5 partners show measurable improvement."""
    results = [
        _improving_result("p1"),
        _improving_result("p2"),
        _flat_result("p3"),
        _flat_result("p4"),
        _flat_result("p5"),
    ]
    report = gate_passed(results)
    assert report.n_partners == 5
    assert report.n_improved == 2
    assert report.passed is True
    assert set(report.improved_partners) == {"p1", "p2"}


def test_gate_fails_with_one_of_five_improving():
    results = [
        _improving_result("p1"),
        _flat_result("p2"),
        _flat_result("p3"),
        _flat_result("p4"),
        _flat_result("p5"),
    ]
    report = gate_passed(results)
    assert report.n_improved == 1
    assert report.passed is False


def test_gate_required_threshold_is_configurable():
    results = [_improving_result("p1"), _improving_result("p2"), _flat_result("p3")]
    assert gate_passed(results, required=2).passed is True
    assert gate_passed(results, required=3).passed is False
