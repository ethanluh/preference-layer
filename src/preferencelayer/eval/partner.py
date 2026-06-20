"""Design-partner before/after relevance measurement harness (Phase 1 WS-C).

This is the *partner-facing* counterpart to the internal
:class:`~preferencelayer.agent.evaluate.IntegrationHarness`. Where that harness
proves the integration thesis on our own synthetic benchmark, this one lets an
external design partner answer the only question the Phase 1 go/no-go gate cares
about, **on their own task and their own data**:

    Does attaching the user's PreferenceLayer credential measurably improve the
    relevance of my agent's recommendations versus running without it?

The design is deliberately minimal so a partner can wire it up against their
existing agent in an afternoon (see ``docs/design-partner-onboarding.md``):

* A partner expresses their task as a list of :class:`PartnerQuery` cases — one
  per ranking decision their agent makes. Each case carries the candidate item
  ids, the candidate attribute matrix (in the schema the credential was issued
  against), the ground-truth relevant ids, and the *use profile* string the QIL
  is queried with.
* The partner supplies a fitted preference model + per-user state (obtained from
  their user's credential via ``GET /preference`` — see the onboarding guide) and
  a :class:`~preferencelayer.qil.query.QualityService` handle (the QIL
  ``/quality`` endpoint, or its MCP tool).
* :func:`measure_partner` runs two conditions on the *same* candidate sets —
  **before** (no credential; quality-only cold-start ranking, ``alpha=0``) and
  **after** (the credential-blended ranking) — and reports NDCG@10, recall@10,
  and MRR for each, the per-query deltas, and a paired-bootstrap p-value.

The "before" condition models an agent that has no portable preference
credential to read: it can still consult community quality (the QIL) but has
nothing user-specific, which is exactly the cold-start situation the credential
is meant to fix. A partner whose own baseline differs (e.g. a popularity prior)
can pass ``before_ranker`` to override it; the *after* condition is always the
α-blend over the credential.

Everything here is pure measurement built on the public surfaces already in the
repo (:mod:`preferencelayer.eval.metrics`,
:class:`~preferencelayer.agent.recommender.AgentRecommender`) — no schema or
protocol changes. The gate logic (:func:`partner_improved`, :func:`gate_passed`)
implements the Phase 1 WS-C criterion verbatim: **at least 2 of 5 design
partners show a measurable improvement in recommendation relevance.**
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..agent.recommender import AgentRecommender
from ..eval import metrics
from ..eval.harness import _paired_bootstrap_p

# A "before" ranker takes (candidate_ids, candidate_attrs, use_profile) and
# returns candidate indices best-first. The default models a credential-less
# agent (quality-only); partners can inject their own production baseline.
BeforeRanker = Callable[[list[str], np.ndarray, str], Sequence[int]]


@dataclass
class PartnerQuery:
    """One ranking decision in a partner's task.

    Parameters
    ----------
    query_id:
        Partner-chosen identifier for this case (for their own bookkeeping).
    candidate_ids:
        Product ids the agent must rank (these are the ids the QIL is keyed on).
    candidate_attrs:
        ``(n_candidates, dim)`` attribute matrix in the schema the credential was
        issued against — the same matrix the preference model scores.
    relevant_ids:
        Ground-truth relevant ids for this query (what the user actually wanted /
        bought / clicked). Binary relevance; supply ``relevance_map`` for graded.
    use_profile:
        The use-profile string passed to the QIL ``/quality`` call (e.g.
        ``"gaming"``, ``"professional"``). Quality signals are always
        use-profile-conditioned — never population aggregates.
    relevance_map:
        Optional graded relevance (id -> gain) overriding binary relevance.
    """

    query_id: str
    candidate_ids: list[str]
    candidate_attrs: np.ndarray
    relevant_ids: list[str]
    use_profile: str
    relevance_map: dict[str, float] | None = None


@dataclass
class ConditionScore:
    """Aggregate metrics for one condition (before or after) over all queries."""

    name: str
    ndcg: float
    recall: float
    mrr: float
    per_query_ndcg: list[float] = field(default_factory=list)


# Default minimum number of queries before a significance verdict is trusted.
# A paired bootstrap over a handful of queries can flag random noise as
# "significant", so below this floor we refuse the verdict and surface the
# result as underpowered instead. 20 is a conventional small-sample floor; a
# partner with fewer queries should gather more before relying on the gate.
DEFAULT_MIN_QUERIES = 20


@dataclass
class PartnerResult:
    """Before/after measurement for a single design partner."""

    partner_id: str
    n_queries: int
    before: ConditionScore
    after: ConditionScore
    k: int
    abs_gain: float          # after.ndcg - before.ndcg
    rel_gain_pct: float      # 100 * abs_gain / before.ndcg
    p_value: float           # paired bootstrap on per-query NDCG@k, two-sided

    @property
    def underpowered(self) -> bool:
        """Convenience: too few queries to trust the significance verdict?

        Compares :attr:`n_queries` against the default :data:`DEFAULT_MIN_QUERIES`
        floor. When ``True``, :attr:`improved` is forced to ``False`` regardless
        of the p-value — see :func:`partner_improved`.
        """
        return self.n_queries < DEFAULT_MIN_QUERIES

    @property
    def improved(self) -> bool:
        """Convenience: did this partner show a measurable improvement?

        Uses :func:`partner_improved` with its default thresholds. See that
        function for the criterion and how to tighten/loosen it.
        """
        return partner_improved(self)


def _score_condition(
    name: str,
    per_query_rankings: list[list[str]],
    queries: Sequence[PartnerQuery],
    k: int,
) -> ConditionScore:
    ndcgs, recalls, mrrs = [], [], []
    for ranking, q in zip(per_query_rankings, queries):
        relevant = set(q.relevant_ids)
        ndcgs.append(metrics.ndcg_at_k(ranking, relevant, k, q.relevance_map))
        recalls.append(metrics.recall_at_k(ranking, relevant, k))
        mrrs.append(metrics.mrr(ranking, relevant))
    return ConditionScore(
        name=name,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        mrr=float(np.mean(mrrs)) if mrrs else 0.0,
        per_query_ndcg=ndcgs,
    )


def _default_before_ranker(agent: AgentRecommender) -> BeforeRanker:
    """Credential-less baseline: rank on community quality alone (``alpha=0``).

    This is the honest "no PreferenceLayer credential" condition — the agent has
    no user-specific signal, so it falls back to use-profile-conditioned quality.
    """

    def ranker(candidate_ids: list[str], candidate_attrs: np.ndarray, use_profile: str):
        res = agent.rank(
            candidate_ids, candidate_attrs, use_profile,
            mean_confidence=0.5, alpha=0.0,
        )
        return res.order

    return ranker


def measure_partner(
    partner_id: str,
    agent: AgentRecommender,
    queries: Sequence[PartnerQuery],
    *,
    mean_confidence: float,
    k: int = 10,
    alpha: float | None = None,
    evidence_aware: bool = False,
    before_ranker: BeforeRanker | None = None,
    bootstrap_seed: int = 0,
) -> PartnerResult:
    """Run the before/after relevance measurement for one design partner.

    Parameters
    ----------
    partner_id:
        Identifier for the partner (appears in the result and the gate report).
    agent:
        A :class:`~preferencelayer.agent.recommender.AgentRecommender` already
        wired to the partner's user credential (preference model + state) and the
        QIL :class:`~preferencelayer.qil.query.QualityService`.
    queries:
        The partner's task, one :class:`PartnerQuery` per ranking decision.
    mean_confidence:
        The credential's mean node confidence, as returned by ``GET /preference``.
        Drives the adaptive blend weight α in the *after* condition.
    k:
        Cutoff for NDCG@k / recall@k (default 10, the gate metric).
    alpha:
        Optional fixed blend weight for the *after* condition. ``None`` (default)
        uses the confidence-adaptive α from ``architecture.md``.
    evidence_aware:
        If ``True`` (and ``alpha`` is ``None``), the *after* condition uses the
        per-candidate evidence-aware α (leans on preference where quality evidence
        is thin). Default ``False`` (confidence-only adaptive α).
    before_ranker:
        Optional override for the baseline. Defaults to a credential-less
        quality-only ranking (``alpha=0``) — see :func:`_default_before_ranker`.
    bootstrap_seed:
        Seed for the paired-bootstrap p-value (reproducibility).

    Returns
    -------
    PartnerResult
        Aggregate before/after metrics, per-query deltas, and significance.
    """
    if not queries:
        raise ValueError("measure_partner needs at least one PartnerQuery")

    before_fn = before_ranker or _default_before_ranker(agent)

    before_rankings: list[list[str]] = []
    after_rankings: list[list[str]] = []
    for q in queries:
        attrs = np.asarray(q.candidate_attrs, dtype=float)
        if attrs.shape[0] != len(q.candidate_ids):
            raise ValueError(
                f"query {q.query_id!r}: candidate_attrs has {attrs.shape[0]} rows "
                f"but {len(q.candidate_ids)} candidate ids"
            )

        order = before_fn(q.candidate_ids, attrs, q.use_profile)
        before_rankings.append([q.candidate_ids[i] for i in order])

        after = agent.rank(
            q.candidate_ids, attrs, q.use_profile,
            mean_confidence=mean_confidence,
            alpha=alpha, evidence_aware=evidence_aware,
        )
        after_rankings.append([q.candidate_ids[i] for i in after.order])

    before = _score_condition("before", before_rankings, queries, k)
    after = _score_condition("after", after_rankings, queries, k)

    abs_gain = after.ndcg - before.ndcg
    rel_gain = 100.0 * abs_gain / before.ndcg if before.ndcg > 0 else float("nan")
    p = _paired_bootstrap_p(after.per_query_ndcg, before.per_query_ndcg, seed=bootstrap_seed)

    return PartnerResult(
        partner_id=partner_id,
        n_queries=len(queries),
        before=before,
        after=after,
        k=k,
        abs_gain=abs_gain,
        rel_gain_pct=rel_gain,
        p_value=p,
    )


def partner_improved(
    result: PartnerResult,
    *,
    min_abs_gain: float = 0.0,
    max_p_value: float = 0.05,
    min_queries: int = DEFAULT_MIN_QUERIES,
) -> bool:
    """Did one partner show a *measurable* improvement in relevance?

    "Measurable" is operationalized as: the after-credential NDCG@k is higher
    than the before condition by more than ``min_abs_gain`` (default: any positive
    gain), **and** that improvement is statistically significant at
    ``max_p_value`` on the paired bootstrap (default p < 0.05), **and** the
    measurement is based on at least ``min_queries`` queries
    (default :data:`DEFAULT_MIN_QUERIES`).

    The ``min_queries`` floor is a power guard: a paired bootstrap over only a
    handful of queries can flag random noise as "significant", so below the floor
    we refuse the verdict outright (return ``False``) rather than trip the gate on
    an underpowered sample. Use :func:`is_underpowered` (or
    :attr:`PartnerResult.underpowered`) to distinguish "no improvement" from "too
    few queries to tell". Pass ``min_queries=0`` to disable the guard.

    Tighten ``min_abs_gain`` (e.g. ``0.02``) if the gate should demand a
    practically meaningful lift, not merely a statistically detectable one.
    """
    if result.n_queries < min_queries:
        return False
    return result.abs_gain > min_abs_gain and result.p_value < max_p_value


def is_underpowered(
    result: PartnerResult, *, min_queries: int = DEFAULT_MIN_QUERIES
) -> bool:
    """Were there too few queries to trust the significance verdict?

    Returns ``True`` when ``result.n_queries < min_queries``. In that case
    :func:`partner_improved` refuses to report an improvement on significance
    alone — the result should be surfaced as underpowered, not as a pass/fail.
    """
    return result.n_queries < min_queries


@dataclass
class GateReport:
    """Phase 1 WS-C go/no-go report across the design-partner cohort."""

    n_partners: int
    n_improved: int
    required: int
    passed: bool
    improved_partners: list[str]
    per_partner: dict[str, bool]
    underpowered_partners: list[str] = field(default_factory=list)


def gate_passed(
    results: Sequence[PartnerResult],
    *,
    required: int = 2,
    min_abs_gain: float = 0.0,
    max_p_value: float = 0.05,
    min_queries: int = DEFAULT_MIN_QUERIES,
) -> GateReport:
    """Evaluate the Phase 1 WS-C gate: ≥ ``required`` partners show improvement.

    The plan's gate is *"at least 2 of 5 design partners report a measurable
    improvement in recommendation relevance."* This counts how many partners in
    ``results`` clear :func:`partner_improved` and compares against ``required``
    (default 2). The denominator is whatever cohort you pass — the gate is about
    an absolute count of improvers, not a fraction, so a 2-of-3 or 2-of-5 cohort
    both pass at ``required=2``.

    The ``min_queries`` power guard is applied per partner: a partner measured on
    fewer than ``min_queries`` queries is **not** counted as an improver on
    significance alone, and is instead listed in
    :attr:`GateReport.underpowered_partners` so the cohort owner can see the gate
    was not tripped by a tiny sample. Pass ``min_queries=0`` to disable.

    Returns a :class:`GateReport` so a caller can see *which* partners improved,
    not just whether the threshold was met.
    """
    per_partner = {
        r.partner_id: partner_improved(
            r,
            min_abs_gain=min_abs_gain,
            max_p_value=max_p_value,
            min_queries=min_queries,
        )
        for r in results
    }
    improved = [pid for pid, ok in per_partner.items() if ok]
    underpowered = [
        r.partner_id for r in results if is_underpowered(r, min_queries=min_queries)
    ]
    return GateReport(
        n_partners=len(results),
        n_improved=len(improved),
        required=required,
        passed=len(improved) >= required,
        improved_partners=improved,
        per_partner=per_partner,
        underpowered_partners=underpowered,
    )
