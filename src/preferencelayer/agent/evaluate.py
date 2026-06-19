"""Integration evaluation: does combining the two layers beat either alone?

Runs four ranking conditions on the integrated benchmark and reports NDCG@10
with paired-bootstrap significance — the same protocol the Phase 0 harness uses
for Claims 1 & 2, applied here to the *combination*:

1. ``preference_only``  (alpha = 1) — the preference graph alone;
2. ``quality_only``     (alpha = 0) — the QIL alone;
3. ``fixed_alpha``      (alpha = 0.5) — a constant 50/50 blend;
4. ``adaptive_alpha``   — the confidence-adaptive blend from ``architecture.md``.

The preference graph and quality service are built once and shared across all
conditions, and every condition is scored on the *same* candidate set per user,
so differences are purely the blend.

What this benchmark does and does not show
------------------------------------------
The **milestone** is the integration thesis itself: an agent that fuses
preference and quality ranks better than one using either signal alone. That is
what ``milestone_pass`` gates on — the α-blend must beat *both* single-signal
baselines, significantly.

A second, subtler question — whether *confidence-adaptive* α beats a *fixed* α —
is reported but **not** gated, because this controlled setting answers it in the
negative and we report that honestly. With z-score-normalized scores and
additive utility, the ideal blend weight is ``1 / (1 + quality_weight)``,
essentially the same for every user; per-user preference-fit noise only nudges it
down a little for sparse users. So the optimal α barely varies across the
population (see :meth:`IntegrationHarness.optimal_alpha_by_cohort`), a fixed
balanced blend is hard to beat, and the documented ``sigmoid(3·(c−0.5))`` swings
*too far* at the extremes. Confidence-adaptation would pay off in regimes this
benchmark deliberately does not model — e.g. products with sparse, uneven quality
evidence, or genuinely zero-history users with no usable prior. The harness still
breaks results out by history cohort and reports the (mild) cold→rich rise in the
optimal α, so the mechanism's *direction* is visible even though its documented
*calibration* is not vindicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.integrated import IntegratedScenario
from ..eval import metrics
from ..eval.harness import _paired_bootstrap_p
from ..qil.aggregate import QualityAggregator
from ..qil.query import QualityService
from ._harness import prepare_preference_model
from .combine import alpha_from_confidence
from .combine import blend as _blend
from .recommender import AgentRecommender

# α grid swept to locate each cohort's empirically-optimal blend weight.
_ALPHA_GRID = np.linspace(0.0, 1.0, 21)

# The four conditions, as (name, fixed-alpha-or-None). None means adaptive.
CONDITIONS: tuple[tuple[str, float | None], ...] = (
    ("preference_only", 1.0),
    ("quality_only", 0.0),
    ("fixed_alpha", 0.5),
    ("adaptive_alpha", None),
)


@dataclass
class ConditionResult:
    name: str
    ndcg: float
    ndcg_std: float
    per_user_ndcg: list[float] = field(default_factory=list)
    mean_alpha: float = float("nan")


@dataclass
class CohortBreakdown:
    cohort: str
    n_users: int
    by_condition: dict[str, float]   # condition name -> mean NDCG@10
    mean_confidence: float


@dataclass
class IntegrationReport:
    conditions: dict[str, ConditionResult]
    cohorts: list[CohortBreakdown]
    comparisons: dict[str, tuple[float, float]]  # other condition -> (abs_gain, p_value)
    milestone_pass: bool
    # Optional per-cohort optimal-alpha curve (filled by run(with_alpha_curve=True)).
    optimal_alpha: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def headline(self) -> ConditionResult:
        return self.conditions["adaptive_alpha"]

    @property
    def adaptive_vs_fixed(self) -> tuple[float, float]:
        """(gain, p) of adaptive over the fixed-0.5 blend — reported, not gated."""
        return self.comparisons["fixed_alpha"]


class IntegrationHarness:
    def __init__(self, scenario: IntegratedScenario, k: int = 10, seed: int = 13):
        self.s = scenario
        self.k = k
        self.seed = seed

    def run(self, with_alpha_curve: bool = False) -> IntegrationReport:
        model, idx, catalog, n_shared = prepare_preference_model(self.s, seed=self.seed)
        service = QualityService(QualityAggregator().fit(self.s.signals))

        # Per-condition per-user NDCG, plus per-cohort accumulation.
        per_user: dict[str, list[float]] = {name: [] for name, _ in CONDITIONS}
        alphas: list[float] = []
        cohort_acc: dict[str, dict[str, list[float]]] = {}
        cohort_conf: dict[str, list[float]] = {}
        # Per-cohort α-sweep NDCG curves (only when the optimal-α analysis is requested).
        cohort_curve: dict[str, list[np.ndarray]] = {}

        for u in self.s.users:
            purchased = np.stack([idx[pid].attributes for pid in u.purchases])
            state = model.fit(purchased, catalog, n_shared)
            agent = AgentRecommender(model, state, service, n_shared)

            cand_ids = u.candidates
            cand_attrs = np.stack([idx[c].attributes for c in cand_ids])
            relevant = set(u.relevant)

            cohort_acc.setdefault(u.cohort, {name: [] for name, _ in CONDITIONS})
            cohort_conf.setdefault(u.cohort, []).append(u.mean_confidence)

            # Compute each stream once, then blend per condition — every condition is
            # the same pref/quality re-weighted by a different α, so there is no need
            # to recompute the preference score or re-query the QIL per condition.
            pref = agent.preference_scores(cand_attrs)
            quality, _evidence = agent.query_quality(cand_ids, u.use_profile)

            for name, alpha in CONDITIONS:
                a = alpha_from_confidence(u.mean_confidence) if alpha is None else alpha
                ranking = [cand_ids[i] for i in np.argsort(-_blend(pref, quality, a))]
                ndcg = metrics.ndcg_at_k(ranking, relevant, self.k)
                per_user[name].append(ndcg)
                cohort_acc[u.cohort][name].append(ndcg)
                if name == "adaptive_alpha":
                    alphas.append(a)

            if with_alpha_curve:
                cohort_curve.setdefault(u.cohort, []).append(np.array([
                    metrics.ndcg_at_k(
                        [cand_ids[i] for i in np.argsort(-_blend(pref, quality, a))], relevant, self.k)
                    for a in _ALPHA_GRID
                ]))

        conditions = {
            name: ConditionResult(
                name=name,
                ndcg=float(np.mean(per_user[name])),
                ndcg_std=float(np.std(per_user[name])),
                per_user_ndcg=per_user[name],
                mean_alpha=float(np.mean(alphas)) if name == "adaptive_alpha" else float("nan"),
            )
            for name, _ in CONDITIONS
        }

        cohorts = [
            CohortBreakdown(
                cohort=c,
                n_users=len(cohort_conf[c]),
                by_condition={name: float(np.mean(vals)) for name, vals in cohort_acc[c].items()},
                mean_confidence=float(np.mean(cohort_conf[c])),
            )
            # Order cohorts cold -> warm -> rich for readability.
            for c in ("cold", "warm", "rich")
            if c in cohort_acc
        ]

        adaptive = conditions["adaptive_alpha"]
        comparisons: dict[str, tuple[float, float]] = {}
        for name, res in conditions.items():
            if name == "adaptive_alpha":
                continue
            gain = adaptive.ndcg - res.ndcg
            p = _paired_bootstrap_p(adaptive.per_user_ndcg, res.per_user_ndcg)
            comparisons[name] = (gain, p)

        # Milestone = the integration thesis: the α-blend beats *both* single
        # layers, significantly. (Adaptive-vs-fixed is reported separately and
        # deliberately not gated; see the module docstring.)
        milestone_pass = all(
            comparisons[name][0] > 0 and comparisons[name][1] < 0.05
            for name in ("preference_only", "quality_only")
        )

        # Empirically optimal α per cohort (the honest adaptive-vs-fixed analysis):
        # the α that maximizes mean NDCG@10, computed from the same per-user streams
        # already swept above. Its mild cold->rich rise is the evidence that adaptive
        # α points the right *direction* even though the documented slope overshoots.
        optimal_alpha: list[tuple[str, float, float]] = []
        if with_alpha_curve:
            for c in ("cold", "warm", "rich"):
                if c not in cohort_curve:
                    continue
                mean_curve = np.mean(cohort_curve[c], axis=0)
                optimal_alpha.append(
                    (c, float(np.mean(cohort_conf[c])), float(_ALPHA_GRID[int(mean_curve.argmax())]))
                )

        return IntegrationReport(
            conditions=conditions,
            cohorts=cohorts,
            comparisons=comparisons,
            milestone_pass=milestone_pass,
            optimal_alpha=optimal_alpha,
        )
