"""How should the agent turn raw quality evidence into a quality score?

The integration milestone showed that *combining* preference and quality is the
big win, and two follow-ups (confidence-adaptive α, then evidence-aware α) showed
that *how the blend is weighted* barely matters — a fixed balanced α is hard to
beat. This module asks the question that actually does move the needle: **how the
quality estimate itself is formed from noisy, unevenly-distributed evidence.**

It compares two estimators, each crossed with two blend weights (a 2×2):

* **estimator** — *Bayesian-shrunk* posteriors (the shipped QIL aggregator, which
  pulls thin-evidence products toward a neutral prior) vs. *raw* confidence-weighted
  sample means (a `QualityAggregator` with ``prior_strength → 0`` — the
  evidence-ignoring ablation);
* **blend weight** — a fixed α=0.5 vs. the per-candidate evidence-aware α.

The headline contrast is the two **fixed-α** cells across a sweep of per-observation
noise (driven by the experiment script): raw averaging is fine — even better — when
review signals are clean, but Bayesian shrinkage is the **noise-robust** choice and
wins once individual signals are noisy (the realistic regime for messy public text).
The two evidence-aware cells are reported alongside to show that α-level
evidence-handling does not help on top of either estimator.

Everything here reuses the existing apparatus: :class:`AgentRecommender`,
:class:`~preferencelayer.qil.aggregate.QualityAggregator` (just re-parameterized),
``eval.metrics`` and the paired bootstrap from ``eval.harness``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.integrated import IntegratedScenario
from ..eval import metrics
from ..eval.harness import _paired_bootstrap_p
from ..qil.aggregate import QualityAggregator
from ..qil.query import QualityService
from . import combine
from ._harness import prepare_preference_model, purchase_matrix
from .recommender import AgentRecommender

# The 2×2 cells: (name, raw_estimator?, evidence_aware_alpha?).
_CELLS: tuple[tuple[str, bool, bool], ...] = (
    ("shrunk_fixed", False, False),   # (a) shipped: Bayesian shrinkage + fixed α
    ("shrunk_evidence", False, True),  # (d) shrinkage + evidence-aware α
    ("raw_fixed", True, False),        # (b) raw means + fixed α (evidence-ignoring)
    ("raw_evidence", True, True),      # (c) raw means + evidence-aware α
)


@dataclass
class CellResult:
    name: str
    ndcg: float
    ndcg_std: float
    per_user_ndcg: list[float] = field(default_factory=list)


@dataclass
class QualityHandlingResult:
    """One run at a fixed observation-noise level."""

    obs_noise: float
    cells: dict[str, CellResult]
    references: dict[str, CellResult]   # preference_only, quality_only (shrunk)

    def contrast(self, a: str, b: str) -> tuple[float, float]:
        """(mean NDCG gain of cell ``a`` over cell ``b``, paired-bootstrap p-value)."""
        ca, cb = self.cells[a], self.cells[b]
        return ca.ndcg - cb.ndcg, _paired_bootstrap_p(ca.per_user_ndcg, cb.per_user_ndcg)


class QualityHandlingHarness:
    """Runs the estimator×blend 2×2 (plus single-signal references) on one scenario.

    ``raw_prior_strength`` makes the "raw" aggregator's Normal-Normal posterior
    collapse to the confidence-weighted sample mean (no shrinkage). ``evidence_pivot``
    and ``evidence_quality_weight`` parameterize the evidence-aware α exactly as
    :meth:`AgentRecommender.rank` does.
    """

    def __init__(
        self,
        scenario: IntegratedScenario,
        k: int = 10,
        seed: int = 13,
        raw_prior_strength: float = 1e-9,
        evidence_pivot: float = 8.0,
        evidence_quality_weight: float = 1.0,
        obs_noise: float = float("nan"),
    ):
        self.s = scenario
        self.k = k
        self.seed = seed
        self.raw_prior_strength = raw_prior_strength
        self.evidence_pivot = evidence_pivot
        self.evidence_quality_weight = evidence_quality_weight
        # Label only — the per-observation noise the scenario was generated with,
        # carried through so a result is self-describing in a noise sweep.
        self.obs_noise = obs_noise

    def run(self) -> QualityHandlingResult:
        model, idx, catalog, n_shared = prepare_preference_model(self.s, seed=self.seed)

        # Two estimators over the *same* extracted signals: shipped shrinkage vs raw.
        shrunk = QualityService(QualityAggregator().fit(self.s.signals))
        raw = QualityService(QualityAggregator(prior_strength=self.raw_prior_strength).fit(self.s.signals))

        per_cell: dict[str, list[float]] = {name: [] for name, _, _ in _CELLS}
        ref: dict[str, list[float]] = {"preference_only": [], "quality_only": []}

        dim = self.s.schema.dim
        for u in self.s.users:
            purchased = purchase_matrix(idx, u.purchases, dim)
            state = model.fit(purchased, catalog, n_shared)
            agent_shrunk = AgentRecommender(model, state, shrunk, n_shared)
            agent_raw = AgentRecommender(model, state, raw, n_shared)

            cand_ids = u.candidates
            cand_attrs = np.stack([idx[c].attributes for c in cand_ids])
            relevant = set(u.relevant)

            # Compute each stream once: preference is estimator-independent; quality
            # and evidence are fetched once per estimator. Every cell is then just a
            # re-blend, avoiding repeated /quality queries and pref recomputation.
            pref = agent_shrunk.preference_scores(cand_attrs)
            streams = {
                False: agent_shrunk.query_quality(cand_ids, u.use_profile),  # shrunk
                True: agent_raw.query_quality(cand_ids, u.use_profile),      # raw
            }

            def ndcg(quality: np.ndarray, alpha) -> float:
                blended = combine.blend(pref, quality, alpha)
                return metrics.ndcg_at_k([cand_ids[i] for i in np.argsort(-blended)], relevant, self.k)

            for name, raw_est, ev in _CELLS:
                quality, evidence = streams[raw_est]
                if ev:
                    r_q = combine.quality_reliability(evidence, pivot=self.evidence_pivot)
                    alpha = combine.evidence_adaptive_alpha(
                        u.mean_confidence, r_q, quality_weight=self.evidence_quality_weight)
                else:
                    alpha = 0.5
                per_cell[name].append(ndcg(quality, alpha))

            # References (shrunk estimator): α=1 is pure preference, α=0 pure quality.
            shrunk_quality = streams[False][0]
            ref["preference_only"].append(ndcg(shrunk_quality, 1.0))
            ref["quality_only"].append(ndcg(shrunk_quality, 0.0))

        def collect(name: str, vals: list[float]) -> CellResult:
            return CellResult(name, float(np.mean(vals)), float(np.std(vals)), vals)

        return QualityHandlingResult(
            obs_noise=self.obs_noise,
            cells={name: collect(name, per_cell[name]) for name, _, _ in _CELLS},
            references={name: collect(name, vals) for name, vals in ref.items()},
        )
