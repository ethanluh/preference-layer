"""Experiment harness: within-category and cross-category transfer evaluation.

The protocol for each user:

* **Within-category:** fit the model on the user's purchases in a category, then
  rank a candidate set (held-out ground-truth relevant items + sampled negatives)
  in the *same* category. Reported as a sanity check.
* **Cross-category transfer (the headline):** fit on the user's purchases in the
  *source* category, then rank candidates in the *target* category. No target-
  category history is ever shown to the model, so any lift over the popularity
  floor is pure transferred preference signal.

Candidate sets are built once per user (independent of model) so every model is
scored on identical ranking problems.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..data.synthetic import CategoryData, SyntheticDataset
from ..models.base import Recommender
from ..models.popularity import PopularityRecommender
from . import metrics


@dataclass
class ModelResult:
    name: str
    ndcg: float
    ndcg_std: float
    recall: float
    hit_rate: float
    per_user_ndcg: list[float] = field(default_factory=list)


@dataclass
class Comparison:
    """Paired comparison of the graph model against a baseline."""

    baseline: str
    graph_ndcg: float
    baseline_ndcg: float
    abs_gain: float
    rel_gain_pct: float
    p_value: float  # paired bootstrap, two-sided


def _paired_bootstrap_p(a: list[float], b: list[float], n_boot: int = 5000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a) - mean(b) != 0 via paired bootstrap."""
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    diff = a_arr - b_arr
    n = len(diff)
    observed = diff.mean()
    if n == 0:
        return 1.0
    # Bootstrap the sampling distribution of the mean difference.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    # Center at 0 to form the null, then count how often |null| >= |observed|.
    centered = boot_means - observed
    p = (np.abs(centered) >= abs(observed)).mean()
    return float(min(1.0, max(p, 1.0 / n_boot)))


class ExperimentHarness:
    def __init__(self, dataset: SyntheticDataset, k: int = 10, n_candidates: int = 100, seed: int = 13):
        self.ds = dataset
        self.k = k
        self.n_candidates = n_candidates
        self.seed = seed

    # ----------------------------------------------------------- candidate sets
    def _candidates(self, target: CategoryData, uid: str, exclude: set[str]) -> tuple[list[str], list[str]]:
        """Use the benchmark's pre-built candidate set (relevant + hard negatives + fill).

        ``exclude`` removes items the model was fit on (within-category split); for
        cross-category transfer it is empty.
        """
        cand = [c for c in target.eval_candidates[uid] if c not in exclude]
        relevant = [r for r in target.relevant[uid] if r not in exclude and r in cand]
        return cand, relevant

    # --------------------------------------------------------------- evaluation
    def _fit_and_score(
        self,
        factory: Callable[[], Recommender],
        source: CategoryData,
        target: CategoryData,
        same_category: bool,
    ) -> ModelResult:
        model = factory()
        attr_index_t = target.item_index()
        src_index = source.item_index()

        # Per-user purchase grouping drives both edge discovery and the prior.
        per_user_purchased = [
            np.stack([src_index[i].attributes for i in source.purchases[uid]])
            for uid in self.ds.user_ids
        ]
        _, source_catalog = source.item_matrix()
        model.prepare(source_catalog, per_user_purchased, source.schema.n_shared)

        per_user = []
        recalls = []
        hits = []

        for uid in self.ds.user_ids:
            purchased_ids = source.purchases[uid]
            if same_category:
                # Fit on a prefix, evaluate on held-out ground truth.
                fit_ids = purchased_ids[: max(1, len(purchased_ids) // 2)]
                exclude = set(fit_ids)
            else:
                fit_ids = purchased_ids
                exclude = set()  # different catalog; nothing to exclude

            purchased = np.stack([src_index[i].attributes for i in fit_ids])
            state = model.fit(purchased, source_catalog, source.schema.n_shared)

            cand, relevant = self._candidates(target, uid, exclude)
            if not relevant:
                continue
            cand_attrs = np.stack([attr_index_t[c].attributes for c in cand])
            scores = model.score(state, cand_attrs, target.schema.n_shared)
            order = np.argsort(-scores)
            ranking = [cand[i] for i in order]

            per_user.append(metrics.ndcg_at_k(ranking, relevant, self.k))
            recalls.append(metrics.recall_at_k(ranking, relevant, self.k))
            hits.append(metrics.hit_rate_at_k(ranking, relevant, self.k))

        return ModelResult(
            name=model.name,
            ndcg=float(np.mean(per_user)),
            ndcg_std=float(np.std(per_user)),
            recall=float(np.mean(recalls)),
            hit_rate=float(np.mean(hits)),
            per_user_ndcg=per_user,
        )

    def _popularity(self, source: CategoryData, target: CategoryData, same_category: bool) -> ModelResult:
        counts = Counter(i for u in self.ds.user_ids for i in target.purchases[u])
        ids = [it.item_id for it in target.items]
        pop = PopularityRecommender(ids, counts)
        per_user, recalls, hits = [], [], []
        for uid in self.ds.user_ids:
            exclude = set(source.purchases[uid][: max(1, len(source.purchases[uid]) // 2)]) if same_category else set()
            cand, relevant = self._candidates(target, uid, exclude)
            if not relevant:
                continue
            scores = pop.score_ids(cand)
            order = np.argsort(-scores)
            ranking = [cand[i] for i in order]
            per_user.append(metrics.ndcg_at_k(ranking, relevant, self.k))
            recalls.append(metrics.recall_at_k(ranking, relevant, self.k))
            hits.append(metrics.hit_rate_at_k(ranking, relevant, self.k))
        return ModelResult("popularity", float(np.mean(per_user)), float(np.std(per_user)),
                           float(np.mean(recalls)), float(np.mean(hits)), per_user)

    # -------------------------------------------------------------- public API
    def run_transfer(
        self,
        factories: dict[str, Callable[[], Recommender]],
        source: str,
        target: str,
    ) -> dict[str, ModelResult]:
        src, tgt = self.ds.categories[source], self.ds.categories[target]
        results = {"popularity": self._popularity(src, tgt, same_category=False)}
        for name, fac in factories.items():
            results[name] = self._fit_and_score(fac, src, tgt, same_category=False)
        return results

    def run_within(
        self,
        factories: dict[str, Callable[[], Recommender]],
        category: str,
    ) -> dict[str, ModelResult]:
        cat = self.ds.categories[category]
        results = {"popularity": self._popularity(cat, cat, same_category=True)}
        for name, fac in factories.items():
            results[name] = self._fit_and_score(fac, cat, cat, same_category=True)
        return results

    @staticmethod
    def compare(results: dict[str, ModelResult], graph_key: str = "preference_graph") -> list[Comparison]:
        graph = results[graph_key]
        out = []
        for name, res in results.items():
            if name == graph_key:
                continue
            abs_gain = graph.ndcg - res.ndcg
            rel = 100.0 * abs_gain / res.ndcg if res.ndcg > 0 else float("nan")
            p = (
                _paired_bootstrap_p(graph.per_user_ndcg, res.per_user_ndcg)
                if len(graph.per_user_ndcg) == len(res.per_user_ndcg)
                else float("nan")
            )
            out.append(Comparison(name, graph.ndcg, res.ndcg, abs_gain, rel, p))
        return out
