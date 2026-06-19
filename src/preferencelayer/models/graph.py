"""Sparse DAG preference graph recommender.

This is the model the whole project is a bet on. It represents a user's
preference not as a flat vector but as a sparse graph over the shared attribute
vocabulary:

* **Nodes** are shared attributes, each with a learned linear weight ``theta``.
* **Edges** connect attribute pairs and carry an interaction weight ``phi``. An
  edge encodes a *conditional* preference / tradeoff — e.g. "battery life matters
  to me only when portability is also high". This is precisely the signal a flat
  (linear / mean) model cannot represent.

Topology is learned once from the source corpus via PMI over attribute
co-activation in purchase sequences (matching the design doc's
"PMI-based initialization, gradient-refined"), giving a *sparse* edge set. The
node and edge weights are then estimated per user with a regularized logistic
ranking objective (purchased items vs. sampled negatives) on standardized
features. Sparse-history users are blended toward a population prior — the
cold-start mechanism named in the PTP credential schema.

Because every node and edge lives in the shared attribute space, a graph fit on
one category transfers directly to another: the same nodes and edges apply to any
category's items. That is the mechanism behind the cross-category transfer claim.
"""

from __future__ import annotations

import numpy as np

from .base import PopulationPrior, Recommender


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SparsePreferenceGraph(Recommender):
    name = "preference_graph"

    def __init__(
        self,
        n_edges: int = 10,
        epochs: int = 300,
        lr: float = 0.3,
        l2: float = 0.02,
        n_negatives: int = 12,
        cold_start_pivot: int = 4,
        seed: int = 0,
    ):
        self.n_edges = n_edges
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.n_negatives = n_negatives
        # Number of purchases at which the per-user fit is trusted ~50/50 vs prior.
        self.cold_start_pivot = cold_start_pivot
        self.seed = seed
        # Populated by prepare().
        self.edges: list[tuple[int, int]] = []
        self.prior_w: np.ndarray | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._n_shared: int = 0

    # ------------------------------------------------------------------ topology
    def prepare(self, catalog: np.ndarray, per_user_purchased: list[np.ndarray], n_shared: int) -> None:
        """Learn the sparse edge topology, feature scaling, and population prior."""
        self._n_shared = n_shared
        self.edges = self._select_edges(per_user_purchased, n_shared)

        all_purchased = np.concatenate(per_user_purchased, axis=0)
        cat_feat = self._features(catalog, n_shared)
        self._feat_mean = cat_feat.mean(axis=0)
        self._feat_std = cat_feat.std(axis=0) + 1e-6

        # Empirical-Bayes-style cold-start prior: one population-level logistic fit.
        self.prior_w = self._logistic_fit(
            pos=self._std(self._features(all_purchased, n_shared)),
            neg=self._std(cat_feat),
            init=np.zeros(cat_feat.shape[1]),
            prior=np.zeros(cat_feat.shape[1]),
            prior_strength=0.0,
        )

    def _select_edges(self, per_user_purchased: list[np.ndarray], n_shared: int) -> list[tuple[int, int]]:
        """Discover interaction edges by cross-user variance of within-user correlation.

        A genuine interaction edge (a, b) shows up as a *conditional* dependence
        whose sign differs across users: a complement for one user (buys items
        high on both), a substitute for another (trades one off against the other).
        Pooled co-occurrence cancels this out, so a plain PMI over the whole corpus
        misses it. Instead we measure, per user, the correlation between attributes
        a and b across that user's purchased items, then rank pairs by the
        *variance* of that correlation across users. Interaction pairs have high
        variance; attributes that merely co-vary with linear taste do not.
        """
        per_user_corr: dict[tuple[int, int], list[float]] = {
            (a, b): [] for a in range(n_shared) for b in range(a + 1, n_shared)
        }
        for pur in per_user_purchased:
            xs = pur[:, :n_shared]
            if len(xs) < 3:
                continue
            # Correlation matrix across this user's purchased items.
            std = xs.std(axis=0)
            valid = std > 1e-6
            c = np.corrcoef(xs, rowvar=False)
            for (a, b) in per_user_corr:
                if valid[a] and valid[b] and np.isfinite(c[a, b]):
                    per_user_corr[(a, b)].append(c[a, b])

        scores: list[tuple[float, tuple[int, int]]] = []
        for pair, vals in per_user_corr.items():
            if len(vals) < 5:
                continue
            scores.append((float(np.var(vals)), pair))
        scores.sort(reverse=True)
        return [pair for _, pair in scores[: self.n_edges]]

    # --------------------------------------------------------------- featurizer
    def _features(self, x: np.ndarray, n_shared: int) -> np.ndarray:
        """Map item attribute rows to ``[shared linear | edge interaction]`` features."""
        lin = x[:, :n_shared]
        if not self.edges:
            return lin
        inter = np.stack([lin[:, a] * lin[:, b] for (a, b) in self.edges], axis=1)
        return np.concatenate([lin, inter], axis=1)

    def _std(self, feat: np.ndarray) -> np.ndarray:
        """Center and scale features using the source-catalog statistics."""
        return (feat - self._feat_mean) / self._feat_std

    # ------------------------------------------------------------------ fitting
    def _logistic_fit(self, pos, neg, init, prior, prior_strength) -> np.ndarray:
        """Pointwise logistic regression: purchased (1) vs sampled negatives (0).

        Log-odds is linear in the standardized node+edge features, so the fitted
        coefficients recover a direction proportional to the user's true utility
        weights (linear terms *and* interaction terms). L2 plus a pull toward
        ``prior`` keeps sparse-history fits stable.
        """
        rng = np.random.default_rng(self.seed)
        n_pos = len(pos)
        w = init.astype(float).copy()
        for _ in range(self.epochs):
            neg_idx = rng.integers(0, len(neg), size=n_pos * self.n_negatives)
            neg_s = neg[neg_idx]
            X = np.concatenate([pos, neg_s], axis=0)
            y = np.concatenate([np.ones(n_pos), np.zeros(len(neg_s))])
            p = _sigmoid(X @ w)
            grad = X.T @ (y - p) / len(y)
            grad -= self.l2 * w
            grad -= prior_strength * (w - prior)
            w += self.lr * grad
        return w

    def fit(self, purchased, catalog, n_shared, population: PopulationPrior | None = None):
        assert self.prior_w is not None, "call prepare() before fit()"
        pos = self._std(self._features(purchased, n_shared))
        neg = self._std(self._features(catalog, n_shared))
        w_user = self._logistic_fit(
            pos=pos, neg=neg,
            init=self.prior_w.copy(),
            prior=self.prior_w,
            prior_strength=0.01,
        )
        # Cold-start blend: trust the per-user fit more as history grows. This is
        # the learned-alpha idea in miniature — sparse credentials lean on the
        # population prior, rich ones on the user's own signal.
        n = len(purchased)
        lam = n / (n + self.cold_start_pivot)
        w = lam * w_user + (1.0 - lam) * self.prior_w
        return {"w": w, "n_shared": n_shared}

    def score(self, state, candidates, n_shared, context_mask: np.ndarray | None = None):
        feat = self._std(self._features(candidates, n_shared))
        w = state["w"]
        if context_mask is not None:
            # Context conditioning: suppress nodes (and incident edges) not active
            # in the current query context. Mirrors PTP context conditioners.
            w = w.copy()
            for i in range(n_shared):
                if not context_mask[i]:
                    w[i] = 0.0
            for k, (a, b) in enumerate(self.edges):
                if not (context_mask[a] and context_mask[b]):
                    w[n_shared + k] = 0.0
        return feat @ w
