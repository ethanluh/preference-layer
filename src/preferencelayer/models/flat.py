"""Flat-vector baselines.

Two variants, both standard "mean of purchase history" recommenders:

* :class:`FlatAttributeRecommender` represents the user as the mean attribute
  vector of their purchases (restricted to the shared block so it transfers
  across categories) and scores by cosine similarity. This is a *strong* baseline:
  it recovers the full linear component of a user's taste and transfers it
  perfectly. It cannot represent attribute interactions.

* :class:`FlatItemEmbeddingRecommender` is the baseline named in the design docs
  ("mean of item embeddings in purchase history"). Item embeddings are a
  per-category random projection of attributes — within a category they capture
  taste well, but across categories the bases are not aligned, so transfer is
  weak. This models the real-world fact that one platform's item-embedding space
  does not align with another's.
"""

from __future__ import annotations

import numpy as np

from .base import PopulationPrior, Recommender


def _cosine(user_vec: np.ndarray, items: np.ndarray) -> np.ndarray:
    un = np.linalg.norm(user_vec) + 1e-9
    inorm = np.linalg.norm(items, axis=1) + 1e-9
    return (items @ user_vec) / (inorm * un)


class FlatAttributeRecommender(Recommender):
    name = "flat_attribute"

    def fit(self, purchased, catalog, n_shared, population: PopulationPrior | None = None):
        # Mean over shared attributes, centered so "preference" has signed direction.
        shared = purchased[:, :n_shared]
        catalog_mean = catalog[:, :n_shared].mean(axis=0)
        user_vec = shared.mean(axis=0) - catalog_mean
        return {"user_vec": user_vec}

    def score(self, state, candidates, n_shared):
        cand = candidates[:, :n_shared] - candidates[:, :n_shared].mean(axis=0)
        return _cosine(state["user_vec"], cand)


class FlatItemEmbeddingRecommender(Recommender):
    """Mean-of-item-embeddings baseline with per-category embedding bases."""

    name = "flat_item_embedding"

    def __init__(self, emb_dim: int = 24, seed: int = 0):
        self.emb_dim = emb_dim
        self.seed = seed
        self._bases: dict[int, np.ndarray] = {}

    def _basis(self, dim: int) -> np.ndarray:
        # A deterministic, category-specific random projection. Keyed by the full
        # attribute dim, which differs per category -> bases do not align across
        # categories, exactly the misalignment we want to model.
        if dim not in self._bases:
            rng = np.random.default_rng(self.seed + dim)
            self._bases[dim] = rng.normal(0.0, 1.0, size=(dim, self.emb_dim))
        return self._bases[dim]

    def fit(self, purchased, catalog, n_shared, population: PopulationPrior | None = None):
        basis = self._basis(purchased.shape[1])
        emb = purchased @ basis
        return {"user_emb": emb.mean(axis=0)}

    def score(self, state, candidates, n_shared):
        basis = self._basis(candidates.shape[1])
        cand_emb = candidates @ basis
        return _cosine(state["user_emb"], cand_emb)
