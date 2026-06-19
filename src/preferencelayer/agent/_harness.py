"""Shared setup for the agent evaluation harnesses.

Both :class:`~preferencelayer.agent.evaluate.IntegrationHarness` and
:class:`~preferencelayer.agent.ablation.QualityHandlingHarness` start from the same
step: fit the sparse preference graph's topology/prior on the scenario's purchase
histories. Factor that here so the two harnesses can't drift in how they prepare the
model. (Each then builds its own QualityService(s) — that part genuinely differs.)
"""

from __future__ import annotations

import numpy as np

from ..data.integrated import IntegratedProduct, IntegratedScenario
from ..models.graph import SparsePreferenceGraph


def purchase_matrix(idx: dict[str, IntegratedProduct], purchases: list[str], dim: int) -> np.ndarray:
    """Stack a user's purchased item attributes; ``(0, dim)`` for a zero-history user.

    ``np.stack([])`` raises, so a brand-new user (empty purchase list) needs an
    explicit empty matrix — which flows safely through ``prepare`` (edge discovery
    skips users with too few purchases) and ``fit`` (returns the population prior).
    """
    if purchases:
        return np.stack([idx[pid].attributes for pid in purchases])
    return np.empty((0, dim))


def prepare_preference_model(
    scenario: IntegratedScenario, seed: int = 13, cold_start_pivot: int = 4
) -> tuple[SparsePreferenceGraph, dict[str, IntegratedProduct], np.ndarray, int]:
    """Build and ``prepare`` the preference graph from a scenario.

    Returns ``(model, product_index, catalog_matrix, n_shared)`` — everything the
    per-user loop needs before calling ``model.fit`` for each user.
    """
    idx = scenario.product_index()
    dim = scenario.schema.dim
    n_shared = scenario.schema.n_shared
    _, catalog = scenario.catalog_matrix()
    per_user_purchased = [purchase_matrix(idx, u.purchases, dim) for u in scenario.users]
    model = SparsePreferenceGraph(cold_start_pivot=cold_start_pivot, seed=seed)
    model.prepare(catalog, per_user_purchased, n_shared)
    return model, idx, catalog, n_shared
