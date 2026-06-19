"""Non-personalized popularity baseline.

Recommends the items most frequently purchased across the whole population in the
*target* category. It ignores the user's own history entirely, so it carries no
cross-category preference signal — it is the floor any personalized model must
clear to justify itself.
"""

from __future__ import annotations

import numpy as np


class PopularityRecommender:
    name = "popularity"

    def __init__(self, item_ids: list[str], counts: dict[str, int]):
        self.item_ids = item_ids
        self.scores = np.array([counts.get(i, 0) for i in item_ids], dtype=float)

    def score_ids(self, candidate_ids: list[str]) -> np.ndarray:
        idx = {i: k for k, i in enumerate(self.item_ids)}
        return np.array([self.scores[idx[c]] if c in idx else 0.0 for c in candidate_ids])
