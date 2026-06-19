"""Ranking metrics for recommendation evaluation.

All functions take a ranking of candidate item ids (best first) and the set of
ids that are actually relevant for the user. Binary relevance is used throughout;
graded relevance can be supplied to :func:`ndcg_at_k` via ``relevance_map``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of an ordered gain list, truncated at ``k``."""
    total = 0.0
    for i, g in enumerate(gains[:k]):
        # rank i is 0-based; discount uses log2(rank + 2)
        total += g / math.log2(i + 2)
    return total


def ndcg_at_k(
    ranking: Sequence[str],
    relevant: Iterable[str],
    k: int = 10,
    relevance_map: dict[str, float] | None = None,
) -> float:
    """Normalized DCG@k.

    With binary relevance, every item in ``relevant`` has gain 1. If
    ``relevance_map`` is given, it overrides the per-item gain (graded relevance).
    Returns 0.0 when there is no attainable gain.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0

    def gain(item: str) -> float:
        if relevance_map is not None:
            return relevance_map.get(item, 0.0)
        return 1.0 if item in relevant else 0.0

    actual = dcg_at_k([gain(item) for item in ranking], k)

    # Ideal ordering: highest gains first.
    if relevance_map is not None:
        ideal_gains = sorted(relevance_map.values(), reverse=True)
    else:
        ideal_gains = [1.0] * len(relevant)
    ideal = dcg_at_k(ideal_gains, k)

    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranking[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    relevant = set(relevant)
    return 1.0 if any(item in relevant for item in ranking[:k]) else 0.0


def mrr(ranking: Sequence[str], relevant: Iterable[str]) -> float:
    """Mean reciprocal rank of the first relevant item."""
    relevant = set(relevant)
    for i, item in enumerate(ranking):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0
