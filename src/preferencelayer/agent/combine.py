"""The α-blend that fuses preference and quality scores.

This is the integration point the whole project builds toward. The preference
graph (``models/``, ``ptp/``) answers *what this user wants*; the Quality
Intelligence Layer (``qil/``) answers *how a product actually performs for this
kind of use*. An agent ranking products needs both, combined into a single
ordering. ``docs/architecture.md`` ("Combined Scoring") specifies exactly how::

    score = alpha * pref_score + (1 - alpha) * quality_score
    alpha = sigmoid(3.0 * (mean_confidence - 0.5))

The key idea is that ``alpha`` is *confidence-adaptive*: when the user's
preference credential is sparse / low-confidence (a cold-start user), ``alpha``
is small and the agent leans on community-derived quality; as the credential
accumulates confidence, ``alpha`` rises and the agent trusts the user's own
learned taste. This module implements those two formulas verbatim, plus the one
modeling decision the doc leaves implicit: the two score streams live on
different scales (preference scores are unbounded logits from ``feat @ w``;
quality scores are posterior means in ``[0, 1]``), so they are standardized
across the candidate set before blending. That normalization is documented and
justified in ``docs/phase1-integration-results.md``.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid for a scalar."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def alpha_from_confidence(mean_confidence: float) -> float:
    """Confidence-adaptive blend weight, exactly as in ``architecture.md``.

    ``alpha = sigmoid(3.0 * (mean_confidence - 0.5))``. At ``mean_confidence ==
    0.5`` this is exactly ``0.5`` (no opinion -> equal weight); it rises toward 1
    as the credential's mean node confidence rises and falls toward 0 as it
    drops. Monotonic in ``mean_confidence``.
    """
    return sigmoid(3.0 * (mean_confidence - 0.5))


def zscore(scores: np.ndarray) -> np.ndarray:
    """Standardize a score stream to mean 0, unit std across the candidate set.

    Preference logits and quality posterior means are not comparable in raw
    units, so each stream is z-scored over the candidate set before blending.
    This makes ``alpha`` a meaningful *relative* weight between two signals
    rather than an accident of their scales. Degenerate (all-equal) streams map
    to all-zeros, contributing nothing to the blend.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    std = scores.std()
    if std < 1e-12:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / std


def blend(pref: np.ndarray, quality: np.ndarray, alpha: float) -> np.ndarray:
    """Combine standardized preference and quality scores with weight ``alpha``.

    ``alpha * z(pref) + (1 - alpha) * z(quality)``. At ``alpha == 1`` this is a
    pure preference ranking; at ``alpha == 0`` a pure quality ranking.
    """
    return alpha * zscore(pref) + (1.0 - alpha) * zscore(quality)
