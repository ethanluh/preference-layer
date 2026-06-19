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


def quality_reliability(evidence_count, pivot: float = 8.0):
    """How much to trust a quality estimate, from its evidence count.

    Saturating reliability in [0, 1): ``evidence / (evidence + pivot)`` — 0 with no
    evidence, rising toward 1 as observations accumulate. This mirrors the
    cold-start blend weight the preference graph uses for *its* own evidence
    (``models/graph.py``: ``lam = n / (n + pivot)``): a product the QIL has barely
    seen yields an unreliable quality posterior (the Normal-Normal aggregator
    shrinks it toward the neutral prior), exactly as a sparse purchase history
    yields an unreliable preference fit. Accepts a scalar or an array.
    """
    e = np.asarray(evidence_count, dtype=float)
    return e / (e + pivot)


def evidence_adaptive_alpha(mean_confidence: float, quality_reliability, *, quality_weight: float = 1.0):
    """Per-candidate blend weight from *both* reliabilities — the evidence-aware α.

    A **generalization** of :func:`alpha_from_confidence`. The documented formula
    keys α off credential confidence alone, implicitly assuming quality evidence is
    uniformly available; in reality some products are heavily reviewed and others
    barely, so the *quality* estimate's reliability varies per candidate. This
    weights the two estimates by their reliabilities:

        alpha = r_p / (r_p + quality_weight * r_q)

    where ``r_p`` is preference reliability (credential confidence, per user) and
    ``r_q`` is quality reliability (per candidate, from evidence). With no quality
    evidence (``r_q -> 0``) α → 1 and the agent ranks on preference alone; as
    evidence accumulates α falls toward quality, the more so when the credential is
    weak. ``quality_weight`` calibrates how strongly quality, once well-evidenced,
    pulls the blend. Accepts an array ``quality_reliability`` and returns an array
    (one α per candidate); the result is clipped into [0, 1].
    """
    r_p = float(mean_confidence)
    r_q = np.asarray(quality_reliability, dtype=float)
    alpha = r_p / (r_p + quality_weight * r_q + 1e-9)
    return np.clip(alpha, 0.0, 1.0)


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
