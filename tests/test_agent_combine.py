"""Tests for the α-blend math (``agent/combine.py``).

These pin the combination to the formula in ``docs/architecture.md`` and the
documented invariants, independent of any data.
"""

import math

import numpy as np

from preferencelayer.agent import combine


def test_alpha_matches_architecture_sigmoid():
    # alpha = sigmoid(3 * (c - 0.5)); exactly 0.5 at c == 0.5.
    assert combine.alpha_from_confidence(0.5) == 0.5
    for c in (0.0, 0.2, 0.37, 0.8, 1.0):
        assert math.isclose(combine.alpha_from_confidence(c), 1.0 / (1.0 + math.exp(-3.0 * (c - 0.5))), rel_tol=1e-9)


def test_alpha_monotonic_in_confidence():
    cs = np.linspace(0.0, 1.0, 25)
    alphas = [combine.alpha_from_confidence(c) for c in cs]
    assert all(b >= a for a, b in zip(alphas, alphas[1:]))
    assert alphas[0] < 0.5 < alphas[-1]


def test_zscore_is_mean_zero_unit_std():
    z = combine.zscore(np.array([1.0, 2.0, 3.0, 10.0]))
    assert math.isclose(z.mean(), 0.0, abs_tol=1e-9)
    assert math.isclose(z.std(), 1.0, rel_tol=1e-9)


def test_zscore_degenerate_stream_is_zeros():
    # An all-equal stream carries no ranking information and must not blow up.
    z = combine.zscore(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(z, 0.0)


def test_blend_reduces_to_single_signals_at_extremes():
    pref = np.array([3.0, 1.0, 2.0, 0.0])
    quality = np.array([0.1, 0.9, 0.5, 0.2])
    # alpha = 1 -> pure preference ranking; alpha = 0 -> pure quality ranking.
    assert np.allclose(combine.blend(pref, quality, 1.0), combine.zscore(pref))
    assert np.allclose(combine.blend(pref, quality, 0.0), combine.zscore(quality))


def test_blend_orders_between_the_two_rankings():
    # Item best on preference but worst on quality should move down as alpha falls.
    pref = np.array([10.0, 0.0, 1.0])
    quality = np.array([0.0, 10.0, 1.0])
    top_pref = np.argmax(combine.blend(pref, quality, 0.9))
    top_qual = np.argmax(combine.blend(pref, quality, 0.1))
    assert top_pref == 0
    assert top_qual == 1


# ---------------------------------------------------------- evidence-aware α
def test_quality_reliability_monotonic_and_bounded():
    e = np.array([0.0, 1.0, 5.0, 20.0, 1000.0])
    r = combine.quality_reliability(e, pivot=8.0)
    assert r[0] == 0.0
    assert np.all((r >= 0.0) & (r < 1.0))
    assert np.all(np.diff(r) > 0)          # rises with evidence
    assert r[-1] > 0.99                    # saturates toward 1


def test_evidence_adaptive_alpha_leans_preference_when_evidence_thin():
    # No quality evidence -> pure preference; rich evidence -> below 1 (uses quality).
    a_thin = combine.evidence_adaptive_alpha(0.5, 0.0)
    a_rich = combine.evidence_adaptive_alpha(0.5, 0.9)
    assert a_thin > 0.999          # ~1.0 (a tiny epsilon guards the division)
    assert a_rich < a_thin
    # Vectorized + clipped into [0, 1].
    a = combine.evidence_adaptive_alpha(0.5, np.array([0.0, 0.3, 0.9]))
    assert a.shape == (3,)
    assert np.all((a >= 0.0) & (a <= 1.0))
    assert a[0] >= a[1] >= a[2]            # falls as quality reliability rises


def test_blend_accepts_a_per_candidate_alpha_vector():
    pref = np.array([1.0, -1.0, 0.5, 2.0])
    quality = np.array([2.0, 0.0, -1.0, 1.0])
    alpha = np.array([0.0, 1.0, 0.5, 0.25])
    expected = alpha * combine.zscore(pref) + (1.0 - alpha) * combine.zscore(quality)
    assert np.allclose(combine.blend(pref, quality, alpha), expected)
