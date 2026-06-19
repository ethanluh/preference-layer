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
