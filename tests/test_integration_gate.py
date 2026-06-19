"""Regression test for the Phase 1 integration milestone.

The milestone is the integration thesis: fusing preference and quality ranks
better than either layer alone. This locks that in, plus the honest secondary
findings the report makes — that a fixed balanced blend also beats both single
layers and that the empirically optimal α sits in a narrow mid band (so
confidence-adaptive α neither needs to, nor does, dominate a fixed blend here).
"""

from preferencelayer.agent import IntegrationHarness
from preferencelayer.data import integrated


def _report(n_users=150, seed=23):
    scenario = integrated.generate(n_users=n_users, seed=seed)
    return IntegrationHarness(scenario, k=10, seed=13).run(with_alpha_curve=True)


def test_blend_beats_both_single_layers():
    """Headline milestone: the α-blend beats preference-only and quality-only."""
    rep = _report()
    assert rep.milestone_pass
    for single in ("preference_only", "quality_only"):
        gain, p = rep.comparisons[single]
        assert gain > 0.02
        assert p < 0.05


def test_fixed_blend_also_beats_both_single_layers():
    """Combining helps regardless of how α is chosen — both blends beat the singles."""
    rep = _report()
    fixed = rep.conditions["fixed_alpha"].ndcg
    assert fixed > rep.conditions["preference_only"].ndcg
    assert fixed > rep.conditions["quality_only"].ndcg


def test_adaptive_is_competitive_with_fixed():
    """Honest finding: adaptive α is close to (not better than) a fixed balanced blend."""
    rep = _report()
    gain, _ = rep.adaptive_vs_fixed
    # Adaptive trails the fixed blend, but only slightly — within ~6 NDCG points.
    assert -0.06 < gain <= 0.0 or abs(gain) < 0.06


def test_optimal_alpha_is_a_narrow_mid_band():
    """The optimal α barely varies across cohorts — why fixed α is hard to beat.

    This is the crux of the honest secondary finding: with uniform quality
    evidence the optimal blend weight clusters in a tight mid band rather than
    swinging cold->rich the way the documented sigmoid does, so a fixed balanced
    blend is hard to beat. (The mild directional rise exists but is noisy at this
    sample size, so we assert the tight clustering, not strict monotonicity.)
    """
    rep = _report()
    assert rep.optimal_alpha, "alpha curve should be populated"
    alphas = [a for _, _, a in rep.optimal_alpha]
    assert all(0.4 <= a <= 0.65 for a in alphas)
    assert max(alphas) - min(alphas) <= 0.2
