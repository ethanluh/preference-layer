import numpy as np
import pytest

from preferencelayer.ptp import (
    AttributeNode,
    BudgetExhausted,
    DPConfig,
    PreferenceCredential,
    PreferenceGraph,
    apply_outcome,
    gaussian_sigma,
    new_user_keypair,
)


def _cred():
    _, did = new_user_keypair(seed=b"1" * 32)
    g = PreferenceGraph(
        category="laptops",
        attributeNodes=[AttributeNode("performance", 0.0, 0.2), AttributeNode("portability", 0.0, 0.2)],
    )
    return PreferenceCredential(did, g)


def test_gaussian_sigma_positive_and_scales_with_epsilon():
    lo = gaussian_sigma(DPConfig(epsilon=8.0))
    hi = gaussian_sigma(DPConfig(epsilon=1.0))
    assert hi > lo > 0  # smaller epsilon -> more noise


def test_outcome_updates_weight_and_metadata():
    cred = _cred()
    rng = np.random.default_rng(0)
    apply_outcome(cred, ["performance"], "purchase", cfg=DPConfig(), rng=rng)
    assert cred.graph.updateCount == 1
    assert cred.graph.privacyBudgetConsumed == pytest.approx(2.0)
    # Confidence must strictly increase toward 1.
    assert cred.graph.attributeNodes[0].confidence > 0.2


def test_purchase_and_return_push_opposite_directions_on_average():
    # Average over many low-noise updates so the DP noise cancels.
    cfg = DPConfig(epsilon=50.0, budget_max=1e9)  # tiny noise for a deterministic-ish check
    rng = np.random.default_rng(42)
    up, down = [], []
    for _ in range(200):
        c = _cred()
        apply_outcome(c, ["performance"], "purchase", cfg=cfg, rng=rng)
        up.append(c.graph.attributeNodes[0].weight)
        c2 = _cred()
        apply_outcome(c2, ["performance"], "return", cfg=cfg, rng=rng)
        down.append(c2.graph.attributeNodes[0].weight)
    assert np.mean(up) > np.mean(down)


def test_budget_exhaustion_raises():
    cred = _cred()
    cfg = DPConfig(epsilon=2.0, budget_max=5.0)
    rng = np.random.default_rng(0)
    apply_outcome(cred, ["performance"], "purchase", cfg=cfg, rng=rng)
    apply_outcome(cred, ["performance"], "purchase", cfg=cfg, rng=rng)
    with pytest.raises(BudgetExhausted):
        apply_outcome(cred, ["performance"], "purchase", cfg=cfg, rng=rng)


def test_rating_outcome_uses_rating_sign():
    cfg = DPConfig(epsilon=50.0, budget_max=1e9)
    rng = np.random.default_rng(7)
    pos = np.mean([
        (lambda c: (apply_outcome(c, ["performance"], "rating", rating=1.0, cfg=cfg, rng=rng),
                    c.graph.attributeNodes[0].weight)[1])(_cred())
        for _ in range(100)
    ])
    neg = np.mean([
        (lambda c: (apply_outcome(c, ["performance"], "rating", rating=0.0, cfg=cfg, rng=rng),
                    c.graph.attributeNodes[0].weight)[1])(_cred())
        for _ in range(100)
    ])
    assert pos > neg
