"""Regression test for the Phase 1 quality-handling finding.

Two robust, falsifiable claims from ``docs/phase1-quality-robustness-results.md``:

1. **Bias–variance crossover.** Raw confidence-weighted sample means beat Bayesian
   shrinkage on *clean* signals, but shrinkage is the noise-robust choice and wins
   once per-observation signals are *noisy* — the realistic regime for messy public
   text. Both ends are significant at this sample size.
2. **Evidence-aware α does not help.** On top of either estimator, the per-candidate
   evidence-aware α does not beat a fixed α — shrinkage + z-scoring already handle
   unreliable evidence.
"""

import pytest

from preferencelayer.agent.ablation import QualityHandlingHarness
from preferencelayer.data import integrated

_CACHE: dict[float, object] = {}


def _run(noise, n_users=200, seed=23):
    if noise not in _CACHE:
        scenario = integrated.generate(
            n_users=n_users, seed=seed, evidence_lo=1, evidence_hi=30, signal_obs_noise=noise)
        _CACHE[noise] = QualityHandlingHarness(scenario, k=10, seed=13, obs_noise=noise).run()
    return _CACHE[noise]


@pytest.fixture(scope="module")
def low():
    return _run(0.2)


@pytest.fixture(scope="module")
def high():
    return _run(1.0)


def test_raw_wins_on_clean_signals(low):
    gain, p = low.contrast("shrunk_fixed", "raw_fixed")
    assert gain < 0 and p < 0.05          # raw averaging beats shrinkage when clean


def test_shrinkage_wins_on_noisy_signals(high):
    gain, p = high.contrast("shrunk_fixed", "raw_fixed")
    assert gain > 0 and p < 0.05          # shrinkage is the noise-robust choice


def test_shrinkage_advantage_grows_with_noise(low, high):
    gain_low, _ = low.contrast("shrunk_fixed", "raw_fixed")
    gain_high, _ = high.contrast("shrunk_fixed", "raw_fixed")
    assert gain_high > gain_low           # the crossover direction


def test_evidence_aware_alpha_does_not_help(low, high):
    # Direction is robust at both noise levels, and significant on clean signals.
    g_low, p_low = low.contrast("shrunk_evidence", "shrunk_fixed")
    assert g_low < 0 and p_low < 0.05
    for res in (low, high):
        assert res.contrast("shrunk_evidence", "shrunk_fixed")[0] <= 0.01
        assert res.contrast("raw_evidence", "raw_fixed")[0] <= 0.01


def test_blend_still_beats_single_signals(high):
    best_blend = max(high.cells["shrunk_fixed"].ndcg, high.cells["raw_fixed"].ndcg)
    assert best_blend > high.references["preference_only"].ndcg
    assert best_blend > high.references["quality_only"].ndcg
